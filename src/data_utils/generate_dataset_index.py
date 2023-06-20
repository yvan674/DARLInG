"""Generate Dataset Index.

Creates an index of file paths to use for the dataset.

The small dataset is made up of:
- The chosen room, user, and torso locations for the specified split
- 10% stratified random set, stratified based on room id, user id, and gesture
- 2 randomly chosen repetitions for each unique sample

The index is in the form of a dictionary.

The dictionary contains keys [`samples`, `num_total_samples`,
 `index_to_csi_index`].

- `samples` is a list of dictionaries where each dictionary item contains:
    - "user": int
    - "room_num": int
    - "torso_location": int
    - "face_orientation": int
    - "gesture": int
    - "repetitions": int
    - "csi_path_{n}": Path
    - "bvp_path_{n}": Path
    where n is the repetitions number.

- `num_total_samples` is the total number of samples, counting repetitions, in
  the split.

- `index_to_csi_index` is used to figure out which sample index and csi index
  a given index refers to. For example, index 1 is sample 0 and csi 1

  4 index files are saved:
    - training_idx.pkl
    - validation_idx.pkl
    - test_room_idx.pkl
    - test_torso_loc_idx.pkl

`_small` is used as a suffix if not the whole dataset is used.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from warnings import warn
import random
import pickle

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_utils import TRAINING_SELECTION, VALIDATION_SELECTION, \
    TEST_ROOM_SELECTION, TEST_LOCATION_SELECTION, \
    ROOM_DATE_MAPPING, DATE_ROOM_MAPPING, SINGLE_DOMAIN_SELECTION


def parse_args() -> Namespace:
    """Parse command line arguments."""
    p = ArgumentParser()
    p.add_argument("DATA_FP", type=Path,
                   help="Path to the Widar3.0 dataset.")
    p.add_argument("-n", "--num_repetitions", type=int, nargs="?", default=None,
                   help="Number of repetitions to choose for each sample."
                        "Giving none will result in a full dataset index being"
                        "generated.")
    p.add_argument("-s", "--single_domain", action="store_true",
                   help="Set to only get samples of a single domain, in this "
                        "case user 2 in room 1.")

    return p.parse_args()


def find_bvp_of_csi(bvp_dir: Path, sample_record: dict,
                    suppress_warnings: bool = False) -> list[Path]:
    """Finds the corresponding BVP file of a CSI file."""
    bvp_search_path = bvp_dir / f"{sample_record['date']}-VS"
    if sample_record["date"] != "20181130":
        bvp_search_path /= "6-link"

    bvp_search_path /= f'user{sample_record["user"] + 1}'
    bvp_paths = []
    for csi_stem in sample_record["csi_stems"]:
        bvp_path = sorted(bvp_search_path.glob(f"{csi_stem}-*"))
        if len(bvp_path) > 1:
            raise ValueError("Number of matching BVP files is "
                             "greater than one.")
        if len(bvp_path) == 0:
            if not suppress_warnings:
                warn(f"No matching BVP file found for {csi_stem}. Skipping...")
            continue
        bvp_paths.extend(bvp_path)
    return bvp_paths


def list_date_dirs(criteria: dict[str, any], csi_dir: Path) -> list[Path]:
    date_dirs = []
    for room_num in criteria["room_num"]:
        for date in ROOM_DATE_MAPPING[room_num]:
            date_dirs.append(csi_dir / date)
    return date_dirs


def create_glob_str(criteria: dict[str, any]) -> str:
    """Creates the glob string for a search."""
    user_str = "".join(criteria["user"])
    torso_str = "".join(criteria["torso_location"])
    gesture_str = "".join(criteria["gesture"])
    if "face_orientation" in criteria:
        face_str = "".join(criteria["face_orientation"])
        glob_str = f"user[{user_str}]/user[{user_str}]-[{gesture_str}]" \
                   f"-[{torso_str}]-[{face_str}]-?-*"
    else:
        glob_str = f"user[{user_str}]/user[{user_str}]-[{gesture_str}]" \
                   f"-[{torso_str}]-?-*"

    return glob_str


def find_matching_in_date_dirs(dirs_to_search: list[Path],
                               glob_str: str,
                               progress_desc: str) -> dict[str, any]:
    """Finds the corresponding files in a dir.

    Args:
        dirs_to_search: Directories to search.
        glob_str: Glob string to use.
        progress_desc: Description to provide to tqdm
    """
    sample_records = {}
    for dir_to_search in tqdm(dirs_to_search, desc=progress_desc):
        matching_files = dir_to_search.glob(glob_str)
        for file in matching_files:
            file_data = file.stem[4:-3].split("-")[:-1]
            file_str = "".join(file_data)
            # user_dir = Path(f"user{file_data[0]}")

            if file_str in sample_records:
                file_stem = file.stem[:-3]
                if file_stem in sample_records[file_str]["csi_stems"]:
                    continue
                else:
                    sample_records[file_str]["csi_stems"].append(file_stem)
                # sample_records[file_str]["bvp_paths"].append(bvp_file)
            else:
                # csi_stems doesn't include the receiver number
                # This way, we have a list of unique repetition file stems.
                # We also turn all numeric values to ints and make them
                # 0-indexed
                sample_records[file_str] = {
                    "user": int(file_data[0]) - 1,
                    "room_num": DATE_ROOM_MAPPING[dir_to_search.stem],
                    "date": dir_to_search.stem,
                    "torso_location": int(file_data[2]) - 1,
                    "face_orientation": int(file_data[3]) - 1,
                    "gesture": int(file_data[1]) - 1,
                    "csi_stems": [file.stem[:-3]]
                }
    return sample_records


def process_samples(sample_records: dict[str, any],
                    progress_desc: str,
                    num_repetitions: int | None,
                    bvp_dir: Path,
                    csi_dir: Path) -> dict[str, any]:
    """Processes read samples.

    Args:
        sample_records: Dictionary of sample records.
        progress_desc: Description to provide to tqdm.
        num_repetitions: Number of repetitions to use. If None, use all.
        bvp_dir: Directory where BVP files are stored.
        csi_dir: Directory where CSI files are stored.
    """
    keys_to_del = []
    total_samples = 0
    for key in tqdm(sample_records.keys(), desc=progress_desc):
        # Sample files to use
        if num_repetitions is not None:
            chosen_files = random.sample(sample_records[key]["csi_stems"],
                                         num_repetitions)
        else:
            chosen_files = sample_records[key]["csi_stems"]
        sample_records[key]["csi_stems"] = sorted(chosen_files)

        bvp_paths = find_bvp_of_csi(bvp_dir, sample_records[key],
                                    suppress_warnings=True)

        if len(bvp_paths) != len(chosen_files):
            keys_to_del.append(key)
        else:
            user_str = f"user{sample_records[key]['user'] + 1}"
            file_dir = csi_dir / sample_records[key]["date"] / user_str
            repetitions = len(sample_records[key]["csi_stems"])
            sample_records[key]["repetitions"] = repetitions

            for rep in range(repetitions):
                total_samples += 1
                stem = sample_records[key]["csi_stems"][rep]
                sample_records[key][f"csi_paths_{rep}"] = [
                    file_dir / f'{stem}-r{i}.dat'
                    for i in range(1, 7)
                ]
                sample_records[key][f"bvp_paths_{rep}"] = bvp_paths[rep]

    if len(keys_to_del) > 0:
        warn(f"No BVP found for {len(keys_to_del)} files.")

    for key in keys_to_del:
        del sample_records[key]

    index_to_sample_index = []
    samples = list(sample_records.values())

    for sample_idx, sample in enumerate(samples):
        for csi_idx in range(sample["repetitions"]):
            index_to_sample_index.append((sample_idx, csi_idx))

    output_data = {"samples": samples,
                   "num_total_samples": total_samples,
                   "index_to_csi_index": index_to_sample_index}

    return output_data


def parse_single_domain(widar_dir: Path):
    """Select files for a single domain dataset.

    Args:
        widar_dir: Path to the Widar3.0 dataset.

    A list of dicts containing file data is saved. The dictionary has keys:
    [user, room_num, date, torso_location, face_orientation, gesture,
    csi_stems, csi_paths_n, bvp_path_n,].
    There may be csi_paths_0, csi_paths_1, etc. based on num_repetitions.
    """
    csi_dir = widar_dir / "CSI"
    bvp_dir = widar_dir / "BVP"

    criteria = SINGLE_DOMAIN_SELECTION

    date_dirs = list_date_dirs(criteria, csi_dir)

    glob_str = create_glob_str(criteria)

    sample_records = find_matching_in_date_dirs(
        date_dirs, glob_str, "Reading dirs for single-domain"
    )

    output_data = process_samples(sample_records,
                                  "Processing samples for single-domain",
                                  None,
                                  bvp_dir,
                                  csi_dir)

    out = {}
    non_test, out["test"] = train_test_split(output_data["index_to_csi_index"],
                                             test_size=0.2,
                                             random_state=0)
    out["train"], out["validation"] = train_test_split(non_test,
                                                       test_size=0.2,
                                                       random_state=42)

    splits = {"train": {"samples": output_data["samples"],
                        "num_total_samples": len(out["train"]),
                        "index_to_csi_index": out["train"]},
              "validation": {"samples": output_data["samples"],
                             "num_total_samples": len(out["validation"]),
                             "index_to_csi_index": out["validation"]},
              "test": {"samples": output_data["samples"],
                       "num_total_samples": len(out["test"]),
                       "index_to_csi_index": out["test"]}
              }

    for split in splits.keys():
        with open(widar_dir / f"{split}_index_single_domain.pkl", "wb") as f:
            pickle.dump(splits[split], f)


def parse_files(widar_dir: Path, num_repetitions: int | None):
    """Select files based on the selection criteria as a Pandas DataFrame.

    Args:
        widar_dir: Path to the Widar3.0 dataset.
        num_repetitions: Number of repetitions to choose for each sample. If
            None, all samples are used.

    A list of dicts containing file data is saved. The dictionary has keys:
    [user, room_num, date, torso_location, face_orientation, gesture,
    csi_stems, csi_paths_n, bvp_path_n,].
    There may be csi_paths_0, csi_paths_1, etc. based on num_repetitions.
    """
    csi_dir = widar_dir / "CSI"
    bvp_dir = widar_dir / "BVP"

    splits = (("train", TRAINING_SELECTION),
              ("validation", VALIDATION_SELECTION),
              ("test_room", TEST_ROOM_SELECTION),
              ("test_location", TEST_LOCATION_SELECTION))

    for split_name, split in splits:
        # Figure out which date directories to look into
        date_dirs = list_date_dirs(split, csi_dir)

        glob_str = create_glob_str(split)

        sample_records = find_matching_in_date_dirs(
            date_dirs, glob_str, f"Reading dirs for {split_name}"
        )

        output_data = process_samples(sample_records,
                                      f"Processing samples for {split_name}",
                                      num_repetitions,
                                      bvp_dir,
                                      csi_dir)

        suffix = "_small" if num_repetitions is not None else ""

        with open(widar_dir / f"{split_name}_index{suffix}.pkl", "wb") as f:
            pickle.dump(output_data, f)


if __name__ == '__main__':
    random.seed(0)
    args = parse_args()
    if args.single_domain:
        parse_single_domain(args.DATA_FP)
    else:
        parse_files(args.DATA_FP, args.num_repetitions)
