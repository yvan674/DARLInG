"""Generate Dataset Index.

Creates an index of file paths to use for the small dataset.

The small dataset is made up of:
- The chosen room, user, and torso locations for the specified split
- 10% stratified random set, stratified based on room id, user id, and gesture
- 2 randomly chosen repetitions for each unique sample

The index is in the form of a list of dictionaries where each dictionary item
contains:
- "user": int
- "room_num": int
- "torso_location": int
- "face_orientation": int
- "gesture": int
- "csi_path": Path
- "bvp_path": Path

4 index files are saved:
- training_idx.pkl
- validation_idx.pkl
- test_room_idx.pkl
- test_torso_loc_idx.pkl

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path
from warnings import warn
from tqdm import tqdm
import random
import pickle

from data_utils import TRAINING_SELECTION, VALIDATION_SELECTION, \
    TEST_ROOM_SELECTION, TEST_LOCATION_SELECTION, \
    ROOM_DATE_MAPPING, DATE_ROOM_MAPPING


def parse_args() -> Namespace:
    """Parse command line arguments."""
    p = ArgumentParser()
    p.add_argument("DATA_FP", type=Path,
                   help="Path to the Widar3.0 dataset.")
    p.add_argument("-n", "--num_repetitions", type=int, nargs="?", default=None,
                   help="Number of repetitions to choose for each sample")

    return p.parse_args()


def find_bvp_of_csi(bvp_dir: Path, sample_record: dict) -> list[Path]:
    """Finds the corresponding BVP file of a CSI file."""
    bvp_search_path = bvp_dir / f"{sample_record['date']}-VS"
    if sample_record["date"] != "20181130":
        bvp_search_path /= "6-link"

    bvp_search_path /= f'user{sample_record["user"]}'
    bvp_paths = []
    for csi_stem in sample_record["csi_stems"]:
        bvp_path = sorted(bvp_search_path.glob(f"{csi_stem}-*"))
        if len(bvp_path) > 1:
            raise ValueError("Number of matching BVP files is "
                             "greater than one.")
        if len(bvp_path) == 0:
            warn(f"No matching BVP file found for {csi_stem}. "
                 f"Skipping...")
            continue
        bvp_paths.extend(bvp_path)
    return bvp_paths


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
    random.seed(0)

    csi_dir = widar_dir / "CSI"
    bvp_dir = widar_dir / "BVP"

    # total_dirs = calculate_total_dirs(csi_dir)
    # prog_bar = tqdm(total=total_dirs, unit="dir")

    splits = (("train", TRAINING_SELECTION),
              ("validation", VALIDATION_SELECTION),
              ("test_room", TEST_ROOM_SELECTION),
              ("test_location", TEST_LOCATION_SELECTION))

    for split_name, split in splits:
        # Figure out which date directories to look into
        date_dirs = []
        for room_num in split["room_num"]:
            for date in ROOM_DATE_MAPPING[room_num]:
                date_dirs.append(csi_dir / date)

        sample_records = {}

        user_str = "".join(split["user"])
        torso_str = "".join(split["torso_location"])
        gesture_str = "".join(split["gesture"])
        glob_str = f"user[{user_str}]/user[{user_str}]-[{gesture_str}]" \
                   f"-[{torso_str}]-?-*"

        for date_dir in tqdm(date_dirs):
            matching_files = date_dir.glob(glob_str)
            for file in matching_files:
                file_data = file.stem[4:-3].split("-")[:-1]
                file_str = "".join(file_data)
                user_dir = Path(f"user{file_data[0]}")

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
                    sample_records[file_str] = {
                        "user": file_data[0],
                        "room_num": DATE_ROOM_MAPPING[date_dir.stem],
                        "date": date_dir.stem,
                        "torso_location": file_data[2],
                        "face_orientation": file_data[3],
                        "gesture": file_data[1],
                        "csi_stems": [file.stem[:-3]]
                    }
        keys_to_del = []
        for key in sample_records.keys():
            # Sample files to use
            if num_repetitions is not None:
                chosen_files = random.sample(sample_records[key]["csi_stems"],
                                             num_repetitions)
            else:
                chosen_files = sample_records[key]["csi_stems"]
            sample_records[key]["csi_stems"] = sorted(chosen_files)

            bvp_paths = find_bvp_of_csi(bvp_dir, sample_records[key])

            if len(bvp_paths) != len(chosen_files):
                keys_to_del.append(key)
            else:
                user_str = f"user{sample_records[key]['user']}"
                file_dir = csi_dir / sample_records[key]["date"] / user_str
                for rep in range(num_repetitions):
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

        sample_list = list(sample_records.values())
        suffix = "_small" if num_repetitions is not None else ""

        with open(widar_dir / f"{split_name}_index{suffix}.pkl", "wb") as f:
            pickle.dump(sample_list, f)


if __name__ == '__main__':
    args = parse_args()
    parse_files(args.DATA_FP, args.num_repetitions)
