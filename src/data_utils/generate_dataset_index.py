"""Generate dataset index.

Generate an index of the entire dataset, so we can quickly analyze dataset
statistics

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm
import pandas as pd

from data_utils import DATE_ROOM_MAPPING, calculate_total_dirs


def parse_args() -> Namespace:
    """Parse command line arguments."""
    p = ArgumentParser()
    p.add_argument("DATA", type=Path,
                   help="Path to the Widar3.0 data dir.")
    p.add_argument("OUT", type=Path,
                   help="Filename of the generated index file, with extension.")

    return p.parse_args()


def extract_dataset_index(widar_dir: Path, out_fp: Path):
    """Parses the dataset to extract an index and puts it in out_fp."""
    csi_dir = widar_dir / "CSI"

    columns = ("date", "user", "gesture", "torso_location", "face_orientation",
               "repetitions", "average_filesize", "room_num")
    index_data = []

    total_dirs = calculate_total_dirs(csi_dir)

    prog_bar = tqdm(total=total_dirs, unit="dir")
    for date_dir in csi_dir.iterdir():
        if not date_dir.is_dir():
            continue
        date_str = date_dir.stem
        room_num = DATE_ROOM_MAPPING[date_dir.stem]
        for user_dir in date_dir.iterdir():
            if not user_dir.is_dir():
                continue
            # We want to keep track of the number of repetitions, so we make
            # a dictionary. It holds as the key the file details as a string.
            # As values, it holds the file_data tuple.
            user_records = {}
            for data_file in user_dir.iterdir():
                if not data_file.is_file():
                    continue
                if not data_file.stem.startswith("user"):
                    # We only process the data files with the correct naming
                    # scheme
                    continue
                # Turn the filename into a list of attributes
                # We remove the 'user' prefix and the receiver number
                file_data = data_file.stem[4:-3].split("-")[:-1]
                file_str = "".join(file_data)
                size = data_file.stat().st_size
                if file_str in user_records:
                    user_records[file_str][5] += 1
                    user_records[file_str][6] += size
                else:
                    user_records[file_str] = [date_str] \
                                             + file_data \
                                             + [1, size, room_num]
            index_data.extend(list(user_records.values()))
            prog_bar.update()

    index_df = pd.DataFrame.from_records(index_data, columns=columns)
    # Divide by 6 since we have 6 receivers and all 6 are considered one sample
    index_df["repetitions"] /= 6
    index_df["repetitions"] = index_df["repetitions"].astype(int)

    index_df["average_filesize"] /= 6
    index_df["average_filesize"] /= 2 ** 20
    index_df.to_csv(widar_dir / out_fp, index=False)

    print(f"Wrote index to {widar_dir / out_fp}.")


if __name__ == '__main__':
    args = parse_args()
    extract_dataset_index(args.DATA, args.OUT)

