"""Generate Small Dataset Index.

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
from tqdm import tqdm
import pandas as pd
import pickle

from data_utils import calculate_total_dirs


def parse_args() -> Namespace:
    """Parse command line arguments."""
    p = ArgumentParser()
    p.add_argument("DATA_FP", type=Path,
                   help="Path to the Widar3.0 dataset.")

    return p.parse_args()


def parse_files(widar_dir: Path) -> pd.DataFrame:
    """Select files based on the selection criteria as a Pandas DataFrame.

    Args:
        widar_dir: Path to the Widar3.0 dataset.

    Returns:
        A dataframe containing file data. The dataframe has columsn:
        [user, room_nu, torso_location, face_orientation, gesture, repetition,
        file_stem, csi_path, bvp_path].
        file_stem does not contain receiver number and repetition number. This,
        way we can sample using a groupby on file_stem for repetitions.
    """
    csi_dir = widar_dir / "CSI"
    bvp_dir = widar_dir / "BVP"

    total_dirs = calculate_total_dirs(csi_dir)
    prog_bar = tqdm(total=total_dirs, unit="dir")



