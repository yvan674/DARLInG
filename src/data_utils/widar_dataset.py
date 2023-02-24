"""Widar3.0 Dataset.

Loads both BVP and CSI information from the Widar3.0 Dataset.
"""
import random
from pathlib import Path
from typing import List

import csiread
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

from data_utils import ROOM_DATE_MAPPING


class WidarDataset(Dataset):
    def __init__(self, root_path: Path, rooms: List[int], users: List[str],
                 random_seed: int = 0):
        """Torch dataset class for Widar3.0.

        Returned values are 4-tuples containing CSI amplitude, CSI phase,
        BVP, gesture.

        BVP is represented as a 3-D tensor of values with the first and second
        dimensions being the x and y velocitiy axes and the 3rd dimension
        being the timestamp.

        A single data point in this dataset consists of:
        1) 4-D array of amplitudes with the shape [pn, cn, an, rn] where:
            - pn: packet number (timestamp)
            - cn: Subcarrier channel number [0,...,29]
            - an: antenna number [0, 1, 2]
            - rn: receiver number [0,...,5]
        2) 4-D array of phase shifts with the same shape.
        3) BVP as 3-D tensor of shape [20, 20, T], where T is the timestep
        4) The gesture target as an int value [0, 21].

        Args:
            root_path: Root path of the data directory
            rooms: The rooms numbers to include.
            users: The users IDs to include. should be in the format `userX`.
            random_seed: Seed used for shuffling the dataset.
        """
        # First build the list of all CSI files that we have
        csi_dirs = []
        self.bvp_dir = root_path / "BVP"  # Used to find corresponding BVP file
        for room in rooms:
            for dir_fp in ROOM_DATE_MAPPING[room]:
                csi_dirs.append(root_path / "CSI" / dir_fp)

        # Calculate total subdirs to go through until user_dir
        csi_user_dirs = [user_dir
                         for user_dir in [list(csi_date_dir.iterdir())
                                          for csi_date_dir in csi_dirs]]


        # Then goes through every file in each dir and checks whether we want
        # to read it or not based on the user ID
        # csi_files is a dictionary mappping of identifiers to all 6 files with
        # that identifier (6 for the 6 receivers)
        self.csi_files = {}
        self.bvp_files = {}
        self.gesture = {}
        for csi_date_dir in csi_dirs:
            for csi_user_dir in csi_date_dir.iterdir():
                if csi_user_dir.stem.startswith("."):
                    # Ignore system files
                    continue
                if csi_user_dir.stem in users:
                    for csi_file in csi_user_dir.iterdir():
                        if csi_file.stem.startswith("."):
                            # Ignore system files
                            continue
                        # identifier is something like user6-1-1-1-1
                        identifier = csi_date_dir.stem + csi_file.stem[:-3]
                        try:
                            bvp_file = self._get_corresponding_bvp_fp(csi_file)
                        except FileNotFoundError:
                            continue
                        if identifier in self.csi_files:
                            self.csi_files[identifier].append(csi_file)
                        else:
                            self.csi_files[identifier] = [csi_file]
                            self.bvp_files[identifier] = bvp_file
                            self.gesture[identifier] = self._gesture(csi_file)
        self.identifiers = sorted(list(self.csi_files.keys()))
        random.Random(random_seed).shuffle(self.identifiers)

    def _get_corresponding_bvp_fp(self, fp: Path) -> Path:
        """Finds the corresponding BVP file from a CSI file path."""
        date_dir = self.bvp_dir / (fp.parents[1].stem + "-VS")
        user_dir = date_dir / "6-link" / fp.parent.stem
        for bvp_file in user_dir.iterdir():
            if bvp_file.stem.startswith(fp.stem[:-3]):
                return bvp_file
        raise FileNotFoundError

    @staticmethod
    def _gesture(fp: Path):
        """Gets the gesture performed from a CSI file path."""
        return int(fp.stem[fp.stem.index("-") + 1])

    @staticmethod
    def _load_csi_file(csi_file_path: Path) -> np.ndarray:
        """Copy-pasted from csiread examples. Reads a single CSI file.

         Returns:
             CSI [pn, cn, an, 1] where pn is the packet number, cn the
             subcarrier channel number, and an the antenna number.
         """
        csidata = csiread.Intel(str(csi_file_path), if_report=False)
        csidata.read()
        csi = csidata.get_scaled_csi_sm(True)[:, :, :, :1]
        return csi

    def __getitem__(self, item):
        """Gets a single datapoint.

        Returns:
            CSI amplitude, CSI phase, BVP, y
        """
        identifier = self.identifiers[item]
        csi_files = [self._load_csi_file(fp)
                     for fp in self.csi_files[identifier]]
        csi = np.stack(csi_files, axis=3)
        amp = np.absolute(csi)
        phase = np.angle(csi)
        y = self.gesture[identifier]
        bvp = loadmat(str(self.bvp_files[identifier]))

        return amp, phase, bvp, y


if __name__ == '__main__':
    # This is testing code for the dataset.
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("FP", type=Path,
                   help="Path to the data root.")
    args = p.parse_args()
    d1 = WidarDataset(args.FP, [1], ["user1"])
    d2 = WidarDataset(args.FP, [1, 2], ["user1", "user6"])
    d3 = WidarDataset(args.FP, [3], ["user3", "user7", "user8", "user9"])

