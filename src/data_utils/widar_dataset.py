"""Widar3.0 Dataset.

Loads both BVP and CSI information from the Widar3.0 Dataset.
"""
import pickle
from pathlib import Path
from time import perf_counter
from typing import List

import csiread
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset


class WidarDataset(Dataset):
    csi_length = 2048

    def __init__(self, root_path: Path, split_name: str, is_small: bool = False,
                 downsample_multiplier: int = 1, return_bvp: bool = True,
                 return_csi: bool = True):
        """Torch dataset class for Widar3.0.

        Returned values are 4-tuples containing CSI amplitude, CSI phase,
        BVP, gesture.

        BVP is represented as a 3-D tensor of values with the first and second
        dimensions being the x and y velocitiy axes and the 3rd dimension
        being the timestamp.

        Note that there is an oversight in the implementation of the small
        dataset generation script where repetitions are turned into one entry.
        We fix this by counting the number of repetitions and multiplying the
        get_item index by the appropriate amount.

        A single data point in this dataset consists of:
        1) 3-D array of amplitudes with the shape [pn, cn, an, rn] where:
            - pn: packet number (timestamp)
            - cn: Subcarrier channel number [0,...,29]
            - an: antenna number [0, ..., 18]
        2) 3-D array of phase shifts with the same shape.
        3) BVP as 3-D tensor of shape [20, 20, T], where T is the timestep
        4) Information about the sample as a dictionary with keys [`user`,
           `room_num`, `date`, `torso_location`, `face_orientation`, `gesture`]

        1 and 2 has a shape of [n, 30, 18], where n represents the time
        series length, 30 the number of subcarrier channels, and 18 the number
        of receiver antennas. n is 2048 / downsampling_multiplier and was
        reached experimentally as being the power of 2 closest to the mean CSI
        length.

        The CSI data is originally sampled at 1000 Hz.

        Args:
            root_path: Root path of the data directory (e.g., DARLInG/data/)
            split_name: Name of the split this dataset should be. Options are
                [`train`, `validation`, `test_room`, `test_location`]
            is_small: True if this is the small version of the dataset.
            downsample_multiplier: If downsampling is desired, the multiplier
                for downsampling (e.g., 2 means only keep every other sample)
            return_bvp: Whether the BVP should be returned. If False,
                then None is provided as the BVP value. Should make it a lot
                faster to load samples if no BVP is necessary.
            return_csi: Whether the CSI amplitude and phase should be returned.
                If False, then None is provided as the amplitude and phase
                values.
        """
        print(f"Loading dataset {split_name}")
        start_time = perf_counter()
        self.data_path = root_path
        if split_name not in ("train", "validation",
                              "test_room", "test_location"):
            raise ValueError(f"`{split_name}` not one of allowed options for "
                             f"parameter `split_name`")
        self.split_name = split_name
        self.is_small = is_small
        self.ts_length = self.csi_length // downsample_multiplier
        self.downsample_multiplier = downsample_multiplier
        self.return_bvp = return_bvp
        self.return_csi = return_csi

        if is_small:
            data_dir = root_path / "widar_small"
            index_fp = data_dir / f"{split_name}_index_small.pkl"
            self.data_path = data_dir / split_name
        else:
            raise NotImplementedError("Normal sized dataset is not yet "
                                      "implemented.")
        with open(index_fp, "rb") as f:
            index_file: dict[str, any] = pickle.load(f)
            self.data_records: list[dict] = index_file["samples"]
            self.total_samples = index_file["num_total_samples"]
            self.index_to_csi_index = index_file["index_to_csi_index"]

        print(f"Loading complete. Took {perf_counter() - start_time:.2f} s.")

    def _load_csi_file(self, csi_file_path: Path) -> np.ndarray:
        """Copy-pasted from csiread examples. Reads a single CSI file.

         Returns:
             CSI [pn, cn, an, 1] where pn is the packet number, cn the
             subcarrier channel number, and an the antenna number.
         """
        if self.is_small:
            # widar_small moves the files to a different location, so we
            # overwrite it here.
            csi_file_path = self.data_path / csi_file_path.name
        csidata = csiread.Intel(str(csi_file_path), if_report=False)
        csidata.read()
        csi = csidata.get_scaled_csi_sm(True)[:, :, :, :1]
        return csi

    def _load_bvp_file(self, bvp_file_path: Path) -> np.ndarray:
        """Loads a BVP file taking into account if this is a small dataset."""
        if self.is_small:
            # widar_small moves the files to a different location, so we
            # overwrite it here.
            bvp_file_path = self.data_path / bvp_file_path.name
        return loadmat(str(bvp_file_path))["velocity_spectrum_ro"]

    def __str__(self):
        return f"WidarDataset: {self.split_name}"

    def __repr__(self):
        return f"WidarDataset({self.split_name}, {self.data_path}, " \
               f"is_small={self.is_small})"

    def __len__(self):
        return self.total_samples

    def _stack_csi_arrays(self, csi_arrays: List[np.ndarray]) -> np.ndarray:
        """Stacks the CSI arrays according to the stack mode specified."""
        stacked_array = np.zeros((self.csi_length, 30, 18), dtype=complex)

        for i, arr in enumerate(csi_arrays):
            if arr.shape[0] >= self.csi_length:
                s = self.ts_length
            else:
                s = arr.shape[0]
            d = self.downsample_multiplier

            reshaped = arr[:self.csi_length:d, :, :, 0]
            stacked_array[:s, :, i * 3:(i + 1) * 3] = reshaped

        return stacked_array

    def __getitem__(self, item):
        """Gets a single datapoint.

        Returns:
            CSI amplitude, CSI phase, BVP, y
        """
        data_records_index, csi_index = self.index_to_csi_index[item]

        data_record = self.data_records[data_records_index]
        if self.return_csi:
            csi_files = [self._load_csi_file(fp)
                         for fp in data_record[f"csi_paths_{csi_index}"]]
            csi = self._stack_csi_arrays(csi_files)
            amp = np.absolute(csi)
            phase = np.angle(csi)
        else:
            amp, phase = None, None

        if self.return_bvp:
            bvp = self._load_bvp_file(data_record[f"bvp_paths_{csi_index}"])
        else:
            bvp = None

        info = {k: data_record[k]
                for k in ("user", "room_num", "date", "torso_location",
                          "face_orientation", "gesture")}

        return amp, phase, bvp, info


if __name__ == '__main__':
    # This is testing code for the dataset.
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("FP", type=Path,
                   help="Path to the data root.")
    args = p.parse_args()
    d1 = WidarDataset(args.FP, "train", is_small=True)
    print(d1)
    breakpoint()
