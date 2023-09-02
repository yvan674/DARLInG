"""Widar3.0 Dataset.

Loads both BVP and CSI information from the Widar3.0 Dataset.
"""
import pickle
import warnings
from pathlib import Path
from time import perf_counter
from typing import List, Optional

import csiread
import numpy as np
import torch
from scipy.io import loadmat
from scipy.io.matlab import MatReadError
from torch.utils.data import Dataset

from signal_processing.pipeline import Pipeline


class WidarDataset(Dataset):
    csi_length = 2000  # Tha max lengths we want to use.

    def __init__(self, root_path: Path, split_name: str, dataset_type: str,
                 return_bvp: bool = True,
                 bvp_agg: Optional[str] = None,
                 return_csi: bool = True,
                 amp_pipeline: Pipeline | None = Pipeline([torch.from_numpy]),
                 phase_pipeline: Pipeline | None = Pipeline([torch.from_numpy]),
                 pregenerated: bool | None = None):
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
        1) 3-D array of amplitudes with the shape [pn, cn, an,] where:
            - pn: packet number (timestamp)
            - cn: Subcarrier channel number [0,...,29]
            - an: antenna number (6 receivers, 3 antennas each) [0, ..., 17]
        2) 3-D array of phase shifts with the same shape.
        3) BVP as 3-D tensor of shape [20, 20, T], where T is the timestep
        4) Information about the sample as a dictionary with keys [`user`,
           `room_num`, `date`, `torso_location`, `face_orientation`, `gesture`,
           `csi_fps`]

        1 and 2 has a shape of [n, 30, 18], where n represents the time
        series length, 30 the number of subcarrier channels, and 18 the number
        of receiver antennas. n is 2048 / downsampling_multiplier and was
        reached experimentally as being the power of 2 closest to the mean CSI
        length.

        The CSI data is originally sampled at 1000 Hz.

        All values are returned as torch tensors.

        Args:
            root_path: Root path of the data directory (e.g., DARLInG/data/)
            split_name: Name of the split this dataset should be. Options are
                [`train`, `validation`, `test_room`, `test_location`, `test`]
            dataset_type: Type of the dataset. Options are [`small`,
                `single_domain`, `full`].
            return_bvp: Whether the BVP should be returned. If False,
                then None is provided as the BVP value. Should make it a lot
                faster to load samples if no BVP is necessary.
            bvp_agg: Aggregation method for BVP. Options are
                [`stack`, `1d`, `sum`].
            return_csi: Whether the CSI amplitude and phase should be returned.
                If False, then None is provided as the amplitude and phase
                values.
            amp_pipeline: Pipeline to transform the amplitude shift signal with.
                If None, no transforms are applied.
            phase_pipeline: Pipeline to transform the phase shift signal with.
                If None, non transforms are applied.
            pregenerated: Whether to use pregenerated CSIs. If None, then
                checks first to see if pregenerated data exists. Useful for
                forcing non-pregenerated CSIs.
        """
        print(f"Loading dataset {split_name}")
        start_time = perf_counter()
        self.data_path = root_path
        if split_name not in ("train", "validation",
                              "test_room", "test_location", "test"):
            raise ValueError(f"`{split_name}` not one of allowed options for "
                             f"parameter `split_name`")
        self.split_name = split_name
        self.dataset_type = dataset_type
        # ts_length is the array size returned given the downsample multiplier.
        # self.ts_length = self.csi_length // downsample_multiplier
        self.return_bvp = return_bvp
        self.return_csi = return_csi

        if (bvp_agg is not None) and (bvp_agg not in ("stack", "1d", "sum")):
            raise ValueError(
                f"Parameter bvp_agg is {bvp_agg} but must be one of"
                f"[`stack`, `1d`, `sum`].")
        if (bvp_agg is None) and return_bvp:
            raise ValueError("Parameter bvp_agg must be filled if returning "
                             "a BVP.")
        self.bvp_agg = bvp_agg

        if amp_pipeline is None:
            amp_pipeline = Pipeline([])
        if phase_pipeline is None:
            phase_pipeline = Pipeline([])
        self.amp_pipeline = amp_pipeline
        self.phase_pipeline = phase_pipeline

        self.pregenerated = False

        if dataset_type not in ("small", "single_domain", "single_user",
                                "full", "single_domain_small",
                                "single_user_small"):
            raise ValueError(f"Dataset type {dataset_type} is not one of the"
                             f"possible options [`small`, `single_domain`, "
                             f"`full`].")
        if dataset_type == "full":
            index_fp = self.data_path / f"{split_name}_index.pkl"
        else:
            # Check for pregenerated
            self.pregen_dir = root_path / f"pregenerated_{dataset_type}"
            if pregenerated is None:
                self.pregenerated = self.pregen_dir.exists()
                if self.pregenerated:
                    print("Found pregenerated data dir. Using pregenerated "
                          "data.")
            else:
                if pregenerated:
                    print("Forcing pregenerated data.")
                else:
                    print("Forcing non-pregenerated data.")
                self.pregenerated = pregenerated
            self.pregen_dir /= split_name

            data_dir = root_path / f"widar_{dataset_type}"
            index_fp = data_dir / f"{split_name}_index_{dataset_type}.pkl"
            self.data_path = data_dir / split_name

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
        if self.dataset_type != "full":
            # widar_small moves the files to a different location, so we
            # overwrite it here.
            csi_file_path = self.data_path / csi_file_path.name
        csidata = csiread.Intel(str(csi_file_path), if_report=False)
        csidata.read()
        csi = csidata.get_scaled_csi_sm(True)[:, :, :, :1]
        return csi

    def _load_bvp_file(self, bvp_file_path: Path) -> np.ndarray:
        """Loads a BVP file taking into account if this is a small dataset.

        Notes:
            Behavior is now changed to sum over the time dimension

        Returns:
            The summed (over the time dimension) BVP with shape (1, 20, 20)
        """
        if self.dataset_type != "full":
            # small and single-domain moves the files to a different location,
            # so we overwrite it here.
            bvp_file_path = self.data_path / bvp_file_path.name

        try:
            # Some BVPs are empty mat files or have a length of 0.
            bvp = loadmat(str(bvp_file_path))["velocity_spectrum_ro"]\
                .astype(np.float32)
        except MatReadError:
            pass
            bvp = np.zeros((20, 20, 28), dtype=np.float32)

        match self.bvp_agg:
            case "stack":
                # Reshape to (time, 20, 20)
                bvp = np.moveaxis(bvp, -1, 0)
                out = np.zeros((28, 20, 20), dtype=np.float32)
                time_dim = min(28, bvp.shape[0])
                out[:time_dim, :, :] = bvp[:time_dim]
            case "1d":
                # Move time axis forward
                bvp = np.moveaxis(bvp, -1, 0)
                # Reshape last two dims together
                bvp = bvp.reshape((bvp.shape[0], 400))
                out = np.zeros((1, 28, 400), dtype=np.float32)
                out[0, :bvp.shape[0], :] = bvp
            case "sum":
                bvp = np.sum(bvp, axis=2, dtype=np.float32)
                out = bvp.reshape((1, 20, 20))
            case _:
                raise ValueError("Never should've come here - Bandit from "
                                 "Skyrim")
        return out

    def __str__(self):
        return f"WidarDataset: {self.split_name}" \
               f"({self.dataset_type})"

    def __repr__(self):
        return f"WidarDataset({self.split_name}, {self.data_path}, " \
               f"({self.dataset_type})"

    def __len__(self):
        return self.total_samples

    def _stack_csi_arrays(self, csi_arrays: List[np.ndarray]) -> np.ndarray:
        """Stacks the ragged CSI arrays."""
        stacked_array = np.zeros((self.csi_length, 30, 18),
                                 dtype=complex)

        for i, arr in enumerate(csi_arrays):
            # Get rid of the last dim
            arr = arr[:, :, :, 0]

            if arr.shape[0] >= self.csi_length:
                # If array is longer than our desired array, we cut
                # it off
                cutoff = self.csi_length
            else:
                # Otherwise, the cutoff is the length of the array
                cutoff = arr.shape[0]

            # Fill the stacked array with this array's values in the appropriate
            # antennas channel dims.
            stacked_array[:cutoff, :, i * 3:(i + 1) * 3] = arr[:cutoff]

        return stacked_array

    def __getitem__(self, item):
        """Gets a single datapoint.

        Returns:
            CSI amplitude, CSI phase, BVP, y
        """
        data_records_index, csi_index = self.index_to_csi_index[item]

        data_record = self.data_records[data_records_index]
        csi_fps = data_record[f"csi_paths_{csi_index}"]

        if self.return_bvp:
            bvp = self._load_bvp_file(data_record[f"bvp_paths_{csi_index}"])
            bvp = torch.from_numpy(bvp)
        else:
            bvp = None

        info = {k: data_record[k]
                for k in ("user", "room_num", "date", "torso_location",
                          "face_orientation")}
        info["csi_fps"] = csi_fps

        # Gesture must be in int format and subtracted by 1 to be 0-indexed.
        info["gesture"] = data_record["gesture"]

        if self.return_csi:
            if self.pregenerated:
                file_name = csi_fps[0].name.split("-r")[0] + ".npz"
                data = np.load(self.pregen_dir / file_name)
                amp, phase = data["x_amp"], data["x_phase"]
                amp, phase = torch.tensor(amp), torch.tensor(phase)
            else:
                csi_files = [self._load_csi_file(fp)
                             for fp in csi_fps]
                csi = self._stack_csi_arrays(csi_files)
                # csi = csi.copy()  # reset strides
                amp = np.absolute(csi).astype(np.float32)
                phase = np.angle(csi).astype(np.float32)

                amp = self.amp_pipeline(amp)
                phase = self.phase_pipeline(phase)

            # convert it to float32
            if isinstance(amp, torch.Tensor):
                amp = amp.to(torch.float32)
            if isinstance(phase, torch.Tensor):
                phase = phase.to(torch.float32)

        else:
            amp, phase = None, None

        return amp, phase, bvp, info


if __name__ == '__main__':
    # This is testing code for the dataset.
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("FP", type=Path,
                   help="Path to the data root.")
    args = p.parse_args()
    d1 = WidarDataset(args.FP, "train",
                      dataset_type="single_user_small",
                      return_bvp=False)
    print(d1[0])
    breakpoint()
