"""Widar3.0 Dataset.

Loads both BVP and CSI information from the Widar3.0 Dataset.
"""
from pathlib import Path

from torch.utils.data import Dataset

from data_utils.csi_dataset import CSIDataset
from data_utils.bvp_dataset import BVPDataset


class WidarDataset(Dataset):
    def __init__(self, root_path: Path, file_set: set):
        """Torch dataset class for Widar3.0.

        Returned values are 4-tuples containing CSI amplitude, CSI phase,
        BVP, gesture.

        Args:
            root_path: Root path of the data CSI data directory
            file_set: The set of files to include in this dataset. Should be a
                set of strings of file paths.
        """
        self.csi = CSIDataset(root_path, file_set)
        self.bvp = BVPDataset(root_path, file_set)

    def __getitem__(self, item):
        amp, phase, y = self.csi[item]
        return amp, phase, self.bvp[item], y
