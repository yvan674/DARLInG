"""BVP Dataset.

Loads BVP information from the Widar3.0 Dataset.
"""
from pathlib import Path

import scipy.io
from torch.utils.data import Dataset


class BVPDataset(Dataset):
    def __init__(self, root_path: Path, file_set: set):
        """Dataset of BVP values.

        BVP is represented as a 3-D tensor of values with the first and second
        dimensions being the x and y velocitiy axes and the 3rd dimension
        being the timestamp.

        We return only the BVP value in this dataset.

        Args:
            root_path: Root path of the data CSI data directory
            file_set: The set of files to include in this dataset. Should be a
                set of strings of file paths.
        """
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
