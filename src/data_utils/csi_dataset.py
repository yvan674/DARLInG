"""CSI Dataset.

Loads CSI information from the Widar3.0 Dataset.
"""
from pathlib import Path

import csiread
import numpy as np
from torch.utils.data import Dataset


class CSIDataset(Dataset):
    def __init__(self, root_path: Path, file_set: set):
        """Dataset of CSI values.

        CSI is represented as a complex number representation of amplitude and
        phase shift. We convert this directly into 2 4-D arrays, one for
        amplitude and one for phase shift.

        A single data point in this dataset consists of:
        1) 4-D array of amplitudes with the shape [pn, cn, an, rn] where:
            - pn: packet number (timestamp)
            - cn: Subcarrier channel number [0,...,29]
            - an: antenna number [0, 1, 2]
            - rn: receiver number [0,...,5]
        2) 4-D array of phase shifts with the same shape.
        3) The gesture target as an int value [0, 21].


        Args:
            root_path: Root path of the data CSI data directory
            file_set: The set of files to include in this dataset. Should be a
                set of strings of file paths.
        """
        # self.csi, self.rate = self.load_csi(root_path)
        raise NotImplementedError

    @staticmethod
    def load_csi_file(csi_file_path: Path):
        """Copy-pasted from csiread examples. Reads a single CSI file.

         Returns:
             CSI as a ? and the data rate
         """
        csidata = csiread.Intel(str(csi_file_path), if_report=False)
        csidata.read()
        csi = csidata.get_scaled_csi_sm(True)[:, :, :, :1]
        return csi

    def __getitem__(self, index):
        raise NotImplementedError


if __name__ == '__main__':
    csi_dataset = CSIDataset(Path("../../data/20181128/"
                                  "user6/user6-1-1-1-1-r1.dat"))

    breakpoint()
