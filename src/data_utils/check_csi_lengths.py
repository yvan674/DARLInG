"""Check CSI Lengths.

Checks the distribution of CSI lengths. Purpose of this script is we want
to either truncate or zero-pad all CSI arrays to the same length. We will use
an array length of 2^n, where 2^n is the closest power of two to the mean
array length. Experimental results shows that n = 11

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import List

from tqdm import tqdm
import csiread
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = ArgumentParser()
    p.add_argument("DATA_FP", type=Path,
                   help="Path to the Widar3.0 directory.")
    return p.parse_args()


def check_csi_lengths(data_fp: Path) -> List[int]:
    lengths = []

    file_list = []
    for split in ("train", "validation", "test_room", "test_location"):
        dir_path = data_fp / split
        file_list.extend(list(dir_path.iterdir()))

    for file in tqdm(file_list):
        if file.name.startswith("user") and file.name.endswith("dat"):
            csidata = csiread.Intel(str(file), if_report=False)
            csidata.read()
            length = csidata.get_scaled_csi_sm(True)[:, :, :, :1].shape[0]
            lengths.append(length)

    return lengths


def save_lengths(data_fp: Path, lengths: List[int]):
    """Saves lengths to a CSV file inside of data_fp."""
    lengths = np.array(lengths).reshape(-1, 1)
    np.savetxt(data_fp / "csi_lengths.csv", lengths, delimiter=",")
    print("Saved lengths to", data_fp / "csi_lengths.csv")


def draw_hist_of_lengths(lengths: List[int]):
    """Draw histograms of the CSI lengths."""
    plt.hist(lengths, bins=100)
    plt.title("Distribution of CSI Lengths")
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    lens = check_csi_lengths(args.DATA_FP)
    save_lengths(args.DATA_FP, lens)
    draw_hist_of_lengths(lens)
