"""BVP Lengths exploration.

Annoyingly, BVP lengths are not all the same. Here we explore the max length
so we can properly pad all BVPs to have the same length.

Research Questions:
    - What is the distribution of BVP lengths?

Answer:
    - There is now a plot in the experimental results dir.
        0: 1
        3: 2
        4: 2
        5: 8
        6: 15
        7: 41
        8: 155
        9: 390
        10: 704
        11: 1143
        12: 1359
        13: 1058
        14: 1096
        15: 1202
        16: 1436
        17: 1668
        18: 1497
        19: 1112
        20: 1007
        21: 1100
        22: 924
        23: 681
        24: 444
        25: 262
        26: 144
        27: 28
        28: 10

Note:
    Since we now pad directly in the WidarDataset class, this doesn't work
    anymore and shows that every BVP has a length of 28.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

from data_utils import WidarDataset


def calc_bvp_lengths(data_fp: Path):
    bvp_lengths = {}
    for split in ("train", "validation", "test_room", "test_location"):
        dataset = WidarDataset(data_fp, split, return_csi=False)
        for _, _, bvp, _ in tqdm(dataset):
            try:
                bvp_len = bvp.shape[2]
                if bvp_len in bvp_lengths:
                    bvp_lengths[bvp_len] += 1
                else:
                    bvp_lengths[bvp_len] = 1
            except AttributeError:
                pass

    return bvp_lengths


def plot_bvp_lengths(bvp_lengths: dict[int, int]):
    x = []
    y = []
    for k, v in bvp_lengths.items():
        x.append(k)
        y.append(v)

    plt.bar(x, y)
    plt.title("BVP Lengths Distribution")
    plt.xlabel("Length")
    plt.ylabel("Num BVPs")
    plt.show()


if __name__ == '__main__':
    bvp_lens = calc_bvp_lengths(Path("../../data"))
    print(bvp_lens)
    plot_bvp_lengths(bvp_lens)
