"""Prepare REFINED.

Generate the mappings for REFINED.
REFINED requires mappings of features to pixels to be pre-calculated.
"""
from argparse import ArgumentParser
from pathlib import Path

from tqdm import trange
import numpy as np

from data_utils import WidarDataset
from signal_to_image.refined.initial_mds import mds_transform
from signal_to_image.refined.hill_climb import hill_climb_master


def parse_args():
    p = ArgumentParser()
    p.add_argument("DATA_FP", type=Path,
                   help="Path to the Widar dataset")
    return p.parse_args()


def precalculate_mappings(dataset: WidarDataset, output_dir: Path,
                          num_iters: int = 5):
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Reading dataset...")
    arrs = [[] for _ in range(18)]
    for i in trange(len(dataset)):
        data = dataset[i]
        for ann_num in range(18):
            arrs[ann_num].append(data[:, :, ann_num])

    print("Running MDS reduction...")
    for i in trange(18):
        mds_transform(np.concatenate(arrs[i], axis=0),
                      output_dir / f"mds_antenna_{i}")

    for i in trange(18):
        print(f"Running hill climb for antenna {i}")
        hill_climb_master(output_dir, output_dir / f"mds_antenna_{i}",
                          i, num_iters)


if __name__ == '__main__':
    args = parse_args()
    d = WidarDataset(args.DATA_FP, "train", True, 4)
    precalculate_mappings(d, args.DATA_FP)
