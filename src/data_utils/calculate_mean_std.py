"""Calculate Mean and Standard Deviation.

Calculates the mean and standard deviation of all samples in the training set.
One mean, std pair is calculated each from amplitude shift and phase shift.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import trange

from data_utils import WidarDataset
from signal_processing.phase_unwrap import PhaseUnwrap
from signal_processing.pipeline import Pipeline


def parse_args():
    p = ArgumentParser()
    p.add_argument("DATA", type=Path, default=Path("../../data"),
                   help="Path to the root data dir")
    return p.parse_args()


def calculate_mean_std(data_dir: Path, out_fp: Path):
    """Calculates the mean and std."""
    dataset = WidarDataset(data_dir, "train", return_bvp=False,
                           dataset_type="full",
                           amp_pipeline=Pipeline([lambda x: x]),
                           phase_pipeline=Pipeline([PhaseUnwrap()]))

    amp_means = []
    amp_vars = []
    phase_means = []
    phase_vars = []

    for i in trange(len(dataset)):
        amp, phase, _, _ = dataset[i]
        amp_means.append(np.mean(amp))
        amp_vars.append(np.var(amp))

        phase_means.append(np.mean(phase))
        phase_vars.append(np.var(phase))

    amp_mean = np.mean(np.array(amp_means))
    amp_std = np.sqrt(np.mean(np.array(amp_vars)))
    phase_means = np.mean(np.array(phase_means))
    phase_std = np.sqrt(np.mean(np.array(phase_vars)))

    # output as csv file with headers
    with open(out_fp, "w") as f:
        f.write("amp_mean,amp_std,phase_mean,phase_std\n")
        f.write(f"{amp_mean},{amp_std},{phase_means},{phase_std}")


if __name__ == "__main__":
    args = parse_args()
    calculate_mean_std(args.DATA, args.DATA / "mean_std.csv")
