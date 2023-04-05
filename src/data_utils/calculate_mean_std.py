"""Calculate Mean and Standard Deviation.

Calculates the mean and standard deviation of all samples in the training set.
One mean, std pair is calculated each from amplitude shift and phase shift.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from pathlib import Path

import numpy as np
from tqdm import trange

from data_utils import WidarDataset
from signal_processing.phase_unwrap import PhaseUnwrap


def calculate_mean_std(data_dir: Path, out_fp: Path):
    """Calculates the mean and std."""
    dataset = WidarDataset(data_dir, "train", return_bvp=False)

    amp_means = []
    amp_vars = []
    phase_means = []
    phase_vars = []
    pu = PhaseUnwrap()

    for i in trange(len(dataset)):
        amp, phase, _, _ = dataset[i]
        amp_means.append(np.mean(amp))
        amp_vars.append(np.var(amp))

        unwrapped = pu(phase)
        phase_means.append(np.mean(unwrapped))
        phase_vars.append(np.var(unwrapped))

    amp_mean = np.mean(np.array(amp_means))
    amp_std = np.sqrt(np.mean(np.array(amp_vars)))
    phase_means = np.mean(np.array(phase_means))
    phase_std = np.sqrt(np.mean(np.array(phase_vars)))

    # output as csv file with headers
    with open(out_fp, "w") as f:
        f.write("amp_mean,amp_std,phase_mean,phase_std\n")
        f.write(f"{amp_mean},{amp_std},{phase_means},{phase_std}")


if __name__ == "__main__":
    calculate_mean_std(Path("../../data"), Path("../data/mean_std.csv"))
