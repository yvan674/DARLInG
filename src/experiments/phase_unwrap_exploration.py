"""Phase Unwrap Exploration.

The purpose of this experiment is to explore how the phase unwrap values are
actually distributed across samples.

Research Questions:
    Are they all roughly monotonically increasing, or is this just x[0]?

    If they are not all monoticially increasing, is there a pattern based on
    gesture?

Answers:
    They are not. Some go up, some go down, some stay around the middle. The
    distribution of final unwrapped phase-shift values is fairly uniform.

    There is no clear pattern of where the phase shift ends based on
    gesture.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from pathlib import Path
import random

from tqdm import tqdm
import numpy as np

from signal_processing import plot_many_signals
from signal_processing.lowpass_filter import LowPassFilter
from signal_processing.phase_derivative import PhaseDerivative
from signal_processing.phase_filter import PhaseFilter
from signal_processing.phase_unwrap import PhaseUnwrap

from data_utils import WidarDataset


def explore_phases(root_path: Path):
    gesture_color_map = {
        0: '#1f77b4',
        1: '#ff7f0d',
        2: '#2ca02c',
        3: '#d62728',
        4: '#9467bd',
        5: '#8c564b',
        6: '#e377c2',
        7: '#7f7f7f',
        8: '#bcbd22',
        9: '#17becf',
    }

    data = WidarDataset(root_path, "train",
                        "single_user_small",
                        return_bvp=False, pregenerated=False)

    sample_idxs = random.sample(range(len(data)), len(data) // 10)

    signals = []
    diff_signals = []
    labels = []
    colors = []

    t_cutoff = 2000
    t = np.arange(t_cutoff)
    pu = PhaseUnwrap()
    filt = PhaseFilter([3, 3, 1], [3, 3, 1])
    lpf = LowPassFilter(25, 1000)
    diff = PhaseDerivative()

    for idx in tqdm(sample_idxs):
        sample = data[idx]
        gesture = sample[3]["gesture"]
        x = sample[1][:t_cutoff, :, :]
        x = lpf(filt(pu(x)))
        signals.append(x[:, 0, 0])

        diff_signals.append(diff(x)[:, 0, 0])
        labels.append(gesture)
        colors.append(gesture_color_map[gesture])

    plot_many_signals(t, signals, labels, colors,
                      "Unwrapped Phase Distribution",
                      show_legend=False)

    plot_many_signals(t, diff_signals, labels, colors,
                      "Derivative of unwrapped phase distribution",
                      show_legend=False)


if __name__ == '__main__':
    explore_phases(Path("../../data"))
