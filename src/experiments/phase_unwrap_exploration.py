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
from matplotlib import pyplot as plt
import numpy as np

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
                        "full",
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

    # Plot the signals as scatter line charts with a rotated histogram
    # to the right indicating the final distribution of the values at
    # t=t_cutoff, with no legend
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True,
                           gridspec_kw={"width_ratios": [3, 1]})
    plt.subplots_adjust(wspace=0.1)
    fig.set_dpi(300)
    ax[0].set_title("Phase Unwrapped Signals")
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("Phase Unwrapped Value")
    ax[0].set_xlim(0, t_cutoff)

    for signal, label, color in zip(signals, labels, colors):
        ax[0].plot(t, signal, label=label, color=color)

    ax[1].set_title("Final Phase Unwrapped Values")
    ax[1].set_xlabel("Count")
    ax[1].hist([signal[-1] for signal in signals],
               bins=40,
               orientation="horizontal")
    # Change histogram so it has a size of 2x1, where 2 is the height
    # and 1 is the width

    plt.show()


if __name__ == '__main__':
    explore_phases(Path("../../data"))
