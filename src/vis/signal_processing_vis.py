"""Signal Processing.

Contains visualization code for the signal processing process.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
    GitHub Copilot.
"""
from pathlib import Path

import numpy as np

from signal_processing import plot_many_signals
from signal_processing.lowpass_filter import LowPassFilter
from signal_processing.standard_scaler import StandardScaler
from signal_processing.downsample import Downsample, interpolate_between_points
from signal_processing.phase_unwrap import PhaseUnwrap
from signal_processing.phase_filter import PhaseFilter
from signal_processing.pipeline import Pipeline
from data_utils import WidarDataset


def visualize():
    # Prep data
    data_dir = Path("../../data")
    data = WidarDataset(Path("../../data/"),
                        "train",
                        "single_user_small",
                        False,
                        None,
                        True,
                        Pipeline([]),
                        Pipeline([]),
                        False)

    amp, phase = data[0][:2]
    co = 600  # cutoff
    amp = amp[:2000]
    phase = phase[:2000]
    t = np.arange(co)

    # Prep preprocessing steps
    lpf = LowPassFilter(250, 1000)
    ss_amp = StandardScaler(data_dir, "amp")
    ds = Downsample(100)

    ss_phase = StandardScaler(data_dir, "phase")
    pu = PhaseUnwrap()
    pf = PhaseFilter([3, 3, 1], [3, 3, 1])

    # Make save directory
    save_dir = Path("../../figures")
    save_dir.mkdir(exist_ok=True)
    # Remove all contents in dir
    for file in save_dir.iterdir():
        file.unlink()

    # Plot original data
    plot_many_signals(t, [amp[:co, 0, 0]], ["Amplitude signal"], None,
                      None, False, True,
                      Path("../../figures/amp_original.png"))
    plot_many_signals(t, [phase[:co, 0, 0]], ["Phase signal"], None,
                      None, False, True,
                      Path("../../figures/phase_original.png"))

    # Plot amplitude steps
    amp_lpf = lpf(amp)
    amp_ss = ss_amp(amp_lpf)
    amp_ds = ds(amp_ss)
    amp_ds = interpolate_between_points(amp_ds, 100)
    plot_many_signals(t, [amp[:co, 0, 0], amp_lpf[:co, 0, 0]],
                      ["Original signal", "Low pass filtered"],
                      None, None, True,
                      True,
                      Path("../../figures/amp_step_1.png"))
    plot_many_signals(t, [amp_lpf[:co, 0, 0],
                          amp_ss[:co, 0, 0]],
                      ["Low pass filtered", "Standard scalar filtered"],
                      None, None, True, True,
                      Path("../../figures/amp_step_2.png"))
    plot_many_signals(t, [amp_ss[:co, 0, 0], amp_ds[:co]],
                      ["Standard scalar filtered", "Downsampled"],
                      None, None, True, True,
                      Path("../../figures/amp_step_3.png"))

    # Plot phase steps
    phase_unwrapped = pu(phase)
    phase_filtered = pf(phase_unwrapped)
    phase_lpf = lpf(phase_filtered)
    phase_ss = ss_phase(phase_lpf)
    phase_ds = ds(phase_ss)
    phase_ds = interpolate_between_points(phase_ds, 100)

    plot_many_signals(t, [phase[:co, 0, 0],
                          phase_unwrapped[:co, 0, 0]],
                      ["Original signal", "Unwrapped phase"],
                      None, None, True, True,
                      Path("../../figures/phase_step_1.png"))
    plot_many_signals(t, [phase_unwrapped[:co, 0, 0],
                          phase_filtered[:co, 0, 0]],
                      ["Unwrapped phase", "Filtered phase"],
                      None, None, True, True,
                      Path("../../figures/phase_step_2.png"))
    plot_many_signals(t, [phase_filtered[:co, 0, 0], phase_lpf[:co, 0, 0]],
                      ["Filtered phase", "Low pass filtered"],
                      None, None, True, True,
                      Path("../../figures/phase_step_3.png"))
    plot_many_signals(t, [phase_lpf[:co, 0, 0], phase_ss[:co, 0, 0]],
                      ["Low pass filtered", "Standard scalar filtered"],
                      None, None, True, True,
                      Path("../../figures/phase_step_4.png"))
    plot_many_signals(t, [phase_ss[:co, 0, 0], phase_ds[:co]],
                      ["Standard scalar filtered", "Downsampled"],
                      None, None, True, True,
                      Path("../../figures/phase_step_5.png"))


if __name__ == '__main__':
    visualize()
