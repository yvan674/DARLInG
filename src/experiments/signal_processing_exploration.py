"""Signal Processing Exploration.

The goal of this experiment is to explore which signal processing methods make
sense to employ.

Research Question:
    - Of all the signal processing methods, which makes sense to employ?
    - Of the signal processing methods, are they all functioning properly?

Answers:
    - The methods that make sense are: Unwrap, Filter, Low Pass, Standard
      Scaler, in that order
    - They are all functioning properly.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import numpy as np

from data_utils import WidarDataset
from pathlib import Path
from signal_processing import plot_many_signals
from signal_processing.lowpass_filter import LowPassFilter
from signal_processing.phase_unwrap import PhaseUnwrap
from signal_processing.phase_filter import PhaseFilter
from signal_processing.phase_derivative import PhaseDerivative
from signal_processing.standard_scaler import StandardScaler


if __name__ == '__main__':
    data = WidarDataset(Path("../../data/"), "train", True)
    x_amp = data[0][0]
    x_phase = data[0][1]
    signal_time_cutoff = 2048
    x_phase = x_phase[:signal_time_cutoff, :, :]
    x_amp = x_amp[:signal_time_cutoff, :, :]
    t = np.arange(signal_time_cutoff)

    mean_std_file = Path("/Users/Yvan/Git/DARLInG/data/mean_std.csv")

    pu = PhaseUnwrap()
    std_scalar = StandardScaler(mean_std_file, "phase")
    filt = PhaseFilter([3, 3, 1], [3, 3, 1])
    low_pass = LowPassFilter(250, 1000)

    x_pu = pu(x_phase)
    x_filt = filt(x_pu)
    x_low = low_pass(x_filt)
    x_std = std_scalar(x_low)

    processed_amp = LowPassFilter(100, 1000)(x_amp)[:, 0, 0]

    plot_many_signals(t,
                      [x_phase[:, 0, 0], x_pu[:, 0, 0], x_std[:, 0, 0],
                       x_filt[:, 0, 0], x_low[:, 0, 0]],
                      ["Original", "Unwrapped", "standardized", "filtered",
                       "low pass"],
                      title="Phase processing")

    plot_many_signals(t,
                      [x_std[:, 0, 0], x_filt[:, 0, 0], x_low[:, 0, 0]],
                      ["Standardized", "filtered", "low pass"],
                      title="Phase processsing with standardization")

    plot_many_signals(t,
                      [x_std[:, 0, 0]],
                      ["Standardized"],
                      title="Only standardized")
