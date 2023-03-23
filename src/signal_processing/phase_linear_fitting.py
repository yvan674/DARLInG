"""Phase linear fitting.

Performs a linear fitting on the CSI Phase values as described in `DensePose
From WiFi` (arXiv:2301.00250v1)

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import numpy as np
from math import pi

from signal_processing.base import SignalProcessor


class PhaseLinearFit(SignalProcessor):
    def __init__(self, num_subcarriers: int = 30):
        super().__init__()
        self.num_subcarriers = num_subcarriers
        self.F = num_subcarriers - 1
        self.denominator = 2 * pi * self.F

    def process(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Applies a linear fitting on the phase."""
        a1 = x[:, self.F, :] - x[:, 0, :]
        a1 = np.repeat(a1[:, :, np.newaxis],
                       self.num_subcarriers,
                       axis=2).transpose(0, 2, 1)
        a0 = np.mean(x, axis=1)
        a0 = np.repeat(a0[:, :, np.newaxis],
                       self.num_subcarriers,
                       axis=2).transpose(0, 2, 1)

        mx = x * a1
        return x - mx - a0


if __name__ == '__main__':
    from data_utils import WidarDataset
    from pathlib import Path
    from signal_processing import plot_signals
    from signal_processing.phase_unwrap import PhaseUnwrap
    from signal_processing.phase_filter import PhaseFilter

    data = WidarDataset(Path("../../data/"), "train", True)
    x = data[0][1]
    signal_time_cutoff = 50
    x = x[:signal_time_cutoff, :, :]
    t = np.arange(int(x.shape[1]))

    pu = PhaseUnwrap()
    filt = PhaseFilter([3, 3, 1], [3, 3, 1])
    fit = PhaseLinearFit()
    unwrapped_signal = pu(x)
    filtered_signal = filt(unwrapped_signal)
    fitted_signal = fit(filtered_signal)

    plot_signals(t, x[0, :, 0], fitted_signal[0, :, 0])
