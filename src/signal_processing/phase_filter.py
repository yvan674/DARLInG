"""Phase Filter.

Applies a median and uniform filter on the CSI phase array.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from typing import List

import numpy as np
from scipy.ndimage import uniform_filter, median_filter

from signal_processing.base import SignalProcessor


class PhaseFilter(SignalProcessor):
    def __init__(self, kernel_size: List[int], filter_size: List[int]):
        """Applies a median and uniform filter on the CSI phase array.

        Args:
            kernel_size: List of kernel sizes for each dimension of the median
                filter.
            filter_size: List of filter sizes for each dimension of the uniform
                filter.
        """
        self.kernel_size = kernel_size
        self.filter_size = filter_size

    def process(self, x: np.ndarray,
                mode: str="nearest", **kwargs) -> np.ndarray:
        """Processes the signal.

        Args:
            x: Signal to process.
            mode: Mode for the filters.

        Returns:
            Processed signal.
        """
        x = median_filter(x, size=self.kernel_size, mode=mode)
        x = uniform_filter(x, self.filter_size, mode=mode)
        return x


if __name__ == '__main__':
    from data_utils import WidarDataset
    from pathlib import Path
    from signal_processing import plot_signals
    from signal_processing.phase_unwrap import PhaseUnwrap

    data = WidarDataset(Path("../../data/"), "train", True)
    x = data[0][1]
    signal_time_cutoff = 50
    x = x[:signal_time_cutoff, :, :]
    t = np.arange(len(x))

    pu = PhaseUnwrap()
    filt = PhaseFilter([3, 3, 1], [3, 3, 1])
    unwrapped_signal = pu(x)
    filtered_signal = filt(unwrapped_signal)
    plot_signals(t, unwrapped_signal[:, 0, 0], filtered_signal[:, 0, 0])