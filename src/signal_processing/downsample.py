"""Downsample.

Performs the simplest downsample, i.e., nearest-neighbor, of a signal.

Author:
    Yvan Satyawan.
"""
import numpy as np

from signal_processing.base import SignalProcessor


class Downsample(SignalProcessor):
    def __init__(self, downsample_multiplier: int):
        """Downsample using striding."""
        super().__init__()
        self.ds_multiplier = downsample_multiplier

    def process(self, x: np.ndarray, axis: int = 0, **kwargs) -> np.ndarray:
        """Processes the signal.

        Args:
            x: Signal to process,
            axis: Axis to apply the processing on.

        Returns:
            Processed signal.
        """
        # First build the slice using np.s_ syntax, figuring out which axis
        # to perform the downsample on.
        s = np.s_[::self.ds_multiplier]
        if axis != 0:
            s = tuple([s if i == axis else np.s_[:]
                       for i in range(x.ndim)])

        return x[s]


def interpolate_between_points(signal: np.ndarray,
                               ds_multiplier: int) -> np.ndarray:
    """Interpolates between downsampled points for plotting.

    Spaces out filtered signals so that the length is also 2000 using linear
    interpolation between data points.
    """
    interpolated = np.zeros(2000)
    for i in range(signal.shape[0] - 1):
        try:
            interpolated[i * 100:(i + 1) * 100] = np.linspace(
                signal[i, 0, 0], signal[i + 1, 0, 0],
                100, endpoint=False
            )
        except IndexError:
            interpolated[i * 100:] = np.repeat(signal[i, 0, 0],
                                               100)
    return interpolated


if __name__ == '__main__':
    from signal_processing import plot_signals
    fs = 1000

    # Try this out with the Widar dataset data
    from data_utils import WidarDataset
    from pathlib import Path
    data = WidarDataset(Path("../../data/"), "train", "single_user_small",
                        return_bvp=False)
    x = data[0][0]
    signal_time_cutoff = 2000
    x = x[:signal_time_cutoff]
    t = np.arange(len(x))

    ds = Downsample(100)
    filtered_signal = ds(x)
    interpolated_signal = interpolate_between_points(filtered_signal,
                                                     100)



    plot_signals(t, x[:, 0, 0], interpolated_signal, "Downsample",
                 "Original", "Downsampled")
