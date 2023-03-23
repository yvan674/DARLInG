"""Phase Unwrap.

Unwraps the CSI Phase array.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

"""
import numpy as np

from signal_processing.base import SignalProcessor


class PhaseUnwrap(SignalProcessor):
    def __init__(self):
        """Unwraps the phase of the CSI array."""
        super().__init__()

    def process(self, x: np.ndarray, axis: int = 0) -> np.ndarray:
        """Processes the signal.

        Args:
            x: Signal to process.
            axis: Axis to apply the processing on.

        Returns:
            Processed signal.
        """
        return np.unwrap(x, axis=axis)


if __name__ == '__main__':
    from data_utils import WidarDataset
    from pathlib import Path
    from signal_processing import plot_signals
    data = WidarDataset(Path("../../data/"), "train", True)
    x = data[0][1]
    signal_time_cutoff = 50
    x = x[:signal_time_cutoff, :, :]
    t = np.arange(len(x))

    pu = PhaseUnwrap()
    unwrapped_signal = pu(x)
    plot_signals(t, x[:, 0, 0], unwrapped_signal[:, 0, 0])
