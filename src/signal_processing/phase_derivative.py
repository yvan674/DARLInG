"""Phase derivative.

Gets the derivative of the phase to avoid an exploding unwrapped phase value.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import numpy as np

from signal_processing.base import SignalProcessor


class PhaseDerivative(SignalProcessor):
    def __init__(self):
        """Calculate the derivative of the phase."""
        super().__init__()

    def process(self, x: np.ndarray, axis: int = 0, **kwargs) -> np.ndarray:
        """Processes the signal.

        Args:
            x: Signal to process.
            axis: Axis to apply the processing on.

        Returns:
            Processed signal.
        """
        return np.gradient(x, axis=axis)


if __name__ == '__main__':
    from data_utils import WidarDataset
    from pathlib import Path
    from signal_processing import plot_signals
    from signal_processing.lowpass_filter import LowPassFilter
    from signal_processing.phase_unwrap import PhaseUnwrap
    from signal_processing.phase_filter import PhaseFilter
    from signal_processing.phase_linear_fitting import PhaseLinearFit

    data = WidarDataset(Path("../../data/"), "train", True)
    x = data[0][1]
    signal_time_cutoff = 100
    x = x[:signal_time_cutoff, :, :]
    t = np.arange(int(x.shape[0]))

    pu = PhaseUnwrap()
    filt = PhaseFilter([3, 3, 1], [3, 3, 1])
    fit = PhaseLinearFit()
    diff = PhaseDerivative()
    lpf = LowPassFilter(100, 1000)
    unwrapped_signal = pu(x)
    plot_signals(t, x[:, 0, 0], unwrapped_signal[:, 0, 0], "Signal Unwrapping",
                 "Raw CSI Phase", "Unwrapped CSI Phase")
    filtered_signal = filt(unwrapped_signal)
    plot_signals(t, unwrapped_signal[:, 0, 0], filtered_signal[:, 0, 0],
                 "Mode/Uniform Filtering", "Unwrapped CSI", "Filtered CSI")
    lowpass_signal = lpf(filtered_signal)
    plot_signals(t, filtered_signal[:, 0, 0], lowpass_signal[:, 0, 0],
                 "Low Pass Filter", "Filtered CSI", "Low pass CSI")
    gradient_signal = diff(lowpass_signal)
    plot_signals(t, lowpass_signal[:, 0, 0], gradient_signal[:, 0, 0],
                 "Derivative", "Low pass CSI", "Derivative CSI")

    lowpass_raw = lpf(x)

    plot_signals(t, lowpass_raw[:, 0, 0], gradient_signal[:, 0, 0],
                 "Derivative", "Low Pass Raw CSI Phase", "Derivative CSI")
