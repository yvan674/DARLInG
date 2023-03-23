"""Pipeline.

Allows for multiple signal processors to be combined into a single pipeline
class.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from typing import List

import numpy as np

from signal_processing.base import SignalProcessor


class Pipeline(SignalProcessor):
    def __init__(self, processors: List[SignalProcessor]):
        """Initializes the pipeline.

        Args:
            processors: List of signal processors to apply.
        """
        super().__init__()
        self.processors = processors

    def process(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Processes the signal.

        Args:
            x: Signal to process.

        Returns:
            Processed signal.
        """
        for p in self.processors:
            x = p(x, **kwargs)
        return x


if __name__ == '__main__':
    from data_utils import WidarDataset
    from pathlib import Path
    from signal_processing import plot_signals
    from signal_processing.lowpass_filter import LowPassFilter
    from signal_processing.phase_unwrap import PhaseUnwrap
    from signal_processing.phase_filter import PhaseFilter
    from signal_processing.phase_linear_fitting import PhaseLinearFit
    from signal_processing.phase_derivative import PhaseDerivative

    data = WidarDataset(Path("../../data/"), "train", True)
    x_csi = data[0][0]
    x_phase = data[0][1]
    signal_time_cutoff = 1952
    x_phase = x_phase[:signal_time_cutoff, :, :]
    x_csi = x_csi[:signal_time_cutoff, :, :]
    t = np.arange(signal_time_cutoff)

    phase_pipeline = Pipeline([PhaseUnwrap(),
                               PhaseFilter([3, 3, 1], [3, 3, 1]),
                               PhaseLinearFit(),
                               LowPassFilter(50, 1000),
                               PhaseDerivative()])

    unwrapped_phase = PhaseUnwrap()(x_phase)[:, 0, 0]
    processed_phase = phase_pipeline(x_phase)[:, 0, 0]

    processed_csi = LowPassFilter(100, 1000)(x_csi)[:, 0, 0]

    plot_signals(t, unwrapped_phase, processed_phase)
    plot_signals(t, x_csi[:, 0, 0], processed_csi)
