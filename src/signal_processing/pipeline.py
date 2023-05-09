"""Pipeline.

Allows for multiple signal processors to be combined into a single pipeline
class.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from pathlib import Path
from typing import Union

import numpy as np
import torch

from signal_processing.base import SignalProcessor
from signal_to_image.base import SignalToImageTransformer
from signal_processing.lowpass_filter import LowPassFilter
from signal_processing.phase_derivative import PhaseDerivative
from signal_processing.phase_unwrap import PhaseUnwrap
from signal_processing.phase_filter import PhaseFilter
from signal_processing.standard_scaler import StandardScaler


class Pipeline(SignalProcessor):
    def __init__(self, processors: list[Union[SignalProcessor,
                                        SignalToImageTransformer,
                                        callable]]):
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

    @staticmethod
    def from_str_list(str_list: list[str],
                      transform: Union[SignalProcessor,
                                       SignalToImageTransformer,
                                       callable],
                      standard_scaler: StandardScaler):
        pipeline = []
        for s in str_list:
            match s:
                case "lowpass_filter":
                    pipeline.append(LowPassFilter(250, 1000))
                case "phase_derivative":
                    pipeline.append(PhaseDerivative())
                case "phase_filter":
                    pipeline.append(PhaseFilter([3, 3, 1], [3, 3, 1]))
                case "phase_unwrap":
                    pipeline.append(PhaseUnwrap())
                case "torch.from_numpy":
                    pipeline.append(torch.from_numpy)
                case "standard_scalar":
                    pipeline.append(standard_scaler)
                case "transform":
                    pipeline.append(transform)

        return Pipeline(pipeline)
