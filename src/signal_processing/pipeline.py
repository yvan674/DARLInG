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
