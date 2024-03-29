"""Base Signal Processing.

Base abstract class for all signal processors.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from abc import ABC, abstractmethod

import numpy as np


class SignalProcessor(ABC):
    """Base class for all signal processors."""
    def __init__(self):
        pass

    @abstractmethod
    def process(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Processes the signal.

        Args:
            x: Signal to process.

        Returns:
            Processed signal.
        """
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Calls the process method.

        Args:
            signal: Signal to process.

        Returns:
            Processed signal.
        """
        return self.process(x)
