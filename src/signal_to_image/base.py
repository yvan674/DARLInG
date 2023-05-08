"""Base Signal to Image Transformer.

Base abstract class for all signal-to-image Transformers.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from abc import ABC, abstractmethod

import numpy as np


class SignalToImageTransformer(ABC):
    """Base class for all signal processors."""
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Processes the signal.

        Args:
            x: Signal to process. Shape should always be
                (time, channels, antennas)

        Returns:
            Processed signal.
        """
        raise NotImplementedError

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Calls the process method.

        Args:
            signal: Signal to process.

        Returns:
            Processed signal.
        """
        return self.transform(signal)
