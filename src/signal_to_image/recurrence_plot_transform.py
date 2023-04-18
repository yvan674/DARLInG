"""Recurrence Plot Transform.

References:
    J.-P Eckmann, S. Oliffson Kamphorst and D Ruelle, “Recurrence Plots of
        Dynamical Systems”. Europhysics Letters (1987).

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import numpy as np
from pyts.image.recurrence import RecurrencePlot

from signal_to_image.base import SignalToImageTransformer


class RP(SignalToImageTransformer):
    def __init__(self,
                 dimension: int | float = 1,
                 time_delay: int | float = 1,
                 threshold: any = None,
                 percentage: int | float = 10,
                 flatten: bool = False):
        super().__init__()
        self.rp = RecurrencePlot(dimension, time_delay, threshold, percentage,
                                 flatten)

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self.rp.transform(x)
