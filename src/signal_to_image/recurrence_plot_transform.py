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
                 time_delay: int | float = 50,
                 threshold: any = None,
                 percentage: int | float = 10,
                 flatten: bool = False):
        super().__init__()
        self.rp = RecurrencePlot(dimension, time_delay, threshold, percentage,
                                 flatten)

    def __str__(self):
        return "RecurrencePlotTransform()"

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # Input is (time, channels, antennas)
        # RP requires (n_samples, n_timestamps)
        # We stack the channel and antennas channels and swap axes
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        x = x.swapaxes(0, 1)
        return self.rp.transform(x)
