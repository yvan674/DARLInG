"""MTS Transform.
Turns the Wi-Fi RSSI Fingerprint into an image using a Markov Transition Field
transformation.
"""
import numpy as np
from pyts.image import MarkovTransitionField

from signal_to_image.base import SignalToImageTransformer


class MTF(SignalToImageTransformer):
    def __init__(self,
                 image_size: int | float = 1.,
                 n_bins: int = 5,
                 strategy: str = "quantile",
                 overlapping: bool = False,
                 flatten: bool = False):
        super().__init__()
        self.mtf = MarkovTransitionField(image_size, n_bins, strategy,
                                         overlapping, flatten)

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self.mtf.transform(x)
