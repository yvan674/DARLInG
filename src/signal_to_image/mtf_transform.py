"""MTF Transform.
Turns the Wi-Fi RSSI Fingerprint into an image using a Markov Transition Field
transformation.
"""
import numpy as np
from pyts.image import MarkovTransitionField

from signal_to_image.base import SignalToImageTransformer


class MTF(SignalToImageTransformer):
    def __init__(self,
                 image_size: int | float = 0.05,
                 n_bins: int = 5,
                 strategy: str = "quantile",
                 overlapping: bool = False,
                 flatten: bool = False):
        super().__init__()
        self.mtf = MarkovTransitionField(image_size, n_bins, strategy,
                                         overlapping, flatten)

    def __str__(self):
        return "MarkovTransitionField()"

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # Input is (time, channels, antennas)
        # MTF requires (n_samples, n_timestamps)
        # We stack the channel and antennas channels and swap axes
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        x = x.swapaxes(0, 1)
        return self.mtf.transform(x)
