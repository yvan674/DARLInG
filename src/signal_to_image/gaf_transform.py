"""GAF Transform.

Turns the Wi-Fi RSSI Fingerprint into an image using a GAF transformation.
"""
import numpy as np
from pyts.image import GramianAngularField

from signal_to_image.base import SignalToImageTransformer


class GAF(SignalToImageTransformer):
    def __init__(self,
                 image_size: int | float = 0.02,
                 normalize: bool = True,
                 method: str = "summation",
                 overlapping: bool = False,
                 flatten: bool = False):
        super().__init__()
        self.normalize = normalize
        self.gaf = GramianAngularField(image_size=image_size,
                                       method=method,
                                       overlapping=overlapping,
                                       flatten=flatten,)

    def __str__(self):
        return "GramianAngularField()"

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # Input is (time, channels, antennas)
        # gaf requires (n_samples, n_timestamps)
        # We stack the channel and antennas channels and swap axes
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        x = x.swapaxes(0, 1)
        x = self.gaf.transform(x)

        # Min max scaling within sample range
        if self.normalize:
            x = (x - x.min()) / (x.max() - x.min())

        return x
