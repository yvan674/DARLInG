"""GAF Transform.

Turns the Wi-Fi RSSI Fingerprint into an image using a GAF transformation.
"""
import numpy as np
from pyts.image import GramianAngularField

from signal_to_image.base import SignalToImageTransformer


class GAF(SignalToImageTransformer):
    def __init__(self,
                 image_size: int | float = 1.,
                 sample_range: tuple | None = (-1, 1),
                 method: str = "summation",
                 overlapping: bool = False,
                 flatten: bool = False):
        super().__init__()
        self.gaf = GramianAngularField(image_size=image_size,
                                       sample_range=sample_range,
                                       method=method,
                                       overlapping=overlapping,
                                       flatten=flatten)

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self.gaf.transform(x)
