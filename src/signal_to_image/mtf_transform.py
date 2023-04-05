"""MTS Transform.
Turns the Wi-Fi RSSI Fingerprint into an image using a Markov Transition Field
transformation.
"""
from pathlib import Path

from pyts.image import MarkovTransitionField
from signal_to_image.pyts_transform import pyts_transform


def mtf_transform(data: dict, output_dir: Path):
    """Performs the transformation to an MTF.
    Args:
        data: The data dict produced by the data ingest functions.
        output_dir: Where to save the generated images.
    """
    mtf = MarkovTransitionField()
    pyts_transform(data, mtf, output_dir)


if __name__ == '__main__':
    pass
