"""GAF Transform.

Turns the Wi-Fi RSSI Fingerprint into an image using a GAF transformation.
"""
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from pyts.image import GramianAngularField


def gaf_transform(data: dict, output_dir: Path):
    """Performs the transformation to a GAF.

    The pyts package is used since it has a (presumably) optimized version of
    this transformation ready to use.

    Args:
        data: The data dict produced by the data ingest functions.
        output_dir: Where to save the generated images.
    """
    gaf = GramianAngularField()
    # pyts_transform(data, gaf, output_dir)


if __name__ == '__main__':
    from data_utils.widar_dataset import WidarDataset
    data_path = Path("/Users/Yvan/Git/DARLInG/data/widar_small")
    gaf = GramianAngularField()
    scalar = StandardScaler()
    dataset = WidarDataset(Path("/Users/Yvan/Git/DARLInG/data"), "train",
                           True)
    for i in dataset:
        pass

