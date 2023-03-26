"""REFINED.

Taken from the Refined repository on GitHub with _significant_ changes.

References:
    Bazgir, O., Zhang, R., Dhruba, S.R. et al. Representation of features as
        images with neighborhood dependencies for compatibility with
        convolutional neural networks. Nat Commun 11, 4391 (2020).
        https://doi.org/10.1038/s41467-020-18197-y.
    REFINED repository on GitHub. <https://github.com/omidbazgirTTU/REFINED>.

Author:
    Yvan Satyawan <y.p.satyawan@student.tue.nl>
"""
import math

from signal_to_image.refined.Toolbox import REFINED_Im_Gen
from signal_to_image.refined.initial_mds import mds_transform
from signal_to_image.refined.hill_climb import hill_climb_master

from pathlib import Path
import pickle
from PIL import Image
import numpy as np


def refined_transform(data: dict, output_dir: Path, num_iters: int = 5):
    """Performs the refined transform introduced in Bazgir et al."""
    output_dir.mkdir(exist_ok=True, parents=True)

    arrs = data["trn_rss"], data["tst_rss"]
    print("Running MDS reduction...")
    mds_transform(arrs[0], output_dir)

    print("Running hill climb...")
    hill_climb_master(output_dir, num_iters)
    with open(output_dir / f"mapping.pkl", "rb") as file:
        coords, map_in_int = pickle.load(file)

    for arr, split in zip(arrs, ("train", "test")):
        output_image_fp = output_dir / f"{split}.pkl"
        if output_image_fp.exists():
            print(f"Images already generated for {split} set. Skipping...")
            continue
        print(f"Generating images for {split} set...")
        image_dim = math.ceil(math.sqrt(arr.shape[1]))
        feat_ids = list(range(arr.shape[1]))
        generated = REFINED_Im_Gen(arr, image_dim, map_in_int, feat_ids, coords)
        generated = generated.reshape((arr.shape[0], image_dim, image_dim))
        generated *= 255
        generated = generated.astype(np.uint8)
        generated = [Image.fromarray(img, mode="L") for img in generated]

        print(f"Saving generated images for {split} set...")
        with open(output_image_fp, "wb") as file:
            pickle.dump(generated, file)


class RefinedTransform:
    def __init__(self, data_fp: Path):
        self.