import pickle
from pathlib import Path

import numpy as np
from signal_to_image.refined.Toolbox import two_d_eq, assign_features_to_pixels
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
import math


def mds_transform(arr: np.ndarray, output_dir: Path):
    """Performs initial transformation using MDS.

    This function calculates euclidean distances between each feature. It then
    uses these values to calculate which features to assign to which pixel and
    saves these values to the output_dir in a pickle file.
    """
    image_dim = math.ceil(math.sqrt(arr.shape[1]))
    num_features = arr.shape[1]

    output_fp = output_dir / f"mds_result.pkl"
    if output_fp.exists():
        print(f"MDS calculation file exists. Skipping...")
        return
    print(f"Performing MDS dimensionality reduction...")
    arr = arr.T
    euc_dist = euclidean_distances(arr)
    euc_dist = np.maximum(euc_dist, euc_dist.T)

    mds = MDS(n_components=2)
    embedding = mds.fit_transform(arr)

    print(f"Performing feature to pixel assignment...")
    eq_xy = two_d_eq(embedding, num_features)
    img = assign_features_to_pixels(eq_xy, image_dim)

    with open(output_fp, "wb") as f:
        pickle.dump((euc_dist, img), f)

