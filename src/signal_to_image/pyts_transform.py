"""Py-Time-Series Transforms.

Base source GAF and MTS transforms, since they are both available in pyts.
"""
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def pyts_transform(data: dict,
                   transform: Any,
                   output_dir: Path):
    """Turns pyts transformed data arrays into images.

    Args:
        data: The data dict produced by the data ingest functions.
        transform: The transform function to use. Should be a class with a
            method called `fit_transform()`.
        output_dir: Where to save the generated images.
    """
    train_arr, test_arr = data["trn_rss"], data["tst_rss"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for arr, split in zip((train_arr, test_arr), ("train", "test")):
        img_arr = transform.fit_transform(arr)

        # img_arr is in the range [-1, 1] and we need to transform it to
        # [0, 255]
        img_arr += 1.
        img_arr /= 2.
        img_arr *= 255
        img_arr = img_arr.astype(np.uint8)
        images = [Image.fromarray(img, mode="L")
                  for img in img_arr]

        with open(output_dir / f"{split}.pkl", "wb") as file:
            pickle.dump(images, file)
