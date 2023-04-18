"""DeepInsight Transform.

Transforms according to the method introduced in DeepInsight.

References:
    Sharma, A., Vans, E., Shigemizu, D. et al. DeepInsight: A methodology to
        transform a non-image data to an image for convolution neural network
        architecture. Sci Rep 9, 11399 (2019). <https://doi.org/10.1038/s41598-
        019-47765-6>.
    pyDeepInsight GitHub Repository. <https://github.com/alok-ai-lab/
        pyDeepInsight>.

Author:
    Yvan Satyawan <y.p.satyawan@student.tue.nl>
"""
import numpy as np
import pickle
from pathlib import Path
from PIL import Image
from pyDeepInsight import ImageTransformer, Norm2Scaler

from signal_to_image.base import SignalToImageTransformer


class DeepInsight(SignalToImageTransformer):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


def deepinsight_transform(data: dict, output_dir: Path):
    """Performs the transformation introduced in the DeepInsight Paper.

    Args:
        data: The data dict produced by the data ingest functions.
        output_dir: Where to save the generated images.
    """
    # Make sure the folder exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Normalizing input data...")
    ln = Norm2Scaler()
    arrs = ln.fit_transform(data["trn_rss"]), ln.fit_transform(data["tst_rss"])
    di = ImageTransformer(feature_extractor="tsne",
                          discretization="bin",
                          pixels=(256, 256))

    di.fit(arrs[0])

    for arr, split in zip(arrs, ("train", "test")):
        print(f"Generating images for {split} set...")
        imgs = di.transform(arr)[:, :, :, 0]
        imgs *= 255.
        imgs = imgs.astype(np.uint8)
        imgs = [Image.fromarray(img) for img in imgs]

        with open(output_dir / f"{split}.pkl", "wb") as f:
            pickle.dump(imgs, f)


if __name__ == '__main__':
    from preprocessing.data_ingest import full_ingest_pipeline
    data_path = Path("../../data/UJI_LIB_DB_v2.2/01")
    out_path = Path("../../data/deepinsight_images")
    deepinsight_transform(full_ingest_pipeline(data_path), out_path)
