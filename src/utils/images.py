"""Images.

Utilities to work with images.
"""
from PIL import Image
from PIL.ImagePalette import ImagePalette
import torch


def tensor_to_image(data: torch.Tensor, img_idxs: tuple,
                    palette: ImagePalette) -> list[Image.Image]:
    """Takes a raw data tensor and turns it into a PIL image.

    Args:
        data: Tensor in shape [batch, n, 20, 20].
        img_idxs: Img indices to turn into the output image.
        palette: Image palette to use.

    Returns:
        An image in mode "rgb" with the given palette.
    """
    img_arrays = data[img_idxs[0]:img_idxs[1], 0, :, :] \
        .detach() \
        .cpu() \
        .numpy() \
        .reshape([-1, 20, 20])
    img_arrays = (img_arrays * 255).astype('uint8')
    imgs = []
    for img_array in img_arrays:
        img = Image.fromarray(img_array, mode="P")
        img.putpalette(palette)
        imgs.append(img.convert(mode="RGB"))

    return img_arrays
