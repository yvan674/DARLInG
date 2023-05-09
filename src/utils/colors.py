"""Colors.

Tools to work with colors.
"""
from PIL.ImagePalette import ImagePalette

import colorcet  # noqa # pylint: disable=import-error


def colorcet_to_image_palette(palette_name: str) -> ImagePalette:
    """Turns a colorcet palette into an ImagePalette object."""
    cc_list = getattr(globals()["colorcet"], palette_name)

    # Create a flattened list of RGB values in ints for use by ImagePalette
    flat_list = []
    for c in cc_list:
        # c is a hex value, e.g. "#FF0000"
        # Convert to RGB tuple
        rgb = tuple(int(c[i:i+2], 16) for i in (1, 3, 5))
        flat_list.extend(rgb)

    return ImagePalette("RGB", flat_list)
