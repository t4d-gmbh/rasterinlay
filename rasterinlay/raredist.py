import numpy as np
from typing import Iterable
# from typing import Union


def block_contours(
        raster_shape: tuple,
        blocks: Iterable[tuple],
        bordervalue: np.uint8 = 255):
    """Create 2D `numpy.ndarray` in the shape of 'raster' with box contours.

    Parameters
    ==========
    raster_shape:
        Height and width (in number of pixles) of the raster
    blocks:
        Collection of block slices for which contours should be made

    Returns
    =======
    numpy.ndarray:
        2D `numpy.ndarray` in the shape of `raster_shape` with all zeros except
        for the contours of the provided blocks.

    """
    b_c = np.zeros(raster_shape, dtype=np.uint8)
    for block in blocks:
        b_c[block] = bordervalue
        vert, horiz = block
        _inner = (
            slice(vert.start+1, vert.stop-1),
            slice(horiz.start+1, horiz.stop-1)
        )
        b_c[_inner] = 0
    return b_c
