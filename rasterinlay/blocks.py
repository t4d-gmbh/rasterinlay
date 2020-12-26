import numpy as np
from typing import (Tuple, Union, Iterable, Generator)
from ._doc import _raster_doc


def get_counts(raster: np.ndarray, outvalue: Union[str, int, float] = 0):
    f"""Count of same valued diagonals (bottom left to top right) in `raster`

    The method creates a 2D ndarray of the same size as raster containing the
    counts of identical values.
    The count is done such that the top left corner of a block with identical
    value is 1, the 2nd element to the right, as well as the 2nd element down,
    has a count of 2, etc.


    Parameters
    ==========
    {_raster_doc}
    outvalue:
        The background value in `raster` that should be ignored in the same
        value counting.

    Returns
    =======
    np.ndarray:
        2D `.numpy.ndarray` holding the counts of same valued diagonals.

    """
    height, width = raster.shape
    _0, _1 = np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8)
    counts = np.zeros((height, width), dtype=np.uint8)
    # count along x axis > width
    _last_col = raster[:, 0]
    counts[:, 0] = np.where(_last_col != outvalue, _1, _0)
    for c in range(1, width):
        _col = raster[:, c]
        _col_adding = np.where(_col != outvalue, _1, _0)
        _col_continue = np.where(_col == _last_col, _1, _0)
        counts[:, c] = _col_continue * counts[:, c-1] + _col_adding
        _last_col = _col
    # count along y axis > height
    _last_row = raster[0, :]
    _last_adding = np.where(_last_row != outvalue, _1, _0)
    for r in range(1, height):
        _row = raster[r, :]
        _row_continue = np.where(_row == _last_row, _1, _0)
        _last_adding *= _row_continue
        counts[r, :] += _last_adding
        _last_adding += np.where(_row != outvalue, _1, _0)
        _last_row = _row
    return counts


def get_block(start: tuple, shape: tuple, offset: tuple = None) -> tuple:
    """Returns slice for accessing the block defined by `start` and `shape`.

    Parameters
    ==========
    start:
        Coordinates of the blocks top left corner
    shape:
        Height and width (in number of pixles) of the block
    offset:
        vertical and horizontal offset of `start`. The offset can be used if
        a block is given in another coordinate system, e.g. when a block is
        detected within another bock.

    Returns:
        Slice to retrieve the block.
    """
    if offset:
        h_offset, w_offset = offset
    else:
        h_offset, w_offset = 0, 0
    return (
            slice(start[0] + h_offset, start[0] + shape[0] + h_offset),
            slice(start[1] + w_offset, start[1] + shape[1] + w_offset)
            )


def offset_block(sblock: tuple, offset: tuple):
    """Adding an offset to a block slice.

    Parameters
    ==========
    sblock:
        Slice of a block as returned by `.get_block`.
    offset:
        vertical and horizontal offset to add to the block slice.
    """
    h_off, w_off = offset
    h_bs, w_bs = sblock
    return (
            slice(h_bs.start + h_off, h_bs.stop + h_off),
            slice(w_bs.start + w_off, w_bs.stop + w_off)
        )


def add_padding(sblock: tuple, padding: int):
    """Adding an padding to a block slice.

    Parameters
    ==========
    sblock:
        Slice of a block as returned by `.get_block`.
    padding:
        number of cells to add around the block slice.
    """
    s_height = max(
                0, sblock[0].start - padding
            ), sblock[0].stop + padding
    s_width = max(
                0, sblock[1].start - padding
            ), sblock[1].stop + padding

    return slice(*s_height), slice(*s_width)


def _superblock_starts(inside_counts):
    return tuple(zip(*np.where(inside_counts == 1)))


def _superblock_shapes(raster, counts, starts):
    # now assess the dimensions of all those blocks
    shapes = []
    _raster_height, _raster_width = raster.shape
    for rs, cs in starts:
        _w = 0
        val = raster[rs, cs]
        while np.logical_and(
                counts[rs, cs + _w + 1] > counts[rs, cs + _w],
                raster[rs, cs + _w + 1] == val):
            _w += 1
            if cs + _w + 1 == _raster_width:
                break
        _h = 0
        while np.logical_and(
                counts[rs + _h + 1, cs] > counts[rs + _h, cs],
                raster[rs + _h + 1, cs] == val):
            _h += 1
            if rs + _h + 1 == _raster_height:
                break
        shapes.append((_h+1, _w+1))
    return shapes


def get_superblocks(
        raster: np.ndarray,
        counts: np.ndarray,
        outvalue: Union[str, int, float] = 0) -> Generator[tuple, None, None]:
    """
    Parameters
    ==========
    raster:
        A 2D `.numpy.ndarray` in which blocks with an identical value should be
        found
    counts:
        2D `.numpy.ndarray` holding the counts of same valued diagonals.
    outvalue:
        The background value in `raster` that should be ignored in the same
        value counting.

    Returns
    =======
    Generator[tuple]:
       Collection of slices for all detected super-blocks. A super-block can
       be a block respecting a certain shape (determined when constructing
       `counts` - see `.get_counts` for details), a multi-block or an
       overloaded block. A multi-block contains only a single value but does
       not respect the limit shape. An overloaded block contains a single value
       in the top row and left column but other blocks within.
    """
    starts = _superblock_starts(counts)
    shapes = _superblock_shapes(raster, counts, starts)
    return (
            get_block(st, sh)
            for st, sh in zip(starts, shapes)
            )


def _classify_block(
        uniques: Union[tuple, list, np.ndarray],
        counts: np.ndarray, sblock: tuple, ok_max: int) -> int:
    """Classifies blocks into normal-, a multi-, or overloaded-blocks.

    Parameters
    ==========
    unique:
        Collection of unique values returned when applying `sblock` to the
        raster
    counts:
        2D `.numpy.ndarray` holding the counts of same valued diagonals.
    sblock:
        Slice of a block as returned by `.get_block`.
    ok_max:
        maximal number of diagonals accepted within a single block.
        If a block is found that contains a single value in `uniques` but
        exceeds this number of same valued diagonals
        it is classified as multi-block.
    """
    if len(uniques) > 1:
        return 2
    elif np.max(counts[sblock]) <= ok_max:
        return 0
    else:
        return 1


def separate_superblocks(
        raster: np.ndarray, counts: np.ndarray,
        superblocks: Iterable[tuple],
        block_shape: tuple,
        tolerance: int = 1) -> Tuple[list]:
    """
    Parameters
    ==========
    raster:
        A 2D array in which blocks with an identical value should be found
    counts:
        2D `.numpy.ndarray` holding the counts of same valued diagonals
    superblocks:
       Collection of supeber-bock slices
    block_shape:
        Height and width (in number of pixles) of a block
    tolerance:
        Accepted _increase_ in height and/or width of a block.

    Returns
    =======
    tuple:
        blocks: list
            Collection of slices for all unique valued blocks respecting the
            `block_shape` (plus tolerance) constriaints
        multi_blocks: list
            Collection of slices for all unique valued blocks that are over
            sized
        overloaded_blocks:
            Collection of slices for all overloaded block. They contains a
            single value in the top row and left column but other blocks
            within.

    """
    # first is ok blocks, second is multiblock and third is overloaded block
    blocks = ([], [], [])
    bheight, bwidth = block_shape
    # 2 * tolerance as both along height and width
    ok_max = bheight+bwidth+2*tolerance
    for sblock in superblocks:
        uniques = np.unique(raster[sblock])
        blocks[_classify_block(uniques, counts, sblock, ok_max)].append(sblock)
    return blocks


def split_multiblock(
        sblock: tuple,
        block_shape: tuple,
        tolerance: int = 1) -> list:
    """Splits up a multi-block into a possible set blocks.

    Parameters
    ==========
    sblock:
        Slice of a block as returned by `.get_block`
    block_shape:
        Height and width (in number of pixles) of a block
    tolerance:
        Accepted _increase_ in height and/or width of a block.

    Returns
    =======
    list:
        Collection of normal blocks that can compose the multi-block specified
        by the slices of `sblock`
    """
    height, width = block_shape
    h_start, h_stop = sblock[0].start, sblock[0].stop
    b_height = h_stop - h_start
    h_residue = b_height % height
    w_start, w_stop = sblock[1].start, sblock[1].stop
    b_width = w_stop - w_start
    w_residue = b_width % width
    h_mult = int((b_height - h_residue) / height)
    w_mult = int((b_width - w_residue) / width)
    # if there is no residue, simply split according to mult
    block_heights = [height for _ in range(h_mult)]
    # NOTE: dealing with tolerance a priory works only for bigger blocks and
    # not for smaller ones.
    if h_residue:
        # how many block have a height slightly off?
        nbr_vertic_dev = int((h_residue - h_residue % tolerance) / tolerance)
        # make sure that the oddly sized blocks do not exceed the tolerance
        assert nbr_vertic_dev * tolerance <= h_residue
        # add tolerance to the last blocks
        i = -1
        while h_residue:
            block_heights[i] += 1
            h_residue -= 1
            i -= 1
    block_widths = [width for _ in range(w_mult)]
    if w_residue:
        # how many block have a height slightly off?
        nbr_horiz_dev = int((w_residue - w_residue % tolerance) / tolerance)
        # make sure that the oddly sized blocks do not exceed the tolerance
        assert nbr_horiz_dev * tolerance <= w_residue
        # add tolerance to the last blocks
        i = -1
        while w_residue:
            block_widths[i] += 1
            w_residue -= 1
            i -= 1
    sb = []
    upper_left = [h_start, w_start]
    block_heights_iter = iter(block_heights)
    for _vert in range(h_mult):
        block_height = next(block_heights_iter)
        block_widths_iter = iter(block_widths)
        for _horiz in range(w_mult):
            block_width = next(block_widths_iter)
            sb.append(
                (
                    slice(upper_left[0], upper_left[0] + block_height),
                    slice(upper_left[1], upper_left[1] + block_width)
                )
            )
            upper_left[1] += block_width
        upper_left[1] = w_start
        upper_left[0] += block_height
    return sb


def breakdown_multiblocks(
        multiblocks: Iterable[tuple], block_shape: tuple, tolerance=1) -> list:
    """Decomposes a collection of multi-blocks into a collection of blocks.

    Parameters
    ==========
    multiblocks:
        Collection of slices that specify multi-blocks.
    block_shape:
        Height and width (in number of pixles) of a block.
    tolerance:
        Accepted _increase_ in height and/or width of a block.

    Returns
    =======
    list:
        Collection of normal blocks found in the collection of  multi-blocks
        in `multiblocks`.

    """
    blocks = []
    for mb in multiblocks:
        blocks.extend(
                split_multiblock(
                    mb, block_shape=block_shape, tolerance=tolerance
                )
            )
    return blocks


def split_overloaded(
        raster: np.ndarray, counts: np.ndarray, overloaded_block: tuple,
        block_shape: tuple,
        outvalue: Union[str, int, float] = 0,
        tolerance: int = 1) -> tuple:
    """Splits an overloaded bock into super-blocks and classifies them.


    Parameters
    ==========
    raster:
        A 2D `.numpy.ndarray` in which blocks with an identical value should be
        found
    counts:
        2D `.numpy.ndarray` holding the counts of same valued diagonals
    overloaded_block:
        Slices that determine an overloaded block in `raster`
    block_shape:
        Height and width (in number of pixles) of a block
    outvalue:
        The background value in `raster` that should be ignored in the same
        value counting
    tolerance:
        Accepted _increase_ in height and/or width of a block.

    Returns
    =======
    tuple:
        blocks: list
            Collection of slices for all unique valued blocks respecting the
            `block_shape` (plus tolerance) constriaints
        multi_blocks: list
            Collection of slices for all unique valued blocks that are over
            sized
        overloaded_blocks:
            Collection of slices for all overloaded block. They contains a
            single value in the top row and left column but other blocks
            within.
    """
    bheight, bwidth = block_shape
    # 2 * tolerance as both along height and width
    ok_max = bheight+bwidth+2*tolerance
    _block_raster = raster[overloaded_block]
    value = _block_raster[0, 0]
    _counts = get_counts(_block_raster, value)
    _starts = _superblock_starts(_counts)
    _shapes = _superblock_shapes(_block_raster, _counts, _starts)
    _inner_blocks = [get_block(_st, _sh) for _st, _sh in zip(_starts, _shapes)]
    ob_height, ob_width = overloaded_block
    h_offset = ob_height.start
    w_offset = ob_width.start
    # first is ok blocks, second is multiblock and third is overloaded block
    blocks = ([], [], [])
    # identify top line as multiblock
    upper_height = min(_starts, key=lambda x: x[0])[0]
    blocks[1].append(
            (
                slice(ob_height.start, ob_height.start + upper_height),
                ob_width
            )
        )
    # identify the (potential) multiblock on the left
    left_width = min(_starts, key=lambda x: x[1])[1]
    blocks[1].append(
            (
                slice(ob_height.start + upper_height, ob_height.stop),
                slice(ob_width.start, ob_width.start + left_width)
            )
        )
    # now check the types of the inner blocks
    for _block in _inner_blocks:
        if _block_raster[_block][0, 0] != outvalue:
            # add it, its either a block a multiblock or a overloaded
            orig_block = offset_block(_block, (h_offset, w_offset))
            uniques = np.unique(raster[orig_block])
            blocks[
                    _classify_block(uniques, _counts, _block, ok_max)
                    ].append(orig_block)
    return blocks


def breakdown_overloadeds(
        raster, counts, overloadeds, block_shape, outvalue=0, tolerance=1):
    """Decomposes a collection of overloaded blocks into classified blocks.

    THIS IS NOT DONE!

    Parameters
    ==========
    overloadeds:
        Collection of slices that specify multi-blocks.
    block_shape:
        Height and width (in number of pixles) of a block.
    tolerance:
        Accepted _increase_ in height and/or width of a block.

    Returns
    =======
    list:
        Collection of normal blocks found in the collection of  multi-blocks
        in `multiblocks`.

    """
    blocks = ([], [], [])
    for ol in overloadeds:
        _okb, _multib, _overloadeds = split_overloaded(
                raster, counts, ol, block_shape,
                outvalue=outvalue, tolerance=tolerance
            )
        blocks[0].extend(_okb)
        blocks[1].extend(_multib)
        blocks[2].extend(_overloadeds)
    return blocks


def find_blocks(
        raster: np.ndarray,
        block_shape: tuple,
        outvalue: Union[str, int, float] = 0):
    """Identify all single-valued blocks within a raster.


    Parameters
    ==========
    raster:
        A 2D `.numpy.ndarray` in which blocks with an identical value should be
        found
    block_shape:
        Height and width (in number of pixles) of a block
    outvalue:
        The background value in `raster` that should be ignored in the same
        value counting.

    Returns
    =======
    list:
        Collection of blocks found in raster.
        Blocks are specified with slices thus a single block can be passed as
        index to `raster`.
    """
    counts = get_counts(raster, outvalue=outvalue)
    blocks = []
    while np.any(counts):
        print(f'looping, with {len(blocks)} blocks')
        min_count = np.min(counts[counts != 0])
        # set smallest non-zero counts to 1
        counts[counts != 0] -= min_count - 1
        superblocks = get_superblocks(raster, counts)
        _blocks, multiblocks, overloaded = separate_superblocks(
                raster, counts, superblocks, block_shape)
        _blocks.extend(_blocks)
        # first handle the overloadeds
        while overloaded:
            _b, _m, overloaded = breakdown_overloadeds(
                    raster, counts, overloaded, block_shape)
            _blocks.extend(_b)
            multiblocks.extend(_m)
        # now break down the multiblocks
        _blocks.extend(
                breakdown_multiblocks(multiblocks, block_shape, tolerance=1)
            )
        # remove the counts within all blocks
        for block in _blocks:
            counts[block] = 0
        blocks.extend(_blocks)
    return blocks
