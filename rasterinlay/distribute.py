import numpy as np
from typing import Union, Tuple, Callable
from enum import IntEnum


class CombineTypes(IntEnum):
    """Specifies how the constraints should act on the raster data.

    For land-coverage the combine type can be :attr:`fill` or :attr:`coverage`.

    """
    fill = 1
    """A cell value indicates the fraction within this cell.
        With this combine type an inlay will first compute the total amount of
        land coverage within a bounding box and then try to redistribute the
        amount within the cells.
    """
    coverage = 2
    """Indicates the fraction of cells within the bounding box that should
        be covered. Using this combine type only bounding boxes in which all
        cells have the same value are processed.
    """


class InvalidCellsIn(IntEnum):
    """Used to indicate where invalid (or outside) cells should be detected.
    """
    none = 0
    """Indicates that there are no invalid (or outside) cells.
    """
    raster = 1
    """Check for invalid cells in the raster block.
    """

    constraints = 2
    """Check for invalid cells in the constraints block.
    """


class MultivaluedBlockError(Exception):
    pass


class InvalidBlockError(Exception):
    pass


class BlockProcessingError(Exception):
    pass


def exclude_invalid(
        inlay_form: Union[Tuple[slice, slice], np.ndarray],
        raster: np.ndarray,
        constraints: np.ndarray,
        invalid_in: InvalidCellsIn,
        invalid_value: Union[int, float],
        inplace: bool = True
        ):
    """Get raster and constraints data in `inlay_form` excluding invalid cells


    .. warning::
        Incomplete and unused function.

    """
    raster_data = None
    return raster_data


def fraction_inlay(
        raster_block: np.ndarray,
        const_block: np.ndarray,
        # inlay_form: Union[Tuple[slice, slice], np.ndarray],
        cell_unit: Union[int, float] = 1,
        cell_max: int = 100,
        invalid_in: InvalidCellsIn = InvalidCellsIn.none,
        invalid_value: int = 254,
        ) -> Tuple[np.ndarray, Union[int, None]]:
    """Redistribute the values in :attr:`raster_block` to avoid collision with
    the values in :attr:`const_block`

    Parameters
    ==========

    Returns
    =======
    tuple:
        np.ndarray:
            The inlayed raster block.
        Union[int, None]:
            The capacity remaining in the raster block.

            .. note::
                A value below zero indicates a jammed block.
                The actual amount gives the excess data from the
                :attr:`raster_block` that could no longer be fitted in.
    """
    # print(f'{raster_data=}')
    data_shape = raster_block.shape
    # print(f'{data_shape=}')

    # TODO: move these tests out into the bulk bbox processing function
    if invalid_in == InvalidCellsIn.none:
        valid_raster_block = raster_block
        valid_const_block = const_block
    else:
        if invalid_in == InvalidCellsIn.constraints:
            invalid_mask = const_block != invalid_value
            _with_invalid = const_block
        elif invalid_in == InvalidCellsIn.raster:
            invalid_mask = raster_block != invalid_value
            _with_invalid = raster_block
        valid_raster_block = raster_block[invalid_mask]
        valid_const_block = const_block[invalid_mask]

    # the number of cells in raster in the inlay form
    raster_nbr = valid_raster_block.size
    if not raster_nbr:
        # There are no valid cells so we stop here.
        return raster_block, None
    const_nbr = valid_raster_block.size
    # get totals
    raster_total = np.sum(valid_raster_block)
    const_total = np.sum(valid_const_block)

    assert raster_nbr == const_nbr, 'not same size for both masked blocks'

    # print(f'{const_view=}')

    # set the form max in within cell units
    # c_data_form_max = raster_nbr * cell_max
    block_max = raster_nbr * cell_max

    # ###
    # if unique value in raster_block
    # ###
    # get the amount to inlay within the form on a per-cell level
    # get unique value of raster data (in the form)
    # block_uniques = np.unique(valid_raster_block)
    # try:
    #     int(block_uniques[0])
    # except TypeError:
    #     raise MultivaluedBlockError(
    #             'Valid cells in raster block contain multiple values')
    # except IndexError:
    #     # block contains no valid cells
    #     # TODO: How should this be treated?
    #     return None
    # ###
    # ###

    # FIXME: remove later - only for testing
    _data_total_initial = raster_total
    # not sure if this is used
    # data_total = np.sum(raster_data)

    if raster_total > block_max:
        # TODO: this could be an error.
        print("WARNING: Valid cells in block cannot carry the required amount"
              ", even without constraints")

    # check if non of the cells is already over-filled by constraints
    assert np.all(valid_const_block <= cell_max), 'constraints block is '\
        'initially overloaded!'
    # NOTE: we do not mind if valid raster block cells are initially overloaded

    # check if data and constrains fit into the form together
    # already set the remaining capacity for the form
    # negative as over-filled
    valid_block_total = const_total + raster_total
    # give the capacity
    block_capacity = block_max - valid_block_total
    if valid_block_total > block_max:
        print("WARNING: Valid cells in the block cannot accommodate the "
              "data along with the constraints!")
        # we can directly set the raster data
        if invalid_in == InvalidCellsIn.none:
            raster_block = cell_max - const_block
        else:
            raster_block = np.where(
                    _with_invalid != invalid_value,
                    cell_max - const_block,
                    raster_block
                )
    else:
        # data_form can be redistributed
        # wipe the form clean
        if invalid_in == InvalidCellsIn.none:
            raster_block[:] = 0
        else:
            # set the block to 0 in all valid cells
            _raster_block_orig = raster_block.copy()
            raster_block[:] = np.where(
                    _with_invalid != invalid_value,
                    0, invalid_value
                )
        # fill incrementally, always the smallest cell
        while raster_total:
            # print(f'{data_total=}')
            # print(f'{raster_data=}')
            # print(f'{const_view=}')
            # NOTE: this only works as long as invalid value is bigger than
            # cell_max!
            min_ind = np.unravel_index(
                    np.argmin(raster_block + const_block, axis=None),
                    data_shape
                )
            raster_block[min_ind] += cell_unit
            assert raster_block[min_ind] <= cell_max
            raster_total -= cell_unit
        if invalid_in != InvalidCellsIn.none:
            raster_block[:] = np.where(
                    _with_invalid != invalid_value,
                    raster_block, _raster_block_orig
                )
    # check if non of the raster data got lost
    if block_capacity >= 0:
        assert _data_total_initial == np.sum(raster_block[invalid_mask])
    return raster_block, block_capacity


def inlay(
        raster: np.ndarray,
        constraints: np.ndarray,
        inlay_bbox: Tuple[slice, slice],
        set_inlay: Callable[
            [np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
        ],
        set_inlay_kwargs: dict
        ) -> Union[float, bool]:
    """

    .. todo::
        Exclude values in the raster for invalid cells defined in constraints.

    Parameters
    ==========
    invalid_value:
        Cells with this value are considered as masked and completely excluded

    Returns
    =======
        float:
            The remaining capacity within the form. A negative value indicates
            that the inlay_form could not accommodate the data from raster
            completely.
    """
    raster_view = raster[inlay_bbox]
    const_view = constraints[inlay_bbox]

    # pass the views to the set_inlay function
    try:
        raster_view[:], block_capacity = set_inlay(
                raster_view, const_view, **set_inlay_kwargs
            )
    except MultivaluedBlockError:
        raise BlockProcessingError
    except InvalidBlockError:
        # Note: unused for now
        return None
    return block_capacity


def imprint_constraints(
        raster,
        constraints,
        bblocks,
        inlay_callback_kwargs: dict,
        data_type: CombineTypes = CombineTypes.fill
        ) -> Tuple[dict, dict]:
    """
    .. warning::
        Function in under construction.

    """
    # TODO: add option to specify data type
    if data_type == CombineTypes.fill:
        inlay_cb = fraction_inlay
    else:
        raise NotImplementedError()
    capacities = {}
    block_report = dict(ok=[], jammed=[], ignored=[], failed=[])
    for i, bblock in enumerate(bblocks):
        try:
            _capacity = inlay(
                    raster, constraints, bblock,
                    inlay_cb, inlay_callback_kwargs)
        except BlockProcessingError:
            block_report['failed'].append(i)
            print(f'{bblock=}')
        else:
            if _capacity is not None:
                capacities[i] = _capacity
                if _capacity < 0:
                    block_report['jammed'].append(i)
                else:
                    block_report['ok'].append(i)
            else:
                block_report['ignored'].append(i)
    return block_report, capacities
