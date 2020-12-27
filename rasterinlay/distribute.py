import numpy as np
from typing import Union, Tuple, Callable
from enum import IntEnum


class DataTypes(IntEnum):
    """
    """
    fraction = 1


class InvalidCellsIn(IntEnum):
    none = 0
    raster = 1
    constraints = 2


class BlockInlayError(Exception):
    pass


class BlockProcessingError(Exception):
    pass


def _distribute_data(
        raster_data: np.ndarray,
        data_type: DataTypes = DataTypes.fraction):
    """UNUSED
    """
    if data_type == DataTypes.fraction:
        return raster_data


def exclude_invalid(
        inlay_form: Union[Tuple[slice, slice], np.ndarray],
        raster: np.ndarray,
        constraints: np.ndarray,
        invalid_in: InvalidCellsIn,
        invalid_value: Union[int, float],
        inplace: bool = True
        ):
    """Get raster and constraints data in `inlay_form` excluding invalid cells


    Parameters
    ==========
    """
    raster_data = None
    return raster_data


def fill_inlay(
        raster_block,
        const_block,
        # inlay_form: Union[Tuple[slice, slice], np.ndarray],
        cell_unit: Union[int, float] = 1,
        cell_max: int = 100,
        invalid_in: InvalidCellsIn = InvalidCellsIn.none,
        invalid_value: int = 254,
        ):
    """Redistribute the values in :attr:`raster_block` to avoid collision with
the values in :attr:`const_block`

    Parameters
    ==========
    """
    # print(f'{raster_data=}')
    data_shape = raster_block.shape
    # print(f'{data_shape=}')

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
    c_data_nbr = valid_raster_block.size
    c_const_nbr = valid_raster_block.size
    # get totals
    raster_total = np.sum(valid_raster_block)
    c_const_total = np.sum(valid_const_block)

    assert c_data_nbr == c_const_nbr, 'not same size for both masked blocks'

    # print(f'{const_view=}')

    # set the form max in within cell units
    # c_data_form_max = c_data_nbr * cell_max
    block_max = c_data_nbr * cell_max

    # ###
    # if unique value in raster_block
    # ###
    # get the amount to inlay within the form on a per-cell level
    # get unique value of raster data (in the form)
    b_uniques = np.unique(valid_raster_block)
    try:
        int(b_uniques[0])
    except TypeError:
        raise BlockInlayError(
                'Valid cells in raster block contain multiple values')
    # ###
    # ###

    # only for testing
    _c_data_total_initial = raster_total
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
    valid_block_total = c_const_total + raster_total
    # give the capacity in form units (e.g. x% of the form remains)
    block_capacity = (block_max - valid_block_total) / c_data_nbr
    if valid_block_total > block_max:
        print("WARNING: All valid cell in the block cannot accommodate the "
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
        assert _c_data_total_initial == np.sum(raster_block[invalid_mask])
    return raster_block, const_block, block_capacity


def inlay(
        raster: np.ndarray,
        constraints: np.ndarray,
        inlay_bbox: Tuple[slice, slice],
        set_inlay: Callable[
            [np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
        ],
        set_inlay_kwargs: dict
        ) -> float:
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
        raster_view, const_view, block_capacity = set_inlay(
                raster_view, const_view, **set_inlay_kwargs
            )
    except BlockInlayError:
        raise BlockProcessingError
    return block_capacity


def inprint_constraints(raster, constraints, inlay_forms):
    capacities = []
    errors = []
    for inlay_form in inlay_forms:
        try:
            capacities.append(inlay(raster, constraints, inlay_form))
        except BlockProcessingError as e:
            print(f'{inlay_form=}')
            errors.append(e.message)
