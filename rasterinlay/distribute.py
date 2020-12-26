import numpy as np
from typing import Union, Tuple
from enum import IntEnum


class DataTypes(IntEnum):
    """
    """
    fraction = 1


class InvalidCellsIn(IntEnum):
    raster = 0
    constraints = 1


class BlockProcessingError(Exception):
    pass


def _distribute_data(
        raster_data: np.ndarray,
        data_type: DataTypes = DataTypes.fraction):
    """UNUSED
    """
    if data_type == DataTypes.fraction:
        return raster_data


def inlay(
        raster: np.ndarray,
        constraints: np.ndarray,
        inlay_form: Union[Tuple[slice, slice], np.ndarray],
        cell_unit: Union[int, float] = 1,
        cell_max: int = 100,
        invalids_in: InvalidCellsIn = InvalidCellsIn.constraints,
        invalid_value: int = 254,
        inplace: bool = True
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
    raster_data = raster[inlay_form]
    # print(f'{raster_data=}')
    data_shape = raster_data.shape
    # print(f'{data_shape=}')
    # the number of cells in raster in the inlay form
    c_data_nbr = raster_data.size

    # get the constraints
    c_const_data = constraints[inlay_form]
    # print(f'{c_const_data=}')
    # the number of cells in the constraints in the inlay from
    c_const_nbr = np.sum(np.where(c_const_data != invalid_value, 1, 0))

    # set the form max in within cell units
    # c_data_form_max = c_data_nbr * cell_max
    c_const_form_max = c_const_nbr * cell_max

    # ###
    # get the amount to inlay within the form on a per-cell level
    # get unique value of raster data (in the form)
    b_uniques = np.unique(raster_data)
    try:
        b_val = int(b_uniques[0])
    except TypeError:
        raise BlockProcessingError(f'Failed to process {inlay_form=}')

    # the amount to distribute in within-cell units
    c_data_total = c_data_nbr * b_val
    c_data_total_initial = c_data_total
    # not sure if this is used
    # data_total = np.sum(raster_data)
    print(f'{c_data_total=}')

    if c_data_total > c_const_form_max:
        # TODO: this could be an error.
        print("WARNING: the form cannot carry the required amount, even "
              "without constraints")

    # check if non of the cells is already over-filled by constraints
    assert np.all(c_const_data[c_const_data != invalid_value] <= cell_max)

    # check if data and constrains fit into the form together
    c_const_total = np.sum(c_const_data[c_const_data != invalid_value])
    # already set the remaining capacity for the form
    # negative as over-filled
    c_form_total = c_const_total + c_data_total
    # give the capacity in form units (e.g. x% of the form remains)
    b_form_capacity = (c_const_form_max - c_form_total) / c_data_nbr
    if c_form_total > c_const_form_max:
        print("WARNING: the form cannot accommodate the data along with "
              "the constraints")
        # we can directly set the raster data
        raster_data = cell_max - c_const_data
    else:
        # data_form can be redistributed
        # wipe the form clean
        raster_data[:] = 0
        # fill incrementally, always the smallest cell
        while c_data_total:
            # print(f'{c_data_total=}')
            # print(f'{raster_data=}')
            # print(f'{c_const_data=}')
            # TODO: solid condition to avoid using invalid cells. As long as
            # invalid value is bigger than cell_max this works.
            min_ind = np.unravel_index(
                    np.argmin(raster_data + c_const_data, axis=None),
                    data_shape
                )
            raster_data[min_ind] += cell_unit
            assert raster_data[min_ind] <= cell_max
            c_data_total -= cell_unit
    # check if non of the raster data got lost
    if b_form_capacity >= 0:
        assert c_data_total_initial == np.sum(raster_data)
    return b_form_capacity


def inprint_constraints(raster, constraints, inlay_forms):
    capacities = []
    errors = []
    for inlay_form in inlay_forms:
        try:
            capacities.append(inlay(raster, constraints, inlay_form))
        except BlockProcessingError as e:
            print(f'{inlay_form=}')
            errors.append(e.message)
