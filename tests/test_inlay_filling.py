import unittest
import numpy as np

from rasterinlay.distribute import imprint_constraints
from rasterinlay.distribute import InvalidCellsIn
# from rasterinlay.blocks import get_block



class TestInlayFill(unittest.TestCase):

    # use self.assetEqual ...
    def setUp(self):
        """Create raster and constraints data and blocks
        """
        # setting some parameters
        self.cell_max = 100
        self.cell_unit = 1
        self.outside_value = 255
        self.inlay_cb_kwargs = dict(
            cell_unit=self.cell_unit,
            cell_max=self.cell_max,
            invalid_in=InvalidCellsIn.constraints,
            invalid_value=self.outside_value
        )
        # ###
        # 9x9 rasters with 4 3x3 blocks
        self.td_shape = (9, 9)
        # the 4 blocks
        self.blocks = {
                'tl': (slice(1, 4), slice(1, 4)),  # top left
                'bl': (slice(5, 8), slice(1, 4)),  # bottom left
                'tr': (slice(1, 4), slice(5, 8)),  # top right 
                'br': (slice(5, 8), slice(5, 8)),  # bottom right
            }
        # ###
        # now create some test data
        # 1st case: no overloaded blocks
        self.test_data = []
        raster1 = np.zeros(self.td_shape, dtype=np.uint8)
        constr1 = np.zeros(self.td_shape, dtype=np.uint8)
        _row = np.array([0, 50, 100])
        _add = np.array([25, 25, 25])
        # all raster blocks have same fill
        for b in self.blocks.values():
            raster1[b] = np.vstack((2*_add, 2*_add, 2*_add))
        # tl block: no constraints
        constr1[self.blocks['tl']] = 0
        # bl block: 3 full and 3 at 50 constraints 
        constr1[self.blocks['bl']] = np.vstack((_row, _row, _row))
        # tr block: equal constraint everywhere
        constr1[self.blocks['tr']] = 25
        # br block: some full with constraints some empty 
        constr1[self.blocks['br']] = np.vstack(
                (0.5*_row, 0.5*_row + _add, 2*_add+0.5*_row)
            )
        self.test_data.append((raster1, constr1))
        # ###
        # 2nd case: all but tl block overloaded
        raster2 = raster1.copy()
        raster2[raster2>0] += 25
        constr2 = constr1.copy()
        self.test_data.append((raster2, constr1))


    def tearDown(self):
        del self.test_data

    def test_ok_blocks(self):
        """Only blocks that can accommodate both raster data and constraints.

        Check if the total sum of raster data is preserved.
        """
        # keep to original raster for comparison
        raster_orig = self.test_data[0][0].copy()
        raster, constraints = self.test_data[0]
        raster_orig = raster.copy()
        report, capacities = imprint_constraints(
                raster=raster, constraints=constraints,
                bblocks=self.blocks.values(),
                inlay_callback_kwargs=self.inlay_cb_kwargs,
            )
        # make sure raster has been modified:
        self.assertFalse(np.all(raster==raster_orig))
        # tl block should remain unchanged:
        self.assertTrue(
                np.all(
                    raster[self.blocks['tl']] == raster_orig[self.blocks['tl']]
                )
            )
        # tr block should remain unchanged:
        self.assertTrue(
                np.all(
                    raster[self.blocks['tr']] == raster_orig[self.blocks['tr']]
                )
            )
        # non of the blocks should be jammed:
        self.assertFalse(report['jammed'])
        # there should be 4 ok blocks
        self.assertEqual(len(report['ok']), 4)

    def test_jammed_blocks(self):
        """Run fill inlay with raster and constraints that'll jam some blocks.
        """
        raster, constraints = self.test_data[1]
        raster_orig = raster.copy()
        print('\n')
        print(raster)
        print(constraints)
        blocks = [*self.blocks.values()]
        report, capacities = imprint_constraints(
                raster=raster, constraints=constraints,
                bblocks=blocks,
                inlay_callback_kwargs=self.inlay_cb_kwargs,
            )
        print(report)
        print(capacities)
        print(raster)
        # make sure raster has been modified:
        self.assertFalse(np.all(raster==raster_orig))
        # check that indeed 'br' and 'bl' block are jammed (1 and 3 in blocks)
        self.assertEqual(set(report['jammed']), set([1, 3]))
        # make sure jammed blocks initial fill is current fill plus the over-
        # flown capacity (inital_fill == current_fill + abs(capacity)
        for i, block in enumerate(blocks):
            initial_fill = raster_orig[block].sum()
            self.assertEqual(
                    initial_fill,
                    raster[block].sum() - min(0, capacities[i])
                )
