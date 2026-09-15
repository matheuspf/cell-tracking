import unittest
import numpy as np
from scipy.ndimage import map_coordinates

from pipeline_error_training.fast_crops import values


class FastCropTests(unittest.TestCase):
    def test_exact_uint16_interpolation_and_boundaries(self):
        rng = np.random.default_rng(19)
        frame = rng.integers(0, 65536, (21, 75, 73), dtype=np.uint16)
        shape = (16, 64, 64)
        offsets = np.stack(np.meshgrid(*[np.arange(n)-(n-1)/2 for n in shape], indexing='ij'))
        for position in [(10, 35, 31), (0, 0, 0), (21, 70, 72), (-1, 74, 73)]:
            for scale in [1, 2]:
                actual, mask = values(frame, position, scale)
                expected = map_coordinates(frame.astype(np.float32), np.asarray(position)[:, None, None, None]+scale*offsets,
                                           order=1, mode='constant', cval=0.)
                np.testing.assert_array_equal(actual*mask, expected)


if __name__ == '__main__':
    unittest.main()
