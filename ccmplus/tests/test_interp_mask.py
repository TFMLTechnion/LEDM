import numpy as np

from ccmplus.grid import RectGrid
from ccmplus.interp import build_interpolation_matrix


class TestMaskedInterpolation:
    def test_disallowed_corner_is_not_used_and_weights_renormalize(self):
        grid = RectGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), delta=1.0)
        point = np.array([[0.25, 0.25, 0.25]])
        allowed = np.ones(grid.size, dtype=bool)
        allowed[0] = False

        A = build_interpolation_matrix(point, grid, allowed_nodes=allowed)
        row_u = A.getrow(0)
        cols = row_u.indices
        vals = row_u.data

        assert 0 not in cols
        np.testing.assert_allclose(vals.sum(), 1.0)

    def test_default_interpolation_keeps_original_behavior(self):
        grid = RectGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), delta=1.0)
        point = np.array([[0.25, 0.25, 0.25]])

        A = build_interpolation_matrix(point, grid)

        np.testing.assert_allclose(A.getrow(0).data.sum(), 1.0)
        assert 0 in A.getrow(0).indices
