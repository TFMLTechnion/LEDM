import numpy as np

from ccmplus.drivers.sphere import median_velocity_filter, residual_outlier_filter
from ccmplus.grid import RectGrid


class TestMedianVelocityFilter:
    def test_removes_local_velocity_outlier(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
                [0.1, 0.1, 0.0],
            ]
        )
        velocities = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ]
        )

        keep = median_velocity_filter(
            positions,
            velocities,
            radius_mm=1.0,
            threshold_ms=0.5,
            min_neighbors=4,
        )

        np.testing.assert_array_equal(keep, [True, True, True, True, False])

    def test_keeps_sparse_points(self):
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        velocities = np.array([[1.0, 0.0, 0.0], [50.0, 0.0, 0.0]])

        keep = median_velocity_filter(
            positions,
            velocities,
            radius_mm=1.0,
            threshold_ms=0.5,
            min_neighbors=3,
        )

        np.testing.assert_array_equal(keep, [True, True])


class TestResidualOutlierFilter:
    def test_removes_tracers_far_from_coarse_field(self):
        grid = RectGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), delta=1.0)
        coarse_velocity = np.tile([1.0, 0.0, 0.0], (grid.size, 1))
        positions = np.array(
            [
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
            ]
        )
        velocities = np.array(
            [
                [1.1, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

        keep, residuals = residual_outlier_filter(
            positions,
            velocities,
            grid,
            coarse_velocity,
            threshold_ms=0.5,
        )

        np.testing.assert_array_equal(keep, [True, False])
        np.testing.assert_allclose(residuals, [0.1, 1.0])

    def test_nonpositive_threshold_keeps_all_points(self):
        grid = RectGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), delta=1.0)
        coarse_velocity = np.zeros((grid.size, 3))
        positions = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
        velocities = np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])

        keep, residuals = residual_outlier_filter(
            positions,
            velocities,
            grid,
            coarse_velocity,
            threshold_ms=0.0,
        )

        np.testing.assert_array_equal(keep, [True, True])
        np.testing.assert_allclose(residuals, [0.0, 0.0])
