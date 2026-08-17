import numpy as np

from ccmplus.config import BodyState, Config, FrameData
from ccmplus.grid import RectGrid
from ccmplus.operators import build_laplacian_smoothing_operator
from ccmplus.reconstruct import CCMPlus


def make_body_far():
    return BodyState(
        X_s=np.array([1000.0, 0.0, 0.0]),
        U_s=np.zeros(3),
        omega_s=np.zeros(3),
        radius=1.0,
        sigma_s=0.1,
    )


def roughness(velocity, grid):
    vals = []
    for comp in range(3):
        arr = velocity[:, comp].reshape(grid.shape, order="F")
        for axis in range(3):
            vals.append(np.diff(arr, axis=axis).ravel())
    vals = np.concatenate(vals)
    return float(np.sqrt(np.mean(vals * vals)))


class TestFieldSmoothingOperators:
    def test_laplacian_zero_for_constant_velocity(self):
        grid = RectGrid((-2, -2, -2), (2, 2, 2), delta=1.0)
        C = np.ones(grid.size, dtype=np.int8)
        config = Config(tuple(grid.domain_min), tuple(grid.domain_max), grid.delta)
        L = build_laplacian_smoothing_operator(grid, C, config).matrix
        velocity = np.tile([2.0, -1.0, 0.5], (grid.size, 1))

        np.testing.assert_allclose(L @ velocity.ravel(), 0.0, atol=1e-12)

    def test_laplacian_near_zero_for_linear_velocity(self):
        grid = RectGrid((-2, -2, -2), (2, 2, 2), delta=1.0)
        C = np.ones(grid.size, dtype=np.int8)
        config = Config(tuple(grid.domain_min), tuple(grid.domain_max), grid.delta)
        L = build_laplacian_smoothing_operator(grid, C, config).matrix
        nodes = grid.nodes
        velocity = np.column_stack(
            [nodes[:, 0] + 2 * nodes[:, 1], -nodes[:, 2], nodes[:, 0] - nodes[:, 1]]
        )

        np.testing.assert_allclose(L @ velocity.ravel(), 0.0, atol=1e-12)

    def test_laplacian_detects_quadratic_curvature(self):
        grid = RectGrid((-2, -2, -2), (2, 2, 2), delta=1.0)
        C = np.ones(grid.size, dtype=np.int8)
        config = Config(tuple(grid.domain_min), tuple(grid.domain_max), grid.delta)
        L = build_laplacian_smoothing_operator(grid, C, config).matrix
        nodes = grid.nodes
        velocity = np.column_stack([nodes[:, 0] ** 2, np.zeros(grid.size), np.zeros(grid.size)])

        vals = L @ velocity.ravel()
        assert np.max(np.abs(vals)) > 1.0

    def test_laplacian_rows_do_not_cross_sphere_mask(self):
        grid = RectGrid((-3, -3, -3), (3, 3, 3), delta=1.0)
        body = BodyState(np.zeros(3), np.zeros(3), np.zeros(3), radius=1.5, sigma_s=0.1)
        r = np.linalg.norm(grid.nodes - body.X_s, axis=1)
        C = np.where(r < body.radius, -1, 1).astype(np.int8)
        config = Config(
            tuple(grid.domain_min),
            tuple(grid.domain_max),
            grid.delta,
            enable_field_smoothing=True,
            smoothing_no_cross_mask=True,
        )

        result = build_laplacian_smoothing_operator(grid, C, config)
        rows, cols = result.matrix.nonzero()
        touched_nodes = cols // 3

        assert np.all(C[touched_nodes] == 1)
        assert result.stats["skipped_mask"] > 0


class TestFieldSmoothingSolve:
    def test_increasing_lambda_reduces_roughness_noisy_field(self):
        grid = RectGrid((-2, -2, -2), (2, 2, 2), delta=1.0)
        rng = np.random.default_rng(2)
        positions = grid.nodes.copy()
        true = np.tile([0.2, 0.0, 0.0], (grid.size, 1))
        velocities = true + rng.normal(0.0, 0.5, true.shape)
        body = make_body_far()

        def solve(lam):
            config = Config(
                tuple(grid.domain_min),
                tuple(grid.domain_max),
                grid.delta,
                kappa=1e-6,
                enable_field_smoothing=True,
                lambda_laplacian=lam,
                solver_rtol=1e-7,
                solver_maxiter=2000,
            )
            frame = FrameData(positions, velocities, np.ones(grid.size), body, t=0.0)
            return CCMPlus(config, grid).reconstruct(frame).velocity

        unsmoothed = solve(0.0)
        smoothed = solve(1.0)

        assert roughness(smoothed, grid) < roughness(unsmoothed, grid)

    def test_lambda_does_not_change_clean_uniform_field(self):
        grid = RectGrid((-2, -2, -2), (2, 2, 2), delta=1.0)
        positions = grid.nodes.copy()
        velocities = np.tile([0.2, -0.1, 0.05], (grid.size, 1))
        body = make_body_far()

        def solve(lam):
            config = Config(
                tuple(grid.domain_min),
                tuple(grid.domain_max),
                grid.delta,
                kappa=1e-6,
                enable_field_smoothing=True,
                lambda_laplacian=lam,
                solver_rtol=1e-8,
                solver_maxiter=2000,
            )
            frame = FrameData(positions, velocities, np.ones(grid.size), body, t=0.0)
            return CCMPlus(config, grid).reconstruct(frame).velocity

        base = solve(0.0)
        strong = solve(1.0)

        np.testing.assert_allclose(strong, base, atol=1e-4)
