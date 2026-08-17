"""Smoke tests for kinematics, prior correction, and CCMPlus reconstruct pipeline."""

import numpy as np
import pytest
from ccmplus.grid import RectGrid
from ccmplus.config import Config, BodyState, FrameData, ReconstructionResult
from ccmplus.kinematics import u_gamma
from ccmplus.classify import classify, near_wall_fluid_set, transition_flags
from ccmplus.prior import apply_prior_correction
from ccmplus.reconstruct import CCMPlus
from ccmplus.sdf import signed_distance_body, signed_distance_body_points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_grid():
    return RectGrid((-4.0, -4.0, -4.0), (4.0, 4.0, 4.0), delta=1.0)


def make_config(grid):
    # These are PIPELINE smoke tests: they check that the stages wire together
    # and that classification / warm-start / shell velocity behave, not that the
    # divergence constraint is converged. A cheap solve is deliberate here.
    #
    # constraint_div_tol is therefore relaxed so the solver's constraint warning
    # stays meaningful elsewhere instead of firing on every smoke test and
    # training readers to ignore it. Constraint QUALITY is asserted, with tight
    # solves and normalized tolerances, in test_solver_onefluid.py.
    return Config(
        domain_min=tuple(grid.domain_min),
        domain_max=tuple(grid.domain_max),
        delta=grid.delta,
        kappa=0.05,
        solver_rtol=1e-6,
        solver_maxiter=2000,
        constraint_div_tol=1.0,
    )


def make_body(center=(0., 0., 0.), radius=1.5, U_s=None, omega_s=None):
    return BodyState(
        X_s=np.array(center, dtype=float),
        U_s=np.zeros(3) if U_s is None else np.array(U_s, dtype=float),
        omega_s=np.zeros(3) if omega_s is None else np.array(omega_s, dtype=float),
        radius=radius,
        sigma_s=0.5,
    )


def make_frame(grid, body, U0=(1., 0., 0.), noise_rng=None, n_override=None):
    pts = grid.nodes.copy()
    n_p = len(pts) if n_override is None else n_override
    if n_override is not None:
        pts = pts[:n_p]
    vel = np.tile(np.array(U0, dtype=float), (n_p, 1))
    if noise_rng is not None:
        vel += noise_rng.normal(0, 0.05, vel.shape)
    unc = np.ones(n_p) * 0.1
    return FrameData(positions=pts, velocities=vel, uncertainties=unc, body=body, t=0.0)


# ---------------------------------------------------------------------------
# kinematics.u_gamma
# ---------------------------------------------------------------------------

class TestKinematics:
    def test_stationary_zero_everywhere(self):
        body = make_body(U_s=[0, 0, 0])
        pts = np.random.default_rng(0).uniform(-3, 3, (20, 3))
        np.testing.assert_allclose(u_gamma(pts, body), 0.0, atol=1e-12)

    def test_pure_translation_everywhere(self):
        U = np.array([2., -1., 0.5])
        body = make_body(U_s=U)
        pts = np.random.default_rng(1).uniform(-3, 3, (20, 3))
        result = u_gamma(pts, body)
        np.testing.assert_allclose(result, np.tile(U, (len(pts), 1)), atol=1e-12)

    def test_rotation_at_center_is_zero(self):
        """omega × (X_s - X_s) = 0."""
        body = make_body(center=(1., 2., 3.), omega_s=[0., 0., 1.])
        pt = np.array([[1., 2., 3.]])
        np.testing.assert_allclose(u_gamma(pt, body), 0.0, atol=1e-12)

    def test_rotation_tangential(self):
        """Spin about z at (R,0,0): velocity is (0, omega*R, 0)."""
        omega = 2.0
        R = 3.0
        body = make_body(center=(0., 0., 0.), omega_s=[0., 0., omega])
        pt = np.array([[R, 0., 0.]])
        result = u_gamma(pt, body)
        np.testing.assert_allclose(result, [[0., omega * R, 0.]], atol=1e-12)

    def test_combined_translation_rotation(self):
        U = np.array([1., 0., 0.])
        omega = np.array([0., 0., 1.])
        body = BodyState(X_s=np.zeros(3), U_s=U, omega_s=omega, radius=1., sigma_s=0.1)
        pt = np.array([[1., 0., 0.]])
        # u_gamma = U + omega × (pt - 0) = [1,0,0] + [0,0,1]×[1,0,0] = [1,0,0]+[0,1,0] = [1,1,0]
        np.testing.assert_allclose(u_gamma(pt, body), [[1., 1., 0.]], atol=1e-12)


    def test_custom_velocity_function(self):
        body = BodyState(
            X_s=np.zeros(3),
            U_s=np.zeros(3),
            omega_s=np.zeros(3),
            radius=1.0,
            sigma_s=0.1,
            velocity_fn=lambda pts, _: np.column_stack(
                [pts[:, 0], np.zeros(len(pts)), -pts[:, 2]]
            ),
        )
        pts = np.array([[2.0, 5.0, 3.0], [-1.0, 0.0, 4.0]])
        np.testing.assert_allclose(
            u_gamma(pts, body),
            [[2.0, 0.0, -3.0], [-1.0, 0.0, -4.0]],
            atol=1e-12,
        )


class TestGeneralBodySDF:
    def test_custom_sdf_function_on_grid_and_points(self):
        grid = make_grid()
        body = BodyState(
            X_s=np.zeros(3),
            U_s=np.zeros(3),
            omega_s=np.zeros(3),
            radius=1.0,
            sigma_s=0.1,
            sdf_fn=lambda pts, _: pts[:, 0] - 0.25,
        )

        phi_grid = signed_distance_body(grid, body)
        phi_points = signed_distance_body_points(
            np.array([[0.25, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            body,
        )

        np.testing.assert_allclose(phi_grid, grid.nodes[:, 0] - 0.25)
        np.testing.assert_allclose(phi_points, [0.0, 0.75])


# ---------------------------------------------------------------------------
# prior.apply_prior_correction
# ---------------------------------------------------------------------------

class TestPriorCorrection:
    def test_first_timestep_gives_zeros(self):
        grid = make_grid()
        body = make_body()
        tau = np.zeros(grid.size, dtype=np.int8)
        x_tilde = apply_prior_correction(None, tau, body, grid)
        assert x_tilde.shape == (3 * grid.size,)
        np.testing.assert_allclose(x_tilde, 0.0, atol=1e-12)

    def test_no_transitions_carries_forward(self):
        grid = make_grid()
        body = make_body()
        x_prev = np.random.default_rng(42).uniform(-1, 1, 3 * grid.size)
        tau = np.zeros(grid.size, dtype=np.int8)
        x_tilde = apply_prior_correction(x_prev, tau, body, grid)
        np.testing.assert_allclose(x_tilde, x_prev, atol=1e-12)

    def test_exposed_nodes_get_body_velocity(self):
        grid = make_grid()
        U_s = np.array([1.5, -0.5, 0.3])
        body = make_body(U_s=U_s)  # omega=0, so u_gamma = U_s everywhere
        x_prev = np.zeros(3 * grid.size)
        tau = np.zeros(grid.size, dtype=np.int8)

        # Mark nodes 5, 10, 20 as exposed
        exposed_nodes = np.array([5, 10, 20])
        tau[exposed_nodes] = 1

        x_tilde = apply_prior_correction(x_prev, tau, body, grid)

        for j in exposed_nodes:
            np.testing.assert_allclose(
                x_tilde[3*j : 3*j+3], U_s, atol=1e-12,
                err_msg=f"Exposed node {j} must carry body velocity"
            )

    def test_non_exposed_nodes_unchanged(self):
        grid = make_grid()
        body = make_body()
        x_prev = np.arange(3 * grid.size, dtype=float)
        tau = np.zeros(grid.size, dtype=np.int8)
        tau[0] = 1  # only node 0 is exposed

        x_tilde = apply_prior_correction(x_prev, tau, body, grid)

        # All nodes except 0 should be unchanged
        for j in range(1, grid.size):
            np.testing.assert_allclose(
                x_tilde[3*j : 3*j+3], x_prev[3*j : 3*j+3], atol=1e-12
            )

    def test_output_shape(self):
        grid = make_grid()
        body = make_body()
        tau = np.zeros(grid.size, dtype=np.int8)
        x_tilde = apply_prior_correction(None, tau, body, grid)
        assert x_tilde.shape == (3 * grid.size,)


# ---------------------------------------------------------------------------
# CCMPlus.reconstruct — output shapes and types
# ---------------------------------------------------------------------------

class TestCCMPlusSmoke:
    def test_reconstruct_returns_correct_types(self):
        grid = make_grid()
        config = make_config(grid)
        body = make_body()
        frame = make_frame(grid, body)
        solver = CCMPlus(config, grid)
        result = solver.reconstruct(frame)
        assert isinstance(result, ReconstructionResult)
        assert isinstance(result.velocity, np.ndarray)
        assert isinstance(result.classification, np.ndarray)
        assert isinstance(result.residual, float)
        assert isinstance(result.iterations, int)

    def test_velocity_shape(self):
        grid = make_grid()
        config = make_config(grid)
        body = make_body()
        frame = make_frame(grid, body)
        result = CCMPlus(config, grid).reconstruct(frame)
        assert result.velocity.shape == (grid.size, 3)

    def test_classification_shape_and_values(self):
        grid = make_grid()
        config = make_config(grid)
        body = make_body(radius=1.5)
        frame = make_frame(grid, body)
        result = CCMPlus(config, grid).reconstruct(frame)
        assert result.classification.shape == (grid.size,)
        assert np.all(np.isin(result.classification, [-1, 0, 1]))

    def test_residual_finite_and_positive(self):
        grid = make_grid()
        config = make_config(grid)
        body = make_body()
        frame = make_frame(grid, body)
        result = CCMPlus(config, grid).reconstruct(frame)
        assert np.isfinite(result.residual)
        assert result.residual >= 0.0

    def test_iterations_positive(self):
        grid = make_grid()
        config = make_config(grid)
        body = make_body()
        frame = make_frame(grid, body)
        result = CCMPlus(config, grid).reconstruct(frame)
        assert result.iterations >= 0

    def test_two_consecutive_timesteps(self):
        """Second call uses cached x_prev without error."""
        grid = make_grid()
        config = make_config(grid)
        body = make_body()
        solver = CCMPlus(config, grid)

        frame1 = make_frame(grid, body, t_val=0.0)
        frame2 = make_frame(grid, body, t_val=0.1)
        result1 = solver.reconstruct(frame1)
        result2 = solver.reconstruct(frame2)
        assert result1.velocity.shape == result2.velocity.shape

    def test_reset_clears_state(self):
        grid = make_grid()
        config = make_config(grid)
        body = make_body()
        solver = CCMPlus(config, grid)
        frame = make_frame(grid, body)
        solver.reconstruct(frame)
        assert solver._x_prev is not None
        solver.reset()
        assert solver._x_prev is None
        assert solver._C_prev is None

    def test_stationary_body_shell_velocity_near_zero(self):
        """No-slip: reconstructed velocity at shell nodes should be close to 0."""
        grid = make_grid()
        config = make_config(grid)
        body = make_body(center=(0., 0., 0.), radius=1.5)  # stationary
        frame = make_frame(grid, body)
        result = CCMPlus(config, grid).reconstruct(frame)
        shell = result.classification == 0
        if shell.any():
            shell_speed = np.linalg.norm(result.velocity[shell], axis=1)
            assert np.mean(shell_speed) < 0.5, (
                f"Mean shell speed {np.mean(shell_speed):.3f} too large"
            )

    def test_moving_sphere_warm_start(self):
        """Three timesteps with moving sphere: solver caches state across calls."""
        grid = make_grid()
        config = make_config(grid)
        solver = CCMPlus(config, grid)
        U_s = np.array([0.2, 0., 0.])
        dt = 0.5
        for k in range(3):
            center = (U_s * k * dt).tolist()
            body = make_body(center=center, radius=1.0, U_s=U_s.tolist())
            frame = make_frame(grid, body)
            result = solver.reconstruct(frame)
            assert result.velocity.shape == (grid.size, 3)
            assert np.isfinite(result.residual)


def make_frame(grid, body, U0=(1., 0., 0.), noise_rng=None, n_override=None, t_val=0.0):
    pts = grid.nodes.copy()
    n_p = len(pts) if n_override is None else n_override
    if n_override is not None:
        pts = pts[:n_p]
    vel = np.tile(np.array(U0, dtype=float), (n_p, 1))
    if noise_rng is not None:
        vel += noise_rng.normal(0, 0.05, vel.shape)
    unc = np.ones(n_p) * 0.1
    return FrameData(positions=pts, velocities=vel, uncertainties=unc, body=body, t=t_val)
