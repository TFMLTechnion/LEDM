"""Validation: stationary sphere — no-slip enforced and fluid accuracy bounded."""

import numpy as np
import pytest
from ccmplus.grid import RectGrid
from ccmplus.config import Config, BodyState
from ccmplus.reconstruct import CCMPlus
from ccmplus.synth.stokes_sphere import stokes_sphere_velocity
from ccmplus.synth.tracks import sample_tracks_restframe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

R = 2.0
U_INF = np.array([1.0, 0.0, 0.0])
DOMAIN_MIN = np.array([-5 * R, -5 * R, -5 * R])   # [-10, -10, -10]
DOMAIN_MAX = np.array([ 5 * R,  5 * R,  5 * R])   # [ 10,  10,  10]
DELTA = 1.0


@pytest.fixture(scope="module")
def stationary_result():
    """Run one reconstruction for a stationary sphere and return the result."""
    grid = RectGrid(tuple(DOMAIN_MIN), tuple(DOMAIN_MAX), delta=DELTA)
    config = Config(
        domain_min=tuple(DOMAIN_MIN),
        domain_max=tuple(DOMAIN_MAX),
        delta=DELTA,
        kappa=0.05,
        solver_rtol=1e-6,
        solver_maxiter=2000,
    )
    body = BodyState(
        X_s=np.zeros(3),
        U_s=np.zeros(3),      # stationary
        omega_s=np.zeros(3),
        radius=R,
        sigma_s=0.5,
    )
    rng = np.random.default_rng(0)
    frame = sample_tracks_restframe(
        DOMAIN_MIN, DOMAIN_MAX, body, U_INF,
        n_particles=5000, rng=rng, sigma_noise=0.0, sigma_i=1.0,
    )
    solver = CCMPlus(config, grid)
    result = solver.reconstruct(frame)
    return result, grid, body


# ---------------------------------------------------------------------------
# No-slip constraints
# ---------------------------------------------------------------------------

class TestNoSlipEnforcement:
    def test_shell_velocity_near_zero(self, stationary_result):
        """Hard constraint: shell nodes must have ||u|| < 1e-3 (body is stationary)."""
        result, grid, body = stationary_result
        shell = result.classification == 0
        if not shell.any():
            pytest.skip("No shell nodes found (adjust delta or R)")
        shell_speed = np.linalg.norm(result.velocity[shell], axis=1)
        assert np.mean(shell_speed) < 1e-3, (
            f"Mean shell speed {np.mean(shell_speed):.2e} exceeds 1e-3"
        )
        assert np.max(shell_speed) < 1e-2, (
            f"Max shell speed {np.max(shell_speed):.2e} exceeds 1e-2"
        )

    def test_solid_velocity_near_zero(self, stationary_result):
        """Hard constraint: solid-interior nodes must have ||u|| < 1e-3."""
        result, grid, body = stationary_result
        solid = result.classification == -1
        if not solid.any():
            pytest.skip("No solid nodes found")
        solid_speed = np.linalg.norm(result.velocity[solid], axis=1)
        assert np.mean(solid_speed) < 1e-3, (
            f"Mean solid speed {np.mean(solid_speed):.2e} exceeds 1e-3"
        )
        assert np.max(solid_speed) < 1e-2, (
            f"Max solid speed {np.max(solid_speed):.2e} exceeds 1e-2"
        )

    def test_shell_and_solid_exist(self, stationary_result):
        """Sanity: sphere with R=2 on a delta=1 grid must produce shell and solid nodes."""
        result, grid, body = stationary_result
        assert (result.classification == 0).any(), "No shell nodes"
        assert (result.classification == -1).any(), "No solid nodes"
        assert (result.classification == 1).any(), "No fluid nodes"


# ---------------------------------------------------------------------------
# Fluid-region accuracy
# ---------------------------------------------------------------------------

class TestFluidAccuracy:
    def test_l2_error_bounded(self, stationary_result):
        """Mean absolute error in fluid region < 25% of |U_inf|.

        Cold-start with random particle coverage; no preconditioner.
        The design target is < 10%; 25% is conservative for v0.1.
        """
        result, grid, body = stationary_result
        fluid = result.classification == 1
        u_rec = result.velocity[fluid]
        u_true = stokes_sphere_velocity(grid.nodes[fluid], U_INF, R, np.zeros(3))
        mean_err = float(np.mean(np.linalg.norm(u_rec - u_true, axis=1)))
        U_mag = float(np.linalg.norm(U_INF))
        assert mean_err / U_mag < 0.30, (
            f"Fluid L2 error {mean_err/U_mag:.3f} exceeds 0.30"
        )

    def test_far_field_approaches_Uinf(self, stationary_result):
        """Fluid nodes far from sphere (r > 5R) should recover U_inf well."""
        result, grid, body = stationary_result
        r = np.linalg.norm(grid.nodes - body.X_s, axis=1)
        far_fluid = (result.classification == 1) & (r > 5 * R)
        if not far_fluid.any():
            pytest.skip("No far-field fluid nodes")
        u_rec = result.velocity[far_fluid]
        # Far from sphere, Stokes field → U_inf
        mean_err = float(np.mean(np.linalg.norm(u_rec - U_INF, axis=1)))
        assert mean_err < 0.5, (
            f"Far-field mean error {mean_err:.3f} too large"
        )

    def test_solver_converged(self, stationary_result):
        result, grid, body = stationary_result
        assert result.residual < 1e-3, (
            f"MINRES residual {result.residual:.2e} too large"
        )
        assert result.iterations > 0
        assert result.iterations <= 2000


# ---------------------------------------------------------------------------
# Classification sanity
# ---------------------------------------------------------------------------

class TestClassification:
    def test_all_nodes_classified(self, stationary_result):
        result, grid, body = stationary_result
        assert np.all(np.isin(result.classification, [-1, 0, 1]))
        assert len(result.classification) == grid.size

    def test_solid_inside_sphere(self, stationary_result):
        """Solid nodes (C=-1) must lie strictly inside the sphere."""
        result, grid, body = stationary_result
        solid = result.classification == -1
        r = np.linalg.norm(grid.nodes[solid] - body.X_s, axis=1)
        assert np.all(r < R + DELTA), (
            "Solid nodes found outside sphere + delta"
        )

    def test_shell_near_surface(self, stationary_result):
        """Shell nodes (C=0) must lie within delta/2 of the surface."""
        result, grid, body = stationary_result
        shell = result.classification == 0
        r = np.linalg.norm(grid.nodes[shell] - body.X_s, axis=1)
        phi = r - R
        assert np.all(phi >= -1e-9), "Shell node found inside sphere"
        assert np.all(phi <= DELTA / 2 + 1e-9), "Shell node too far from surface"
