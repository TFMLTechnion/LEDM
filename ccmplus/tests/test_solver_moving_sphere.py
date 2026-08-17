"""Validation: translating sphere — no-slip tracks U_s, warm-start reduces error."""

import numpy as np
import pytest
from ccmplus.grid import RectGrid
from ccmplus.config import Config, BodyState
from ccmplus.reconstruct import CCMPlus
from ccmplus.synth.stokes_sphere import stokes_sphere_lab_frame
from ccmplus.synth.tracks import sample_tracks_labframe


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

R = 2.0
U_S = np.array([1.0, 0.0, 0.0])
DOMAIN_MIN = np.array([-5 * R, -5 * R, -5 * R])
DOMAIN_MAX = np.array([ 5 * R,  5 * R,  5 * R])
DELTA = 1.0
N_STEPS = 5
DT = 0.1


def _make_body(t):
    center = U_S * t
    return BodyState(
        X_s=center.copy(),
        U_s=U_S.copy(),
        omega_s=np.zeros(3),
        radius=R,
        sigma_s=0.5,
    )


@pytest.fixture(scope="module")
def moving_results():
    """Run 5 timesteps of a translating sphere, return list of (result, body)."""
    grid = RectGrid(tuple(DOMAIN_MIN), tuple(DOMAIN_MAX), delta=DELTA)
    config = Config(
        domain_min=tuple(DOMAIN_MIN),
        domain_max=tuple(DOMAIN_MAX),
        delta=DELTA,
        kappa=0.05,
        solver_rtol=1e-6,
        solver_maxiter=2000,
    )
    solver = CCMPlus(config, grid)
    rng = np.random.default_rng(7)
    results = []
    for k in range(N_STEPS):
        t = round(k * DT, 6)
        body = _make_body(t)
        frame = sample_tracks_labframe(
            DOMAIN_MIN, DOMAIN_MAX, body,
            n_particles=3000, rng=rng, sigma_noise=0.0, sigma_i=1.0, t=t,
        )
        result = solver.reconstruct(frame)
        results.append((result, body))
    return results, grid


# ---------------------------------------------------------------------------
# No-slip: shell and solid track U_s at every timestep
# ---------------------------------------------------------------------------

class TestNoSlipMoving:
    def test_shell_velocity_equals_Us(self, moving_results):
        """Shell nodes must have mean ||u - U_s|| < 1e-3 at every timestep."""
        results, grid = moving_results
        for k, (result, body) in enumerate(results):
            shell = result.classification == 0
            if not shell.any():
                continue
            err = np.linalg.norm(result.velocity[shell] - body.U_s, axis=1)
            assert np.mean(err) < 1e-3, (
                f"Step {k}: mean shell no-slip error {np.mean(err):.2e} > 1e-3"
            )

    def test_solid_velocity_equals_Us(self, moving_results):
        """Solid interior nodes must have mean ||u - U_s|| < 1e-3."""
        results, grid = moving_results
        for k, (result, body) in enumerate(results):
            solid = result.classification == -1
            if not solid.any():
                continue
            err = np.linalg.norm(result.velocity[solid] - body.U_s, axis=1)
            assert np.mean(err) < 1e-3, (
                f"Step {k}: mean solid no-slip error {np.mean(err):.2e} > 1e-3"
            )

    def test_shell_and_solid_exist_every_step(self, moving_results):
        """Sphere must produce shell and solid nodes at all 5 timesteps."""
        results, grid = moving_results
        for k, (result, body) in enumerate(results):
            assert (result.classification == 0).any(),  f"Step {k}: no shell nodes"
            assert (result.classification == -1).any(), f"Step {k}: no solid nodes"
            assert (result.classification == 1).any(),  f"Step {k}: no fluid nodes"


# ---------------------------------------------------------------------------
# Fluid accuracy improves with warm-starting
# ---------------------------------------------------------------------------

class TestWarmStart:
    def test_late_steps_more_accurate_than_cold(self, moving_results):
        """Mean fluid L2 error at steps 3-4 must be <= error at step 0.

        Warm-starting seeds x_prior from the previous solution; after a few
        steps the prior is close to the true field and accuracy improves.
        """
        results, grid = moving_results
        U_mag = float(np.linalg.norm(U_S))

        def fluid_l2(k):
            result, body = results[k]
            fluid = result.classification == 1
            if not fluid.any():
                return float("nan")
            u_true = stokes_sphere_lab_frame(grid.nodes[fluid], U_S, R, body.X_s)
            return float(np.mean(np.linalg.norm(result.velocity[fluid] - u_true, axis=1))) / U_mag

        err_cold = fluid_l2(0)
        err_warm = np.mean([fluid_l2(k) for k in range(3, N_STEPS)])
        assert err_warm <= err_cold * 1.05, (
            f"Warm-start did not help: cold={err_cold:.3f}, warm_avg={err_warm:.3f}"
        )

    def test_all_steps_fluid_error_bounded(self, moving_results):
        """Fluid L2 error < 35% of |U_s| at every timestep (cold-start tolerance)."""
        results, grid = moving_results
        U_mag = float(np.linalg.norm(U_S))
        for k, (result, body) in enumerate(results):
            fluid = result.classification == 1
            if not fluid.any():
                continue
            u_true = stokes_sphere_lab_frame(grid.nodes[fluid], U_S, R, body.X_s)
            err = float(np.mean(np.linalg.norm(result.velocity[fluid] - u_true, axis=1))) / U_mag
            assert err < 0.35, f"Step {k}: fluid L2 error {err:.3f} > 0.35"


# ---------------------------------------------------------------------------
# Transition flags: exposed nodes seeded with U_s
# ---------------------------------------------------------------------------

class TestTransitionFlags:
    def test_solver_converges_all_steps(self, moving_results):
        """MINRES must converge at every timestep (residual < 1e-3)."""
        results, grid = moving_results
        for k, (result, body) in enumerate(results):
            assert result.residual < 1e-3, (
                f"Step {k}: MINRES residual {result.residual:.2e} > 1e-3"
            )
            assert result.iterations > 0
            assert result.iterations <= 2000

    def test_classification_moves_with_sphere(self, moving_results):
        """The centroid of solid nodes must track the sphere center."""
        results, grid = moving_results
        for k, (result, body) in enumerate(results):
            solid = result.classification == -1
            if not solid.any():
                continue
            centroid = grid.nodes[solid].mean(axis=0)
            # Centroid should be within 2*delta of the true sphere center
            dist = np.linalg.norm(centroid - body.X_s)
            assert dist < 2 * DELTA, (
                f"Step {k}: solid centroid {centroid} too far from center {body.X_s} (dist={dist:.2f})"
            )
