"""Solver tests: no body (all-fluid), reduces to standard CCM."""

import numpy as np
import pytest
from ccmplus.grid import RectGrid
from ccmplus.config import Config, BodyState
from ccmplus.sdf import signed_distance_sphere_points
from ccmplus.classify import classify, near_wall_fluid_set
from ccmplus.interp import build_interpolation_matrix
from ccmplus.constraints import build_constraints
from ccmplus.solver import build_weight_matrix, solve_saddle_point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**kwargs):
    base = dict(
        domain_min=(-3.0, -3.0, -3.0),
        domain_max=(3.0, 3.0, 3.0),
        delta=1.0,
        kappa=0.05,
        solver_rtol=1e-7,
        solver_maxiter=2000,
    )
    base.update(kwargs)
    return Config(**base)


def make_body_far():
    return BodyState(
        X_s=np.array([1000.0, 0.0, 0.0]),
        U_s=np.zeros(3), omega_s=np.zeros(3),
        radius=1.0, sigma_s=0.1,
    )


def cell_center_particles(grid):
    """Particles at all cell centres of the grid."""
    xs = grid.x_coords[:-1] + grid.delta / 2
    ys = grid.y_coords[:-1] + grid.delta / 2
    zs = grid.z_coords[:-1] + grid.delta / 2
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def node_particles(grid):
    """Particles at all grid node positions.

    With particles at nodes, A is block-diagonal with 1-weight per node per component,
    so A^T W A = W (diagonal) — a perfectly conditioned data term.
    """
    return grid.nodes.copy()


def run_solver(grid, positions, velocities, uncertainties, body, config,
               kernel="wide"):
    """Full pipeline: classify → interpolation → weights → constraints → solve."""
    phi_grid = np.full(grid.size, 1000.0)   # all fluid
    C = classify(phi_grid, grid.delta)
    N_mask = near_wall_fluid_set(C, grid)

    A = build_interpolation_matrix(positions, grid, kernel=kernel)
    phi_particles = np.full(len(positions), 1000.0)
    W = build_weight_matrix(phi_particles, uncertainties, sigma_s=0.1)
    B, g_vec = build_constraints(C, N_mask, phi_grid, body, grid)

    y = velocities.ravel()
    x_prior = np.zeros(3 * grid.size)
    return solve_saddle_point(A, W, y, B, g_vec, x_prior, config), C


def tight_config(**kwargs):
    """A config whose MINRES solve is tight enough to judge the CONSTRAINTS.

    The default (rtol 1e-6..1e-7, maxiter 2000) stops while the normalized
    divergence residual is still ~1e-3. That is a property of how far MINRES
    was run, not of the discretisation: tightening the solve drives it to
    ~1e-5 with no change to the operators. Constraint-satisfaction tests
    therefore tighten the solve rather than relaxing the physics tolerance.
    """
    base = dict(solver_rtol=1e-10, solver_maxiter=20000,
                use_jacobi_precond=True)
    base.update(kwargs)
    return make_config(**base)


# Documented constraint tolerances asserted by the tests below.
#   Delta * rms(div u) / U_ref  -- dimensionless divergence error per cell
DIV_RMS_NORM_TOL = 1e-4
#   Delta * max|div u| / U_ref
DIV_MAX_NORM_TOL = 1e-3


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_minres_converges_uniform_field(self):
        """MINRES must converge for a small, well-posed all-fluid problem."""
        grid = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        pts = cell_center_particles(grid)
        n_p = len(pts)
        U0 = np.array([1.0, 0.0, 0.0])
        vel = np.tile(U0, (n_p, 1))
        unc = np.ones(n_p)
        body = make_body_far()
        config = make_config()

        (x, lam, info), C = run_solver(grid, pts, vel, unc, body, config)
        assert info.converged, (
            f"MINRES did not converge (flag). "
            f"Residual={info.residual:.3e}, iters={info.iterations}"
        )

    def test_residual_small_after_convergence(self):
        grid = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        pts = cell_center_particles(grid)
        n_p = len(pts)
        vel = np.tile([0.5, -0.3, 0.1], (n_p, 1))
        unc = np.ones(n_p)
        body = make_body_far()
        config = make_config()

        (x, lam, info), C = run_solver(grid, pts, vel, unc, body, config)
        assert info.residual < 1e-4, (
            f"Relative residual too large: {info.residual:.3e}"
        )


# ---------------------------------------------------------------------------
# Accuracy: reconstructed field close to ground truth
# ---------------------------------------------------------------------------

class TestAccuracy:
    def _interior_fluid_mask(self, grid, C):
        boundary = np.array([grid.is_boundary_node(i) for i in range(grid.size)])
        return (C == 1) & ~boundary

    def test_uniform_field_recovered(self):
        """Uniform noiseless data → reconstructed velocity ≈ U0 at all nodes.

        Particles at grid nodes make A block-identity → A^T W A = (1/sigma^2)*I.
        Bias error ≈ kappa/(1/sigma^2 + kappa) * |U0| = 0.01/(1+0.01) ≈ 0.01.
        """
        grid = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        pts = node_particles(grid)
        n_p = len(pts)
        U0 = np.array([1.0, 0.0, 0.0])
        vel = np.tile(U0, (n_p, 1))
        unc = np.ones(n_p)          # sigma=1 → W_ii=1, balanced with kappa=0.01

        body = make_body_far()
        config = make_config(kappa=0.01)

        (x, lam, info), C = run_solver(grid, pts, vel, unc, body, config)
        assert info.converged, f"MINRES did not converge: res={info.residual:.2e}"
        u_rec = x.reshape(-1, 3)
        mask = self._interior_fluid_mask(grid, C)
        err = np.linalg.norm(u_rec[mask] - U0, axis=1)
        assert np.mean(err) < 0.05, (
            f"Mean error {np.mean(err):.4f} too large for uniform field recovery"
        )

    def test_y_direction_uniform_field(self):
        """Same as above but for y-direction to ensure no axis bias."""
        grid = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        pts = node_particles(grid)
        n_p = len(pts)
        U0 = np.array([0.0, 1.0, 0.0])
        vel = np.tile(U0, (n_p, 1))
        unc = np.ones(n_p)

        body = make_body_far()
        config = make_config(kappa=0.01)

        (x, lam, info), C = run_solver(grid, pts, vel, unc, body, config)
        assert info.converged
        u_rec = x.reshape(-1, 3)
        mask = self._interior_fluid_mask(grid, C)
        err = np.linalg.norm(u_rec[mask] - U0, axis=1)
        assert np.mean(err) < 0.05

    def _noisy_case(self, seed=7, sigma_noise=0.1):
        rng = np.random.default_rng(seed)
        grid = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        pts = node_particles(grid)
        n_p = len(pts)
        U0 = np.array([1.0, 0.0, 0.0])
        vel = np.tile(U0, (n_p, 1)) + rng.normal(0, sigma_noise, (n_p, 3))
        unc = np.full(n_p, sigma_noise)
        return grid, pts, U0, vel, unc

    def test_noisy_data_bounded_error(self):
        """Error is O(sigma_noise) when the interpolation operator is the identity.

        The premise is that with particles ON the nodes and the COMPACT
        (trilinear) kernel, A is exactly the identity, so A^T W A is diagonal
        and W_ii = 1/sigma^2 = 100 >> kappa = 0.1 lets the data through nearly
        unchanged: error ~ sigma_noise per node. That is a statement about the
        solver, so the kernel is pinned rather than left at the default -- see
        test_noisy_data_wide_kernel_is_unbiased for what the default kernel
        does, which is a different (and legitimately noisier) question.
        """
        sigma_noise = 0.1
        grid, pts, U0, vel, unc = self._noisy_case(sigma_noise=sigma_noise)
        config = tight_config(kappa=0.1)

        (x, lam, info), C = run_solver(grid, pts, vel, unc, make_body_far(),
                                       config, kernel="trilinear")
        assert info.converged
        u_rec = x.reshape(-1, 3)
        mask = self._interior_fluid_mask(grid, C)
        err = np.linalg.norm(u_rec[mask] - U0, axis=1)
        assert np.mean(err) < 3 * sigma_noise, (
            f"Reconstruction error {np.mean(err):.4f} exceeds "
            f"3*sigma_noise={3 * sigma_noise}"
        )

    def test_noisy_data_wide_kernel_is_unbiased(self):
        """The default wide kernel amplifies noise but stays unbiased.

        The cubic B-spline spreads each track over 4x4x4 nodes, so recovering
        nodal values from track values is a deconvolution. Deconvolution
        amplifies high-wavenumber noise -- with only kappa = 0.1 of Tikhonov
        damping the per-node error is several times sigma_noise, which is a
        real property of the estimator and not a solver defect. What must still
        hold is that the amplified error is zero-mean (no systematic bias) and
        that it is suppressed by stronger regularisation.
        """
        sigma_noise = 0.1
        grid, pts, U0, vel, unc = self._noisy_case(sigma_noise=sigma_noise)

        errs = {}
        for kappa in (0.1, 10.0):
            (x, lam, info), C = run_solver(
                grid, pts, vel, unc, make_body_far(), tight_config(kappa=kappa)
            )
            assert info.converged
            u_rec = x.reshape(-1, 3)
            mask = self._interior_fluid_mask(grid, C)
            errs[kappa] = float(
                np.mean(np.linalg.norm(u_rec[mask] - U0, axis=1))
            )
            # Unbiased: the SPATIAL MEAN of the field recovers U0 even though
            # individual nodes are noisy.
            bias = np.linalg.norm(u_rec[mask].mean(axis=0) - U0)
            assert bias < 3 * sigma_noise, (
                f"kappa={kappa}: field mean is biased by {bias:.4f}"
            )

        # Stronger Tikhonov damping must reduce the amplified noise.
        assert errs[10.0] < errs[0.1], (
            f"raising kappa did not damp the noise: {errs}"
        )

    def test_warm_start_does_not_degrade_accuracy(self):
        """Starting from x_prior = ground truth should give same or better accuracy."""
        grid = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        pts = cell_center_particles(grid)
        n_p = len(pts)
        U0 = np.array([1.0, 0.0, 0.0])
        vel = np.tile(U0, (n_p, 1))
        unc = np.ones(n_p) * 0.01

        phi_grid = np.full(grid.size, 1000.0)
        C = classify(phi_grid, grid.delta)
        N_mask = near_wall_fluid_set(C, grid)
        body = make_body_far()
        config = make_config(kappa=1e-3)

        A = build_interpolation_matrix(pts, grid)
        phi_pts = np.full(n_p, 1000.0)
        W = build_weight_matrix(phi_pts, unc, sigma_s=0.1)
        B, g_vec = build_constraints(C, N_mask, phi_grid, body, grid)
        y = vel.ravel()

        # Warm start from U0
        x_prior = np.zeros(3 * grid.size)
        x_prior[0::3] = U0[0]
        x_prior[1::3] = U0[1]
        x_prior[2::3] = U0[2]

        x, lam, info = solve_saddle_point(A, W, y, B, g_vec, x_prior, config)
        assert info.converged
        u_rec = x.reshape(-1, 3)
        boundary = np.array([grid.is_boundary_node(i) for i in range(grid.size)])
        mask = (C == 1) & ~boundary
        err = np.linalg.norm(u_rec[mask] - U0, axis=1)
        assert np.mean(err) < 0.05


# ---------------------------------------------------------------------------
# Divergence-free enforcement
# ---------------------------------------------------------------------------

class TestDivergenceFreeEnforcement:
    def test_reconstructed_field_divergence_free_at_interior(self):
        """Divergence is zero to the documented NORMALIZED tolerance.

        A divergence row has units of velocity/length, so an absolute threshold
        on ``B @ x`` is not a physically meaningful statement -- it moves with
        the grid spacing and the flow speed. The defensible quantity is the
        dimensionless divergence error per cell,

            Delta * rms(div u) / U_ref ,

        which the solver reports directly in ``SolverInfo.constraints``.
        """
        grid = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        pts = cell_center_particles(grid)
        n_p = len(pts)
        U0 = np.array([1.0, 0.5, -0.3])
        vel = np.tile(U0, (n_p, 1))
        unc = np.ones(n_p) * 0.01

        (x, lam, info), C = run_solver(grid, pts, vel, unc, make_body_far(),
                                       tight_config(kappa=1e-3))
        assert info.converged

        cd = info.constraints
        assert cd is not None
        assert cd.n_div_rows > 0
        assert cd.div_rms_norm < DIV_RMS_NORM_TOL, (
            f"Delta*rms(div u)/U_ref = {cd.div_rms_norm:.3e} exceeds "
            f"{DIV_RMS_NORM_TOL:.1e}"
        )
        assert cd.div_max_norm < DIV_MAX_NORM_TOL, (
            f"Delta*max|div u|/U_ref = {cd.div_max_norm:.3e} exceeds "
            f"{DIV_MAX_NORM_TOL:.1e}"
        )
        # And the solver agrees it met its own configured tolerance.
        assert cd.div_ok

    def test_divergence_tolerance_is_reported_not_assumed(self):
        """A deliberately under-converged solve is FLAGGED, not silently passed."""
        grid = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        pts = cell_center_particles(grid)
        n_p = len(pts)
        vel = np.tile([1.0, 0.5, -0.3], (n_p, 1))
        unc = np.ones(n_p) * 0.01

        # Stop MINRES far too early, and demand a tight divergence tolerance.
        config = make_config(kappa=1e-3, solver_rtol=1e-2, solver_maxiter=3,
                             constraint_div_tol=1e-12)
        with pytest.warns(RuntimeWarning, match="divergence constraint not met"):
            (x, lam, info), C = run_solver(grid, pts, vel, unc,
                                           make_body_far(), config)
        assert info.constraints is not None
        assert not info.constraints.div_ok

    def test_constraint_satisfaction_general(self):
        """Every constraint family is satisfied to its own documented tolerance.

        Divergence rows are judged by the normalized measure; body identity
        (no-slip) rows are Dirichlet rows with an O(1) right-hand side and are
        judged by a relative residual. This case is all-fluid, so it also
        pins down that the body block is genuinely empty rather than
        vacuously "satisfied".
        """
        grid = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        pts = cell_center_particles(grid)
        n_p = len(pts)
        vel = np.tile([0.7, -0.4, 0.2], (n_p, 1))
        unc = np.ones(n_p) * 0.01

        (x, lam, info), C = run_solver(grid, pts, vel, unc, make_body_far(),
                                       tight_config(kappa=1e-3))
        assert info.converged

        cd = info.constraints
        assert cd is not None
        assert cd.n_body_rows == 0, "all-fluid case must have no body rows"
        assert cd.n_div_rows > 0
        assert cd.div_rms_norm < DIV_RMS_NORM_TOL
        assert cd.div_max_norm < DIV_MAX_NORM_TOL
        assert cd.ok

    def test_body_rows_hold_to_solver_tolerance(self):
        """With a body present, the no-slip identity rows hold tightly.

        Unlike the divergence rows these are exact algebraic conditions
        (u_j = u_Gamma(x_j)), so they should be satisfied to roughly the MINRES
        tolerance and get a relative, not normalized, criterion.
        """
        grid = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=1.0)
        body = BodyState(
            X_s=np.zeros(3), U_s=np.array([0.0, 2.0, 0.0]),
            omega_s=np.zeros(3), radius=2.0, sigma_s=0.1,
        )
        pts = cell_center_particles(grid)
        keep = np.linalg.norm(pts - body.X_s, axis=1) > body.radius * 1.2
        pts = pts[keep]
        n_p = len(pts)
        vel = np.tile([0.0, 2.0, 0.0], (n_p, 1))
        unc = np.ones(n_p) * 0.01

        phi_grid = signed_distance_sphere_points(grid.nodes, body)
        C = classify(phi_grid, grid.delta)
        N_mask = near_wall_fluid_set(C, grid)
        A = build_interpolation_matrix(pts, grid, allowed_nodes=(C == 1))
        W = build_weight_matrix(np.full(n_p, 1000.0), unc, sigma_s=0.1)
        B, g_vec = build_constraints(C, N_mask, phi_grid, body, grid)

        config = tight_config(kappa=1e-3)
        x, lam, info = solve_saddle_point(
            A, W, vel.ravel(), B, g_vec, np.zeros(3 * grid.size), config
        )

        cd = info.constraints
        assert cd is not None
        assert cd.n_body_rows > 0, "sphere should produce shell/solid rows"
        assert cd.body_ok, (
            f"no-slip relative residual {cd.body_rel:.3e} exceeds "
            f"{cd.body_tol:.1e}"
        )


# ---------------------------------------------------------------------------
# Weight matrix
# ---------------------------------------------------------------------------

class TestWeightMatrix:
    def test_no_body_weights_are_inverse_variance(self):
        """Far from body (phi >> sigma_s), proximity factor → 1, W_ii = 1/sigma_i^2."""
        phi = np.full(10, 100.0)
        sigma = np.array([0.5, 1.0, 2.0, 0.1, 0.2, 3.0, 0.3, 0.7, 1.5, 0.8])
        W = build_weight_matrix(phi, sigma, sigma_s=0.1)
        w_diag = W.diagonal()
        expected = np.repeat(1.0 / sigma ** 2, 3)
        np.testing.assert_allclose(w_diag, expected, rtol=1e-6)

    def test_at_surface_weight_is_zero(self):
        """phi = 0 (particle at body surface): proximity factor = 0, W_ii = 0."""
        phi = np.array([0.0])
        sigma = np.array([1.0])
        W = build_weight_matrix(phi, sigma, sigma_s=0.5)
        np.testing.assert_allclose(W.diagonal(), 0.0, atol=1e-12)

    def test_inside_body_weight_is_zero(self):
        """phi < 0 (inside body): clamped to 0, same as surface."""
        phi = np.array([-1.0, -5.0])
        sigma = np.ones(2)
        W = build_weight_matrix(phi, sigma, sigma_s=0.5)
        np.testing.assert_allclose(W.diagonal(), 0.0, atol=1e-12)

    def test_weight_matrix_shape(self):
        n_p = 15
        phi = np.ones(n_p)
        sigma = np.ones(n_p)
        W = build_weight_matrix(phi, sigma, sigma_s=0.5)
        assert W.shape == (3 * n_p, 3 * n_p)

    def test_weight_repeats_per_component(self):
        """Each particle's weight is repeated 3 times (u, v, w)."""
        phi = np.array([1.0, 2.0])
        sigma = np.array([0.5, 1.0])
        W = build_weight_matrix(phi, sigma, sigma_s=0.5)
        d = W.diagonal()
        assert d[0] == d[1] == d[2]
        assert d[3] == d[4] == d[5]
        assert d[0] != d[3]
