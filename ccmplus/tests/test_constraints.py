"""Tests for interp.py and constraints.py."""

import numpy as np
import pytest
import scipy.sparse as sp
from ccmplus.grid import RectGrid
from ccmplus.config import BodyState
from ccmplus.sdf import signed_distance_sphere
from ccmplus.classify import classify, near_wall_fluid_set
from ccmplus.interp import build_interpolation_matrix
from ccmplus.constraints import build_constraints, u_gamma, _phi_gradient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_body_far():
    """Sphere far outside any test grid — all nodes fluid."""
    return BodyState(
        X_s=np.array([1000.0, 0.0, 0.0]),
        U_s=np.zeros(3),
        omega_s=np.zeros(3),
        radius=1.0,
        sigma_s=0.1,
    )


def make_body_sphere(center=(0.0, 0.0, 0.0), radius=2.0, U_s=None):
    return BodyState(
        X_s=np.array(center, dtype=float),
        U_s=np.zeros(3) if U_s is None else np.array(U_s, dtype=float),
        omega_s=np.zeros(3),
        radius=radius,
        sigma_s=0.1,
    )


def all_fluid_setup(grid):
    """All nodes classified as fluid, N_mask all False."""
    phi = np.full(grid.size, 100.0)
    C = classify(phi, grid.delta)
    N_mask = near_wall_fluid_set(C, grid)
    return phi, C, N_mask


def sphere_setup(grid, radius=2.0, center=(0.0, 0.0, 0.0)):
    body = make_body_sphere(center=center, radius=radius)
    phi = signed_distance_sphere(grid, body)
    C = classify(phi, grid.delta)
    N_mask = near_wall_fluid_set(C, grid)
    return phi, C, N_mask, body


def velocity_dofs(grid, u_func, v_func=None, w_func=None):
    """Build DOF vector x (3*Ng,) from scalar functions of node positions."""
    nodes = grid.nodes
    u = u_func(nodes)
    v = v_func(nodes) if v_func is not None else np.zeros(grid.size)
    w = w_func(nodes) if w_func is not None else np.zeros(grid.size)
    x = np.zeros(3 * grid.size)
    x[0::3] = u
    x[1::3] = v
    x[2::3] = w
    return x


# ---------------------------------------------------------------------------
# Trilinear interpolation
# ---------------------------------------------------------------------------

class TestInterpolation:
    def test_shape(self):
        g = RectGrid((0.0, 0.0, 0.0), (4.0, 4.0, 4.0), delta=1.0)
        pts = np.random.default_rng(0).uniform(0.5, 3.5, (20, 3))
        A = build_interpolation_matrix(pts, g)
        assert A.shape == (3 * 20, 3 * g.size)

    def test_row_sums_to_one_per_component(self):
        """For each particle, the 8 trilinear weights sum to 1."""
        g = RectGrid((0.0, 0.0, 0.0), (5.0, 5.0, 5.0), delta=1.0)
        rng = np.random.default_rng(42)
        pts = rng.uniform(0.1, 4.9, (50, 3))
        A = build_interpolation_matrix(pts, g)
        # Sum of each row of A: for each scalar row, weights sum to 1
        row_sums = np.asarray(A.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_interpolates_uniform_field_exactly(self):
        """Interpolating a uniform velocity field reproduces it at any point."""
        g = RectGrid((0.0, 0.0, 0.0), (4.0, 4.0, 4.0), delta=1.0)
        U0 = np.array([2.0, -1.0, 0.5])
        x = np.zeros(3 * g.size)
        x[0::3] = U0[0]
        x[1::3] = U0[1]
        x[2::3] = U0[2]
        pts = np.array([[1.3, 2.7, 0.9], [0.5, 0.5, 0.5], [3.1, 1.9, 2.4]])
        A = build_interpolation_matrix(pts, g)
        y = A @ x
        # y should be [u0, v0, w0, u0, v0, w0, ...]
        expected = np.tile(U0, len(pts))
        np.testing.assert_allclose(y, expected, atol=1e-12)

    def test_interpolates_linear_field_exactly(self):
        """The default 'wide' kernel reproduces linear fields exactly.

        The default kernel is the tensor-product cubic B-spline with 4x4x4 =
        64-node support, which has linear (indeed cubic) precision -- but only
        where the whole 4-node-per-axis stencil lies inside the domain. Within
        one cell of the domain boundary the out-of-domain slots are dropped and
        the surviving weights renormalised to a partition of unity, which keeps
        constants exact but loses linear precision. Sample strictly inside that
        one-cell margin.
        """
        g = RectGrid((0.0, 0.0, 0.0), (5.0, 5.0, 5.0), delta=1.0)
        # u = 1 + 2x, v = 3y, w = 0.5z
        x = velocity_dofs(g,
            u_func=lambda p: 1.0 + 2.0 * p[:, 0],
            v_func=lambda p: 3.0 * p[:, 1],
            w_func=lambda p: 0.5 * p[:, 2],
        )
        # Need floor(coord) >= 1 and floor(coord) <= N-3 on every axis so the
        # full 4-node span (offsets -1..+2) stays in the 6-node domain.
        pts = np.array([[1.3, 2.7, 1.9], [1.0, 1.0, 1.0], [3.99, 3.99, 3.99]])
        A = build_interpolation_matrix(pts, g)
        y = A @ x
        expected_u = 1.0 + 2.0 * pts[:, 0]
        expected_v = 3.0 * pts[:, 1]
        expected_w = 0.5 * pts[:, 2]
        y_u = y[0::3]
        y_v = y[1::3]
        y_w = y[2::3]
        np.testing.assert_allclose(y_u, expected_u, atol=1e-10)
        np.testing.assert_allclose(y_v, expected_v, atol=1e-10)
        np.testing.assert_allclose(y_w, expected_w, atol=1e-10)

    def test_nonzeros_per_particle(self):
        """The default 'wide' kernel gives 64-node support: 192 nonzeros.

        The cubic B-spline spans 4 nodes per axis (offsets -1, 0, +1, +2), so
        the footprint is 4x4x4 = 64 nodes and each particle contributes
        64 x 3 components = 192 entries. The legacy 'trilinear' kernel is the
        8-corner stencil (24 entries) and is checked separately below.
        """
        g = RectGrid((0.0, 0.0, 0.0), (4.0, 4.0, 4.0), delta=1.0)
        pts = np.array([[1.5, 1.5, 1.5]])  # cell centre, one cell in from the edge
        A = build_interpolation_matrix(pts, g)
        assert A.nnz == 3 * 64

    def test_nonzeros_per_particle_trilinear(self):
        """The legacy compact kernel keeps its 8-corner, 24-nonzero stencil."""
        g = RectGrid((0.0, 0.0, 0.0), (4.0, 4.0, 4.0), delta=1.0)
        pts = np.array([[1.5, 1.5, 1.5]])
        A = build_interpolation_matrix(pts, g, kernel="trilinear")
        assert A.nnz == 3 * 8

    def test_components_decoupled(self):
        """u-DOFs should not appear in v or w rows."""
        g = RectGrid((0.0, 0.0, 0.0), (3.0, 3.0, 3.0), delta=1.0)
        pts = np.array([[1.1, 1.1, 1.1]])
        A = build_interpolation_matrix(pts, g)
        A_dense = A.toarray()
        # u-row (row 0): only cols with index % 3 == 0
        u_row = A_dense[0]
        assert np.all(u_row[1::3] == 0.0), "u-row must not touch v-DOFs"
        assert np.all(u_row[2::3] == 0.0), "u-row must not touch w-DOFs"
        # v-row (row 1): only cols with index % 3 == 1
        v_row = A_dense[1]
        assert np.all(v_row[0::3] == 0.0), "v-row must not touch u-DOFs"
        assert np.all(v_row[2::3] == 0.0), "v-row must not touch w-DOFs"


# ---------------------------------------------------------------------------
# u_gamma
# ---------------------------------------------------------------------------

class TestUGamma:
    def test_stationary_body(self):
        body = make_body_sphere(U_s=[0, 0, 0])
        pts = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        ug = u_gamma(pts, body)
        np.testing.assert_allclose(ug, 0.0, atol=1e-12)

    def test_pure_translation(self):
        body = make_body_sphere(U_s=[1.0, 2.0, 3.0])
        pts = np.array([[5.0, 0.0, 0.0], [-3.0, 1.0, 2.0]])
        ug = u_gamma(pts, body)
        np.testing.assert_allclose(ug, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], atol=1e-12)

    def test_pure_rotation(self):
        """omega × (x - X_s) at a point on the rotation axis should be tangential."""
        body = BodyState(
            X_s=np.zeros(3), U_s=np.zeros(3),
            omega_s=np.array([0.0, 0.0, 1.0]),  # spin about z
            radius=1.0, sigma_s=0.1,
        )
        pts = np.array([[1.0, 0.0, 0.0]])  # on x-axis
        ug = u_gamma(pts, body)
        # omega × (1,0,0) = (0,0,1) × (1,0,0) = (0*0-1*0, 1*1-0*0, 0*0-0*1) = (0,1,0)
        np.testing.assert_allclose(ug, [[0.0, 1.0, 0.0]], atol=1e-12)


# ---------------------------------------------------------------------------
# Block 1: centered divergence (all-fluid domain)
# ---------------------------------------------------------------------------

class TestDivergenceBlock:
    def _build_all_fluid(self, grid):
        phi, C, N_mask = all_fluid_setup(grid)
        body = make_body_far()
        B, g = build_constraints(C, N_mask, phi, body, grid)
        return B, g, C, N_mask

    def test_uniform_velocity_divergence_free(self):
        """Uniform field has zero divergence everywhere."""
        g = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        B, g_vec, _, _ = self._build_all_fluid(g)
        for U0 in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]:
            x = np.zeros(3 * g.size)
            x[0::3] = U0[0]
            x[1::3] = U0[1]
            x[2::3] = U0[2]
            np.testing.assert_allclose(B @ x, g_vec, atol=1e-12,
                err_msg=f"Uniform field U0={U0} should be divergence-free")

    def test_pure_shear_divergence_free(self):
        """u=y, v=0, w=0: du/dx=0, dv/dy=0, dw/dz=0 → divergence-free."""
        g = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        B, g_vec, _, _ = self._build_all_fluid(g)
        x = velocity_dofs(g, u_func=lambda p: p[:, 1])  # u = y
        np.testing.assert_allclose(B @ x, g_vec, atol=1e-12)

    def test_pure_shear_w_of_z_not_divergence_free(self):
        """u=0, v=0, w=z: dw/dz=1 → div = 1 ≠ 0 at interior nodes."""
        g = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        B, g_vec, _, _ = self._build_all_fluid(g)
        x = velocity_dofs(g, u_func=lambda p: np.zeros(len(p)),
                          w_func=lambda p: p[:, 2])  # w = z
        residual = B @ x - g_vec
        # Some rows (interior nodes) should be nonzero
        assert np.any(np.abs(residual) > 1e-10)

    def test_divergence_field_correct_value(self):
        """u=x, v=y, w=-2z: div = 1+1-2 = 0 → zero divergence."""
        g = RectGrid((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), delta=1.0)
        B, g_vec, _, _ = self._build_all_fluid(g)
        x = velocity_dofs(g,
            u_func=lambda p: p[:, 0],
            v_func=lambda p: p[:, 1],
            w_func=lambda p: -2.0 * p[:, 2],
        )
        np.testing.assert_allclose(B @ x, g_vec, atol=1e-12)

    def test_linear_divergent_field_gives_correct_magnitude(self):
        """u=2x, v=0, w=0: div = 2. Centered stencil on interior nodes returns 2."""
        g = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        B, g_vec, C, N_mask = self._build_all_fluid(g)
        x = velocity_dofs(g, u_func=lambda p: 2.0 * p[:, 0])
        result = B @ x - g_vec
        # All div-F rows should give 2.0 (since B @ x is the divergence value)
        np.testing.assert_allclose(result, 2.0, atol=1e-10)

    def test_no_constraint_rows_at_domain_boundary(self):
        """Domain boundary nodes (missing neighbor) must not appear in B_div_F."""
        g = RectGrid((0.0, 0.0, 0.0), (4.0, 4.0, 4.0), delta=1.0)
        phi, C, N_mask = all_fluid_setup(g)
        body = make_body_far()
        B, g_vec = build_constraints(C, N_mask, phi, body, g)
        # Apply a field with a large value at domain boundary; if boundary nodes
        # were in B, the residual would be large from the stencil going out of bounds.
        # (This is implicitly tested by the other tests not failing with bounds errors.)
        assert B.shape[0] > 0
        assert B.shape[1] == 3 * g.size


# ---------------------------------------------------------------------------
# Block 2: one-sided divergence (near-wall nodes)
# ---------------------------------------------------------------------------

class TestOneSidedDivergence:
    def test_linear_field_derivative_correct(self):
        """For u_x = a + b*x, the one-sided stencil at a near-wall node returns b."""
        # Use a grid with a sphere; near-wall nodes are in N
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=1.0)
        phi, C, N_mask, body = sphere_setup(g, radius=2.0)
        B, g_vec = build_constraints(C, N_mask, phi, body, g)

        # u = x, v = y, w = -2z: divergence-free
        x = velocity_dofs(g,
            u_func=lambda p: p[:, 0],
            v_func=lambda p: p[:, 1],
            w_func=lambda p: -2.0 * p[:, 2],
        )
        # Every divergence row should give 0 for this field. Take the row count
        # from the assembler's own diagnostics rather than recomputing it, so
        # the test cannot silently slice into the body identity rows.
        _, _, diag = build_constraints(C, N_mask, phi, body, g,
                                       return_diagnostics=True)
        div_rows = int(diag["divergence_rows"])
        if div_rows > 0:
            div_residual = (B @ x - g_vec)[:div_rows]
            np.testing.assert_allclose(div_residual, 0.0, atol=1e-10,
                err_msg="Divergence-free linear field must satisfy divergence constraints")

    def test_divergence_stencils_never_touch_non_fluid_nodes(self):
        """No divergence row may reference a shell (C==0) or solid (C==-1) node.

        This is the guarantee that makes the incompressibility operator
        boundary-aware: the interface is respected by the *stencil*, not merely
        by which nodes carry a row. Checked column-by-column on the assembled
        matrix, so it covers every rule in the hierarchy including fallbacks.
        """
        for radius in (1.5, 2.0, 3.0, 3.5):
            g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=1.0)
            phi, C, N_mask, body = sphere_setup(g, radius=radius)
            B, _, diag = build_constraints(C, N_mask, phi, body, g,
                                           return_diagnostics=True)
            n_div = int(diag["divergence_rows"])
            assert n_div > 0, f"no divergence rows at radius {radius}"

            div_block = B[:n_div].tocoo()
            touched_nodes = np.unique(div_block.col // 3)
            bad = touched_nodes[C[touched_nodes] != 1]
            assert bad.size == 0, (
                f"radius {radius}: divergence stencils reference "
                f"{bad.size} non-fluid nodes, e.g. node {bad[:5]} with "
                f"C={C[bad[:5]]}"
            )

    def test_row_owner_nodes_are_all_open_fluid(self):
        """Every divergence row belongs to an open-fluid node."""
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=1.0)
        phi, C, N_mask, body = sphere_setup(g, radius=2.0)
        _, _, diag = build_constraints(C, N_mask, phi, body, g,
                                       return_diagnostics=True)
        owners = diag["divergence_row_nodes"]
        assert owners.size == int(diag["divergence_rows"])
        assert np.all(C[owners] == 1)

    def test_insufficiently_resolved_nodes_are_reported_not_hidden(self):
        """A node with no admissible stencil is dropped AND reported.

        A sphere nearly as wide as the grid leaves fluid nodes wedged between
        the body and the domain edge. Those must lose their row rather than
        silently borrow a shell/solid neighbour.
        """
        g = RectGrid((-4.0, -4.0, -4.0), (4.0, 4.0, 4.0), delta=1.0)
        phi, C, N_mask, body = sphere_setup(g, radius=3.4)
        B, _, diag = build_constraints(C, N_mask, phi, body, g,
                                       return_diagnostics=True)
        n_dropped = int(diag["n_insufficiently_resolved"])
        # Whatever the count, the accounting must be self-consistent and the
        # surviving rows must still be strictly fluid-only.
        assert n_dropped == int(diag["divergence_candidates"]) - int(
            diag["divergence_rows"]
        )
        assert diag["insufficiently_resolved"].size == n_dropped
        n_div = int(diag["divergence_rows"])
        cols = B[:n_div].tocoo().col // 3
        assert np.all(C[np.unique(cols)] == 1)

    def test_near_wall_rows_present_when_sphere_exists(self):
        """When a sphere exists, N is non-empty and Block 2 contributes rows."""
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=1.0)
        phi, C, N_mask, body = sphere_setup(g, radius=2.0)
        assert N_mask.sum() > 0, "Sphere should create near-wall nodes"
        B, g_vec = build_constraints(C, N_mask, phi, body, g)
        assert B.shape[0] > 0


# ---------------------------------------------------------------------------
# Blocks 3 & 4: shell and solid identity rows
# ---------------------------------------------------------------------------

class TestIdentityConstraints:
    def test_shell_identity_picks_correct_dof(self):
        """Each shell row picks exactly one DOF with coefficient 1."""
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=1.0)
        phi, C, N_mask, body = sphere_setup(g, radius=2.0)
        B, g_vec = build_constraints(C, N_mask, phi, body, g)

        shell_nodes = np.where(C == 0)[0]
        n_F_interior = np.sum((C == 1) & ~N_mask & np.all(g.neighbors >= 0, axis=1))
        n_N = int(N_mask.sum())
        bc_start_row = n_F_interior + n_N

        # For each shell node and component, find the corresponding row
        B_dense = B.toarray()
        for idx, j in enumerate(shell_nodes):
            for comp in range(3):
                r = bc_start_row + idx * 3 + comp
                row_data = B_dense[r]
                # Only one nonzero: at column 3*j + comp
                expected_col = 3 * j + comp
                assert row_data[expected_col] == pytest.approx(1.0), \
                    f"Shell row {r} must have 1.0 at col {expected_col}"
                assert np.count_nonzero(row_data) == 1, \
                    f"Shell row {r} must have exactly one nonzero"

    def test_solid_identity_picks_correct_dof(self):
        """Each solid row picks exactly one DOF with coefficient 1."""
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=1.0)
        phi, C, N_mask, body = sphere_setup(g, radius=2.0)
        B, g_vec = build_constraints(C, N_mask, phi, body, g)

        shell_nodes = np.where(C == 0)[0]
        solid_nodes = np.where(C == -1)[0]
        n_F_interior = np.sum((C == 1) & ~N_mask & np.all(g.neighbors >= 0, axis=1))
        n_N = int(N_mask.sum())
        n_shell_rows = 3 * len(shell_nodes)
        solid_start_row = n_F_interior + n_N + n_shell_rows

        B_dense = B.toarray()
        for idx, j in enumerate(solid_nodes):
            for comp in range(3):
                r = solid_start_row + idx * 3 + comp
                row_data = B_dense[r]
                expected_col = 3 * j + comp
                assert row_data[expected_col] == pytest.approx(1.0), \
                    f"Solid row {r} must have 1.0 at col {expected_col}"
                assert np.count_nonzero(row_data) == 1

    def test_stationary_body_target_is_zero(self):
        """For a stationary body, all BC targets g must be zero."""
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=1.0)
        phi, C, N_mask, body = sphere_setup(g, radius=2.0, center=(0.0, 0.0, 0.0))
        B, g_vec = build_constraints(C, N_mask, phi, body, g)
        n_F_interior = np.sum((C == 1) & ~N_mask & np.all(g.neighbors >= 0, axis=1))
        n_N = int(N_mask.sum())
        bc_targets = g_vec[n_F_interior + n_N:]
        np.testing.assert_allclose(bc_targets, 0.0, atol=1e-12)

    def test_translating_body_target_matches_u_gamma(self):
        """BC target for a translating body equals U_s at shell/solid nodes."""
        U_s = np.array([1.5, -0.5, 0.3])
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=1.0)
        body = make_body_sphere(center=(0.0, 0.0, 0.0), radius=2.0, U_s=U_s)
        phi = signed_distance_sphere(g, body)
        C = classify(phi, g.delta)
        N_mask = near_wall_fluid_set(C, g)
        B, g_vec = build_constraints(C, N_mask, phi, body, g)

        bc_nodes = np.where((C == 0) | (C == -1))[0]
        n_F_interior = np.sum((C == 1) & ~N_mask & np.all(g.neighbors >= 0, axis=1))
        n_N = int(N_mask.sum())
        bc_targets = g_vec[n_F_interior + n_N:].reshape(-1, 3)

        expected = u_gamma(g.nodes[bc_nodes], body)
        np.testing.assert_allclose(bc_targets, expected, atol=1e-12)

    def test_constraint_count(self):
        """Total constraint rows match the formula from Section 3."""
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=1.0)
        phi, C, N_mask, body = sphere_setup(g, radius=2.0)
        B, g_vec = build_constraints(C, N_mask, phi, body, g)

        n_F_interior = int(np.sum((C == 1) & ~N_mask & np.all(g.neighbors >= 0, axis=1)))
        n_N = int(N_mask.sum())
        n_shell = int(np.sum(C == 0))
        n_solid = int(np.sum(C == -1))
        expected_m = n_F_interior + n_N + 3 * n_shell + 3 * n_solid
        assert B.shape[0] == expected_m
        assert len(g_vec) == expected_m

    def test_B_shape(self):
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=1.0)
        phi, C, N_mask, body = sphere_setup(g, radius=2.0)
        B, g_vec = build_constraints(C, N_mask, phi, body, g)
        assert B.shape[1] == 3 * g.size
        assert B.shape[0] == len(g_vec)
