"""Tests for sdf, classify, near_wall_fluid_set, and transition_flags."""

import numpy as np
import pytest
from ccmplus.grid import RectGrid
from ccmplus.config import BodyState
from ccmplus.sdf import signed_distance_sphere, signed_distance_sphere_points
from ccmplus.classify import classify, near_wall_fluid_set, transition_flags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_body(radius=2.0, center=(0.0, 0.0, 0.0)):
    return BodyState(
        X_s=np.array(center, dtype=float),
        U_s=np.zeros(3),
        omega_s=np.zeros(3),
        radius=radius,
        sigma_s=0.1,
    )


# ---------------------------------------------------------------------------
# signed_distance_sphere
# ---------------------------------------------------------------------------

class TestSignedDistanceSphere:
    def test_center_is_negative(self):
        g = RectGrid((-4.0, -4.0, -4.0), (4.0, 4.0, 4.0), delta=1.0)
        body = make_body(radius=2.0)
        phi = signed_distance_sphere(g, body)
        idx_center = g.idx_from_ijk(4, 4, 4)
        assert phi[idx_center] < 0

    def test_far_node_is_positive(self):
        g = RectGrid((-4.0, -4.0, -4.0), (4.0, 4.0, 4.0), delta=1.0)
        body = make_body(radius=2.0)
        phi = signed_distance_sphere(g, body)
        idx_far = g.idx_from_ijk(0, 0, 0)  # corner at (-4,-4,-4)
        assert phi[idx_far] > 0

    def test_surface_node_near_zero(self):
        """Node at exactly (R, 0, 0) has phi ≈ 0."""
        R = 2.0
        g = RectGrid((-4.0, -4.0, -4.0), (4.0, 4.0, 4.0), delta=1.0)
        body = make_body(radius=R)
        phi = signed_distance_sphere(g, body)
        # Node (i=6,j=4,k=4) → position (2, 0, 0), r=2 → phi=0
        idx = g.idx_from_ijk(6, 4, 4)
        np.testing.assert_allclose(phi[idx], 0.0, atol=1e-12)

    def test_phi_sign_convention(self):
        """phi < 0 inside, phi > 0 outside."""
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=0.5)
        body = make_body(radius=2.0)
        phi = signed_distance_sphere(g, body)
        r = np.linalg.norm(g.nodes - body.X_s, axis=1)
        np.testing.assert_array_less(phi[r < 2.0], 0.0)
        np.testing.assert_array_less(0.0, phi[r > 2.0])

    def test_phi_equals_r_minus_R(self):
        g = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        body = make_body(radius=1.5, center=(0.0, 0.0, 0.0))
        phi = signed_distance_sphere(g, body)
        r = np.linalg.norm(g.nodes - body.X_s, axis=1)
        np.testing.assert_allclose(phi, r - body.radius, atol=1e-12)

    def test_off_center_sphere(self):
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=1.0)
        body = make_body(radius=1.0, center=(2.0, 0.0, 0.0))
        phi = signed_distance_sphere(g, body)
        # Center of sphere: nearest grid node is (2,0,0) → phi = -1
        idx = g.idx_from_ijk(7, 5, 5)  # position (2,0,0)
        np.testing.assert_allclose(phi[idx], -1.0, atol=1e-12)

    def test_signed_distance_sphere_points(self):
        body = make_body(radius=2.0)
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        phi = signed_distance_sphere_points(pts, body)
        np.testing.assert_allclose(phi, [-2.0, 0.0, 1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

class TestClassify:
    def test_solid_nodes(self):
        """Nodes with phi < 0 → C = -1."""
        phi = np.array([-1.0, -0.1, -0.001])
        C = classify(phi, delta=1.0)
        np.testing.assert_array_equal(C, [-1, -1, -1])

    def test_shell_nodes(self):
        """Nodes with 0 <= phi <= delta/2 → C = 0."""
        phi = np.array([0.0, 0.25, 0.5])
        C = classify(phi, delta=1.0)
        np.testing.assert_array_equal(C, [0, 0, 0])

    def test_fluid_nodes(self):
        """Nodes with phi > delta/2 → C = 1."""
        phi = np.array([0.501, 1.0, 10.0])
        C = classify(phi, delta=1.0)
        np.testing.assert_array_equal(C, [1, 1, 1])

    def test_boundary_between_shell_and_fluid(self):
        """phi = delta/2 is still shell (C=0); phi = delta/2 + eps is fluid (C=1)."""
        delta = 1.0
        phi_shell = np.array([delta / 2])
        phi_fluid = np.array([delta / 2 + 1e-9])
        assert classify(phi_shell, delta)[0] == 0
        assert classify(phi_fluid, delta)[0] == 1

    def test_dtype(self):
        phi = np.array([-1.0, 0.0, 1.0])
        C = classify(phi, delta=1.0)
        assert C.dtype == np.int8

    def test_sphere_solid_interior(self):
        """Nodes with |x| < R - delta/2 must classify as solid."""
        R = 2.0
        delta = 0.5
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=delta)
        body = make_body(radius=R)
        phi = signed_distance_sphere(g, body)
        C = classify(phi, delta)
        r = np.linalg.norm(g.nodes - body.X_s, axis=1)
        deep_solid = r < R - delta / 2
        assert np.all(C[deep_solid] == -1), "All deep interior nodes must be solid"

    def test_sphere_open_fluid(self):
        """Nodes with |x| > R + delta/2 must classify as fluid."""
        R = 2.0
        delta = 0.5
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=delta)
        body = make_body(radius=R)
        phi = signed_distance_sphere(g, body)
        C = classify(phi, delta)
        r = np.linalg.norm(g.nodes - body.X_s, axis=1)
        deep_fluid = r > R + delta / 2
        assert np.all(C[deep_fluid] == 1), "All deep fluid nodes must be C=+1"

    def test_sphere_shell_band_nonempty(self):
        """Shell band (C=0) should be non-empty for a sphere."""
        R = 2.0
        delta = 0.5
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=delta)
        body = make_body(radius=R)
        phi = signed_distance_sphere(g, body)
        C = classify(phi, delta)
        assert np.any(C == 0), "Shell band should be non-empty"

    def test_sphere_shell_band_thickness(self):
        """Shell nodes (C=0) must lie in [R, R + delta/2] range of r."""
        R = 2.0
        delta = 0.5
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=delta)
        body = make_body(radius=R)
        phi = signed_distance_sphere(g, body)
        C = classify(phi, delta)
        r = np.linalg.norm(g.nodes - body.X_s, axis=1)
        shell = C == 0
        assert np.all(r[shell] >= R - 1e-12)
        assert np.all(r[shell] <= R + delta / 2 + 1e-12)

    def test_all_nodes_classified(self):
        """Every node must be exactly one of {-1, 0, 1}."""
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=0.5)
        body = make_body(radius=2.0)
        phi = signed_distance_sphere(g, body)
        C = classify(phi, delta=0.5)
        assert np.all(np.isin(C, [-1, 0, 1]))


# ---------------------------------------------------------------------------
# near_wall_fluid_set
# ---------------------------------------------------------------------------

class TestNearWallFluidSet:
    def test_fluid_node_adjacent_to_shell_is_near_wall(self):
        """A C=+1 node adjacent to a C=0 node must be in N."""
        # Manual 1D-like setup: 5 nodes in x, trivial y,z
        g = RectGrid((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), delta=1.0)
        # phi arranged so: solid, solid, shell, fluid, fluid
        phi = np.array([-1.0, -0.5, 0.2, 0.8, 1.5])
        C = classify(phi, delta=1.0)
        np.testing.assert_array_equal(C, [-1, -1, 0, 1, 1])
        N = near_wall_fluid_set(C, g)
        # Node 3 (fluid) has node 2 (shell, C=0) as -x neighbor → in N
        assert N[3], "Node adjacent to shell must be near-wall"
        # Node 4 (fluid) has node 3 (fluid) as -x neighbor; no solid/shell neighbors → not in N
        assert not N[4], "Node with only fluid neighbors must not be near-wall"

    def test_fluid_node_adjacent_to_solid_is_near_wall(self):
        """A C=+1 node with a solid (C=-1) neighbor must also be in N."""
        g = RectGrid((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), delta=1.0)
        phi = np.array([-0.5, 1.0, 2.0])
        C = classify(phi, delta=1.0)
        np.testing.assert_array_equal(C, [-1, 1, 1])
        N = near_wall_fluid_set(C, g)
        assert N[1], "Fluid node next to solid must be near-wall"
        assert not N[2], "Fluid node with only fluid neighbor must not be near-wall"

    def test_solid_node_not_in_N(self):
        """C=-1 nodes must never appear in N."""
        g = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        body = make_body(radius=1.5)
        phi = signed_distance_sphere(g, body)
        C = classify(phi, delta=1.0)
        N = near_wall_fluid_set(C, g)
        assert not np.any(N & (C == -1)), "Solid nodes must not be in N"

    def test_shell_node_not_in_N(self):
        """C=0 nodes must never appear in N."""
        g = RectGrid((-3.0, -3.0, -3.0), (3.0, 3.0, 3.0), delta=1.0)
        body = make_body(radius=1.5)
        phi = signed_distance_sphere(g, body)
        C = classify(phi, delta=1.0)
        N = near_wall_fluid_set(C, g)
        assert not np.any(N & (C == 0)), "Shell nodes must not be in N"

    def test_deep_fluid_not_in_N(self):
        """Fluid nodes far from the sphere (all neighbors fluid) must not be in N."""
        g = RectGrid((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0), delta=1.0)
        body = make_body(radius=1.5)
        phi = signed_distance_sphere(g, body)
        C = classify(phi, delta=1.0)
        N = near_wall_fluid_set(C, g)
        r = np.linalg.norm(g.nodes - body.X_s, axis=1)
        deep_fluid = (C == 1) & (r > body.radius + 3.0)
        assert not np.any(N[deep_fluid]), "Deep fluid nodes must not be near-wall"

    def test_N_is_subset_of_fluid(self):
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=0.5)
        body = make_body(radius=2.0)
        phi = signed_distance_sphere(g, body)
        C = classify(phi, delta=0.5)
        N = near_wall_fluid_set(C, g)
        assert np.all((C == 1)[N]), "N must be a subset of open-fluid nodes"

    def test_domain_boundary_not_counted_as_solid(self):
        """Nodes at the domain edge should not gain near-wall status due to missing neighbors."""
        # Pure fluid domain, no body: all phi > 0 and large
        g = RectGrid((0.0, 0.0, 0.0), (4.0, 4.0, 4.0), delta=1.0)
        phi = np.full(g.size, 100.0)  # all deeply fluid
        C = classify(phi, delta=1.0)
        N = near_wall_fluid_set(C, g)
        assert not np.any(N), "No near-wall nodes in a purely fluid domain"


# ---------------------------------------------------------------------------
# transition_flags
# ---------------------------------------------------------------------------

class TestTransitionFlags:
    def test_first_timestep_all_zero(self):
        C = np.array([-1, 0, 1], dtype=np.int8)
        tau = transition_flags(C, None)
        np.testing.assert_array_equal(tau, [0, 0, 0])

    def test_exposed_node(self):
        """C was -1, now >= 0 → tau = +1."""
        C_prev = np.array([-1, -1, 1], dtype=np.int8)
        C_curr = np.array([1, 0, 1], dtype=np.int8)
        tau = transition_flags(C_curr, C_prev)
        assert tau[0] == 1, "Node going from solid to fluid must have tau=+1"
        assert tau[1] == 1, "Node going from solid to shell must have tau=+1"
        assert tau[2] == 0, "Unchanged fluid node must have tau=0"

    def test_covered_node(self):
        """C was >= 0, now < 0 → tau = -1."""
        C_prev = np.array([1, 0, -1], dtype=np.int8)
        C_curr = np.array([-1, -1, -1], dtype=np.int8)
        tau = transition_flags(C_curr, C_prev)
        assert tau[0] == -1, "Node going from fluid to solid must have tau=-1"
        assert tau[1] == -1, "Node going from shell to solid must have tau=-1"
        assert tau[2] == 0, "Already-solid node must have tau=0"

    def test_unchanged_nodes_have_tau_zero(self):
        C_prev = np.array([-1, 0, 1], dtype=np.int8)
        C_curr = np.array([-1, 0, 1], dtype=np.int8)
        tau = transition_flags(C_curr, C_prev)
        np.testing.assert_array_equal(tau, [0, 0, 0])

    def test_dtype(self):
        C = np.array([1, -1, 0], dtype=np.int8)
        tau = transition_flags(C, C.copy())
        assert tau.dtype == np.int8

    def test_sphere_moving_through_grid(self):
        """When sphere moves, newly uncovered nodes should get tau=+1."""
        g = RectGrid((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), delta=0.5)
        # Sphere at x=-1 → classify
        body1 = make_body(radius=1.0, center=(-1.0, 0.0, 0.0))
        phi1 = signed_distance_sphere(g, body1)
        C1 = classify(phi1, delta=0.5)

        # Sphere at x=+1 → some previously solid nodes are now fluid
        body2 = make_body(radius=1.0, center=(1.0, 0.0, 0.0))
        phi2 = signed_distance_sphere(g, body2)
        C2 = classify(phi2, delta=0.5)

        tau = transition_flags(C2, C1)
        # Exposed: C_prev == -1 and C_curr >= 0
        exposed = (C1 < 0) & (C2 >= 0)
        assert np.all(tau[exposed] == 1)
        # Covered: C_prev >= 0 and C_curr < 0
        covered = (C1 >= 0) & (C2 < 0)
        assert np.all(tau[covered] == -1)
        # Unchanged fluid stays zero
        both_fluid = (C1 == 1) & (C2 == 1)
        assert np.all(tau[both_fluid] == 0)
