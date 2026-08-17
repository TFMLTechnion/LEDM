"""Tests for RectGrid: construction, idx<->ijk roundtrip, neighbor table."""

import numpy as np
import pytest
from ccmplus.grid import RectGrid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid():
    """4x3x2 grid over [0,3]x[0,2]x[0,1] with delta=1."""
    return RectGrid((0.0, 0.0, 0.0), (3.0, 2.0, 1.0), delta=1.0)


@pytest.fixture
def cube_grid():
    """3x3x3 grid over [-1,1]^3 with delta=1."""
    return RectGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), delta=1.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_shape(self, small_grid):
        assert small_grid.Nx == 4
        assert small_grid.Ny == 3
        assert small_grid.Nz == 2

    def test_size(self, small_grid):
        assert small_grid.size == 4 * 3 * 2

    def test_nodes_shape(self, small_grid):
        assert small_grid.nodes.shape == (small_grid.size, 3)

    def test_node_coordinates_corner(self, small_grid):
        # idx 0 should be (i=0,j=0,k=0) -> position (0,0,0)
        np.testing.assert_allclose(small_grid.nodes[0], [0.0, 0.0, 0.0])

    def test_node_coordinates_last(self, small_grid):
        # last node (i=3,j=2,k=1) -> position (3,2,1)
        np.testing.assert_allclose(small_grid.nodes[-1], [3.0, 2.0, 1.0])

    def test_x_coords(self, small_grid):
        np.testing.assert_allclose(small_grid.x_coords, [0.0, 1.0, 2.0, 3.0])

    def test_y_coords(self, small_grid):
        np.testing.assert_allclose(small_grid.y_coords, [0.0, 1.0, 2.0])

    def test_z_coords(self, small_grid):
        np.testing.assert_allclose(small_grid.z_coords, [0.0, 1.0])

    def test_non_unit_delta(self):
        g = RectGrid((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), delta=0.5)
        assert g.Nx == 5
        assert g.Ny == 5
        assert g.Nz == 5
        assert g.size == 125


# ---------------------------------------------------------------------------
# idx <-> ijk roundtrip
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_ijk_from_idx_origin(self, small_grid):
        assert small_grid.ijk_from_idx(0) == (0, 0, 0)

    def test_ijk_from_idx_known(self, small_grid):
        # idx = i + Nx*(j + Ny*k) = 1 + 4*(0 + 3*0) = 1
        assert small_grid.ijk_from_idx(1) == (1, 0, 0)
        # idx = 0 + 4*(1 + 3*0) = 4
        assert small_grid.ijk_from_idx(4) == (0, 1, 0)
        # idx = 0 + 4*(0 + 3*1) = 12
        assert small_grid.ijk_from_idx(12) == (0, 0, 1)

    def test_idx_from_ijk_origin(self, small_grid):
        assert small_grid.idx_from_ijk(0, 0, 0) == 0

    def test_idx_from_ijk_known(self, small_grid):
        assert small_grid.idx_from_ijk(1, 0, 0) == 1
        assert small_grid.idx_from_ijk(0, 1, 0) == 4
        assert small_grid.idx_from_ijk(0, 0, 1) == 12

    def test_roundtrip_idx_to_ijk_to_idx(self, small_grid):
        for idx in range(small_grid.size):
            i, j, k = small_grid.ijk_from_idx(idx)
            assert small_grid.idx_from_ijk(i, j, k) == idx

    def test_roundtrip_ijk_to_idx_to_ijk(self, small_grid):
        Nx, Ny, Nz = small_grid.Nx, small_grid.Ny, small_grid.Nz
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    idx = small_grid.idx_from_ijk(i, j, k)
                    assert small_grid.ijk_from_idx(idx) == (i, j, k)

    def test_vectorised_roundtrip(self, small_grid):
        idx = np.arange(small_grid.size)
        ijk = small_grid.ijk_from_idx_array(idx)
        idx2 = small_grid.idx_from_ijk_array(ijk)
        np.testing.assert_array_equal(idx, idx2)

    def test_all_nodes_covered(self, small_grid):
        """Every node appears exactly once in the nodes array."""
        Nx, Ny, Nz = small_grid.Nx, small_grid.Ny, small_grid.Nz
        expected = set()
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    expected.add((i, j, k))
        recovered = set()
        for idx in range(small_grid.size):
            recovered.add(small_grid.ijk_from_idx(idx))
        assert expected == recovered


# ---------------------------------------------------------------------------
# Neighbor table
# ---------------------------------------------------------------------------

class TestNeighborTable:
    def test_shape(self, small_grid):
        assert small_grid.neighbors.shape == (small_grid.size, 6)

    def test_interior_node_no_minus_ones(self, small_grid):
        # Node (1,1,0) in a 4x3x2 grid is interior in x,y but boundary in z
        idx = small_grid.idx_from_ijk(1, 1, 0)
        nb = small_grid.neighbors[idx]
        # -x, +x, -y, +y neighbors all valid
        assert nb[0] != -1  # -x
        assert nb[1] != -1  # +x
        assert nb[2] != -1  # -y
        assert nb[3] != -1  # +y
        # -z boundary
        assert nb[4] == -1
        # +z valid (k=0 has k=1 neighbor)
        assert nb[5] != -1

    def test_corner_node_neighbors(self, small_grid):
        # Origin corner (0,0,0): -x,-y,-z all missing
        idx = small_grid.idx_from_ijk(0, 0, 0)
        nb = small_grid.neighbors[idx]
        assert nb[0] == -1  # -x missing
        assert nb[1] != -1  # +x present
        assert nb[2] == -1  # -y missing
        assert nb[3] != -1  # +y present
        assert nb[4] == -1  # -z missing
        assert nb[5] != -1  # +z present

    def test_far_corner_neighbors(self, small_grid):
        # Far corner (Nx-1, Ny-1, Nz-1): +x,+y,+z all missing
        idx = small_grid.idx_from_ijk(small_grid.Nx - 1, small_grid.Ny - 1, small_grid.Nz - 1)
        nb = small_grid.neighbors[idx]
        assert nb[0] != -1  # -x present
        assert nb[1] == -1  # +x missing
        assert nb[2] != -1  # -y present
        assert nb[3] == -1  # +y missing
        assert nb[4] != -1  # -z present
        assert nb[5] == -1  # +z missing

    def test_neighbor_values_correct(self, small_grid):
        # Node (2, 1, 0): -x neighbor is (1,1,0), +x is (3,1,0), -y is (2,0,0), +y is (2,2,0)
        idx = small_grid.idx_from_ijk(2, 1, 0)
        nb = small_grid.neighbors[idx]
        assert nb[0] == small_grid.idx_from_ijk(1, 1, 0)
        assert nb[1] == small_grid.idx_from_ijk(3, 1, 0)
        assert nb[2] == small_grid.idx_from_ijk(2, 0, 0)
        assert nb[3] == small_grid.idx_from_ijk(2, 2, 0)
        assert nb[4] == -1   # -z: k=0 is boundary
        assert nb[5] == small_grid.idx_from_ijk(2, 1, 1)

    def test_neighbor_symmetry(self, small_grid):
        """If A's +x neighbor is B, then B's -x neighbor is A."""
        for idx in range(small_grid.size):
            nb = small_grid.neighbors[idx]
            # +x neighbor
            if nb[1] != -1:
                assert small_grid.neighbors[nb[1], 0] == idx
            # +y neighbor
            if nb[3] != -1:
                assert small_grid.neighbors[nb[3], 2] == idx
            # +z neighbor
            if nb[5] != -1:
                assert small_grid.neighbors[nb[5], 4] == idx

    def test_boundary_node_detection(self, small_grid):
        assert small_grid.is_boundary_node(small_grid.idx_from_ijk(0, 0, 0))
        # (1,1,0) has k=0 → -z is missing, so IS a boundary node
        assert small_grid.is_boundary_node(small_grid.idx_from_ijk(1, 1, 0))

    def test_fully_interior_node_in_larger_grid(self):
        g = RectGrid((0.0, 0.0, 0.0), (4.0, 4.0, 4.0), delta=1.0)
        idx = g.idx_from_ijk(2, 2, 2)
        assert not g.is_boundary_node(idx)


# ---------------------------------------------------------------------------
# Node position
# ---------------------------------------------------------------------------

class TestNodePosition:
    def test_node_position_origin(self, small_grid):
        np.testing.assert_allclose(small_grid.node_position(0), [0.0, 0.0, 0.0])

    def test_node_position_matches_nodes_array(self, small_grid):
        for idx in range(small_grid.size):
            np.testing.assert_allclose(
                small_grid.node_position(idx), small_grid.nodes[idx]
            )

    def test_node_position_known(self, small_grid):
        idx = small_grid.idx_from_ijk(2, 1, 0)
        np.testing.assert_allclose(small_grid.node_position(idx), [2.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# cell_containing
# ---------------------------------------------------------------------------

class TestCellContaining:
    def test_at_origin(self, small_grid):
        ijk0, frac = small_grid.cell_containing(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(ijk0[0], [0, 0, 0])
        np.testing.assert_allclose(frac[0], [0.0, 0.0, 0.0])

    def test_interior_point(self, small_grid):
        ijk0, frac = small_grid.cell_containing(np.array([1.3, 0.7, 0.5]))
        np.testing.assert_array_equal(ijk0[0], [1, 0, 0])
        np.testing.assert_allclose(frac[0], [0.3, 0.7, 0.5], atol=1e-14)

    def test_batch_points(self, small_grid):
        pts = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 0.5], [2.9, 1.9, 0.9]])
        ijk0, frac = small_grid.cell_containing(pts)
        assert ijk0.shape == (3, 3)
        assert frac.shape == (3, 3)
        np.testing.assert_array_equal(ijk0[1], [1, 1, 0])
        np.testing.assert_allclose(frac[1], [0.5, 0.5, 0.5], atol=1e-14)

    def test_fraction_in_unit_interval(self):
        g = RectGrid((0.0, 0.0, 0.0), (5.0, 5.0, 5.0), delta=1.0)
        rng = np.random.default_rng(42)
        pts = rng.uniform(0.0, 4.999, size=(1000, 3))
        _, frac = g.cell_containing(pts)
        assert np.all(frac >= 0.0)
        assert np.all(frac < 1.0 + 1e-10)
