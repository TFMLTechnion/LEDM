"""3D uniform Cartesian grid with node indexing and neighbor table."""

from __future__ import annotations

import numpy as np


class RectGrid:
    """Uniform Cartesian grid over a rectangular domain.

    Node ordering: flat index idx = i + Nx*(j + Ny*k)
    where i, j, k are indices along x, y, z axes respectively.

    Velocity DOFs: for node idx, u-DOF = 3*idx, v-DOF = 3*idx+1, w-DOF = 3*idx+2.
    """

    def __init__(
        self,
        domain_min: tuple[float, float, float],
        domain_max: tuple[float, float, float],
        delta: float,
    ) -> None:
        self.domain_min = np.asarray(domain_min, dtype=float)
        self.domain_max = np.asarray(domain_max, dtype=float)
        self.delta = float(delta)

        # Number of nodes along each axis (inclusive of both endpoints)
        extents = self.domain_max - self.domain_min
        self.Nx = int(round(extents[0] / delta)) + 1
        self.Ny = int(round(extents[1] / delta)) + 1
        self.Nz = int(round(extents[2] / delta)) + 1
        self.shape = (self.Nx, self.Ny, self.Nz)
        self.size = self.Nx * self.Ny * self.Nz  # Ng

        # Node coordinates
        self.x_coords = self.domain_min[0] + np.arange(self.Nx) * delta
        self.y_coords = self.domain_min[1] + np.arange(self.Ny) * delta
        self.z_coords = self.domain_min[2] + np.arange(self.Nz) * delta

        # (Ng, 3) array of node positions
        # idx = i + Nx*(j + Ny*k) means i varies fastest → Fortran-order ravel
        ii, jj, kk = np.meshgrid(
            np.arange(self.Nx), np.arange(self.Ny), np.arange(self.Nz), indexing="ij"
        )
        self.nodes = np.stack(
            [
                self.x_coords[ii.ravel(order="F")],
                self.y_coords[jj.ravel(order="F")],
                self.z_coords[kk.ravel(order="F")],
            ],
            axis=1,
        )  # (Ng, 3)

        self.neighbors = self._build_neighbor_table()

    # ------------------------------------------------------------------
    # Index conversion
    # ------------------------------------------------------------------

    def idx_from_ijk(self, i: int, j: int, k: int) -> int:
        """Flat index from (i, j, k) grid indices. No bounds check."""
        return i + self.Nx * (j + self.Ny * k)

    def ijk_from_idx(self, idx: int) -> tuple[int, int, int]:
        """(i, j, k) from flat index."""
        i = idx % self.Nx
        remainder = idx // self.Nx
        j = remainder % self.Ny
        k = remainder // self.Ny
        return i, j, k

    def idx_from_ijk_array(self, ijk: np.ndarray) -> np.ndarray:
        """Vectorised flat index from (N, 3) integer array of (i, j, k)."""
        return ijk[:, 0] + self.Nx * (ijk[:, 1] + self.Ny * ijk[:, 2])

    def ijk_from_idx_array(self, idx: np.ndarray) -> np.ndarray:
        """Vectorised (N, 3) ijk from flat index array."""
        i = idx % self.Nx
        remainder = idx // self.Nx
        j = remainder % self.Ny
        k = remainder // self.Ny
        return np.stack([i, j, k], axis=1)

    # ------------------------------------------------------------------
    # Neighbor table
    # ------------------------------------------------------------------

    def _build_neighbor_table(self) -> np.ndarray:
        """Build (Ng, 6) neighbor index table.

        Column order: [-x, +x, -y, +y, -z, +z].
        Value is -1 if the neighbor is outside the domain.
        """
        Ng = self.size
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        neighbors = -np.ones((Ng, 6), dtype=np.intp)

        all_idx = np.arange(Ng)
        ijk = self.ijk_from_idx_array(all_idx)
        i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]

        # -x neighbor
        mask = i > 0
        neighbors[all_idx[mask], 0] = self.idx_from_ijk_array(
            np.stack([i[mask] - 1, j[mask], k[mask]], axis=1)
        )
        # +x neighbor
        mask = i < Nx - 1
        neighbors[all_idx[mask], 1] = self.idx_from_ijk_array(
            np.stack([i[mask] + 1, j[mask], k[mask]], axis=1)
        )
        # -y neighbor
        mask = j > 0
        neighbors[all_idx[mask], 2] = self.idx_from_ijk_array(
            np.stack([i[mask], j[mask] - 1, k[mask]], axis=1)
        )
        # +y neighbor
        mask = j < Ny - 1
        neighbors[all_idx[mask], 3] = self.idx_from_ijk_array(
            np.stack([i[mask], j[mask] + 1, k[mask]], axis=1)
        )
        # -z neighbor
        mask = k > 0
        neighbors[all_idx[mask], 4] = self.idx_from_ijk_array(
            np.stack([i[mask], j[mask], k[mask] - 1], axis=1)
        )
        # +z neighbor
        mask = k < Nz - 1
        neighbors[all_idx[mask], 5] = self.idx_from_ijk_array(
            np.stack([i[mask], j[mask], k[mask] + 1], axis=1)
        )

        return neighbors

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def node_position(self, idx: int) -> np.ndarray:
        """Return (3,) position of node idx."""
        return self.nodes[idx]

    def is_boundary_node(self, idx: int) -> bool:
        """True if any neighbor is missing (domain boundary node)."""
        return bool((self.neighbors[idx] == -1).any())

    def cell_containing(self, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find the lower-corner cell index (i0, j0, k0) and fractional offsets (fx, fy, fz).

        point: (3,) or (N, 3) array.
        Returns (ijk0, frac) both shape (N, 3).
        Clamps to valid cell range [0, N-2].
        """
        point = np.atleast_2d(point)
        rel = (point - self.domain_min) / self.delta
        ijk0 = np.clip(np.floor(rel).astype(np.intp), 0,
                       np.array([self.Nx - 2, self.Ny - 2, self.Nz - 2]))
        frac = rel - ijk0
        return ijk0, frac
