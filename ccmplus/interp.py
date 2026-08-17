"""Particle-to-grid interpolation matrix A (LE-DM v2).

A maps grid velocity DOFs (3*Ng) to particle velocities (3*n_p): rows are
particles, columns are node DOFs. Three kernels are available:

  "trilinear"  compact 8-corner stencil (1-cell support). A cell with no
               particle inside it receives zero data information -- the
               structural gap the wider kernels close.

  "wide"       DEFAULT: the tensor-product CUBIC B-SPLINE. 4 nodes per axis
               (offsets -1, 0, +1, +2) gives a 4x4x4 = 64-node footprint
               spanning a 2-cell radius. It has polynomial (linear, indeed
               cubic) precision, so data-rich cells are not over-smoothed while
               a single sparse particle still informs a whole neighbourhood --
               restoring the "interpolation volume" of the original local-Taylor
               CCM. This is NOT a Gaussian and takes no radius/sigma parameter.

  "gaussian"   an explicit Shepard-Gaussian footprint parameterised by
               ``radius_cells`` / ``sigma_cells``. Reproduces constants but not
               linear fields; offered as an alternative, not the default.

Both kernels:
  * are masked by ``allowed_nodes`` (the SDF classification C==1): a particle
    never contributes to shell/solid nodes and vice versa;
  * are renormalised to a partition of unity per particle over the allowed
    nodes, so a constant velocity field is reproduced exactly (the rows of A
    sum to 1). This keeps A a genuine *interpolation*, not a blurring, operator.

Node ordering matches RectGrid: idx = i + Nx*(j + Ny*k) (i fastest).
"""

from __future__ import annotations
import itertools
import numpy as np
import scipy.sparse as sp
from ccmplus.grid import RectGrid


# ---------------------------------------------------------------------------
# Shared assembly: given (n_p, K) corner indices + weights, build A
# ---------------------------------------------------------------------------

def _assemble_A(corner_idx: np.ndarray,
                weights: np.ndarray,
                grid: RectGrid,
                allowed_nodes: np.ndarray | None,
                renormalize_allowed: bool) -> sp.csr_matrix:
    """Build CSR A of shape (3*n_p, 3*Ng) from per-particle stencils.

    corner_idx : (n_p, K) int flat node indices (already clamped in-range)
    weights    : (n_p, K) float kernel weights (0 for unused slots)
    """
    n_p, K = corner_idx.shape
    Ng = grid.size
    Ndof = 3 * Ng

    if allowed_nodes is not None:
        allowed = np.asarray(allowed_nodes, dtype=bool)
        if allowed.shape != (Ng,):
            raise ValueError(f"allowed_nodes must have shape ({Ng},), got {allowed.shape}")
        weights = np.where(allowed[corner_idx], weights, 0.0)

    if renormalize_allowed:
        row_sum = weights.sum(axis=1)
        good = row_sum > 1e-15
        weights[good] /= row_sum[good, None]

    n_entries = n_p * K * 3
    row_arr = np.empty(n_entries, dtype=np.intp)
    col_arr = np.empty(n_entries, dtype=np.intp)
    val_arr = np.empty(n_entries, dtype=float)

    p_idx = np.arange(n_p)
    for c in range(3):
        start = c * n_p * K
        end = start + n_p * K
        row_arr[start:end] = np.repeat(3 * p_idx + c, K)
        col_arr[start:end] = (3 * corner_idx + c).ravel()
        val_arr[start:end] = weights.ravel()

    A = sp.coo_matrix((val_arr, (row_arr, col_arr)), shape=(3 * n_p, Ndof)).tocsr()
    A.eliminate_zeros()
    return A


# ---------------------------------------------------------------------------
# Trilinear kernel (v1 behaviour, kept for base/regression)
# ---------------------------------------------------------------------------

def _trilinear_stencils(positions: np.ndarray, grid: RectGrid):
    ijk0, frac = grid.cell_containing(positions)
    i0, j0, k0 = ijk0[:, 0], ijk0[:, 1], ijk0[:, 2]
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
    weights = np.array([
        (1 - fx) * (1 - fy) * (1 - fz),
        fx * (1 - fy) * (1 - fz),
        (1 - fx) * fy * (1 - fz),
        (1 - fx) * (1 - fy) * fz,
        fx * fy * (1 - fz),
        fx * (1 - fy) * fz,
        (1 - fx) * fy * fz,
        fx * fy * fz,
    ]).T  # (n_p, 8)
    di = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    dj = np.array([0, 0, 1, 0, 1, 0, 1, 1])
    dk = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    corner_idx = (i0[:, None] + di[None, :]) + grid.Nx * (
        (j0[:, None] + dj[None, :]) + grid.Ny * (k0[:, None] + dk[None, :])
    )
    return corner_idx, weights


# ---------------------------------------------------------------------------
# Wide kernels (v2)
# ---------------------------------------------------------------------------

def _bspline3_weights(f: np.ndarray) -> np.ndarray:
    """1D cubic B-spline weights for the 4 nodes at offsets (-1, 0, 1, 2).

    f : (n_p,) fractional coordinate in [0, 1) within the cell. Each row of the
    returned (n_p, 4) array sums to 1 and reproduces linear (and up to cubic)
    fields exactly. All weights are non-negative.
    """
    f = np.asarray(f, dtype=float)
    w_m1 = (1.0 - f) ** 3 / 6.0
    w_0 = (3.0 * f ** 3 - 6.0 * f ** 2 + 4.0) / 6.0
    w_1 = (-3.0 * f ** 3 + 3.0 * f ** 2 + 3.0 * f + 1.0) / 6.0
    w_2 = f ** 3 / 6.0
    return np.stack([w_m1, w_0, w_1, w_2], axis=1)   # (n_p, 4)


def _bspline_stencils(positions: np.ndarray, grid: RectGrid):
    """Tensor-product cubic B-spline footprint (4 nodes/axis -> 64-node support).

    Spans a 2-cell radius and reproduces linear fields exactly (polynomial
    precision), so data-rich cells are not over-smoothed while a single sparse
    particle still informs a 4x4x4 neighbourhood of nodes.
    """
    n_p = positions.shape[0]
    delta = grid.delta
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    domain_min = grid.domain_min

    rel = (positions - domain_min) / delta
    base = np.floor(rel).astype(np.intp)             # (n_p, 3)
    frac = rel - base                                # (n_p, 3) in [0,1)

    wx = _bspline3_weights(frac[:, 0])               # (n_p, 4)
    wy = _bspline3_weights(frac[:, 1])
    wz = _bspline3_weights(frac[:, 2])

    node_off = np.array([-1, 0, 1, 2])
    offsets = list(itertools.product(range(4), repeat=3))   # indices into the 4
    K = len(offsets)                                          # 64

    corner_idx = np.empty((n_p, K), dtype=np.intp)
    weights = np.zeros((n_p, K), dtype=float)
    Nmax = np.array([Nx - 1, Ny - 1, Nz - 1])

    for col, (a, b, c) in enumerate(offsets):
        node_ijk = base + np.array([node_off[a], node_off[b], node_off[c]])
        in_dom = np.all((node_ijk >= 0) & (node_ijk <= Nmax), axis=1)
        clamped = np.clip(node_ijk, 0, Nmax)
        ci, cj, ck = clamped[:, 0], clamped[:, 1], clamped[:, 2]
        corner_idx[:, col] = ci + Nx * (cj + Ny * ck)
        w = wx[:, a] * wy[:, b] * wz[:, c]
        weights[:, col] = np.where(in_dom, w, 0.0)

    return corner_idx, weights


def _gaussian_stencils(positions: np.ndarray, grid: RectGrid,
                       radius_cells: float, sigma_cells: float):
    """Shepard-Gaussian footprint over a box of half-width ceil(radius_cells).

    Reproduces constants (after renormalisation) but not linear fields; provided
    as an alternative to the cubic B-spline. Out-of-domain / out-of-radius slots
    get weight 0.
    """
    n_p = positions.shape[0]
    delta = grid.delta
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    domain_min = grid.domain_min

    rel = (positions - domain_min) / delta
    base = np.floor(rel).astype(np.intp)

    rad = int(np.ceil(radius_cells))
    R2 = (radius_cells * delta) ** 2
    two_sig2 = 2.0 * (sigma_cells * delta) ** 2

    offsets = list(itertools.product(range(-rad, rad + 1), repeat=3))
    K = len(offsets)
    corner_idx = np.empty((n_p, K), dtype=np.intp)
    weights = np.zeros((n_p, K), dtype=float)
    Nmax = np.array([Nx - 1, Ny - 1, Nz - 1])

    for col, (di, dj, dk) in enumerate(offsets):
        node_ijk = base + np.array([di, dj, dk])
        in_dom = np.all((node_ijk >= 0) & (node_ijk <= Nmax), axis=1)
        clamped = np.clip(node_ijk, 0, Nmax)
        ci, cj, ck = clamped[:, 0], clamped[:, 1], clamped[:, 2]
        corner_idx[:, col] = ci + Nx * (cj + Ny * ck)
        node_xyz = domain_min + clamped * delta
        dist2 = np.sum((positions - node_xyz) ** 2, axis=1)
        w = np.exp(-dist2 / two_sig2)
        weights[:, col] = np.where(in_dom & (dist2 <= R2), w, 0.0)

    return corner_idx, weights


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def build_interpolation_matrix(
    positions: np.ndarray,
    grid: RectGrid,
    allowed_nodes: np.ndarray | None = None,
    *,
    renormalize_allowed: bool = True,
    kernel: str = "wide",
    radius_cells: float = 2.0,
    sigma_cells: float = 1.0,
) -> sp.csr_matrix:
    """Build sparse interpolation matrix A of shape (3*n_p, 3*Ng).

    kernel       : "wide" (default; cubic B-spline, 4x4x4 = 64-node support),
                   "gaussian" (Shepard-Gaussian footprint), or "trilinear"
                   (compact 8-corner stencil).
    radius_cells : footprint radius in grid cells -- "gaussian" kernel ONLY,
                   ignored by "wide" (whose support is fixed at 4 nodes/axis).
    sigma_cells  : Gaussian width in grid cells -- "gaussian" kernel ONLY.
    allowed_nodes: optional (Ng,) bool mask (SDF classification C==1). Weights on
                   disallowed nodes are zeroed and the remaining allowed weights
                   renormalised per particle.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.shape[0] == 0:
        return sp.csr_matrix((0, 3 * grid.size))

    k = kernel.strip().lower()
    if k == "trilinear":
        corner_idx, weights = _trilinear_stencils(positions, grid)
    elif k in ("wide", "bspline"):
        corner_idx, weights = _bspline_stencils(positions, grid)
    elif k == "gaussian":
        corner_idx, weights = _gaussian_stencils(positions, grid, radius_cells, sigma_cells)
    else:
        raise ValueError(f"Unknown interpolation kernel {kernel!r}; "
                         f"use 'wide', 'gaussian', or 'trilinear'.")

    return _assemble_A(corner_idx, weights, grid, allowed_nodes, renormalize_allowed)
