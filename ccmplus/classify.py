"""Ternary node classification, near-wall fluid set, and transition flags."""

from __future__ import annotations
import numpy as np
from ccmplus.grid import RectGrid


def classify(phi: np.ndarray, delta: float) -> np.ndarray:
    """Ternary classification of grid nodes.

    C_j = -1  if phi_j < 0               (solid interior)
    C_j =  0  if 0 <= phi_j <= delta/2   (boundary shell)
    C_j = +1  if phi_j > delta/2         (open fluid)

    Returns int8 array of shape (Ng,).
    """
    C = np.empty(phi.shape, dtype=np.int8)
    C[:] = 1
    C[phi <= delta / 2] = 0
    C[phi < 0] = -1
    return C


def near_wall_fluid_set(C: np.ndarray, grid: RectGrid) -> np.ndarray:
    """Boolean mask of near-wall fluid nodes.

    j is in N(tk) if C_j == +1 AND at least one axis-aligned neighbor has C_n <= 0.
    Domain-boundary slots (neighbor index -1) are ignored (free-slip, not solid).

    Returns bool array of shape (Ng,).
    """
    nb = grid.neighbors   # (Ng, 6), -1 for missing
    has_solid_nb = np.zeros(grid.size, dtype=bool)

    for col in range(6):
        n_idx = nb[:, col]
        valid = n_idx >= 0
        nb_le0 = np.zeros(grid.size, dtype=bool)
        nb_le0[valid] = C[n_idx[valid]] <= 0
        has_solid_nb |= nb_le0

    return (C == 1) & has_solid_nb


def transition_flags(C_current: np.ndarray, C_prev: np.ndarray | None) -> np.ndarray:
    """Per-node transition flags between two consecutive timesteps (Eq. 7).

    tau_j = +1  if C_j(tk) >= 0 AND C_j(tk-1) <  0  (exposed this step)
    tau_j = -1  if C_j(tk) <  0 AND C_j(tk-1) >= 0  (covered this step)
    tau_j =  0  otherwise

    If C_prev is None (first timestep), all tau = 0.
    Returns int8 array of shape (Ng,).
    """
    if C_prev is None:
        return np.zeros(C_current.shape, dtype=np.int8)

    tau = np.zeros(C_current.shape, dtype=np.int8)
    tau[(C_current >= 0) & (C_prev < 0)] = 1
    tau[(C_current < 0) & (C_prev >= 0)] = -1
    return tau
