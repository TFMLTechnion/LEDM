"""Prior correction: handle nodes transitioning in/out of the solid body (Eq. 8)."""

from __future__ import annotations
import numpy as np
from ccmplus.grid import RectGrid
from ccmplus.config import BodyState
from ccmplus.kinematics import u_gamma


def apply_prior_correction(
    x_prev: np.ndarray | None,
    tau: np.ndarray,
    body: BodyState,
    grid: RectGrid,
) -> np.ndarray:
    """Build the corrected prior x_prev_tilde for the current timestep (Eq. 8).

    tau_j == +1 (exposed this step): seed with body velocity u_Gamma(x_j).
    Otherwise: carry x_prev forward unchanged.
    First timestep (x_prev is None): initialise to zero everywhere.

    Returns x_prev_tilde of shape (3*Ng,).
    """
    if x_prev is None:
        x_tilde = np.zeros(3 * grid.size)
    else:
        x_tilde = x_prev.copy()

    exposed = np.where(tau == 1)[0]
    if len(exposed) > 0:
        ug = u_gamma(grid.nodes[exposed], body)   # (n_exp, 3)
        x_tilde[3 * exposed]     = ug[:, 0]
        x_tilde[3 * exposed + 1] = ug[:, 1]
        x_tilde[3 * exposed + 2] = ug[:, 2]

    return x_tilde


def apply_coverage_gated_prior(
    x_prev: np.ndarray | None,
    tau: np.ndarray,
    C: np.ndarray,
    support: np.ndarray,
    body: BodyState,
    grid: RectGrid,
    *,
    enable_lema: bool,
    decay_factor: float = 0.5,
) -> np.ndarray:
    """Coverage-gated temporal prior (v2 final-run fix for the prior-dominated wake).

    The v1 rule wrote the body velocity u_Gamma into every freshly exposed node and
    carried the warm-start prior forward unchanged. In the sparse wake those values
    have no particle support, are never overwritten by data, and freeze through the
    warm-start chain. The gated rule:

      * DECAY (both modes): every FLUID node (C==1) with zero local track support
        has its inherited prior multiplied by ``decay_factor`` this snapshot. Across
        consecutive unsupported snapshots this compounds (val * decay^k -> 0),
        preventing multi-frame freezing. Data-rich cells (support >= 1) are left
        completely untouched.

      * EXPOSED SEEDING (LE-DM only): at tau==+1 nodes, write u_Gamma only where
        local support >= 1; where support == 0 set the prior to zero (quiescent
        fluid) instead of u_Gamma.

    support : (Ng,) integer count of tracks within the support radius of each node.
    Returns x_prev_tilde of shape (3*Ng,).
    """
    Ng = grid.size
    if x_prev is None:
        x_tilde = np.zeros(3 * Ng)
    else:
        x_tilde = x_prev.copy()

    support = np.asarray(support)
    C = np.asarray(C)

    # DECAY: unsupported fluid nodes (applies whether or not boundary
    # constraints are enabled)
    unsup = np.where((C == 1) & (support == 0))[0]
    if len(unsup) > 0:
        f = float(decay_factor)
        x_tilde[3 * unsup]     *= f
        x_tilde[3 * unsup + 1] *= f
        x_tilde[3 * unsup + 2] *= f

    # EXPOSED SEEDING: LE-DM only
    if enable_lema:
        exposed = np.where(tau == 1)[0]
        if len(exposed) > 0:
            ug = u_gamma(grid.nodes[exposed], body)      # (n_exp, 3)
            seed_mask = support[exposed] >= 1            # u_Gamma where supported
            ug_seed = np.where(seed_mask[:, None], ug, 0.0)
            x_tilde[3 * exposed]     = ug_seed[:, 0]
            x_tilde[3 * exposed + 1] = ug_seed[:, 1]
            x_tilde[3 * exposed + 2] = ug_seed[:, 2]

    return x_tilde
