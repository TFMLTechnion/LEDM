"""Rigid-body kinematics: surface velocity evaluator."""

from __future__ import annotations
import numpy as np
from ccmplus.config import BodyState


def u_gamma(positions: np.ndarray, body: BodyState) -> np.ndarray:
    """Body velocity at arbitrary positions.

    If ``body.velocity_fn`` is supplied it defines the body velocity field.
    Otherwise the rigid-body expression is used:

        u_Gamma(x) = U_s + omega_s × (x - X_s)

    positions: (N, 3)
    Returns: (N, 3)
    """
    positions = np.asarray(positions, dtype=float)
    if body.velocity_fn is not None:
        values = np.asarray(body.velocity_fn(positions, body), dtype=float)
        if values.shape != positions.shape:
            raise ValueError(
                "body.velocity_fn must return an array with shape "
                f"{positions.shape}, got {values.shape}"
            )
        return values

    return body.U_s + np.cross(body.omega_s, positions - body.X_s)
