"""Signed distance functions."""

from __future__ import annotations
import numpy as np
from ccmplus.grid import RectGrid
from ccmplus.config import BodyState


def signed_distance_sphere(grid: RectGrid, body: BodyState) -> np.ndarray:
    """Signed distance from each grid node to a sphere surface.

    phi > 0: fluid (outside), phi < 0: solid (inside), phi = 0: surface.

    Returns shape (Ng,).
    """
    r_vec = grid.nodes - body.X_s   # (Ng, 3)
    r = np.linalg.norm(r_vec, axis=1)  # (Ng,)
    return r - body.radius


def signed_distance_body(grid: RectGrid, body: BodyState) -> np.ndarray:
    """Signed distance from grid nodes to the active body geometry.

    ``body.sdf_fn`` may provide an arbitrary stationary, moving, or composite
    solid geometry. If it is absent, the analytic sphere SDF is used.
    """
    return signed_distance_body_points(grid.nodes, body)


def signed_distance_body_points(points: np.ndarray, body: BodyState) -> np.ndarray:
    """Signed distance from arbitrary points to the active body geometry."""
    points = np.asarray(points, dtype=float)
    if body.sdf_fn is not None:
        phi = np.asarray(body.sdf_fn(points, body), dtype=float)
        expected = (len(points),)
        if phi.shape != expected:
            raise ValueError(
                f"body.sdf_fn must return an array with shape {expected}, "
                f"got {phi.shape}"
            )
        return phi

    return signed_distance_sphere_points(points, body)


def signed_distance_sphere_points(points: np.ndarray, body: BodyState) -> np.ndarray:
    """Signed distance from arbitrary points (N, 3) to a sphere surface.

    Returns shape (N,).
    """
    points = np.asarray(points, dtype=float)
    r_vec = points - body.X_s
    r = np.linalg.norm(r_vec, axis=1)
    return r - body.radius
