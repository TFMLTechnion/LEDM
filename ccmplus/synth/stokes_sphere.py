"""Analytical Stokes flow around a sphere.

Reference: Happel & Brenner, "Low Reynolds Number Hydrodynamics", Ch. 4;
           Kim & Karrila, "Microhydrodynamics", Ch. 3.

Formula (rest frame — sphere stationary, uniform U_inf at infinity):

  u = U_inf
      - (3R)/(4r)  * (U_inf + (U_inf·r̂) r̂)
      - (R³)/(4r³) * (U_inf - 3(U_inf·r̂) r̂)

Verified properties (see test_stokes_sphere.py):
  - u = 0 at r = R  (no-slip)
  - u → U_inf as r → ∞
  - ∇·u = 0 everywhere outside the sphere
"""

from __future__ import annotations
import numpy as np


def stokes_sphere_velocity(
    points: np.ndarray,
    U_inf: np.ndarray,
    R: float,
    center: np.ndarray,
) -> np.ndarray:
    """Stokes flow in the sphere's rest frame.

    Sphere stationary at `center`, uniform velocity U_inf at infinity.

    points: (N, 3)
    U_inf:  (3,) far-field velocity
    R:      sphere radius
    center: (3,) sphere centre

    Returns (N, 3) velocity field.  Points inside the sphere are clamped to
    r = R*(1+eps) before evaluation (result is not physically meaningful there).
    """
    r_vec = points - center                                    # (N, 3)
    r = np.linalg.norm(r_vec, axis=1, keepdims=True)          # (N, 1)
    r = np.maximum(r, R * (1.0 + 1e-9))                       # clamp inside
    r_hat = r_vec / r                                          # (N, 3)
    U = np.asarray(U_inf, dtype=float)
    U_dot_rhat = (U * r_hat).sum(axis=1, keepdims=True)       # (N, 1)

    u = (U
         - (3.0 * R) / (4.0 * r) * (U + U_dot_rhat * r_hat)
         - (R ** 3) / (4.0 * r ** 3) * (U - 3.0 * U_dot_rhat * r_hat))
    return u


def stokes_sphere_lab_frame(
    points: np.ndarray,
    U_s: np.ndarray,
    R: float,
    center: np.ndarray,
) -> np.ndarray:
    """Stokes flow in the lab frame.

    Sphere at `center` translating at U_s; fluid at rest at infinity.

    Galilean transform from rest frame (U_inf = -U_s):
      u_lab = u_rest_frame + U_s

    Boundary conditions:
      u_lab = U_s  at r = R  (no-slip, sphere surface moves at U_s)
      u_lab → 0   as r → ∞  (fluid at rest far away)
    """
    U_inf = -np.asarray(U_s, dtype=float)
    return stokes_sphere_velocity(points, U_inf, R, center) + U_s
