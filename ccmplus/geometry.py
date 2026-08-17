"""Pluggable body geometry + rotation conventions for the LE-DM input layer.

The solver core is untouched: this module only produces a ``BodyState`` whose
``sdf_fn`` (occupancy) and ``X_s/U_s/omega_s`` (rigid kinematics) match the
contract in INTERFACE.md. Every geometry lives in its own BODY frame and
implements one method, ``signed_distance(pts_body)`` (negative inside). The
framework applies pose generically: ``X_body = Rᵀ(X − c)`` then
``signed_distance``; the surface velocity ``u_Γ = U + ω×(x − c)`` is computed
once for all shapes by ccmplus, never per shape.
"""
from __future__ import annotations

import numpy as np

from ccmplus.config import BodyState

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
GEOMETRY_REGISTRY: dict[str, type["Geometry"]] = {}


def register(name):
    """Class decorator: register a Geometry subclass under a header ``type``."""
    def deco(cls):
        GEOMETRY_REGISTRY[name] = cls
        return cls
    return deco


class Geometry:
    """Body shape in its own body frame."""

    def signed_distance(self, pts_body: np.ndarray) -> np.ndarray:
        """Signed distance at BODY-frame points (N,3). Negative inside. Returns (N,)."""
        raise NotImplementedError

    def bounding_radius(self) -> float:
        """Radius of a bounding sphere (proximity length / fast culling)."""
        raise NotImplementedError

    def characteristic_length(self) -> float:
        """Characteristic body length used for ROI padding (a body-diameter).

        Default is the bounding-sphere diameter ``2 * bounding_radius`` which gives
        ``2r`` for a sphere and ``2*max(a,b,c)`` for an ellipsoid. Shapes with a more
        natural length scale may override.
        """
        return 2.0 * float(self.bounding_radius())

    def aabb_half_extents(self, R: np.ndarray) -> np.ndarray:
        """World-frame axis-aligned bounding-box half-extents for pose rotation
        ``R`` (3x3, ``x_world = c + R @ x_body``). Returns (3,), so the body AABB at
        center ``c`` is ``[c - half, c + half]``.

        Default is the rotation-invariant bounding sphere (``bounding_radius`` on
        every axis) -- always a valid, conservative enclosure for any shape.
        Subclasses with an anisotropic support function may override for a tighter
        box.
        """
        r = float(self.bounding_radius())
        return np.array([r, r, r], dtype=float)

    @classmethod
    def from_params(cls, params: dict) -> "Geometry":
        """Build from the geometry-header ``params`` dict (e.g. {'a':5,'b':5,'c':8})."""
        raise NotImplementedError


@register("sphere")
class Sphere(Geometry):
    """phi = ‖p‖ − r. Reuses the exact distance ccmplus' analytic sphere uses, so
    a sphere driven through this layer matches the old driver bit-for-bit."""

    def __init__(self, r: float) -> None:
        self.r = float(r)

    @classmethod
    def from_params(cls, params: dict) -> "Sphere":
        return cls(r=float(params["r"]))

    def signed_distance(self, pts_body: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_body, dtype=float)
        # Identical ops to sdf.signed_distance_sphere_points under identity pose:
        # norm(pts) - r, with pts = (X - c) already the body-frame offset.
        return np.linalg.norm(pts, axis=1) - self.r

    def bounding_radius(self) -> float:
        return self.r


@register("ellipsoid")
class Ellipsoid(Geometry):
    """Axis-aligned ellipsoid (a,b,c) in the body frame.

    Sign is exact from the quadric ``(x/a)²+(y/b)²+(z/c)² − 1``; the near-surface
    magnitude is the true Euclidean point-to-ellipsoid distance found by the
    Lagrange root ``x_i = y_i e_i²/(e_i²+t)`` (Eberly), solved with a vectorised
    bracketed Newton/bisection on the scalar ``t``. One compact function.
    """

    def __init__(self, a: float, b: float, c: float) -> None:
        self.e = np.array([float(a), float(b), float(c)])
        if np.any(self.e <= 0):
            raise ValueError(f"ellipsoid semi-axes must be > 0; got {self.e.tolist()}")

    @classmethod
    def from_params(cls, params: dict) -> "Ellipsoid":
        return cls(a=float(params["a"]), b=float(params["b"]), c=float(params["c"]))

    def bounding_radius(self) -> float:
        return float(self.e.max())

    def aabb_half_extents(self, R: np.ndarray) -> np.ndarray:
        """Tight world-frame AABB half-extents of the rotated ellipsoid: the
        support function h_k = sqrt(sum_j (R[k,j] * e_j)^2) along each world axis k
        (exact; reduces to (a,b,c) at identity pose)."""
        R = np.asarray(R, dtype=float)
        return np.sqrt(np.sum((R * self.e[None, :]) ** 2, axis=1))

    def signed_distance(self, pts_body: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_body, dtype=float)
        e = self.e
        e2 = e * e
        y = np.abs(p)                              # first-octant reduction (symmetry)
        s = np.sum((p / e) ** 2, axis=1)           # quadric value; <1 inside
        sign = np.where(s < 1.0, -1.0, 1.0)

        n = len(p)
        t = np.zeros(n)
        outside = s > 1.0
        inside = s < 1.0

        # Outside: root t*>0 with G(0)=s-1>0, G(∞)=-1<0. Grow an upper bracket.
        if np.any(outside):
            lo = np.zeros(outside.sum())
            hi = np.full(outside.sum(), e2.max())
            po = y[outside]
            for _ in range(60):                    # widen until G(hi)<0
                bad = np.sum((po * e[None, :] / (e2[None, :] + hi[:, None])) ** 2,
                             axis=1) - 1.0 > 0.0
                if not np.any(bad):
                    break
                hi[bad] *= 2.0
            for _ in range(80):                    # bisection
                mid = 0.5 * (lo + hi)
                g = np.sum((po * e[None, :] / (e2[None, :] + mid[:, None])) ** 2,
                           axis=1) - 1.0
                pos = g > 0.0
                lo = np.where(pos, mid, lo)
                hi = np.where(pos, hi, mid)
            t[outside] = 0.5 * (lo + hi)

        # Inside: root in (−e_min², 0). G(0)=s-1<0, G(−e_min²⁺)=+∞. Bisect.
        # Zero/degenerate components (e.g. the exact centre) make the closest-point
        # solve singular, so the interior point is nudged off any axis plane by a
        # negligible ε·e before solving; the distance is still measured from the
        # true point, giving the correct min-axis distance at the centre.
        yi = y.copy()
        if np.any(inside):
            emin2 = e2.min()
            yin = np.maximum(y[inside], 1e-9 * e[None, :])
            yi[inside] = yin
            lo = np.full(inside.sum(), -emin2 + 1e-15 * emin2)
            hi = np.zeros(inside.sum())
            for _ in range(100):
                mid = 0.5 * (lo + hi)
                g = np.sum((yin * e[None, :] / (e2[None, :] + mid[:, None])) ** 2,
                           axis=1) - 1.0
                pos = g > 0.0
                lo = np.where(pos, mid, lo)
                hi = np.where(pos, hi, mid)
            t[inside] = 0.5 * (lo + hi)

        closest = yi * e2[None, :] / (e2[None, :] + t[:, None])  # first-octant foot
        dist = np.linalg.norm(y - closest, axis=1)
        return sign * dist


# --------------------------------------------------------------------------- #
# Rotation convention (built strictly from euler_seq / angle_unit / handedness)
# --------------------------------------------------------------------------- #
_AXIS = {"X": 0, "Y": 1, "Z": 2}


def _elem_R(axis: int, ang: float) -> np.ndarray:
    """Right-handed active elemental rotation about ``axis`` by ``ang`` (radians)."""
    c, s = np.cos(ang), np.sin(ang)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _to_rad(angles, angle_unit: str) -> np.ndarray:
    a = np.asarray(angles, dtype=float)
    return np.deg2rad(a) if str(angle_unit).lower().startswith("deg") else a


def build_rotation(angles, euler_seq="ZYX", angle_unit="rad",
                   handedness="right") -> np.ndarray:
    """R such that ``x_world = c + R @ x_body`` from an intrinsic Euler sequence.

    Intrinsic composition: ``R = R_{seq0}(a0) @ R_{seq1}(a1) @ R_{seq2}(a2)``.
    Left-handed frames negate the angle sense.
    """
    a = _to_rad(angles, angle_unit)
    if str(handedness).lower().startswith("left"):
        a = -a
    R = np.eye(3)
    for ch, ang in zip(euler_seq.upper(), a):
        R = R @ _elem_R(_AXIS[ch], ang)
    return R


def euler_rates_to_omega(angles, rates, euler_seq="ZYX", angle_unit="rad",
                         handedness="right", frame="world") -> np.ndarray:
    """Euler kinematic map: angular-velocity vector from Euler angles + their rates.

    For an intrinsic sequence the world-frame axis of the i-th rotation is the
    product of the preceding rotations applied to the base axis, so
    ``ω_world = Σ_i ȧ_i · (R_{<i} · ê_{axis_i})``. Returns ``ω`` in ``world`` or
    ``body`` frame. This is NOT ``d(angles)/dt`` except for single-axis motion —
    it is exactly what the load-time consistency check compares against ``ω``.
    """
    a = _to_rad(angles, angle_unit)
    da = _to_rad(rates, angle_unit)
    if str(handedness).lower().startswith("left"):
        a, da = -a, -da
    R = np.eye(3)
    omega = np.zeros(3)
    for ch, ang, drate in zip(euler_seq.upper(), a, da):
        omega = omega + drate * (R @ np.eye(3)[_AXIS[ch]])
        R = R @ _elem_R(_AXIS[ch], ang)
    if str(frame).lower() == "body":
        return R.T @ omega
    return omega


# --------------------------------------------------------------------------- #
# Framework: pose application -> BodyState (physics-generic)
# --------------------------------------------------------------------------- #
def make_body(geometry: Geometry, center, R, U, omega_world, sigma_s: float) -> BodyState:
    """Assemble the ccmplus ``BodyState`` for one timestep.

    ``sdf_fn`` applies the pose generically (``X_body = Rᵀ(X − c)``) and defers to
    ``geometry.signed_distance``. ``velocity_fn`` is left ``None`` so ccmplus uses
    the frozen rigid law ``u_Γ = U + ω×(x − c)`` — identical for every shape.
    """
    c = np.asarray(center, dtype=float)
    Rm = np.asarray(R, dtype=float)

    def sdf_fn(points, body):
        pts_body = (np.asarray(points, dtype=float) - c) @ Rm   # = Rᵀ(X − c), rows
        return geometry.signed_distance(pts_body)

    return BodyState(
        X_s=c.copy(),
        U_s=np.asarray(U, dtype=float),
        omega_s=np.asarray(omega_world, dtype=float),
        radius=float(geometry.bounding_radius()),
        sigma_s=float(sigma_s),
        sdf_fn=sdf_fn,
    )
