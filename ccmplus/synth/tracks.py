"""Sample synthetic particle tracks from an analytical velocity field."""

from __future__ import annotations
import numpy as np
from ccmplus.config import BodyState, FrameData
from ccmplus.synth.stokes_sphere import stokes_sphere_velocity, stokes_sphere_lab_frame


def sample_tracks_restframe(
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    body: BodyState,
    U_inf: np.ndarray,
    n_particles: int,
    rng: np.random.Generator,
    sigma_noise: float = 0.0,
    sigma_i: float = 1.0,
    t: float = 0.0,
) -> FrameData:
    """Sample tracks from Stokes flow in the sphere's rest frame.

    Particles are drawn uniformly from the domain and rejected if inside the sphere.

    U_inf:       (3,) far-field velocity (sphere is stationary)
    sigma_noise: Gaussian noise added to velocities (std dev)
    sigma_i:     reported per-particle uncertainty (stored in FrameData)
    """
    positions = _sample_outside_sphere(domain_min, domain_max, body, n_particles, rng)
    velocities = stokes_sphere_velocity(positions, U_inf, body.radius, body.X_s)
    if sigma_noise > 0:
        velocities += rng.normal(0.0, sigma_noise, velocities.shape)
    uncertainties = np.full(n_particles, sigma_i)
    return FrameData(positions=positions, velocities=velocities,
                     uncertainties=uncertainties, body=body, t=t)


def sample_tracks_labframe(
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    body: BodyState,
    n_particles: int,
    rng: np.random.Generator,
    sigma_noise: float = 0.0,
    sigma_i: float = 1.0,
    t: float = 0.0,
) -> FrameData:
    """Sample tracks from Stokes flow in the lab frame.

    Sphere translates at body.U_s; fluid at rest at infinity.
    """
    positions = _sample_outside_sphere(
        domain_min, domain_max, body, n_particles, rng
    )
    velocities = stokes_sphere_lab_frame(positions, body.U_s, body.radius, body.X_s)
    if sigma_noise > 0:
        velocities += rng.normal(0.0, sigma_noise, velocities.shape)
    uncertainties = np.full(n_particles, sigma_i)
    return FrameData(positions=positions, velocities=velocities,
                     uncertainties=uncertainties, body=body, t=t)


def _sample_outside_sphere(
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    body: BodyState,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rejection-sample n points uniformly in the domain outside the sphere."""
    dmin = np.asarray(domain_min, dtype=float)
    dmax = np.asarray(domain_max, dtype=float)
    accepted: list[np.ndarray] = []
    batch = max(n * 4, 1000)
    while len(accepted) < n:
        pts = rng.uniform(dmin, dmax, (batch, 3))
        r = np.linalg.norm(pts - body.X_s, axis=1)
        outside = pts[r > body.radius]
        accepted.append(outside)
    return np.vstack(accepted)[:n]
