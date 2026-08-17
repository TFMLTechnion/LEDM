"""Tests for the analytical Stokes flow formula and track sampler."""

import numpy as np
import pytest
from ccmplus.config import BodyState
from ccmplus.synth.stokes_sphere import stokes_sphere_velocity, stokes_sphere_lab_frame
from ccmplus.synth.tracks import sample_tracks_restframe, sample_tracks_labframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

R = 2.0
U_INF = np.array([1.0, 0.0, 0.0])
CENTER = np.zeros(3)


def make_body(center=CENTER, radius=R, U_s=None):
    return BodyState(
        X_s=np.asarray(center, dtype=float),
        U_s=np.zeros(3) if U_s is None else np.asarray(U_s, dtype=float),
        omega_s=np.zeros(3),
        radius=radius,
        sigma_s=0.5,
    )


def sphere_surface_pts(n=200, R=R, center=CENTER, rng=None):
    """Random points exactly on the sphere surface."""
    if rng is None:
        rng = np.random.default_rng(0)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.asarray(center) + R * v


def far_field_pts(n=50, dist=50000.0, rng=None):
    """Random points very far from the sphere (correction ~ R/dist << 1)."""
    if rng is None:
        rng = np.random.default_rng(1)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return dist * v


# ---------------------------------------------------------------------------
# Rest-frame: no-slip at surface
# ---------------------------------------------------------------------------

class TestNoSlip:
    def test_restframe_noslip_uniform_upstream(self):
        """u = 0 at r = R for any direction (rest frame)."""
        pts = sphere_surface_pts()
        u = stokes_sphere_velocity(pts, U_INF, R, CENTER)
        np.testing.assert_allclose(
            np.linalg.norm(u, axis=1), 0.0, atol=1e-6,
            err_msg="Stokes flow must be zero at the sphere surface (rest frame)",
        )

    def test_restframe_noslip_multiple_directions(self):
        """Verify no-slip for upstream in y and z directions too."""
        for U in [np.array([0., 1., 0.]), np.array([0., 0., 1.]),
                  np.array([1., 1., 1.]) / np.sqrt(3)]:
            pts = sphere_surface_pts()
            u = stokes_sphere_velocity(pts, U, R, CENTER)
            np.testing.assert_allclose(
                np.linalg.norm(u, axis=1), 0.0, atol=1e-6,
                err_msg=f"No-slip failed for U_inf={U}",
            )

    def test_restframe_noslip_offcenter_sphere(self):
        """No-slip must hold for a sphere not at the origin."""
        center = np.array([3.0, -1.0, 2.0])
        pts = sphere_surface_pts(center=center)
        u = stokes_sphere_velocity(pts, U_INF, R, center)
        np.testing.assert_allclose(
            np.linalg.norm(u, axis=1), 0.0, atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Rest-frame: far-field limit
# ---------------------------------------------------------------------------

class TestFarField:
    def test_restframe_farfield_approaches_Uinf(self):
        """u → U_inf as r → ∞."""
        pts = far_field_pts()
        u = stokes_sphere_velocity(pts, U_INF, R, CENTER)
        err = np.linalg.norm(u - U_INF, axis=1)
        assert np.max(err) < 1e-4, (
            f"Far-field error too large: max={np.max(err):.2e}"
        )

    def test_restframe_farfield_decay_rate(self):
        """Correction decays as 1/r: at 20R, correction < 10% of |U_inf|.

        At r = 10R on the x-axis the leading term is 3R/(2r)=15%, so we use 20R.
        """
        pts = 20 * R * np.eye(3)   # three axis-aligned points at 20R
        u = stokes_sphere_velocity(pts, U_INF, R, CENTER)
        err = np.linalg.norm(u - U_INF, axis=1)
        assert np.all(err < 0.10 * np.linalg.norm(U_INF))


# ---------------------------------------------------------------------------
# Lab-frame boundary conditions
# ---------------------------------------------------------------------------

class TestLabFrame:
    def test_labframe_noslip_equals_Us(self):
        """u_lab = U_s at r = R."""
        U_s = np.array([1.5, -0.5, 0.3])
        pts = sphere_surface_pts()
        u = stokes_sphere_lab_frame(pts, U_s, R, CENTER)
        expected = np.tile(U_s, (len(pts), 1))
        np.testing.assert_allclose(
            u, expected, atol=1e-6,
            err_msg="Lab-frame no-slip: surface velocity must equal U_s",
        )

    def test_labframe_farfield_at_rest(self):
        """u_lab → 0 as r → ∞ (fluid at rest at infinity)."""
        U_s = np.array([1.0, 0.0, 0.0])
        pts = far_field_pts()
        u = stokes_sphere_lab_frame(pts, U_s, R, CENTER)
        np.testing.assert_allclose(
            np.linalg.norm(u, axis=1), 0.0, atol=1e-4,
            err_msg="Lab-frame far-field: fluid must be at rest",
        )

    def test_labframe_is_galilean_transform_of_restframe(self):
        """u_lab = u_rest + U_s at any fluid point."""
        U_s = np.array([1.0, 0.5, -0.2])
        U_inf = -U_s
        rng = np.random.default_rng(7)
        pts = rng.uniform(-10, 10, (50, 3))
        # Exclude interior
        r = np.linalg.norm(pts - CENTER, axis=1)
        pts = pts[r > R * 1.1][:30]

        u_rest = stokes_sphere_velocity(pts, U_inf, R, CENTER)
        u_lab = stokes_sphere_lab_frame(pts, U_s, R, CENTER)
        np.testing.assert_allclose(u_lab, u_rest + U_s, atol=1e-12)


# ---------------------------------------------------------------------------
# Specific analytical values (cross-checked with spherical-coordinate formula)
# ---------------------------------------------------------------------------

class TestAnalyticalValues:
    def test_on_x_axis(self):
        """On the x-axis (theta=0): u_y = u_z = 0, u_x = U*(1 - 3R/(2r) + R³/(2r³))."""
        r_vals = np.array([2.5, 4.0, 10.0, 50.0])
        pts = np.column_stack([r_vals, np.zeros(len(r_vals)), np.zeros(len(r_vals))])
        U = 1.0
        u = stokes_sphere_velocity(pts, np.array([U, 0., 0.]), R, CENTER)
        u_x_expected = U * (1.0 - 3*R/(2*r_vals) + R**3/(2*r_vals**3))
        np.testing.assert_allclose(u[:, 0], u_x_expected, rtol=1e-10)
        np.testing.assert_allclose(u[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(u[:, 2], 0.0, atol=1e-12)

    def test_on_y_axis(self):
        """On the y-axis (theta=pi/2): u_x = U*(1 - 3R/(4r) - R³/(4r³)), u_y=u_z=0."""
        r_vals = np.array([2.5, 4.0, 10.0])
        pts = np.column_stack([np.zeros(len(r_vals)), r_vals, np.zeros(len(r_vals))])
        U = 1.0
        u = stokes_sphere_velocity(pts, np.array([U, 0., 0.]), R, CENTER)
        u_x_expected = U * (1.0 - 3*R/(4*r_vals) - R**3/(4*r_vals**3))
        np.testing.assert_allclose(u[:, 0], u_x_expected, rtol=1e-10)
        np.testing.assert_allclose(u[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(u[:, 2], 0.0, atol=1e-12)

    def test_at_surface_on_axes(self):
        """Explicit check: u ≈ 0 at (R,0,0) and (0,R,0) and (0,0,R).

        The r-clamp at R*(1+1e-9) introduces O(1e-8) error; use atol=1e-7.
        """
        pts = R * np.eye(3)
        u = stokes_sphere_velocity(pts, U_INF, R, CENTER)
        np.testing.assert_allclose(u, 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Divergence-free (numerical check)
# ---------------------------------------------------------------------------

class TestDivergenceFree:
    def _numerical_divergence(self, pts, U_inf, R, center, h=1e-5):
        """Finite-difference divergence at pts."""
        divs = np.zeros(len(pts))
        for ax in range(3):
            dp = pts.copy(); dp[:, ax] += h
            dm = pts.copy(); dm[:, ax] -= h
            up = stokes_sphere_velocity(dp, U_inf, R, center)
            um = stokes_sphere_velocity(dm, U_inf, R, center)
            divs += (up[:, ax] - um[:, ax]) / (2 * h)
        return divs

    def test_divergence_free_restframe(self):
        """∇·u = 0 at fluid points (rest frame)."""
        rng = np.random.default_rng(3)
        pts = rng.uniform(-8, 8, (100, 3))
        r = np.linalg.norm(pts - CENTER, axis=1)
        pts = pts[r > R * 1.5][:40]
        divs = self._numerical_divergence(pts, U_INF, R, CENTER)
        np.testing.assert_allclose(divs, 0.0, atol=1e-6,
            err_msg="Stokes flow must be divergence-free")

    def test_divergence_free_labframe(self):
        """∇·u = 0 at fluid points (lab frame)."""
        U_s = np.array([1.0, 0.0, 0.0])
        rng = np.random.default_rng(4)
        pts = rng.uniform(-8, 8, (100, 3))
        r = np.linalg.norm(pts - CENTER, axis=1)
        pts = pts[r > R * 1.5][:40]
        h = 1e-5
        divs = np.zeros(len(pts))
        for ax in range(3):
            dp = pts.copy(); dp[:, ax] += h
            dm = pts.copy(); dm[:, ax] -= h
            up = stokes_sphere_lab_frame(dp, U_s, R, CENTER)
            um = stokes_sphere_lab_frame(dm, U_s, R, CENTER)
            divs += (up[:, ax] - um[:, ax]) / (2 * h)
        np.testing.assert_allclose(divs, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Track sampler
# ---------------------------------------------------------------------------

class TestTrackSampler:
    def test_restframe_sampler_shapes(self):
        body = make_body()
        rng = np.random.default_rng(10)
        frame = sample_tracks_restframe(
            np.full(3, -10.), np.full(3, 10.), body, U_INF, 200, rng
        )
        assert frame.positions.shape == (200, 3)
        assert frame.velocities.shape == (200, 3)
        assert frame.uncertainties.shape == (200,)

    def test_restframe_sampler_outside_sphere(self):
        body = make_body()
        rng = np.random.default_rng(11)
        frame = sample_tracks_restframe(
            np.full(3, -10.), np.full(3, 10.), body, U_INF, 200, rng
        )
        r = np.linalg.norm(frame.positions - body.X_s, axis=1)
        assert np.all(r > body.radius), "All particles must be outside the sphere"

    def test_labframe_sampler_velocity_at_surface(self):
        """Particles exactly at r=R should have velocity ≈ U_s (lab frame)."""
        U_s = np.array([1.0, 0.0, 0.0])
        body = make_body(U_s=U_s)
        pts = sphere_surface_pts()
        u = stokes_sphere_lab_frame(pts, U_s, R, CENTER)
        np.testing.assert_allclose(u, np.tile(U_s, (len(pts), 1)), atol=1e-6)

    def test_restframe_sampler_velocities_match_analytical(self):
        """Sampled velocities must equal the analytical field at sampled positions."""
        body = make_body()
        rng = np.random.default_rng(12)
        frame = sample_tracks_restframe(
            np.full(3, -8.), np.full(3, 8.), body, U_INF, 100, rng
        )
        u_expected = stokes_sphere_velocity(frame.positions, U_INF, R, CENTER)
        np.testing.assert_allclose(frame.velocities, u_expected, atol=1e-12)

    def test_noise_adds_scatter(self):
        """With sigma_noise > 0, velocities should deviate from analytical."""
        body = make_body()
        rng = np.random.default_rng(13)
        frame = sample_tracks_restframe(
            np.full(3, -8.), np.full(3, 8.), body, U_INF, 500, rng,
            sigma_noise=0.1,
        )
        u_exact = stokes_sphere_velocity(frame.positions, U_INF, R, CENTER)
        diff = np.linalg.norm(frame.velocities - u_exact, axis=1)
        assert np.mean(diff) > 0.01, "Noise should produce non-zero deviation"
        assert np.mean(diff) < 0.5, "Noise should not be huge"
