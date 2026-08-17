"""Unit tests for the LE-DM geometry layer (Phase 1): shapes + rotation map."""

import numpy as np
import pytest

from ccmplus.config import BodyState
from ccmplus.sdf import signed_distance_sphere_points
from ccmplus.geometry import (
    GEOMETRY_REGISTRY, Sphere, Ellipsoid,
    build_rotation, euler_rates_to_omega, make_body,
)


def test_registry_has_backends():
    assert set(GEOMETRY_REGISTRY) >= {"sphere", "ellipsoid"}


def test_sphere_sdf_bit_for_bit_vs_ccmplus():
    """Sphere driven through the geometry layer (identity pose) reproduces
    ccmplus' analytic sphere distance byte-for-byte."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 3)) * 4.0
    c = np.array([0.7, -1.3, 2.1])
    r = 1.9
    body = make_body(Sphere(r), c, build_rotation([0, 0, 0], "ZYX", "deg"),
                     U=[1, 0, 0], omega_world=[0, 0, 0], sigma_s=0.5)
    phi_new = body.sdf_fn(X, body)
    ref = BodyState(X_s=c, U_s=np.zeros(3), omega_s=np.zeros(3), radius=r, sigma_s=0.5)
    phi_ref = signed_distance_sphere_points(X, ref)
    assert np.array_equal(phi_new, phi_ref)


def test_ellipsoid_sign_and_surface():
    e = Ellipsoid(3.0, 2.0, 1.0)
    # exact sign from the quadric; centre distance = shortest semi-axis (1.0)
    assert e.signed_distance(np.array([[0.0, 0, 0]]))[0] == pytest.approx(-1.0, abs=1e-4)
    assert e.signed_distance(np.array([[6.0, 0, 0]]))[0] > 0        # outside
    assert e.signed_distance(np.array([[2.0, 0, 0]]))[0] < 0        # interior, off-centre
    # on-surface points have |phi| ~ 0
    rng = np.random.default_rng(2)
    u = rng.normal(size=(400, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    surf = u * np.array([3.0, 2.0, 1.0])
    assert np.max(np.abs(e.signed_distance(surf))) < 1e-9
    # exterior magnitude matches the known axis distances
    assert e.signed_distance(np.array([[6.0, 0, 0]]))[0] == pytest.approx(3.0, abs=1e-6)
    assert e.signed_distance(np.array([[0.0, 0, 4.0]]))[0] == pytest.approx(3.0, abs=1e-6)


def test_ellipsoid_distance_vs_bruteforce():
    e = Ellipsoid(3.0, 2.0, 1.0)
    th = np.linspace(0, np.pi, 500)
    ph = np.linspace(0, 2 * np.pi, 1000)
    T, P = np.meshgrid(th, ph)
    S = np.stack([(3 * np.sin(T) * np.cos(P)).ravel(),
                  (2 * np.sin(T) * np.sin(P)).ravel(),
                  (1 * np.cos(T)).ravel()], axis=1)
    for p in ([5.0, 3.0, 2.0], [0.0, 0.0, 4.0], [4.0, 1.0, 0.0]):
        brute = np.min(np.linalg.norm(S - np.array(p), axis=1))
        assert e.signed_distance(np.array([p]))[0] == pytest.approx(brute, abs=2e-2)


def test_euler_map_matches_numerical_derivative():
    """The Euler kinematic map equals R_dot R^T (world) / R^T R_dot (body)."""
    seq, unit = "ZYX", "deg"
    ang = np.array([20.0, 35.0, -15.0])
    rate = np.array([1.3, -0.7, 2.1])
    dt = 1e-6
    R0 = build_rotation(ang, seq, unit)
    R1 = build_rotation(ang + rate * dt, seq, unit)
    Rdot = (R1 - R0) / dt
    Ow = Rdot @ R0.T
    w_num = np.array([Ow[2, 1], Ow[0, 2], Ow[1, 0]])
    w_map = euler_rates_to_omega(ang, rate, seq, unit, "right", "world")
    assert np.allclose(w_map, w_num, atol=1e-5)
    # world/body consistency: R @ omega_body == omega_world
    w_body = euler_rates_to_omega(ang, rate, seq, unit, "right", "body")
    assert np.allclose(R0 @ w_body, w_map, atol=1e-12)


def test_single_axis_omega_equals_rate():
    w = euler_rates_to_omega([10, 0, 0], [2, 0, 0], "ZYX", "deg", "right", "world")
    assert np.allclose(w, [0, 0, np.deg2rad(2)], atol=1e-12)
