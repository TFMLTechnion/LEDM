"""Unit tests: trajectory sign correction propagates correctly through
sphere-position and sphere-velocity lookups."""

import numpy as np
import pytest

from ccmplus.drivers.sphere import (
    apply_coordinate_signs,
    coordinate_signs_from_params,
    lookup_sphere_state,
)


def _make_traj(positions: np.ndarray, times: np.ndarray | None = None) -> dict:
    if times is None:
        times = np.arange(len(positions), dtype=float)
    return {"diameter_mm": 11.11, "times": times, "positions": positions.copy()}


class TestTrajectorySignDefault:
    """Default (all +1) should leave positions and velocities unchanged."""

    def test_default_signs_are_all_positive(self):
        signs = coordinate_signs_from_params({}, "trajectory")
        np.testing.assert_array_equal(signs, [1.0, 1.0, 1.0])

    def test_default_positions_unchanged(self):
        pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        signs = coordinate_signs_from_params({}, "trajectory")
        result = apply_coordinate_signs(pos, signs)
        np.testing.assert_array_equal(result, pos)

    def test_default_velocity_unchanged(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        traj = _make_traj(pos)
        # no sign flip applied
        p, v = lookup_sphere_state(traj, t_value=1.0, vel_scale=1.0)
        np.testing.assert_allclose(p, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(v, [1.0, 2.0, 3.0])


class TestTrajectoryXFlip:
    """trajectory_x_sign = -1 should negate x positions and x velocities."""

    def test_x_sign_loaded(self):
        signs = coordinate_signs_from_params({"trajectory_x_sign": -1}, "trajectory")
        np.testing.assert_array_equal(signs, [-1.0, 1.0, 1.0])

    def test_x_position_negated(self):
        pos = np.array([[5.96, -5.34, 3.87]])
        signs = coordinate_signs_from_params({"trajectory_x_sign": -1}, "trajectory")
        result = apply_coordinate_signs(pos, signs)
        np.testing.assert_allclose(result, [[-5.96, -5.34, 3.87]])

    def test_x_velocity_negated(self):
        # positions: x goes +1 per step → raw dx/dt = +1 mm/unit
        # after x-flip applied to positions: dx_flipped/dt = -1 mm/unit
        pos_raw = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        signs = coordinate_signs_from_params({"trajectory_x_sign": -1}, "trajectory")
        pos_flipped = apply_coordinate_signs(pos_raw, signs)
        traj = _make_traj(pos_flipped)
        _, vel = lookup_sphere_state(traj, t_value=1.0, vel_scale=1.0)
        np.testing.assert_allclose(vel[0], -1.0)   # x flipped
        np.testing.assert_allclose(vel[1],  2.0)   # y unchanged
        np.testing.assert_allclose(vel[2],  3.0)   # z unchanged


class TestTrajectoryYFlip:
    """trajectory_y_sign = -1 should negate y positions and y velocities."""

    def test_y_sign_loaded(self):
        signs = coordinate_signs_from_params({"trajectory_y_sign": -1}, "trajectory")
        np.testing.assert_array_equal(signs, [1.0, -1.0, 1.0])

    def test_y_position_negated(self):
        pos = np.array([[1.0, 5.0, 3.0]])
        signs = coordinate_signs_from_params({"trajectory_y_sign": -1}, "trajectory")
        result = apply_coordinate_signs(pos, signs)
        np.testing.assert_allclose(result, [[1.0, -5.0, 3.0]])

    def test_y_velocity_negated(self):
        pos_raw = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        signs = coordinate_signs_from_params({"trajectory_y_sign": -1}, "trajectory")
        pos_flipped = apply_coordinate_signs(pos_raw, signs)
        traj = _make_traj(pos_flipped)
        _, vel = lookup_sphere_state(traj, t_value=1.0, vel_scale=1.0)
        np.testing.assert_allclose(vel[0],  1.0)   # x unchanged
        np.testing.assert_allclose(vel[1], -2.0)   # y flipped
        np.testing.assert_allclose(vel[2],  3.0)   # z unchanged


class TestTrajectoryZFlip:
    """trajectory_z_sign = -1 should negate z positions and z velocities."""

    def test_z_sign_loaded(self):
        signs = coordinate_signs_from_params({"trajectory_z_sign": -1}, "trajectory")
        np.testing.assert_array_equal(signs, [1.0, 1.0, -1.0])

    def test_z_position_negated(self):
        pos = np.array([[1.0, 2.0, 7.0]])
        signs = coordinate_signs_from_params({"trajectory_z_sign": -1}, "trajectory")
        result = apply_coordinate_signs(pos, signs)
        np.testing.assert_allclose(result, [[1.0, 2.0, -7.0]])

    def test_z_velocity_negated(self):
        pos_raw = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        signs = coordinate_signs_from_params({"trajectory_z_sign": -1}, "trajectory")
        pos_flipped = apply_coordinate_signs(pos_raw, signs)
        traj = _make_traj(pos_flipped)
        _, vel = lookup_sphere_state(traj, t_value=1.0, vel_scale=1.0)
        np.testing.assert_allclose(vel[0],  1.0)   # x unchanged
        np.testing.assert_allclose(vel[1],  2.0)   # y unchanged
        np.testing.assert_allclose(vel[2], -3.0)   # z flipped
