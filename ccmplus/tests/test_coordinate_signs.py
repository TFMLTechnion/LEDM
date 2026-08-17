import numpy as np
import pytest

from ccmplus.drivers.sphere import (
    apply_coordinate_signs,
    coordinate_signs_from_params,
)


class TestCoordinateSigns:
    def test_defaults_to_unmodified_handedness(self):
        signs = coordinate_signs_from_params({}, "trajectory")

        np.testing.assert_array_equal(signs, [1.0, 1.0, 1.0])

    def test_applies_axis_signs_to_coordinates(self):
        values = np.array(
            [
                [1.0, 2.0, 3.0],
                [-4.0, -5.0, -6.0],
            ]
        )
        signs = coordinate_signs_from_params(
            {
                "trajectory_x_sign": -1,
                "trajectory_y_sign": 1,
                "trajectory_z_sign": -1,
            },
            "trajectory",
        )

        transformed = apply_coordinate_signs(values, signs)

        np.testing.assert_allclose(
            transformed,
            [
                [-1.0, 2.0, -3.0],
                [4.0, -5.0, 6.0],
            ],
        )

    def test_rejects_non_sign_values(self):
        with pytest.raises(ValueError, match="must be \\+1 or -1"):
            coordinate_signs_from_params({"tracks_x_sign": 0}, "tracks")
