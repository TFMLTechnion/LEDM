import numpy as np

from ccmplus.track_denoise import (
    METHOD_CENTRAL,
    METHOD_ONESIDED,
    METHOD_POLY,
    apply_mad_outlier_confidence,
    confidence_to_uncertainty,
    denoise_frame_velocities,
)


def make_frame(t, pos, track_ids=(7,)):
    return {
        "time_s": float(t),
        "positions_mm": np.asarray(pos, dtype=float),
        "velocities_ms": np.zeros((len(track_ids), 3), dtype=float),
        "track_ids": np.asarray(track_ids, dtype=np.int64),
    }


class TestTrackDenoise:
    def test_polynomial_velocity_from_derivative(self):
        # x(t) = 1000*t + 500*t^2 mm, so dx/dt at t=2 is 3000 mm/s = 3 m/s.
        frames = []
        for t in [0, 1, 2, 3, 4]:
            x = 1000.0 * t + 500.0 * t * t
            frames.append(make_frame(t, [[x, 2.0 * t, 0.0]]))

        result = denoise_frame_velocities(frames, 2, poly_order=2, filter_length=5)

        np.testing.assert_allclose(result.velocities_ms[0], [3.0, 0.002, 0.0], rtol=1e-10)
        assert result.method_code[0] == METHOD_POLY
        assert result.confidence[0] == 1.0

    def test_central_difference_fallback_with_two_neighbors(self):
        frames = [
            make_frame(0.0, [[0.0, 0.0, 0.0]]),
            make_frame(1.0, [[1.0, 0.0, 0.0]]),
            make_frame(2.0, [[4000.0, 0.0, 0.0]]),
        ]

        result = denoise_frame_velocities(frames, 1, poly_order=3, filter_length=5)

        np.testing.assert_allclose(result.velocities_ms[0], [2.0, 0.0, 0.0])
        assert result.method_code[0] == METHOD_CENTRAL

    def test_one_sided_difference_is_lower_confidence(self):
        frames = [
            make_frame(0.0, [[0.0, 0.0, 0.0]]),
            make_frame(1.0, [[1000.0, 0.0, 0.0]]),
        ]

        result = denoise_frame_velocities(frames, 1, poly_order=2, filter_length=5)

        np.testing.assert_allclose(result.velocities_ms[0], [1.0, 0.0, 0.0])
        assert result.method_code[0] == METHOD_ONESIDED
        assert result.confidence[0] < 0.35

    def test_mad_outlier_confidence_downweights_large_velocity(self):
        velocities = np.array(
            [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.9, 0.0, 0.0], [10.0, 0.0, 0.0]]
        )
        confidence = np.ones(4)

        updated, outliers, threshold = apply_mad_outlier_confidence(
            velocities, confidence, threshold_mad=5.0, multiplier=0.1
        )

        assert threshold < 10.0
        np.testing.assert_array_equal(outliers, [False, False, False, True])
        np.testing.assert_allclose(updated, [1.0, 1.0, 1.0, 0.1])

    def test_confidence_to_uncertainty_lowers_weight(self):
        unc = confidence_to_uncertainty(0.01, np.array([1.0, 0.25]))

        np.testing.assert_allclose(unc, [0.01, 0.02])
