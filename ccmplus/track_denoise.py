"""Track-based VIC-style polynomial velocity denoising."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


METHOD_RAW = 0
METHOD_ONESIDED = 1
METHOD_CENTRAL = 2
METHOD_POLY = 3


@dataclass
class TrackDenoiseResult:
    velocities_ms: np.ndarray
    confidence: np.ndarray
    method_code: np.ndarray
    sample_count: np.ndarray


def denoise_frame_velocities(
    frames: list[dict],
    target_index: int,
    *,
    poly_order: int = 2,
    filter_length: int = 5,
    raw_velocity_confidence: float = 0.25,
    one_sided_confidence: float = 0.35,
    central_confidence: float = 0.65,
) -> TrackDenoiseResult:
    """Estimate target-frame velocities from track-position history.

    Parameters
    ----------
    frames
        Sequence of dictionaries with ``time_s``, ``positions_mm``,
        ``velocities_ms``, and ``track_ids`` arrays.
    target_index
        Index into ``frames`` for the frame being reconstructed.
    poly_order
        Polynomial order for fitting x(t), y(t), z(t). VIC# setting is 2.
    filter_length
        Number of frames in the centered temporal fit window. VIC# setting is 5.

    Returns
    -------
    TrackDenoiseResult
        Velocities are in m/s. Confidence is in [0, 1] and can be converted to
        particle uncertainty by ``sigma_i / sqrt(confidence)``.
    """
    if not frames:
        raise ValueError("frames must not be empty")
    if target_index < 0 or target_index >= len(frames):
        raise IndexError("target_index out of range")

    target = frames[target_index]
    target_ids = np.asarray(target["track_ids"], dtype=np.int64)
    target_positions = np.asarray(target["positions_mm"], dtype=float)
    raw_velocities = np.asarray(target["velocities_ms"], dtype=float)
    n_target = len(target_ids)

    velocities = raw_velocities.copy()
    confidence = np.full(n_target, raw_velocity_confidence, dtype=float)
    method_code = np.full(n_target, METHOD_RAW, dtype=np.int8)
    sample_count = np.ones(n_target, dtype=np.int16)

    radius = max(0, int(filter_length) // 2)
    start = max(0, target_index - radius)
    stop = min(len(frames), target_index + radius + 1)
    window_indices = list(range(start, stop))
    index_maps = [
        {int(track_id): i for i, track_id in enumerate(frames[j]["track_ids"])}
        for j in window_indices
    ]
    target_time = float(target["time_s"])

    for i, track_id in enumerate(target_ids):
        times = []
        positions = []
        for j, lookup in zip(window_indices, index_maps):
            idx = lookup.get(int(track_id))
            if idx is None:
                continue
            pos = np.asarray(frames[j]["positions_mm"][idx], dtype=float)
            if not np.isfinite(pos).all():
                continue
            times.append(float(frames[j]["time_s"]))
            positions.append(pos)

        if not times:
            continue

        times_arr = np.asarray(times, dtype=float)
        pos_arr = np.asarray(positions, dtype=float)
        order = np.argsort(times_arr)
        times_arr = times_arr[order]
        pos_arr = pos_arr[order]
        sample_count[i] = len(times_arr)

        unique_time_count = len(np.unique(times_arr))
        if unique_time_count >= max(3, int(poly_order) + 1):
            degree = min(int(poly_order), unique_time_count - 1)
            tau = times_arr - target_time
            deriv_mm_s = np.empty(3, dtype=float)
            for comp in range(3):
                coeff = np.polyfit(tau, pos_arr[:, comp], degree)
                dcoeff = np.polyder(coeff)
                deriv_mm_s[comp] = np.polyval(dcoeff, 0.0)
            velocities[i] = deriv_mm_s * 1e-3
            confidence[i] = min(1.0, len(times_arr) / max(float(filter_length), 1.0))
            method_code[i] = METHOD_POLY
            continue

        before = np.where(times_arr < target_time)[0]
        after = np.where(times_arr > target_time)[0]
        if len(before) and len(after):
            ib = before[-1]
            ia = after[0]
            dt = times_arr[ia] - times_arr[ib]
            if dt > 0:
                velocities[i] = (pos_arr[ia] - pos_arr[ib]) * 1e-3 / dt
                confidence[i] = central_confidence * min(
                    1.0, len(times_arr) / max(float(filter_length), 1.0)
                )
                method_code[i] = METHOD_CENTRAL
                continue

        target_pos = target_positions[i]
        if len(before):
            ib = before[-1]
            dt = target_time - times_arr[ib]
            if dt > 0:
                velocities[i] = (target_pos - pos_arr[ib]) * 1e-3 / dt
                confidence[i] = one_sided_confidence * min(
                    1.0, len(times_arr) / max(float(filter_length), 1.0)
                )
                method_code[i] = METHOD_ONESIDED
                continue
        if len(after):
            ia = after[0]
            dt = times_arr[ia] - target_time
            if dt > 0:
                velocities[i] = (pos_arr[ia] - target_pos) * 1e-3 / dt
                confidence[i] = one_sided_confidence * min(
                    1.0, len(times_arr) / max(float(filter_length), 1.0)
                )
                method_code[i] = METHOD_ONESIDED

    confidence = np.clip(confidence, 0.0, 1.0)
    return TrackDenoiseResult(
        velocities_ms=velocities,
        confidence=confidence,
        method_code=method_code,
        sample_count=sample_count,
    )


def apply_mad_outlier_confidence(
    velocities_ms: np.ndarray,
    confidence: np.ndarray,
    *,
    threshold_mad: float = 5.0,
    multiplier: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Downweight velocities whose magnitude exceeds median + k*MAD."""
    vmag = np.linalg.norm(np.asarray(velocities_ms, dtype=float), axis=1)
    conf = np.asarray(confidence, dtype=float).copy()
    finite = np.isfinite(vmag)
    if not finite.any():
        return conf, np.zeros(len(vmag), dtype=bool), float("nan")

    med = float(np.median(vmag[finite]))
    mad = float(np.median(np.abs(vmag[finite] - med)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 1e-12:
        threshold = float(np.percentile(vmag[finite], 99.5))
    else:
        threshold = med + float(threshold_mad) * robust_sigma
    outliers = finite & (vmag > threshold)
    conf[outliers] *= float(multiplier)
    return np.clip(conf, 0.0, 1.0), outliers, threshold


def confidence_to_uncertainty(
    base_sigma: float,
    confidence: np.ndarray,
    *,
    min_confidence: float = 0.05,
) -> np.ndarray:
    """Convert confidence weights to per-particle velocity uncertainties."""
    conf = np.clip(np.asarray(confidence, dtype=float), min_confidence, 1.0)
    return float(base_sigma) / np.sqrt(conf)
