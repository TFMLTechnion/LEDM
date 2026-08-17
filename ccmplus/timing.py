"""Explicit-SI time pipeline for LE-DM v2 (Problem 1 fix).

Every velocity in this code base is in metres per second. Every time is in
seconds. The trajectory file stores a "Time" column in some file-specific unit
(for the B1111 rising-sphere data it is milliseconds, advancing 4.166667 per row
because the camera runs at 240 fps). Two parameters describe the mapping:

    trajectory_dt_units    row-to-row spacing of the Time column, in column units
                           (must equal the actual spacing seen in the file)
    trajectory_dt_seconds  physical seconds spanned by one trajectory row

From these:

    seconds_per_unit = trajectory_dt_seconds / trajectory_dt_units      [s / unit]
    vel_scale        = 1e-3 [m/mm] / seconds_per_unit                   [ (m/s) / (mm/unit) ]

A centred finite difference of position[mm] over Time[unit] gives mm/unit; times
vel_scale yields m/s. For the B1111 data the Time column is in ms
(1 unit = 1 ms = 0.001 s) so seconds_per_unit = 0.001 and vel_scale = 1.0.

THE v1 BUG: some parameter files stored the camera frame period (0.004167 s) as
`trajectory_dt_seconds` while leaving `trajectory_dt_units = 1`, giving
seconds_per_unit = 0.004167 and vel_scale = 0.24, so U_s came out ~4x too low
(~0.079 m/s instead of ~0.23 m/s). Others stored 0.001 s with units = 4.166667,
giving vel_scale = 4.17 (too high). v2 fixes this at the source: the loader
computes seconds_per_unit explicitly, validates `trajectory_dt_units` against the
file, and `validate_trajectory_timing` recomputes the mean rise speed from the
trajectory and compares it to `expected_rise_speed_ms`.
"""
from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger(__name__)


def seconds_per_unit(trajectory_dt_seconds: float,
                     trajectory_dt_units: float) -> float:
    """Physical seconds per one Time-column unit. Explicit SI."""
    if trajectory_dt_units == 0:
        raise ValueError("trajectory_dt_units must be non-zero.")
    return float(trajectory_dt_seconds) / float(trajectory_dt_units)


def velocity_scale(trajectory_dt_seconds: float,
                   trajectory_dt_units: float) -> float:
    """(m/s) per (mm / Time-column-unit). = 1e-3 / seconds_per_unit."""
    spu = seconds_per_unit(trajectory_dt_seconds, trajectory_dt_units)
    if spu <= 0:
        raise ValueError(f"seconds_per_unit must be positive, got {spu}.")
    return 1.0e-3 / spu


def _axis_index(rise_axis: str) -> int:
    a = rise_axis.strip().lstrip("+-").lower()
    return {"x": 0, "y": 1, "z": 2}[a]


def trajectory_velocities_ms(times: np.ndarray,
                             positions_mm: np.ndarray,
                             vel_scale: float) -> np.ndarray:
    """Centred finite-difference velocity [m/s] at every trajectory row.

    times        : (N,) Time column in file units (already monotone, NaNs dropped)
    positions_mm : (N, 3) positions in mm
    vel_scale    : from velocity_scale(...)
    """
    times = np.asarray(times, dtype=float)
    positions_mm = np.asarray(positions_mm, dtype=float)
    n = len(times)
    vel = np.zeros_like(positions_mm)
    if n < 2:
        return vel
    # interior: centred
    dt_c = (times[2:] - times[:-2])
    vel[1:-1] = (positions_mm[2:] - positions_mm[:-2]) / dt_c[:, None]
    # endpoints: one-sided
    vel[0] = (positions_mm[1] - positions_mm[0]) / (times[1] - times[0])
    vel[-1] = (positions_mm[-1] - positions_mm[-2]) / (times[-1] - times[-2])
    return vel * vel_scale


def validate_trajectory_timing(
    traj: dict,
    trajectory_dt_units: float,
    trajectory_dt_seconds: float,
    rise_axis: str,
    expected_rise_speed_ms: float | None,
    *,
    rel_tol: float = 0.30,
    raise_on_fail: bool = False,
) -> dict:
    """Load-time sanity check on the time pipeline.

    Recomputes the mean rise speed directly from the detected trajectory using the
    explicit-SI conversion and compares it to ``expected_rise_speed_ms``. Warns
    LOUDLY (and optionally raises) when:
      * the parameter `trajectory_dt_units` disagrees with the file row spacing, or
      * the recovered mean rise speed differs from the expectation by > rel_tol.

    Returns a diagnostics dict (also suitable for the gate report).
    """
    times = np.asarray(traj["times"], dtype=float)
    pos = np.asarray(traj["positions"], dtype=float)

    spu = seconds_per_unit(trajectory_dt_seconds, trajectory_dt_units)
    vscale = velocity_scale(trajectory_dt_seconds, trajectory_dt_units)

    # File row spacing (median of finite diffs)
    file_dt = float(np.median(np.diff(times))) if len(times) >= 2 else float("nan")
    units_mismatch = (
        np.isfinite(file_dt) and trajectory_dt_units > 0
        and abs(file_dt - trajectory_dt_units) / trajectory_dt_units > 0.01
    )

    vel = trajectory_velocities_ms(times, pos, vscale)   # (N, 3) m/s
    ax = _axis_index(rise_axis)
    sign = -1.0 if rise_axis.strip().startswith("-") else 1.0
    rise_speed = sign * vel[:, ax]
    mean_rise = float(np.mean(rise_speed))
    speed_mag = float(np.mean(np.linalg.norm(vel, axis=1)))
    # robust "terminal" estimate: median of the second half (after transient)
    half = len(rise_speed) // 2
    terminal_rise = float(np.median(rise_speed[half:])) if half > 0 else mean_rise

    diag = {
        "seconds_per_unit": spu,
        "vel_scale": vscale,
        "file_row_dt_units": file_dt,
        "param_trajectory_dt_units": float(trajectory_dt_units),
        "units_mismatch": bool(units_mismatch),
        "mean_rise_speed_ms": mean_rise,
        "terminal_rise_speed_ms": terminal_rise,
        "mean_speed_mag_ms": speed_mag,
        "expected_rise_speed_ms": expected_rise_speed_ms,
        "ok": True,
    }

    if units_mismatch:
        msg = (f"TIMING: trajectory_dt_units = {trajectory_dt_units:g} but the file "
               f"row spacing is {file_dt:g} (ratio {file_dt/trajectory_dt_units:.4f}). "
               f"seconds_per_unit and vel_scale are likely WRONG.")
        log.warning("!" * 8 + " " + msg)
        diag["ok"] = False

    if expected_rise_speed_ms is not None and expected_rise_speed_ms > 0:
        rel = abs(terminal_rise - expected_rise_speed_ms) / expected_rise_speed_ms
        diag["rel_error_vs_expected"] = rel
        if rel > rel_tol:
            msg = (f"TIMING SANITY CHECK FAILED: recovered terminal rise speed "
                   f"{terminal_rise:.4f} m/s (mean {mean_rise:.4f}) disagrees with "
                   f"expected_rise_speed_ms = {expected_rise_speed_ms:.4f} m/s "
                   f"(rel error {rel*100:.1f}% > {rel_tol*100:.0f}%). "
                   f"Check trajectory_dt_seconds / trajectory_dt_units. "
                   f"seconds_per_unit={spu:.6g}, vel_scale={vscale:.4f}.")
            log.warning("!" * 8 + " " + msg)
            diag["ok"] = False
            if raise_on_fail:
                raise ValueError(msg)
        else:
            log.info("TIMING OK: terminal rise speed %.4f m/s vs expected %.4f m/s "
                     "(rel %.1f%%), vel_scale=%.4f.",
                     terminal_rise, expected_rise_speed_ms, rel * 100, vscale)

    return diag
