"""CCMPlus generalised sphere driver.

Works for any sphere experiment whose data follow the Tecplot trajectory +
tracks format used in R12.  All case-specific values (diameter, axis, file
patterns, time units) are supplied through the ``p`` parameter dict.

Entry point: ``run_sphere(p)``
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ccmplus.io_tecplot import read_trajectory, read_tracks_zone
from ccmplus.grid import RectGrid
from ccmplus.config import Config, BodyState, FrameData
from ccmplus.reconstruct import CCMPlus
from ccmplus.interp import build_interpolation_matrix
from ccmplus.operators import build_laplacian_smoothing_operator, operator_rms
from ccmplus.solver import build_weight_matrix
from ccmplus.sdf import signed_distance_sphere_points
from ccmplus.track_denoise import (
    METHOD_CENTRAL,
    METHOD_ONESIDED,
    METHOD_POLY,
    METHOD_RAW,
    apply_mad_outlier_confidence,
    confidence_to_uncertainty,
    denoise_frame_velocities,
)

log = logging.getLogger(__name__)

_VALS_PER_LINE = 8


# ---------------------------------------------------------------------------
# Coordinate-system helpers
# ---------------------------------------------------------------------------

def coordinate_signs_from_params(p: dict, prefix: str) -> np.ndarray:
    """Return per-axis coordinate signs for ``trajectory`` or ``tracks`` data."""
    signs = np.array(
        [
            p.get(f"{prefix}_x_sign", 1),
            p.get(f"{prefix}_y_sign", 1),
            p.get(f"{prefix}_z_sign", 1),
        ],
        dtype=float,
    )
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError(
            f"{prefix}_*_sign values must be +1 or -1; got {signs.tolist()}."
        )
    return signs


def apply_coordinate_signs(values: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Apply a handedness/sign correction to an ``(N, 3)`` coordinate array."""
    return np.asarray(values, dtype=float) * signs.reshape(1, 3)


# ---------------------------------------------------------------------------
# Trajectory-index resolver
# ---------------------------------------------------------------------------

def _resolve_trajectory_index(
    track_data: dict,
    local_step_idx: int,
    trajectory_times: np.ndarray,
    p: dict,
) -> tuple[int, float, float]:
    """Return (trajectory row index, raw trajectory time, physical seconds).

    Numeric ``ZONE T`` files are matched by value. Snapshot-labelled files are
    mapped by file order: the first processed file maps to trajectory row 1.
    """
    seconds_per_unit = p["trajectory_dt_seconds"] / p["trajectory_dt_units"]

    if track_data["zone_time_is_numeric"]:
        target = float(track_data["zone_time"] - p.get("zone_time_offset", 0))
        tol = max(1e-9, 1e-6 * abs(target))
        diffs = np.abs(trajectory_times.astype(float) - target)
        hits = np.where(diffs <= tol)[0]
        if len(hits) == 0:
            raise ValueError(
                f"No trajectory entry matches zone_time={track_data['zone_time']} "
                f"minus zone_time_offset={p.get('zone_time_offset', 0)}. "
                f"Target={target}, trajectory range=[{trajectory_times[0]}, "
                f"{trajectory_times[-1]}]."
            )
        traj_idx = int(hits[0])
    else:
        traj_idx = local_step_idx - 1
        if traj_idx < 0 or traj_idx >= len(trajectory_times):
            raise ValueError(
                f"Snapshot trajectory row {local_step_idx} is out of range "
                f"[1, {len(trajectory_times)}]."
            )

    traj_time = float(trajectory_times[traj_idx])
    return traj_idx, traj_time, traj_time * seconds_per_unit


# ---------------------------------------------------------------------------
# Rise-axis helper
# ---------------------------------------------------------------------------

_AXIS_MAP: dict[str, tuple[int, int]] = {
    "+x": (0, +1), "-x": (0, -1),
    "+y": (1, +1), "-y": (1, -1),
    "+z": (2, +1), "-z": (2, -1),
}


def _axis_index_and_sign(rise_axis: str) -> tuple[int, int]:
    """Return (axis_index, sign) for a rise-axis string like '+y' or '-x'."""
    key = rise_axis.strip().lower()
    if key not in _AXIS_MAP:
        raise ValueError(
            f"Invalid rise_axis {rise_axis!r}. "
            f"Must be one of: {', '.join(sorted(_AXIS_MAP))}."
        )
    return _AXIS_MAP[key]


# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------

def _block(fh, arr: np.ndarray, fmt: str = "{:.6g}") -> None:
    n = len(arr)
    for start in range(0, n, _VALS_PER_LINE):
        chunk = arr[start : start + _VALS_PER_LINE]
        fh.write(" ".join(fmt.format(v) for v in chunk) + "\r\n")


def write_output(
    path: Path,
    grid: RectGrid,
    vel: np.ndarray,   # (Ng, 3) [m/s]
    C: np.ndarray,     # (Ng,) int8  -1/0/+1
    t_ms: int,
    case_name: str,
    solution_time_s: float,
) -> None:
    """Write one CCM+ timestep in Tecplot ASCII BLOCK format.

    mask: 0 = fluid, 1 = solid interior, 2 = body shell.
    Node order: idx = i + Nx*(j + Ny*k), i fastest.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    x = grid.nodes[:, 0]
    y = grid.nodes[:, 1]
    z = grid.nodes[:, 2]
    u, v, w = vel[:, 0], vel[:, 1], vel[:, 2]
    vmag = np.linalg.norm(vel, axis=1)
    mask = np.where(C == 1, 0.0, np.where(C == -1, 1.0, 2.0))

    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f'TITLE = "{case_name} CCM+ reconstruction t={t_ms}ms"\r\n')
        fh.write(
            'VARIABLES = "x[mm]" "y[mm]" "z[mm]"'
            ' "Vx[m/s]" "Vy[m/s]" "Vz[m/s]" "|V|[m/s]" "mask"\r\n'
        )
        fh.write(f'ZONE T="{t_ms}"\r\n')
        fh.write(f"STRANDID=1, SOLUTIONTIME={solution_time_s:.6f}\r\n")
        fh.write(f"I={Nx}, J={Ny}, K={Nz}, ZONETYPE=Ordered\r\n")
        fh.write("DATAPACKING=BLOCK\r\n")
        _block(fh, x,    "{:.4f}")
        _block(fh, y,    "{:.4f}")
        _block(fh, z,    "{:.4f}")
        _block(fh, u)
        _block(fh, v)
        _block(fh, w)
        _block(fh, vmag)
        _block(fh, mask, "{:.0f}")


# ---------------------------------------------------------------------------
# Sphere kinematics
# ---------------------------------------------------------------------------

def lookup_sphere_state(
    traj: dict,
    t_value: float,
    vel_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sphere position [mm] and velocity [m/s] at trajectory time.

    Raw velocity from the centred finite difference is pos[mm] / traj_units.
    ``vel_scale`` converts that to m/s:
        vel_scale = 1e-3 [m/mm] / (seconds_per_traj_unit [s/unit])
    For R12 (1 unit = 1 ms): vel_scale = 1e-3 / 0.001 = 1.0 (no conversion).
    """
    times     = traj["times"]      # (N,) trajectory time units
    positions = traj["positions"]  # (N, 3) [mm]

    matches = np.where(np.isclose(times, float(t_value), rtol=0.0, atol=1e-9))[0]
    if len(matches) == 0:
        raise ValueError(
            f"Trajectory has no entry for t={t_value}. "
            f"Range available: {times[0]}-{times[-1]}."
        )
    idx = int(matches[0])
    pos = positions[idx].copy()

    if idx == 0:
        dt  = float(times[1] - times[0])
        vel = (positions[1] - positions[0]) / dt
    elif idx >= len(times) - 1:
        dt  = float(times[-1] - times[-2])
        vel = (positions[-1] - positions[-2]) / dt
    else:
        dt  = float(times[idx + 1] - times[idx - 1])
        vel = (positions[idx + 1] - positions[idx - 1]) / dt

    return pos, vel * vel_scale   # pos [mm], vel [m/s]


def median_velocity_filter(
    positions_mm: np.ndarray,
    velocities_ms: np.ndarray,
    radius_mm: float,
    threshold_ms: float,
    min_neighbors: int,
) -> np.ndarray:
    """Return a mask that keeps velocities close to their local spatial median.

    For each tracer, neighbours within ``radius_mm`` are gathered in physical
    space. The component-wise median neighbour velocity is computed, and the
    tracer is rejected when ``||v_i - median(v_neighbours)|| > threshold_ms``.
    Points with too few neighbours are kept because their local median is not
    reliable.
    """
    n = len(positions_mm)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if radius_mm <= 0.0 or threshold_ms <= 0.0:
        return np.ones(n, dtype=bool)

    min_neighbors = max(1, int(min_neighbors))
    tree = cKDTree(positions_mm)
    neighbours = tree.query_ball_point(positions_mm, r=radius_mm)
    keep = np.ones(n, dtype=bool)

    for i, idx in enumerate(neighbours):
        if len(idx) < min_neighbors:
            continue
        local_median = np.median(velocities_ms[idx], axis=0)
        if np.linalg.norm(velocities_ms[i] - local_median) > threshold_ms:
            keep[i] = False

    return keep


def interpolate_velocity_at_points(
    positions_mm: np.ndarray,
    grid: RectGrid,
    velocity_ms: np.ndarray,
) -> np.ndarray:
    """Interpolate a grid velocity field to tracer positions."""
    if len(positions_mm) == 0:
        return np.empty((0, 3), dtype=float)
    A = build_interpolation_matrix(positions_mm, grid)
    return (A @ velocity_ms.ravel()).reshape(-1, 3)


def residual_outlier_filter(
    positions_mm: np.ndarray,
    velocities_ms: np.ndarray,
    grid: RectGrid,
    coarse_velocity_ms: np.ndarray,
    threshold_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep tracers whose residual against a coarse field is within threshold."""
    n = len(positions_mm)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    if threshold_ms <= 0.0:
        return np.ones(n, dtype=bool), np.zeros(n, dtype=float)

    coarse_at_particles = interpolate_velocity_at_points(
        positions_mm, grid, coarse_velocity_ms
    )
    residuals = np.linalg.norm(velocities_ms - coarse_at_particles, axis=1)
    return residuals <= threshold_ms, residuals


def make_body_with_sigma(body: BodyState, sigma_s_mm: float) -> BodyState:
    """Copy a body state while replacing the near-body weighting length."""
    return BodyState(
        X_s=body.X_s.copy(),
        U_s=body.U_s.copy(),
        omega_s=body.omega_s.copy(),
        radius=body.radius,
        sigma_s=sigma_s_mm,
        sdf_fn=body.sdf_fn,
        velocity_fn=body.velocity_fn,
    )


def parse_float_list(value) -> list[float]:
    """Parse a comma-separated parameter value into floats."""
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    return [float(v) for v in value]


def parse_int_list(value) -> list[int]:
    """Parse a comma-separated parameter value into integers."""
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, float):
        return [int(round(value))]
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(v) for v in value]


def support_count_grid(
    grid: RectGrid,
    positions_mm: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    """Count particles within ``radius_mm`` of every grid node."""
    if len(positions_mm) == 0:
        return np.zeros(grid.size, dtype=np.int32)
    tree = cKDTree(positions_mm)
    try:
        return tree.query_ball_point(
            grid.nodes, r=radius_mm, return_length=True, workers=-1
        ).astype(np.int32)
    except TypeError:
        return np.array(
            [len(idx) for idx in tree.query_ball_point(grid.nodes, r=radius_mm)],
            dtype=np.int32,
        )


def support_statistics_from_counts(
    counts: np.ndarray,
    valid_nodes: np.ndarray,
    radius_mm: float,
) -> dict[str, float]:
    """Compute support-count statistics from a precomputed count field."""
    vals = counts[np.asarray(valid_nodes, dtype=bool)]
    if len(vals) == 0:
        return {
            "support_radius_mm": float(radius_mm),
            "support_min": float("nan"),
            "support_mean": float("nan"),
            "support_median": float("nan"),
            "support_p5": float("nan"),
            "support_p95": float("nan"),
            "support_max": float("nan"),
            "support_pct_eq0": float("nan"),
            "support_pct_lt3": float("nan"),
            "support_pct_lt5": float("nan"),
        }
    return {
        "support_radius_mm": float(radius_mm),
        "support_min": float(np.min(vals)),
        "support_mean": float(np.mean(vals)),
        "support_median": float(np.median(vals)),
        "support_p5": float(np.percentile(vals, 5)),
        "support_p95": float(np.percentile(vals, 95)),
        "support_max": float(np.max(vals)),
        "support_pct_eq0": float(100.0 * np.mean(vals == 0)),
        "support_pct_lt3": float(100.0 * np.mean(vals < 3)),
        "support_pct_lt5": float(100.0 * np.mean(vals < 5)),
    }


def support_statistics(
    grid: RectGrid,
    positions_mm: np.ndarray,
    valid_nodes: np.ndarray,
    radius_mm: float,
) -> dict[str, float]:
    """Compute particle support-count statistics for grid nodes."""
    counts = support_count_grid(grid, positions_mm, radius_mm)
    return support_statistics_from_counts(counts, valid_nodes, radius_mm)


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _safe_mean(values: np.ndarray) -> float:
    vals = _finite(values)
    return float(np.mean(vals)) if len(vals) else float("nan")


def _safe_median(values: np.ndarray) -> float:
    vals = _finite(values)
    return float(np.median(vals)) if len(vals) else float("nan")


def _safe_percentile(values: np.ndarray, q: float) -> float:
    vals = _finite(values)
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def _safe_max(values: np.ndarray) -> float:
    vals = _finite(values)
    return float(np.max(vals)) if len(vals) else float("nan")


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-300:
        return float("nan")
    return float(num / den)


def reshape_grid(values: np.ndarray, grid: RectGrid) -> np.ndarray:
    """Reshape flat node data into (Nx, Ny, Nz) using CCM+ node ordering."""
    return np.asarray(values).reshape(grid.shape, order="F")


def compute_roughness_metric(
    velocity: np.ndarray,
    grid: RectGrid,
    valid_fluid: np.ndarray,
) -> float:
    """RMS of spacing-scaled first differences within connected fluid pairs."""
    U = reshape_grid(velocity[:, 0], grid)
    V = reshape_grid(velocity[:, 1], grid)
    W = reshape_grid(velocity[:, 2], grid)
    valid = reshape_grid(valid_fluid.astype(bool), grid)
    diffs = []
    for arr in (U, V, W):
        for axis in range(3):
            d = np.diff(arr, axis=axis) / grid.delta
            v0 = np.take(valid, indices=range(valid.shape[axis] - 1), axis=axis)
            v1 = np.take(valid, indices=range(1, valid.shape[axis]), axis=axis)
            mask = v0 & v1 & np.isfinite(d)
            if np.any(mask):
                diffs.append(d[mask].ravel())
    if not diffs:
        return float("nan")
    vals = np.concatenate(diffs)
    return float(np.sqrt(np.mean(vals * vals)))


def compute_patchiness_metric(
    velocity: np.ndarray,
    grid: RectGrid,
    valid_fluid: np.ndarray,
) -> float:
    """Percentage of valid nodes whose local |V| deviation exceeds 3 sigma."""
    vmag = np.linalg.norm(velocity, axis=1)
    valid = valid_fluid & np.isfinite(vmag)
    if not np.any(valid):
        return float("nan")
    V3 = reshape_grid(vmag, grid)
    valid3 = reshape_grid(valid_fluid.astype(bool), grid)
    fill = float(np.nanmedian(vmag[valid]))
    Vfill = np.where(np.isfinite(V3), V3, fill)
    local_med = ndimage.median_filter(Vfill, size=3, mode="nearest")
    deviation = np.abs(V3 - local_med)
    vals = deviation[valid3 & np.isfinite(deviation)]
    if len(vals) == 0:
        return float("nan")
    sigma = float(np.std(vals))
    return 0.0 if sigma <= 0.0 else float(100.0 * np.mean(vals > 3.0 * sigma))


def compute_divergence_rms(
    velocity: np.ndarray,
    grid: RectGrid,
    valid_fluid: np.ndarray,
) -> float:
    """RMS centered divergence on interior fluid nodes with fluid neighbours."""
    U = reshape_grid(velocity[:, 0], grid)
    V = reshape_grid(velocity[:, 1], grid)
    W = reshape_grid(velocity[:, 2], grid)
    valid = reshape_grid(valid_fluid.astype(bool), grid)
    vals = []
    for i in range(1, grid.Nx - 1):
        for j in range(1, grid.Ny - 1):
            for k in range(1, grid.Nz - 1):
                if not valid[i, j, k]:
                    continue
                if not (
                    valid[i - 1, j, k]
                    and valid[i + 1, j, k]
                    and valid[i, j - 1, k]
                    and valid[i, j + 1, k]
                    and valid[i, j, k - 1]
                    and valid[i, j, k + 1]
                ):
                    continue
                div = (
                    (U[i + 1, j, k] - U[i - 1, j, k])
                    + (V[i, j + 1, k] - V[i, j - 1, k])
                    + (W[i, j, k + 1] - W[i, j, k - 1])
                ) / (2.0 * grid.delta)
                vals.append(div)
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def compute_laplacian_rms(
    velocity: np.ndarray,
    grid: RectGrid,
    C: np.ndarray,
    p: dict,
) -> float:
    """RMS of the same spacing-scaled mask-aware Laplacian used for smoothing."""
    config = Config(
        domain_min=tuple(grid.domain_min),
        domain_max=tuple(grid.domain_max),
        delta=grid.delta,
        enable_field_smoothing=True,
        smoothing_exclude_mask=p["smoothing_exclude_mask"],
        smoothing_exclude_shell=p["smoothing_exclude_shell"],
        smoothing_no_cross_mask=p["smoothing_no_cross_mask"],
        smoothing_spacing_scaled=p["smoothing_spacing_scaled"],
        smoothing_componentwise=p["smoothing_componentwise"],
    )
    op = build_laplacian_smoothing_operator(grid, C, config).matrix
    return operator_rms(op, velocity.ravel())


# ---------------------------------------------------------------------------
# Condition-number diagnostic
# ---------------------------------------------------------------------------

def estimate_condition(
    A: sp.csr_matrix,
    W: sp.csr_matrix,
    kappa: float,
    Ng3: int,
) -> float:
    """Estimate cond(H) ≈ lam_max(H) / kappa, H = A^T W A + kappa I."""
    H = A.T @ (W @ A) + kappa * sp.eye(Ng3, format="csr")
    try:
        lam_max = eigsh(
            H, k=1, which="LM",
            return_eigenvectors=False,
            tol=1e-3, maxiter=500,
        )[0]
        return float(lam_max / kappa)
    except Exception as exc:
        log.warning("eigsh failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Diagnostic PNG  (always slices perpendicular to x; shows y-z plane)
# ---------------------------------------------------------------------------

def make_diagnostic_png(
    step_n: int,
    t_ms: int,
    grid: RectGrid,
    vel: np.ndarray,                # (Ng, 3)
    C: np.ndarray,                  # (Ng,)
    positions_noise_mm: np.ndarray, # (N, 3) post-active filters
    sphere_center: np.ndarray,      # (3,) [mm]
    R_mm: float,
    case_name: str,
    output_dir: Path,
) -> Path:
    """Save two-panel midplane y-z diagnostic PNG (slice perpendicular to x)."""
    x_s, y_s, z_s = sphere_center

    i_s = int(np.argmin(np.abs(grid.x_coords - x_s)))
    x_slice = grid.x_coords[i_s]
    if abs(x_slice - x_s) > 3 * grid.delta:
        log.warning(
            "Slice x=%.1f mm is far from sphere x=%.1f mm; "
            "sphere may be outside grid x-range.", x_slice, x_s,
        )

    jj, kk = np.meshgrid(np.arange(grid.Ny), np.arange(grid.Nz), indexing="ij")
    sl = i_s + grid.Nx * (jj + grid.Ny * kk)

    Y2D  = grid.y_coords[jj]
    Z2D  = grid.z_coords[kk]
    Vy2D = vel[sl, 1]
    Vz2D = vel[sl, 2]
    C2D  = C[sl]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax = axes[0]
    fluid_2d = C2D == 1
    q  = max(1, min(grid.Ny, grid.Nz) // 15)
    Yq  = Y2D[::q, ::q];  Zq  = Z2D[::q, ::q]
    Vyq = Vy2D[::q, ::q]; Vzq = Vz2D[::q, ::q]
    mq  = fluid_2d[::q, ::q]

    if mq.any():
        ax.quiver(Yq[mq], Zq[mq], Vyq[mq], Vzq[mq], color="steelblue", alpha=0.8)

    for label, val, clr in (("shell", 0, "gold"), ("solid", -1, "tomato")):
        mask_2d = C2D == val
        if mask_2d.any():
            ax.scatter(Y2D[mask_2d], Z2D[mask_2d], s=4, c=clr,
                       alpha=0.5, label=label, zorder=2)

    circ = mpatches.Circle((y_s, z_s), R_mm,
                            fill=False, color="red", linewidth=2, label="sphere")
    ax.add_patch(circ)
    ax.set_xlim(grid.y_coords[0], grid.y_coords[-1])
    ax.set_ylim(grid.z_coords[0], grid.z_coords[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("y [mm]"); ax.set_ylabel("z [mm]")
    ax.set_title(
        f"Reconstructed Vy-Vz -- step {step_n}, t={t_ms} ms\n"
        f"slice x = {x_slice:.1f} mm  (sphere x = {x_s:.1f} mm)"
    )
    ax.legend(fontsize=8, loc="upper right")

    ax2 = axes[1]
    in_slab = np.abs(positions_noise_mm[:, 0] - x_slice) <= grid.delta
    n_slab  = int(in_slab.sum())
    ax2.scatter(positions_noise_mm[in_slab, 1], positions_noise_mm[in_slab, 2],
                s=1, c="darkorange", alpha=0.5)
    circ2 = mpatches.Circle((y_s, z_s), R_mm, fill=False, color="red", linewidth=2)
    ax2.add_patch(circ2)
    ax2.set_xlim(grid.y_coords[0], grid.y_coords[-1])
    ax2.set_ylim(grid.z_coords[0], grid.z_coords[-1])
    ax2.set_aspect("equal")
    ax2.set_xlabel("y [mm]"); ax2.set_ylabel("z [mm]")
    ax2.set_title(
        f"Tracers in slab |x - {x_slice:.1f}| <= {grid.delta:.0f} mm\n"
        f"(n = {n_slab}, post-active filters)"
    )

    plt.tight_layout()
    out = output_dir / f"{case_name}_slice_step{step_n:02d}_t{t_ms:04d}ms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(target))))


def _save_scalar_zplane_png(
    path: Path,
    grid: RectGrid,
    field: np.ndarray,
    z_value: float,
    title: str,
    colorbar_label: str,
    valid: np.ndarray | None = None,
    cmap: str = "turbo",
) -> Path:
    """Save a compact x-y scalar plot at the nearest z plane."""
    k = _nearest_index(grid.z_coords, z_value)
    X, Y = np.meshgrid(grid.x_coords, grid.y_coords, indexing="ij")
    data = reshape_grid(field, grid)[:, :, k].astype(float)
    if valid is not None:
        valid2 = reshape_grid(valid.astype(bool), grid)[:, :, k]
        data = data.copy()
        data[~valid2] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    if np.all(~np.isfinite(data)):
        ax.text(0.5, 0.5, "No finite data on selected plane",
                transform=ax.transAxes, ha="center", va="center")
    else:
        im = ax.contourf(X, Y, data, levels=40, cmap=cmap)
        fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{title}, z={grid.z_coords[k]:.5g} mm")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_components_zplane_png(
    path: Path,
    grid: RectGrid,
    velocity: np.ndarray,
    z_value: float,
    valid: np.ndarray,
    title: str,
) -> Path:
    """Save U, V, W component panels at the nearest z plane."""
    k = _nearest_index(grid.z_coords, z_value)
    X, Y = np.meshgrid(grid.x_coords, grid.y_coords, indexing="ij")
    valid2 = reshape_grid(valid.astype(bool), grid)[:, :, k]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, comp, label in zip(axes, range(3), ("U", "V", "W")):
        data = reshape_grid(velocity[:, comp], grid)[:, :, k].astype(float)
        data = data.copy()
        data[~valid2] = np.nan
        if np.all(~np.isfinite(data)):
            ax.text(0.5, 0.5, "No finite data", transform=ax.transAxes,
                    ha="center", va="center")
        else:
            im = ax.contourf(X, Y, data, levels=40, cmap="coolwarm")
            fig.colorbar(im, ax=ax, label=f"{label} [m/s]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(label)
    fig.suptitle(f"{title}, z={grid.z_coords[k]:.5g} mm")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_particle_scatter_zplane_png(
    path: Path,
    grid: RectGrid,
    positions_mm: np.ndarray,
    velocities_ms: np.ndarray,
    z_value: float,
    sphere_center: np.ndarray,
    R_mm: float,
    title: str,
) -> Path:
    """Save particle |V| scatter in a slab around the selected z plane."""
    k = _nearest_index(grid.z_coords, z_value)
    z_slice = float(grid.z_coords[k])
    slab = np.abs(positions_mm[:, 2] - z_slice) <= grid.delta
    vmag = np.linalg.norm(velocities_ms, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    if np.any(slab):
        sc = ax.scatter(
            positions_mm[slab, 0],
            positions_mm[slab, 1],
            c=vmag[slab],
            s=5,
            cmap="turbo",
            alpha=0.8,
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax, label="particle |V| [m/s]")
    else:
        ax.text(0.5, 0.5, "No particles in selected z slab",
                transform=ax.transAxes, ha="center", va="center")
    sphere_xy = mpatches.Circle(
        (sphere_center[0], sphere_center[1]), R_mm,
        fill=False, color="red", linewidth=1.5, label="sphere projection"
    )
    ax.add_patch(sphere_xy)
    ax.set_xlim(grid.x_coords[0], grid.x_coords[-1])
    ax.set_ylim(grid.y_coords[0], grid.y_coords[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        f"{title}, |z - {z_slice:.5g}| <= {grid.delta:g} mm, n={int(np.sum(slab))}"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_selected_frame_diagnostics(
    output_dir: Path,
    case_name: str,
    step_n: int,
    t_ms: int,
    grid: RectGrid,
    velocity: np.ndarray,
    C: np.ndarray,
    support_count: np.ndarray,
    positions_mm: np.ndarray,
    velocities_ms: np.ndarray,
    sphere_center: np.ndarray,
    R_mm: float,
) -> list[Path]:
    """Save the compact selected-frame diagnostic PNG set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{case_name}_step{step_n:02d}_t{t_ms:04d}ms"
    z_value = float(sphere_center[2])
    valid = C == 1
    vmag = np.linalg.norm(velocity, axis=1)
    mask = np.where(C == 1, 0.0, np.where(C == -1, 1.0, 2.0))

    paths = [
        _save_scalar_zplane_png(
            output_dir / f"{prefix}_vmag_zplane.png",
            grid, vmag, z_value, "|V|", "|V| [m/s]", valid=valid,
        ),
        _save_components_zplane_png(
            output_dir / f"{prefix}_components_zplane.png",
            grid, velocity, z_value, valid, "Velocity components",
        ),
        _save_scalar_zplane_png(
            output_dir / f"{prefix}_support_zplane.png",
            grid, support_count, z_value, "Support count", "count",
            valid=valid, cmap="viridis",
        ),
        _save_scalar_zplane_png(
            output_dir / f"{prefix}_mask_zplane.png",
            grid, mask, z_value, "Mask", "mask", cmap="viridis",
        ),
        _save_particle_scatter_zplane_png(
            output_dir / f"{prefix}_particle_scatter_zplane.png",
            grid, positions_mm, velocities_ms, z_value, sphere_center, R_mm,
            "Particle |V|",
        ),
    ]
    return paths


def write_full_run_summary(path: Path, rows: list[dict]) -> None:
    """Write a compact text summary for production runs."""
    completed = [r for r in rows if not r.get("failed", False)]
    failed = [r for r in rows if r.get("failed", False)]

    def stats_line(key: str) -> str:
        vals = np.array(
            [r.get(key, float("nan")) for r in completed],
            dtype=float,
        )
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return f"{key}: min=nan mean=nan max=nan"
        return (
            f"{key}: min={np.min(vals):.6g} "
            f"mean={np.mean(vals):.6g} max={np.max(vals):.6g}"
        )

    low = [
        r["frame"] for r in completed
        if r.get("ratio_grid_p95_to_particle_p95", 1.0) < 0.3
        or r.get("ratio_grid_p99_to_particle_p99", 1.0) < 0.3
    ]
    high = [r["frame"] for r in completed if r.get("grid_vmag_max", 0.0) > 1.0]
    poor_support = [
        r["frame"] for r in completed
        if r.get("support_percent_zero", 0.0) > 20.0
        or r.get("support_percent_lt3", 0.0) > 50.0
    ]
    unconverged = [
        r["frame"] for r in completed
        if not bool(r.get("solver_converged", False))
    ]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("CCM+ full run summary\n")
        fh.write("=====================\n\n")
        fh.write(f"frames_completed: {len(completed)}\n")
        fh.write(
            "failed_frames: "
            + (", ".join(str(r.get("frame", "?")) for r in failed) if failed else "none")
            + "\n\n"
        )
        for key in (
            "grid_vmag_p95",
            "grid_vmag_p99",
            "ratio_grid_p95_to_particle_p95",
            "support_percent_zero",
            "support_percent_lt3",
        ):
            fh.write(stats_line(key) + "\n")
        fh.write("\n")
        fh.write(
            "frames_with_suspiciously_low_velocity_magnitude: "
            + (", ".join(map(str, low)) if low else "none") + "\n"
        )
        fh.write(
            "frames_with_suspiciously_high_velocity_magnitude: "
            + (", ".join(map(str, high)) if high else "none") + "\n"
        )
        fh.write(
            "frames_with_poor_support: "
            + (", ".join(map(str, poor_support)) if poor_support else "none") + "\n"
        )
        fh.write(
            "frames_where_solver_did_not_converge: "
            + (", ".join(map(str, unconverged)) if unconverged else "none") + "\n"
        )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_sphere(p: dict) -> None:
    """Run the CCMPlus sphere reconstruction pipeline.

    Parameters
    ----------
    p : dict
        Parsed parameter dict (``ccmplus.params.read_parameters`` + defaults).
        All mandatory keys must be present; optional keys must have been
        filled in by ``apply_defaults`` before this call.
    """
    tracks_dir     = Path(p["tracks_dir"])
    output_dat_dir = Path(p["output_dat_dir"])
    output_fig_dir = Path(p["output_fig_dir"])
    case_name      = p["case_name"]

    # Resolve trajectory file: explicit path takes priority; otherwise derive
    # from tracks_dir parent and trajectory_filename.
    if "traj_file" in p and p["traj_file"]:
        traj_file = Path(p["traj_file"])
    else:
        traj_file = tracks_dir.parent / p["trajectory_filename"]

    # ------------------------------------------------------------------
    # Rise-axis setup
    # ------------------------------------------------------------------
    axis_idx, sign = _axis_index_and_sign(p["rise_axis"])
    axis_label = "xyz"[axis_idx]          # "x", "y", or "z"
    axis_name  = p["rise_axis"]           # e.g. "+y"

    # Velocity scale: converts pos[mm]/traj_unit → m/s
    seconds_per_unit = p["trajectory_dt_seconds"] / p["trajectory_dt_units"]
    vel_scale = 1e-3 / seconds_per_unit   # = 1.0 for R12 (1 unit = 1 ms)
    trajectory_signs = coordinate_signs_from_params(p, "trajectory")
    tracks_signs = coordinate_signs_from_params(p, "tracks")

    # ------------------------------------------------------------------
    # 0. Path validation
    # ------------------------------------------------------------------
    errors = []
    if not tracks_dir.exists():
        errors.append(f"  tracks_dir not found: {tracks_dir}")
    if not traj_file.exists():
        errors.append(f"  traj_file  not found: {traj_file}")
    if errors:
        for e in errors:
            log.error("%s", e)
        raise FileNotFoundError("Data paths missing:\n" + "\n".join(errors))

    output_dat_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_png_steps = set(parse_int_list(p.get("diagnostic_png_steps", "")))
    if p["write_png"] or diagnostic_png_steps:
        output_fig_dir.mkdir(parents=True, exist_ok=True)

    step_indices = list(range(p["first_step"], p["last_step"] + 1))

    # ------------------------------------------------------------------
    # 1. Trajectory
    # ------------------------------------------------------------------
    traj = read_trajectory(str(traj_file))
    traj["positions"] = apply_coordinate_signs(traj["positions"], trajectory_signs)

    if not np.all(trajectory_signs == 1.0):
        log.info(
            "Trajectory coordinate signs: x=%+.0f y=%+.0f z=%+.0f",
            trajectory_signs[0], trajectory_signs[1], trajectory_signs[2],
        )
    if not np.all(tracks_signs == 1.0):
        log.info(
            "Track coordinate signs: x=%+.0f y=%+.0f z=%+.0f",
            tracks_signs[0], tracks_signs[1], tracks_signs[2],
        )

    if p["diameter_source"] == "trajectory_header":
        D = traj["diameter_mm"]
    elif p["diameter_source"] == "parameter_file":
        if "diameter_mm_override" not in p:
            raise KeyError(
                "diameter_source = parameter_file but diameter_mm_override "
                "is not set in parameters.txt."
            )
        D = float(p["diameter_mm_override"])
    else:
        raise ValueError(
            f"Unknown diameter_source {p['diameter_source']!r}. "
            "Use 'trajectory_header' or 'parameter_file'."
        )

    R_mm = D / 2.0
    log.info(
        "Case: %s  |  Trajectory: %d entries, t = [%.6g, %.6g]  |  "
        "D = %.1f mm (source: %s)  |  rise_axis = %s",
        case_name, len(traj["times"]), traj["times"][0], traj["times"][-1],
        D, p["diameter_source"], axis_name,
    )

    # Warn when the trajectory file spacing and parameter-file conversion do
    # not agree. The run still proceeds because some exports store time labels
    # rather than integer frame indices.
    if len(traj["times"]) >= 2:
        file_dt = float(traj["times"][1] - traj["times"][0])
        expect_dt = float(p["trajectory_dt_units"])
        if expect_dt > 0 and abs(file_dt - expect_dt) / expect_dt > 0.01:
            log.warning(
                "Trajectory dt mismatch: row spacing in file = %.6g, "
                "trajectory_dt_units = %.6g (ratio = %.4f). "
                "Verify trajectory_dt_units in parameters.txt matches the file.",
                file_dt, expect_dt, file_dt / expect_dt,
            )

    # ------------------------------------------------------------------
    # 2. Pre-load and noise-filter all track files
    # ------------------------------------------------------------------
    log.info(
        "\nPre-loading steps %d-%d ...\n"
        "  Pattern : %s\n"
        "  Noise filter: discard tracers with |V| < %.0e m/s",
        p["first_step"], p["last_step"],
        p["tracks_filename_pattern"], p["noise_thresh_ms"],
    )

    cached: dict[int, dict] = {}
    frame_records: list[dict] = []
    snapshot_format_logged = False
    for local_step_idx, step_n in enumerate(step_indices, start=1):
        fname = p["tracks_filename_pattern"].format(N=step_n)
        fpath = tracks_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Track file missing: {fpath}")

        t_io  = time.perf_counter()
        raw   = read_tracks_zone(str(fpath))
        _, traj_time, traj_time_s = _resolve_trajectory_index(
            raw, local_step_idx, traj["times"], p
        )
        raw_positions = apply_coordinate_signs(raw["positions_mm"], tracks_signs)
        raw_velocities = apply_coordinate_signs(raw["velocities_ms"], tracks_signs)
        raw_vmag = np.linalg.norm(raw_velocities, axis=1)
        n_tot = len(raw_vmag)

        if (local_step_idx == 1 and not raw["zone_time_is_numeric"]
                and not snapshot_format_logged):
            log.info(
                "Snapshot format detected: track files map to trajectory rows "
                "by processed file order."
            )
            snapshot_format_logged = True

        cached[step_n] = {
            "zone_time_is_numeric": raw["zone_time_is_numeric"],
            "zone_time": raw.get("zone_time"),
            "zone_label": raw.get("zone_label"),
            "raw_positions_mm": raw_positions,
            "raw_velocities_ms": raw_velocities,
            "raw_vmag_ms": raw_vmag,
            "raw_track_ids": raw["track_ids"],
            "traj_time": traj_time,
            "traj_time_s": traj_time_s,
            "n_total": n_tot,
        }
        frame_records.append({
            "step_n": step_n,
            "time_s": traj_time_s,
            "positions_mm": raw_positions,
            "velocities_ms": raw_velocities,
            "track_ids": raw["track_ids"],
        })

        if raw["zone_time_is_numeric"]:
            expected_t = (
                p["zone_time_offset"]
                + p["zone_time_step_mult"] * (step_n + p["zone_time_step_offset"])
            )
            mismatch = "" if raw["zone_time"] == expected_t else (
                f"  *** MISMATCH: expected {expected_t} "
                f"from offset+mult*(N+step_offset) ***"
            )
        else:
            mismatch = ""
        zone_tag = str(raw["zone_time"]) if raw["zone_time_is_numeric"] else raw.get("zone_label", "?")
        log.info(
            "  step %2d  zone=%s  %7d tracers total  read in %.1fs%s",
            step_n, zone_tag, n_tot, time.perf_counter() - t_io, mismatch,
        )

    if p["enable_track_denoising"]:
        log.info(
            "Track denoising: polynomial order=%d  filter_length=%d frames  "
            "MAD threshold=median+%.1f*MAD  outlier_action=%s",
            p["track_denoise_poly_order"],
            p["track_denoise_filter_length"],
            p["track_denoise_mad_threshold"],
            p["track_denoise_outlier_action"],
        )

    log.info("Applying velocity/confidence filters ...")
    for record_idx, record in enumerate(frame_records):
        step_n = record["step_n"]
        c = cached[step_n]
        raw_positions = c["raw_positions_mm"]
        raw_velocities = c["raw_velocities_ms"]

        if p["enable_track_denoising"]:
            den = denoise_frame_velocities(
                frame_records,
                record_idx,
                poly_order=p["track_denoise_poly_order"],
                filter_length=p["track_denoise_filter_length"],
                raw_velocity_confidence=p["track_denoise_raw_confidence"],
                one_sided_confidence=p["track_denoise_one_sided_confidence"],
                central_confidence=p["track_denoise_central_confidence"],
            )
            velocities = den.velocities_ms
            confidence = den.confidence
            method_code = den.method_code
            sample_count = den.sample_count
        else:
            velocities = raw_velocities.copy()
            confidence = np.ones(len(velocities), dtype=float)
            method_code = np.full(len(velocities), METHOD_RAW, dtype=np.int8)
            sample_count = np.ones(len(velocities), dtype=np.int16)

        confidence, mad_outliers, mad_threshold = apply_mad_outlier_confidence(
            velocities,
            confidence,
            threshold_mad=p["track_denoise_mad_threshold"],
            multiplier=p["track_denoise_outlier_confidence_multiplier"],
        )

        finite = np.isfinite(raw_positions).all(axis=1) & np.isfinite(velocities).all(axis=1)
        vmag = np.linalg.norm(velocities, axis=1)
        noise_keep = vmag >= p["noise_thresh_ms"]
        outlier_keep = np.ones(len(vmag), dtype=bool)
        if str(p["track_denoise_outlier_action"]).lower() == "remove":
            outlier_keep = ~mad_outliers
        keep = finite & noise_keep & outlier_keep

        uncertainties = confidence_to_uncertainty(
            p["sigma_i_ms"],
            confidence,
            min_confidence=p["track_denoise_min_confidence"],
        )

        c.update({
            "positions_mm": raw_positions[keep],
            "velocities_ms": velocities[keep],
            "velocity_confidence": confidence[keep],
            "velocity_uncertainties_ms": uncertainties[keep],
            "velocity_mad_outlier": mad_outliers[keep],
            "velocity_method_code": method_code[keep],
            "track_sample_count": sample_count[keep],
            "track_ids": c["raw_track_ids"][keep],
            "n_kept": int(keep.sum()),
            "n_noise": int((finite & ~noise_keep).sum()),
            "n_mad_outlier": int(mad_outliers.sum()),
            "mad_threshold_ms": mad_threshold,
        })

        n_tot = c["n_total"]
        method_counts = {
            "poly": int(np.sum(method_code == METHOD_POLY)),
            "central": int(np.sum(method_code == METHOD_CENTRAL)),
            "oneside": int(np.sum(method_code == METHOD_ONESIDED)),
            "raw": int(np.sum(method_code == METHOD_RAW)),
        }
        log.info(
            "  step %2d  %7d kept (%.1f%%)  %7d noise  %5d MAD-outlier  "
            "methods poly/central/one/raw=%d/%d/%d/%d",
            step_n, c["n_kept"], 100 * c["n_kept"] / n_tot, c["n_noise"],
            c["n_mad_outlier"], method_counts["poly"], method_counts["central"],
            method_counts["oneside"], method_counts["raw"],
        )

    # ------------------------------------------------------------------
    # 3. Build fixed reconstruction grid
    # ------------------------------------------------------------------
    all_pos = np.vstack([c["positions_mm"] for c in cached.values()])

    # Track bounding box along each axis
    tr_min = all_pos.min(axis=0)   # (3,)
    tr_max = all_pos.max(axis=0)

    # Sphere trajectory extent along the rise axis
    sphere_axis_coords = []
    for local_step_idx, step_n in enumerate(step_indices, start=1):
        _, traj_time, _ = _resolve_trajectory_index(
            cached[step_n], local_step_idx, traj["times"], p
        )
        spos, _ = lookup_sphere_state(traj, traj_time, vel_scale)
        sphere_axis_coords.append(spos[axis_idx])
    sph_axis_min = min(sphere_axis_coords)
    sph_axis_max = max(sphere_axis_coords)

    half_width   = p["domain_truncation_factor"] * D
    axis_min_raw = sph_axis_min - half_width
    axis_max_raw = sph_axis_max + half_width

    if p["clip_y_to_tracks"]:
        axis_min = max(axis_min_raw, tr_min[axis_idx] - p["bbox_pad_mm"])
        axis_max = min(axis_max_raw, tr_max[axis_idx] + p["bbox_pad_mm"])
        if axis_min > axis_min_raw:
            log.info(
                "[INFO] %s_min clipped from %.1f to %.1f mm (track bbox floor)",
                axis_label, axis_min_raw, axis_min,
            )
        if axis_max < axis_max_raw:
            log.info(
                "[INFO] %s_max clipped from %.1f to %.1f mm (track bbox ceiling)",
                axis_label, axis_max_raw, axis_max,
            )
    else:
        axis_min = axis_min_raw
        axis_max = axis_max_raw

    # All axes: track bbox + padding; override rise axis with sphere-tracked bounds
    domain_min_arr = tr_min - p["bbox_pad_mm"]
    domain_max_arr = tr_max + p["bbox_pad_mm"]
    domain_min_arr[axis_idx] = axis_min
    domain_max_arr[axis_idx] = axis_max
    domain_min = tuple(domain_min_arr.tolist())
    domain_max = tuple(domain_max_arr.tolist())

    grid = RectGrid(domain_min, domain_max, p["delta_mm"])

    log.info(
        "\nTrack bbox (noise-filtered): "
        "x=[%.1f, %.1f]  y=[%.1f, %.1f]  z=[%.1f, %.1f] mm",
        tr_min[0], tr_max[0], tr_min[1], tr_max[1], tr_min[2], tr_max[2],
    )
    log.info(
        "Sphere %s over steps %d-%d: [%.2f, %.2f] mm  "
        "-> +/-%.1fD window: [%.1f, %.1f] mm",
        axis_label, p["first_step"], p["last_step"],
        sph_axis_min, sph_axis_max, p["domain_truncation_factor"],
        sph_axis_min - half_width, sph_axis_max + half_width,
    )
    log.info(
        "Grid domain: x=[%.1f, %.1f]  y=[%.1f, %.1f]  z=[%.1f, %.1f] mm",
        domain_min[0], domain_max[0],
        domain_min[1], domain_max[1],
        domain_min[2], domain_max[2],
    )
    log.info(
        "Grid nodes:  %d x %d x %d = %s  (delta = %.1f mm)",
        grid.Nx, grid.Ny, grid.Nz, f"{grid.size:,}", p["delta_mm"],
    )
    log.info(
        "Parameters:  kappa=%s  sigma_s=%.2f mm  sigma_i=%.4f m/s  "
        "noise_thresh=%.0e m/s",
        p["kappa"], p["sigma_s_mm"], p["sigma_i_ms"], p["noise_thresh_ms"],
    )
    if p["enable_median_filter"]:
        log.info(
            "Median filter: radius=%.2f mm  threshold=%.4f m/s  min_neighbors=%d",
            p["median_filter_radius_mm"],
            p["median_filter_threshold_ms"],
            p["median_filter_min_neighbors"],
        )
    if p["enable_multipass"]:
        log.info(
            "Multipass coarse residual filter: delta=%.2f mm  kappa=%s  "
            "sigma_s=%.2f mm  residual_threshold=%.4f m/s  min_tracers=%d",
            p["multipass_coarse_delta_mm"],
            p["multipass_coarse_kappa"],
            p["multipass_coarse_sigma_s_mm"],
            p["multipass_residual_threshold_ms"],
            p["multipass_min_tracers"],
        )
    if p["enable_field_smoothing"]:
        log.info(
            "Field smoothing: type=%s  lambda_laplacian=%.3g  "
            "lambda_gradient=%.3g  no_cross_mask=%s  spacing_scaled=%s",
            p["field_smoothing_type"],
            p["lambda_laplacian"],
            p["lambda_gradient"],
            p["smoothing_no_cross_mask"],
            p["smoothing_spacing_scaled"],
        )
    support_radii = parse_float_list(p.get("support_radius_candidates_mm", p["support_radius_mm"]))
    if p["support_radius_mm"] not in support_radii:
        support_radii.insert(0, float(p["support_radius_mm"]))
    log.info(
        "Support diagnostics radii [mm]: %s",
        ", ".join(f"{r:g}" for r in support_radii),
    )

    # ------------------------------------------------------------------
    # 4. Solver setup
    # ------------------------------------------------------------------
    config = Config(
        domain_min=domain_min,
        domain_max=domain_max,
        delta=p["delta_mm"],
        kappa=p["kappa"],
        solver_rtol=p["solver_rtol"],
        solver_maxiter=p["solver_maxiter"],
        output_dir=str(output_dat_dir),
        enable_field_smoothing=p["enable_field_smoothing"],
        field_smoothing_type=p["field_smoothing_type"],
        lambda_laplacian=p["lambda_laplacian"],
        lambda_gradient=p["lambda_gradient"],
        smoothing_exclude_mask=p["smoothing_exclude_mask"],
        smoothing_exclude_shell=p["smoothing_exclude_shell"],
        smoothing_no_cross_mask=p["smoothing_no_cross_mask"],
        smoothing_spacing_scaled=p["smoothing_spacing_scaled"],
        smoothing_componentwise=p["smoothing_componentwise"],
        laplacian_taper=p["laplacian_taper"],
        lap_taper_mm=float(p["lap_taper_mm"]),
        enable_irls=p["enable_irls"],
        irls_loss=str(p["irls_loss"]),
        irls_threshold_sigma=float(p["irls_threshold_sigma"]),
        irls_max_outer=int(p["irls_max_outer"]),
        irls_tol=float(p["irls_tol"]),
        irls_min_weight=float(p["irls_min_weight"]),
        enable_lema=bool(p["enable_lema"]),
    )
    solver = CCMPlus(config, grid)

    coarse_grid = None
    coarse_solver = None
    if p["enable_multipass"]:
        coarse_grid = RectGrid(domain_min, domain_max, p["multipass_coarse_delta_mm"])
        coarse_config = Config(
            domain_min=domain_min,
            domain_max=domain_max,
            delta=p["multipass_coarse_delta_mm"],
            kappa=p["multipass_coarse_kappa"],
            solver_rtol=p["solver_rtol"],
            solver_maxiter=p["solver_maxiter"],
            output_dir=str(output_dat_dir),
            enable_field_smoothing=p["enable_field_smoothing"],
            field_smoothing_type=p["field_smoothing_type"],
            lambda_laplacian=p["lambda_laplacian"],
            lambda_gradient=p["lambda_gradient"],
            smoothing_exclude_mask=p["smoothing_exclude_mask"],
            smoothing_exclude_shell=p["smoothing_exclude_shell"],
            smoothing_no_cross_mask=p["smoothing_no_cross_mask"],
            smoothing_spacing_scaled=p["smoothing_spacing_scaled"],
            smoothing_componentwise=p["smoothing_componentwise"],
            laplacian_taper=p["laplacian_taper"],
            lap_taper_mm=float(p["lap_taper_mm"]),
        )
        coarse_solver = CCMPlus(coarse_config, coarse_grid)
        log.info(
            "Coarse grid nodes: %d x %d x %d = %s  (delta = %.1f mm)",
            coarse_grid.Nx, coarse_grid.Ny, coarse_grid.Nz,
            f"{coarse_grid.size:,}", coarse_grid.delta,
        )

    # Leading-edge bound (the side where domain is truncated / undisturbed fluid lies)
    leading_bound = axis_max if sign == +1 else axis_min

    # ------------------------------------------------------------------
    # 5. Per-step loop
    # ------------------------------------------------------------------
    hdr = (
        f"{'step':>4}  {'t':>7}  {'#fld':>7}  {'#shl':>5}  {'#sol':>5}  "
        f"{'#tr_in':>7}  {'#tr_ex':>6}  {'#med':>5}  {'#mp':>5}  "
        f"{'cond':>8}  {'data_r':>8}  {'reg_r':>8}  "
        f"{'iters':>5}  {'mnres':>8}  {'wall':>6}"
    )
    log.info("\n%s\n%s", hdr, "-" * len(hdr))

    summary = []
    support_summary = []

    for local_step_idx, step_n in enumerate(step_indices, start=1):
        t0     = time.perf_counter()
        c      = cached[step_n]
        _, traj_time, traj_time_s = _resolve_trajectory_index(
            c, local_step_idx, traj["times"], p
        )
        t_ms   = int(round(traj_time))
        pos_mm = c["positions_mm"].copy()
        vel_ms = c["velocities_ms"].copy()
        unc_ms = c["velocity_uncertainties_ms"].copy()
        conf = c["velocity_confidence"].copy()
        mad_flags = c["velocity_mad_outlier"].copy()

        # ---- Per-step in-domain filter ----
        # Exclude tracers beyond the leading (undisturbed-side) domain bound.
        # sign * pos <= sign * leading_bound + tol
        #   +y: y <= y_max + tol  (exclude high-y particles)
        #   -y: y >= y_min - tol  (exclude low-y particles)
        in_domain  = (
            sign * pos_mm[:, axis_idx]
            <= sign * leading_bound + p["in_domain_tolerance_mm"]
        )
        n_excluded = int((~in_domain).sum())
        if p["enable_indomain_filter"] and n_excluded > 0:
            frac = n_excluded / len(pos_mm)
            if frac > p["exclusion_warn_fraction"]:
                log.warning(
                    "Step %d: %.0f%% of tracers (n=%d) are outside "
                    "%s_bound=%.1f mm + %.1f mm tolerance -- "
                    "truncation may be too aggressive.",
                    step_n, 100 * frac, n_excluded,
                    axis_label, leading_bound, p["in_domain_tolerance_mm"],
            )
            pos_mm = pos_mm[in_domain]
            vel_ms = vel_ms[in_domain]
            unc_ms = unc_ms[in_domain]
            conf = conf[in_domain]
            mad_flags = mad_flags[in_domain]
        elif n_excluded > 0 and n_excluded / len(pos_mm) > p["exclusion_warn_fraction"]:
            log.warning(
                "Step %d: %.0f%% of tracers (n=%d) lie beyond "
                "%s_bound=%.1f mm "
                "(in-domain filter disabled; set enable_indomain_filter = true to exclude).",
                step_n, 100 * n_excluded / len(pos_mm), n_excluded,
                axis_label, leading_bound,
            )

        # ---- Optional local median velocity outlier filter ----
        n_med = 0
        if p["enable_median_filter"] and len(pos_mm) > 0:
            med_keep = median_velocity_filter(
                pos_mm,
                vel_ms,
                radius_mm=p["median_filter_radius_mm"],
                threshold_ms=p["median_filter_threshold_ms"],
                min_neighbors=p["median_filter_min_neighbors"],
            )
            n_med = int((~med_keep).sum())
            if n_med > 0:
                pos_mm = pos_mm[med_keep]
                vel_ms = vel_ms[med_keep]
                unc_ms = unc_ms[med_keep]
                conf = conf[med_keep]
                mad_flags = mad_flags[med_keep]

        # ---- Sphere state ----
        sph_pos, sph_vel = lookup_sphere_state(traj, traj_time, vel_scale)
        body = BodyState(
            X_s=sph_pos.copy(),
            U_s=sph_vel.copy(),
            omega_s=np.zeros(3),
            radius=R_mm,
            sigma_s=p["sigma_s_mm"],
        )

        # ---- Exclude solid-interior tracers (phi < 0) ----
        phi_p  = signed_distance_sphere_points(pos_mm, body)
        in_fld = phi_p >= 0.0
        n_in_before_mp = int(in_fld.sum())
        n_ex   = int((~in_fld).sum())
        pos_in = pos_mm[in_fld]
        vel_in = vel_ms[in_fld]
        unc_in = unc_ms[in_fld]
        conf_in = conf[in_fld]
        mad_in = mad_flags[in_fld]

        # ---- Optional coarse-to-fine multipass residual filter ----
        n_mp = 0
        if p["enable_multipass"] and len(pos_in) >= p["multipass_min_tracers"]:
            coarse_body = make_body_with_sigma(body, p["multipass_coarse_sigma_s_mm"])
            coarse_frame = FrameData(
                positions=pos_in,
                velocities=vel_in,
                uncertainties=unc_in,
                body=coarse_body,
                t=traj_time_s,
            )
            if not p["warm_start"] and coarse_solver is not None:
                coarse_solver._x_prev = None

            assert coarse_solver is not None
            assert coarse_grid is not None
            coarse_result = coarse_solver.reconstruct(coarse_frame)
            mp_keep, mp_residuals = residual_outlier_filter(
                pos_in,
                vel_in,
                coarse_grid,
                coarse_result.velocity,
                p["multipass_residual_threshold_ms"],
            )
            n_mp_candidate = int((~mp_keep).sum())
            if n_mp_candidate == len(pos_in):
                log.warning(
                    "Step %d: multipass residual filter would remove all %d "
                    "fluid tracers; keeping them. Increase "
                    "multipass_residual_threshold_ms.",
                    step_n, len(pos_in),
                )
            elif n_mp_candidate > 0:
                n_mp = n_mp_candidate
                log.info(
                    "      multipass residuals: max=%.4f m/s  p95=%.4f m/s",
                    float(np.max(mp_residuals)),
                    float(np.percentile(mp_residuals, 95)),
                )
                pos_in = pos_in[mp_keep]
                vel_in = vel_in[mp_keep]
                unc_in = unc_in[mp_keep]
                conf_in = conf_in[mp_keep]
                mad_in = mad_in[mp_keep]

        n_in = len(pos_in)
        unc = unc_in

        frame = FrameData(
            positions=pos_in,
            velocities=vel_in,
            uncertainties=unc,
            body=body,
            t=traj_time_s,
        )

        x_prior_snap = (
            solver._x_prev.copy() if solver._x_prev is not None
            else np.zeros(3 * grid.size)
        )

        if not p["warm_start"]:
            solver._x_prev = None

        result = solver.reconstruct(frame)
        u_rec  = result.velocity.ravel()
        C      = result.classification

        n_fld = int((C ==  1).sum())
        n_shl = int((C ==  0).sum())
        n_sol = int((C == -1).sum())
        valid_fluid = C == 1
        vmag_grid = np.linalg.norm(result.velocity, axis=1)
        vmag_particle = np.linalg.norm(vel_in, axis=1)
        main_support_count = support_count_grid(grid, pos_in, p["support_radius_mm"])
        main_support = support_statistics_from_counts(
            main_support_count, valid_fluid, p["support_radius_mm"]
        )

        step_support_stats = [main_support]
        if p["write_support_stats_csv"]:
            for radius_mm in support_radii:
                if abs(float(radius_mm) - float(p["support_radius_mm"])) < 1e-12:
                    stats = main_support.copy()
                else:
                    stats = support_statistics(grid, pos_in, valid_fluid, radius_mm)
                stats.update({"step_n": step_n, "t_ms": t_ms})
                support_summary.append(stats)

        # ---- Diagnostics ----
        A_d   = build_interpolation_matrix(pos_in, grid, allowed_nodes=(C == 1))
        phi_d = signed_distance_sphere_points(pos_in, body)
        W_d   = build_weight_matrix(phi_d, unc, p["sigma_s_mm"])
        Ng3   = 3 * grid.size

        cond   = estimate_condition(A_d, W_d, p["kappa"], Ng3)
        b      = vel_in.ravel()
        Au     = A_d @ u_rec
        data_r = float(np.linalg.norm(Au - b) / (np.linalg.norm(b) + 1e-300))
        reg_r  = float(np.linalg.norm(u_rec - x_prior_snap))

        particle_vmag_mean = _safe_mean(vmag_particle)
        particle_vmag_p95 = _safe_percentile(vmag_particle, 95)
        particle_vmag_p99 = _safe_percentile(vmag_particle, 99)
        grid_vmag_valid = vmag_grid[valid_fluid]
        grid_vmag_mean = _safe_mean(grid_vmag_valid)
        grid_vmag_p95 = _safe_percentile(grid_vmag_valid, 95)
        grid_vmag_p99 = _safe_percentile(grid_vmag_valid, 99)
        grid_vmag_max = _safe_max(grid_vmag_valid)
        ratio_p95 = _safe_ratio(grid_vmag_p95, particle_vmag_p95)
        ratio_p99 = _safe_ratio(grid_vmag_p99, particle_vmag_p99)
        smoothing_stats = result.smoothing_stats or {}
        n_smoothing_rows = int(smoothing_stats.get("smoothing_rows", 0))
        n_smoothing_rows_skipped_mask = int(
            smoothing_stats.get("smoothing_rows_skipped_mask", 0)
        )
        roughness_metric = compute_roughness_metric(result.velocity, grid, valid_fluid)
        patchiness_metric = compute_patchiness_metric(result.velocity, grid, valid_fluid)
        divergence_rms = compute_divergence_rms(result.velocity, grid, valid_fluid)
        laplacian_rms = compute_laplacian_rms(result.velocity, grid, C, p)
        n_downweighted = (
            int(np.sum(mad_in))
            if str(p["track_denoise_outlier_action"]).lower() == "downweight"
            else 0
        )
        mean_conf = _safe_mean(conf_in)
        median_conf = _safe_median(conf_in)
        p10_conf = _safe_percentile(conf_in, 10)

        frame_warnings: list[str] = []

        def add_frame_warning(condition: bool, message: str) -> None:
            if condition:
                frame_warnings.append(message)
                log.warning("Step %d: %s", step_n, message)

        add_frame_warning(
            np.isfinite(ratio_p95) and ratio_p95 < 0.3,
            f"ratio_grid_p95_to_particle_p95 is low ({ratio_p95:.3g})",
        )
        add_frame_warning(
            np.isfinite(ratio_p99) and ratio_p99 < 0.3,
            f"ratio_grid_p99_to_particle_p99 is low ({ratio_p99:.3g})",
        )
        add_frame_warning(
            np.isfinite(grid_vmag_max) and grid_vmag_max > 1.0,
            f"grid_vmag_max exceeds 1.0 m/s ({grid_vmag_max:.3g})",
        )
        add_frame_warning(
            np.isfinite(main_support["support_pct_eq0"])
            and main_support["support_pct_eq0"] > 20.0,
            f"support_percent_zero exceeds 20% ({main_support['support_pct_eq0']:.1f}%)",
        )
        add_frame_warning(
            np.isfinite(main_support["support_pct_lt3"])
            and main_support["support_pct_lt3"] > 50.0,
            f"support_percent_lt3 exceeds 50% ({main_support['support_pct_lt3']:.1f}%)",
        )
        add_frame_warning(
            not bool(result.converged),
            "solver_converged is false",
        )
        add_frame_warning(
            p["enable_field_smoothing"] and n_smoothing_rows == 0,
            "n_smoothing_rows is zero while field smoothing is enabled",
        )
        add_frame_warning(
            n_smoothing_rows > 0
            and n_smoothing_rows_skipped_mask > 0.5 * n_smoothing_rows,
            "n_smoothing_rows_skipped_mask is large compared with n_smoothing_rows",
        )

        wall = time.perf_counter() - t0

        log.info(
            "%4d  %7d  %7d  %5d  %5d  %7d  %6d  %5d  %5d  "
            "%8.2e  %8.3e  %8.3e  %5d  %8.2e  %6.1fs",
            step_n, t_ms, n_fld, n_shl, n_sol,
            n_in, n_ex, n_med, n_mp,
            cond, data_r, reg_r,
            result.iterations, result.residual, wall,
        )
        if step_support_stats:
            main_support = min(
                step_support_stats,
                key=lambda s: abs(s["support_radius_mm"] - p["support_radius_mm"]),
            )
            log.info(
                "      support r=%.2fmm: mean=%.1f median=%.1f "
                "pct(<3)=%.1f pct(0)=%.1f",
                main_support["support_radius_mm"],
                main_support["support_mean"],
                main_support["support_median"],
                main_support["support_pct_lt3"],
                main_support["support_pct_eq0"],
            )
        if p["enable_field_smoothing"] and result.smoothing_stats:
            log.info(
                "      smoothing rows=%d  skipped_mask_rows=%d  crosses_mask=%s",
                result.smoothing_stats.get("smoothing_rows", 0),
                result.smoothing_stats.get("smoothing_rows_skipped_mask", 0),
                result.smoothing_stats.get("smoothing_crosses_mask", False),
            )

        if p["write_dat"]:
            out_dat = (
                output_dat_dir
                / f"{case_name}_ccm_timeStep_{step_n}_t{t_ms:04d}ms.dat"
            )
            write_output(
                out_dat, grid, result.velocity, C, t_ms, case_name, traj_time_s
            )
            log.info("      %s", out_dat.name)

        if p["write_png"]:
            out_png = make_diagnostic_png(
                step_n, t_ms, grid, result.velocity, C,
                pos_mm,
                sph_pos, R_mm, case_name, output_fig_dir,
            )
            log.info("      %s", out_png.name)

        if step_n in diagnostic_png_steps:
            selected_paths = save_selected_frame_diagnostics(
                output_fig_dir,
                case_name,
                step_n,
                t_ms,
                grid,
                result.velocity,
                C,
                main_support_count,
                pos_in,
                vel_in,
                sph_pos,
                R_mm,
            )
            log.info(
                "      selected diagnostic PNGs: %s",
                ", ".join(path.name for path in selected_paths),
            )

        # Guard: warn if sphere approaches within 2δ of rise-axis domain bounds
        for label, edge in (
            (f"{axis_label}_min", domain_min[axis_idx]),
            (f"{axis_label}_max", domain_max[axis_idx]),
        ):
            if abs(sph_pos[axis_idx] - edge) < 2 * p["delta_mm"]:
                log.warning(
                    "Sphere %s=%.1f mm is within 2δ of grid %s=%.1f mm "
                    "-- wake may be clipped.",
                    axis_label, sph_pos[axis_idx], label, edge,
                )

        summary.append({
            "frame": step_n,
            "time": traj_time_s,
            "step_n": step_n,
            "t_ms":   t_ms,
            f"sph_{axis_label}": sph_pos[axis_idx],
            f"U_s{axis_label}":  sph_vel[axis_idx],
            "n_fld": n_fld, "n_shl": n_shl, "n_sol": n_sol,
            "n_in":  n_in,  "n_ex":  n_ex,
            "n_med": n_med, "n_mp": n_mp,
            "n_in_before_mp": n_in_before_mp,
            "n_particles_loaded": c["n_total"],
            "n_particles_after_filters": n_in,
            "n_particles_removed_median": n_med,
            "n_particles_removed_multipass": n_mp,
            "n_particles_downweighted": n_downweighted,
            "mean_particle_confidence": mean_conf,
            "median_particle_confidence": median_conf,
            "p10_particle_confidence": p10_conf,
            "particle_vmag_mean": particle_vmag_mean,
            "particle_vmag_p95": particle_vmag_p95,
            "particle_vmag_p99": particle_vmag_p99,
            "grid_vmag_mean": grid_vmag_mean,
            "grid_vmag_p95": grid_vmag_p95,
            "grid_vmag_p99": grid_vmag_p99,
            "grid_vmag_max": grid_vmag_max,
            "ratio_grid_p95_to_particle_p95": ratio_p95,
            "ratio_grid_p99_to_particle_p99": ratio_p99,
            "support_median": main_support["support_median"],
            "support_percent_zero": main_support["support_pct_eq0"],
            "support_percent_lt3": main_support["support_pct_lt3"],
            "support_percent_lt5": main_support["support_pct_lt5"],
            "n_smoothing_rows": n_smoothing_rows,
            "n_smoothing_rows_skipped_mask": n_smoothing_rows_skipped_mask,
            "roughness_metric": roughness_metric,
            "patchiness_metric": patchiness_metric,
            "divergence_rms": divergence_rms,
            "laplacian_rms": laplacian_rms,
            "solver_iterations": result.iterations,
            "solver_final_residual": result.residual,
            "solver_converged": bool(result.converged),
            "warnings": "; ".join(frame_warnings),
            "cond": cond, "data_r": data_r, "reg_r": reg_r,
            "iters": result.iterations, "mnres": result.residual, "wall": wall,
            "support_mean": main_support["support_mean"],
            "smoothing_rows": (
                result.smoothing_stats.get("smoothing_rows", 0)
                if result.smoothing_stats else 0
            ),
            "smoothing_rows_skipped_mask": (
                result.smoothing_stats.get("smoothing_rows_skipped_mask", 0)
                if result.smoothing_stats else 0
            ),
        })

    # ------------------------------------------------------------------
    # 6. Summary table
    # ------------------------------------------------------------------
    log.info("\n%s\nSummary", "=" * 75)
    log.info(
        "%4s  %7s  %8s  %10s  %7s  %8s  %8s  %5s  %6s",
        "step", "t", f"sph_{axis_label}", f"U_s{axis_label} m/s",
        "#fluid", "cond", "data_r", "iters", "wall_s",
    )
    log.info("-" * 68)
    for r in summary:
        log.info(
            "%4d  %7d  %8.2f  %10.5f  %7d  %8.2e  %8.3e  %5d  %6.1fs",
            r["step_n"], r["t_ms"],
            r[f"sph_{axis_label}"], r[f"U_s{axis_label}"],
            r["n_fld"], r["cond"], r["data_r"], r["iters"], r["wall"],
        )
    total = sum(r["wall"] for r in summary)
    log.info("\nTotal wall time: %.1fs  (%.1fs/step)", total, total / len(summary))
    log.info("Output .dat : %s", output_dat_dir.resolve())
    if p["write_png"] or diagnostic_png_steps:
        log.info("Output .png : %s", output_fig_dir.resolve())

    if p["write_summary_csv"]:
        csv_path = output_dat_dir / f"{case_name}_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        log.info("Summary CSV : %s", csv_path)

    full_summary_path = output_dat_dir / "full_run_summary.txt"
    write_full_run_summary(full_summary_path, summary)
    log.info("Full run summary : %s", full_summary_path)

    if p["write_support_stats_csv"] and support_summary:
        csv_path = output_dat_dir / f"{case_name}_support_stats.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(support_summary[0].keys()))
            writer.writeheader()
            writer.writerows(support_summary)
        log.info("Support CSV : %s", csv_path)
