"""LE-DM - single clean entry point.

The body boundary conditions are controlled by one flag in the parameter file:

    boundary_constraints = on    ->  no-slip body constraints + prior seeding
                                     and body-aware (fluid-side) divergence
                                     stencils. This is LE-DM.
    boundary_constraints = off   ->  the shell/solid Dirichlet rows are not
                                     assembled and divergence is enforced
                                     uniformly over all interior nodes.

`off` is an ABLATION of the boundary conditions, not an all-fluid
reconstruction: interpolation and smoothing are still masked by the body
classification in both cases, so tracks never write into the body interior.
(`enable_lema` is the deprecated spelling of this flag and still works, with a
warning.)

Everything else - the wide cubic-B-spline kernel, the coverage-adaptive
smoothness penalty, the time pipeline, the saddle-point MINRES solver - is
shared, so outside a 2-cell shell around the body the two modes agree by
construction.

Usage:
    # two-file / four-file input contract (recommended, see LEDM_input_spec.md)
    python run_ledm.py --four-file configs/example_synthetic_sphere.txt

    # legacy Tecplot trajectory + tracks driver
    python run_ledm.py <parameters.txt> [--first-step N] [--last-step M]

Worked configs:
    configs/example_synthetic_sphere.txt   runnable, self-contained example
    paper_templates/*.txt                  illustrative starting points for the
                                           published cases (NOT exact recipes)

Outputs are written under the parameter file's output_dir / output_dat_dir. All
input/output locations are paths in the config, relative to the working
directory by default; edit those keys to point anywhere you like.
"""
from __future__ import annotations
import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ccmplus.params import read_parameters, apply_defaults, resolve_domain_truncation
from ccmplus.grid import RectGrid
from ccmplus.config import Config, BodyState, FrameData
from ccmplus.reconstruct import CCMPlus
from ccmplus.io_tecplot import read_trajectory, read_tracks_zone, write_tecplot_volume
from ccmplus.sdf import signed_distance_sphere_points
from ccmplus.timing import (
    velocity_scale, trajectory_velocities_ms, validate_trajectory_timing,
)

log = logging.getLogger("ledm_v2")

# ---- v2 frozen global constants (see VERSION / GATE report) ----
# Provisional; finalise on the Wo benchmark before the CFD rerun. The coverage
# penalty is the gradient (first-difference) form, so lambda is O(1), not O(0.01).
LAMBDA_COVERAGE_FROZEN = 1.0    # one global weight, identical for base & LE-DM
COVERAGE_REF_COUNT = 1.0
INTERP_KERNEL = "wide"

_DEFAULTS = {
    "kappa": 1.0,
    "solver_rtol": 1e-6,
    "solver_maxiter": 2000,
    "sigma_s_mm": 0.5,
    "sigma_i_ms": 0.01,
    "noise_thresh_ms": 1e-6,
    "bbox_pad_mm": 2.0,
    "boundary_constraints": True,
    "rise_axis": "+y",
    "trajectory_dt_units": 1.0,
    "trajectory_dt_seconds": 1.0,
    "expected_rise_speed_ms": 0.0,
    "tracks_filename_pattern": "B{N:05d}.dat",
    "lambda_coverage": LAMBDA_COVERAGE_FROZEN,
    "coverage_ref_count": COVERAGE_REF_COUNT,
    "interp_kernel": INTERP_KERNEL,
    "domain_truncation": False,
}


def _axis_index_and_sign(rise_axis: str):
    a = rise_axis.strip()
    sign = -1.0 if a.startswith("-") else 1.0
    idx = {"x": 0, "y": 1, "z": 2}[a.lstrip("+-").lower()]
    return idx, sign


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("parameters")
    ap.add_argument("--first-step", type=int, default=None)
    ap.add_argument("--last-step", type=int, default=None)
    ap.add_argument("--four-file", action="store_true",
                    help="Consume the open-source four-file input contract "
                         "(particles + geometry + kinematics + parameters) via "
                         "the geometry layer; see LEDM_input_spec.md.")
    args = ap.parse_args(argv)

    if args.four_file:
        from ccmplus.io_ledm import run_four_file
        run, results = run_four_file(
            args.parameters, first=args.first_step, last=args.last_step,
            log=log.info)
        log.info("four-file run: %d timesteps, geometry=%s -> %s",
                 len(results), run.meta["type"], run.meta["output_dir"])
        return

    p = read_parameters(args.parameters)

    # Accept the deprecated enable_lema spelling for existing parameter files.
    # This must run BEFORE apply_defaults, which would otherwise insert
    # boundary_constraints itself and mask the fact that the user set neither.
    if "enable_lema" in p:
        warnings.warn(
            "The 'enable_lema' parameter is deprecated; rename it to "
            "'boundary_constraints = on|off'. Note that 'off' removes the "
            "no-slip constraint rows but still masks interpolation and "
            "smoothing by the body classification -- it is not an all-fluid "
            "baseline.",
            DeprecationWarning, stacklevel=2)
        if "boundary_constraints" not in p:
            p["boundary_constraints"] = p["enable_lema"]

    apply_defaults(p, _DEFAULTS)

    # Domain truncation toggle (validated in the parser): when enabled, the
    # explicit x/y rectangle below replaces any other domain bounds.
    truncate, trunc_lim = resolve_domain_truncation(p)

    first = args.first_step if args.first_step is not None else int(p["first_step"])
    last = args.last_step if args.last_step is not None else int(p["last_step"])

    traj_file = Path(p["traj_file"])
    tracks_dir = Path(p["tracks_dir"])
    out_dir = Path(p["output_dat_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- trajectory + EXPLICIT-SI time pipeline (Problem 1) -----
    traj = read_trajectory(str(traj_file))
    sign_x = float(p.get("trajectory_x_sign", 1))
    sign_y = float(p.get("trajectory_y_sign", 1))
    sign_z = float(p.get("trajectory_z_sign", 1))
    traj["positions"] = traj["positions"] * np.array([sign_x, sign_y, sign_z])

    vscale = velocity_scale(p["trajectory_dt_seconds"], p["trajectory_dt_units"])
    timing_diag = validate_trajectory_timing(
        traj, p["trajectory_dt_units"], p["trajectory_dt_seconds"],
        p["rise_axis"], float(p["expected_rise_speed_ms"]) or None,
    )
    log.info("vel_scale=%.4f  seconds_per_unit=%.6g  terminal_rise=%.4f m/s",
             timing_diag["vel_scale"], timing_diag["seconds_per_unit"],
             timing_diag["terminal_rise_speed_ms"])

    traj_vel = trajectory_velocities_ms(traj["times"], traj["positions"], vscale)

    D = float(traj["diameter_mm"]); R = D / 2.0
    delta = float(p["delta_mm"])

    boundary_constraints = bool(p["boundary_constraints"])
    log.info("boundary_constraints=%s  |  kernel=%s  lambda_c=%.4g  kappa=%g",
             "on" if boundary_constraints else "off",
             p["interp_kernel"], p["lambda_coverage"], p["kappa"])

    # Domain grid bounds:
    #   domain_truncation = yes -> explicit x/y rectangle [x_min,x_max]x[y_min,y_max]
    #                              (tracks outside are dropped; z spans the kept tracks).
    #   domain_truncation = no  -> no truncation: full track bounding box (+ pad).
    pad = float(p["bbox_pad_mm"])
    if truncate:
        log.info("domain_truncation: ON  x=[%.3f, %.3f]  y=[%.3f, %.3f] mm",
                 trunc_lim["x_min"], trunc_lim["x_max"],
                 trunc_lim["y_min"], trunc_lim["y_max"])
    else:
        log.info("domain_truncation: OFF (grid = full track bounding box + %.1f mm pad)", pad)

    solver = None
    cfg = None
    for step in range(first, last + 1):
        tf = tracks_dir / p["tracks_filename_pattern"].format(N=step)
        if not tf.exists():
            log.warning("step %d: tracks file missing (%s); skipping", step, tf.name)
            continue
        tr = read_tracks_zone(str(tf))
        pos = tr["positions_mm"]; vel = tr["velocities_ms"]
        vmag = np.linalg.norm(vel, axis=1)
        keep = (np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1)
                & (vmag >= float(p["noise_thresh_ms"])))
        pos, vel = pos[keep], vel[keep]

        sph_pos = traj["positions"][step - 1]
        sph_vel = traj_vel[step - 1]

        # ----- domain truncation (gated by the domain_truncation flag) -----
        if truncate:
            in_rect = (
                (pos[:, 0] >= trunc_lim["x_min"]) & (pos[:, 0] <= trunc_lim["x_max"]) &
                (pos[:, 1] >= trunc_lim["y_min"]) & (pos[:, 1] <= trunc_lim["y_max"])
            )
            pos, vel = pos[in_rect], vel[in_rect]
            if len(pos) == 0:
                log.warning("step %d: no tracks inside the truncation rectangle; skipping", step)
                continue
            dmin = (trunc_lim["x_min"], trunc_lim["y_min"], float(pos[:, 2].min()) - pad)
            dmax = (trunc_lim["x_max"], trunc_lim["y_max"], float(pos[:, 2].max()) + pad)
        else:
            # no truncation: full (noise-filtered) track bounding box + pad
            if len(pos) == 0:
                log.warning("step %d: no tracks after filtering; skipping", step)
                continue
            dmin = tuple((pos.min(axis=0) - pad).tolist())
            dmax = tuple((pos.max(axis=0) + pad).tolist())
        grid = RectGrid(dmin, dmax, delta)

        cfg = Config(
            domain_min=dmin, domain_max=dmax, delta=delta,
            kappa=float(p["kappa"]), solver_rtol=float(p["solver_rtol"]),
            solver_maxiter=int(p["solver_maxiter"]),
            boundary_constraints=boundary_constraints,
            interp_kernel=str(p["interp_kernel"]),
            lambda_coverage=float(p["lambda_coverage"]),
            coverage_ref_count=float(p["coverage_ref_count"]),
            expected_rise_speed_ms=float(p["expected_rise_speed_ms"]),
        )
        body = BodyState(X_s=np.asarray(sph_pos, float),
                         U_s=np.asarray(sph_vel, float),
                         omega_s=np.zeros(3), radius=R,
                         sigma_s=float(p["sigma_s_mm"]))
        # boundary constraints off: no-body SDF, so no shell/solid rows
        if not boundary_constraints:
            body.sdf_fn = lambda pts, b: np.full(len(pts), 1e9)

        # fresh solver per step here (warm-start handled by caller if desired)
        if solver is None:
            solver = CCMPlus(cfg, grid)
        else:
            solver.config = cfg
            solver.grid = grid

        unc = np.full(len(pos), float(p["sigma_i_ms"]))
        fr = FrameData(positions=pos, velocities=vel, uncertainties=unc,
                       body=body, t=float(step))
        res = solver.reconstruct(fr)

        # write volume
        X = grid.nodes
        out = out_dir / f"{p.get('case_name','case')}_ledmv2_step{step}.dat"
        write_tecplot_volume(
            str(out), X[:, 0], X[:, 1], X[:, 2],
            res.velocity[:, 0], res.velocity[:, 1], res.velocity[:, 2],
            signed_distance_sphere_points(X, body), res.classification,
            title=f"LE-DM v2 step {step}", zone_time=step,
            Nx=grid.Nx, Ny=grid.Ny, Nz=grid.Nz,
        )
        log.info("step %d: |U_s|=%.4f m/s  iters=%d  converged=%s  -> %s",
                 step, float(np.linalg.norm(sph_vel)), res.iterations,
                 res.converged, out.name)

    log.info("done.")


if __name__ == "__main__":
    main()
