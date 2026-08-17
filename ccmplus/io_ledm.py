"""LE-DM four-file input layer: readers, load-time validation, run assembly.

Implements the contract in ``LEDM_input_spec.md``: per-timestep particle files,
one geometry file, one kinematics file, one parameter file. Turns them into the
``(grid, config, frames)`` the UNCHANGED ccmplus solver consumes (see INTERFACE.md).
Stdlib + numpy only; no new dependencies.
"""
from __future__ import annotations

import glob
import re
import warnings
from pathlib import Path

import numpy as np

from ccmplus.grid import RectGrid
from ccmplus.config import Config, FrameData
from ccmplus.params import read_parameters
from ccmplus.geometry import (
    GEOMETRY_REGISTRY, build_rotation, euler_rates_to_omega, make_body,
)

_ALLOWED_SEQ_AXES = set("XYZ")
_ALLOWED_HANDEDNESS = {"right", "left"}
_ALLOWED_OMEGA_FRAME = {"world", "body"}


# --------------------------------------------------------------------------- #
# Low-level text helpers
# --------------------------------------------------------------------------- #
def _split(line: str) -> list[str]:
    return line.replace(",", " ").split()


def _data_lines(path: Path) -> list[str]:
    """Non-comment, non-blank lines (``#`` starts a comment)."""
    out = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def _header_kv(path: Path) -> dict[str, str]:
    """Parse ``# key: value`` header comment lines into a dict."""
    hdr: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s.startswith("#") and ":" in s:
            key, _, val = s.lstrip("#").strip().partition(":")
            hdr[key.strip().lower()] = val.strip()
    return hdr


def _numeric(lines: list[str], path) -> np.ndarray:
    try:
        return np.array([[float(x) for x in _split(l)] for l in lines], dtype=float)
    except ValueError as exc:
        raise ValueError(f"{path}: non-numeric data row ({exc}).") from None


# --------------------------------------------------------------------------- #
# 1. Particle field (one file per timestep)
# --------------------------------------------------------------------------- #
def read_particle_file(path) -> dict:
    """Read one particle snapshot. Columns are mapped by the required one-line name
    header: x y z u v w (required), ax ay az / su sv sw (optional). Natural void —
    no mask is read.

    An OPTIONAL ``# units: length=mm, velocity=mm/s`` comment line may precede
    the column-name header. When present it is checked against the parameter
    file's declared units by :func:`validate_run`, which is the only way a
    mixed ``mm`` positions + ``m/s`` velocities file can be caught -- the
    numbers alone are indistinguishable from a correct file.
    """
    path = Path(path)
    hdr = _header_kv(path)
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if not lines:
        raise ValueError(f"{path}: empty particle file.")
    names = [t.lower() for t in _split(lines[0])]
    data = _numeric(lines[1:], path)
    if data.ndim != 2 or data.shape[1] != len(names):
        raise ValueError(
            f"{path}: header has {len(names)} columns but data has "
            f"{0 if data.ndim < 2 else data.shape[1]}.")
    col = {nm: data[:, i] for i, nm in enumerate(names)}
    for need in ("x", "y", "z", "u", "v", "w"):
        if need not in col:
            raise ValueError(f"{path}: required column '{need}' missing "
                             f"(header: {names}).")
    out = {
        "positions": np.stack([col["x"], col["y"], col["z"]], axis=1),
        "velocities": np.stack([col["u"], col["v"], col["w"]], axis=1),
        "accel": None,
        "sigma": None,
        "columns": names,
        "units": _parse_units_field(hdr["units"]) if "units" in hdr else {},
    }
    if all(k in col for k in ("ax", "ay", "az")):
        out["accel"] = np.stack([col["ax"], col["ay"], col["az"]], axis=1)
    if all(k in col for k in ("su", "sv", "sw")):
        out["sigma"] = np.stack([col["su"], col["sv"], col["sw"]], axis=1)
    return out


def discover_particle_files(directory, pattern="particles_*.dat") -> list[Path]:
    """Return particle files sorted by their zero-padded integer index."""
    files = [Path(p) for p in glob.glob(str(Path(directory) / pattern))]
    if not files:
        raise FileNotFoundError(
            f"No particle files matching '{pattern}' in {directory}.")

    def key(p: Path) -> int:
        m = re.findall(r"\d+", p.stem)
        return int(m[-1]) if m else 0

    return sorted(files, key=key)


# --------------------------------------------------------------------------- #
# 2. Geometry file (shape header + pose rows)
# --------------------------------------------------------------------------- #
def _parse_params_field(text: str) -> dict:
    """Parse ``a=5.0 b=5.0 c=8.0`` / ``path=mesh.obj spacing=0.5`` into a dict.
    Numeric values are floats; everything else stays a string."""
    out: dict = {}
    for tok in text.split():
        if "=" not in tok:
            continue
        k, _, v = tok.partition("=")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            out[k.strip()] = v.strip()
    return out


def read_geometry_file(path) -> dict:
    """Read the geometry file: header (type/params/units/columns) + pose rows."""
    path = Path(path)
    hdr = _header_kv(path)
    for req in ("type", "params", "units", "columns"):
        if req not in hdr:
            raise ValueError(f"{path}: geometry header missing '# {req}:'.")
    gtype = hdr["type"].lower()
    if gtype == "cylinder":
        raise ValueError(
            f"{path}: geometry type 'cylinder' is not yet implemented; only "
            "'sphere' and 'ellipsoid' are supported by this input layer.")
    if gtype not in GEOMETRY_REGISTRY:
        raise ValueError(
            f"{path}: geometry type '{gtype}' is not registered "
            f"(known: {sorted(GEOMETRY_REGISTRY)}).")
    columns = [c.lower() for c in _split(hdr["columns"])]
    data = _numeric(_data_lines(path), path)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(f"{path}: {len(columns)} columns declared, data has "
                         f"{0 if data.ndim < 2 else data.shape[1]}.")
    cmap = {nm: data[:, i] for i, nm in enumerate(columns)}
    if "t" not in cmap or not all(a in cmap for a in ("x", "y", "z")):
        raise ValueError(f"{path}: geometry columns must include t x y z "
                         f"(got {columns}).")
    out = {
        "type": gtype,
        "params": _parse_params_field(hdr["params"]),
        "units": hdr["units"].strip().lower(),
        "columns": columns,
        "t": cmap["t"],
        "centers": np.stack([cmap["x"], cmap["y"], cmap["z"]], axis=1),
        "angles": None,
        "quats": None,
    }
    if all(a in cmap for a in ("alpha", "beta", "gamma")):
        out["angles"] = np.stack([cmap["alpha"], cmap["beta"], cmap["gamma"]], axis=1)
    elif all(q in cmap for q in ("qw", "qx", "qy", "qz")):
        out["quats"] = np.stack([cmap["qw"], cmap["qx"], cmap["qy"], cmap["qz"]], axis=1)
    else:
        raise ValueError(f"{path}: pose columns must be alpha beta gamma OR "
                         f"qw qx qy qz (got {columns}).")
    return out


# --------------------------------------------------------------------------- #
# 3. Kinematics file (t u v w omega_x omega_y omega_z)
# --------------------------------------------------------------------------- #
def read_kinematics_file(path) -> dict:
    """Read the kinematics file: body linear + angular velocity per timestep."""
    path = Path(path)
    hdr = _header_kv(path)
    if "columns" not in hdr:
        raise ValueError(f"{path}: kinematics header missing '# columns:'.")
    columns = [c.lower() for c in _split(hdr["columns"])]
    data = _numeric(_data_lines(path), path)
    if data.ndim != 2 or data.shape[1] != len(columns):
        raise ValueError(f"{path}: {len(columns)} columns declared, data has "
                         f"{0 if data.ndim < 2 else data.shape[1]}.")
    cmap = {nm: data[:, i] for i, nm in enumerate(columns)}
    need = ("t", "u", "v", "w", "omega_x", "omega_y", "omega_z")
    if not all(k in cmap for k in need):
        raise ValueError(f"{path}: kinematics columns must be {list(need)} "
                         f"(got {columns}).")
    units = {}
    if "units" in hdr:
        units = _parse_units_field(hdr["units"])
    return {
        "units": units,
        "columns": columns,
        "t": cmap["t"],
        "U": np.stack([cmap["u"], cmap["v"], cmap["w"]], axis=1),
        "omega": np.stack([cmap["omega_x"], cmap["omega_y"], cmap["omega_z"]], axis=1),
    }


def _parse_units_field(text: str) -> dict:
    """Parse ``velocity=mm/s, omega=rad/s`` into {'velocity':'mm/s','omega':'rad/s'}."""
    out = {}
    for tok in text.replace(",", " ").split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k.strip().lower()] = v.strip().lower()
    return out


# --------------------------------------------------------------------------- #
# 4. Parameter file  (reuse ccmplus.params; add spec-name -> Config mapping)
# --------------------------------------------------------------------------- #
def read_param_file(path) -> dict:
    return read_parameters(path)


def _get(params: dict, *names, default=None):
    """First present key among ``names`` (spec name then ccmplus alias)."""
    for nm in names:
        if nm in params:
            return params[nm]
    return default


# --------------------------------------------------------------------------- #
# Validation (load-time consistency checks from the spec)
# --------------------------------------------------------------------------- #
def _finite_diff(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """d(values)/dt per row: central interior, one-sided at the two ends."""
    v = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    d = np.zeros_like(v)
    if len(t) == 1:
        return d
    d[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])[:, None]
    d[0] = (v[1] - v[0]) / (t[1] - t[0])
    d[-1] = (v[-1] - v[-2]) / (t[-1] - t[-2])
    return d


def validate_run(geom: dict, kin: dict | None, n_particle_files: int,
                 conv: dict, tol: dict,
                 particle_units: list[dict] | None = None) -> list[str]:
    """Run the load-time checks. Raise ValueError with a clear message on
    failure; return a list of non-fatal informational notes.

    particle_units : optional list of ``{"path": ..., "units": {...}}`` from the
    particle files' optional ``# units:`` headers. Files that declare no units
    are skipped (the header is optional for backwards compatibility), so pass
    what you have and only declared units get enforced.
    """
    notes: list[str] = []

    # 1. dx>0 and enum fields valid
    if not (conv["dx"] > 0):
        raise ValueError(f"dx must be > 0 (got {conv['dx']}).")
    seq = conv["euler_seq"].upper()
    if len(seq) != 3 or any(ch not in _ALLOWED_SEQ_AXES for ch in seq) \
            or seq[0] == seq[1] or seq[1] == seq[2]:
        raise ValueError(
            f"euler_seq '{conv['euler_seq']}' invalid: need 3 axes from X/Y/Z "
            "with no two consecutive equal (e.g. ZYX).")
    if conv["handedness"] not in _ALLOWED_HANDEDNESS:
        raise ValueError(f"handedness must be one of {_ALLOWED_HANDEDNESS}, "
                         f"got '{conv['handedness']}'.")
    if conv["omega_frame"] not in _ALLOWED_OMEGA_FRAME:
        raise ValueError(f"omega_frame must be one of {_ALLOWED_OMEGA_FRAME}, "
                         f"got '{conv['omega_frame']}'.")

    # 2. position (& kinematics, if present) t match; particle-file count matches.
    #    ``kin is None`` is the position-only (single-file) path.
    tg = geom["t"]
    if kin is not None:
        tk = kin["t"]
        if len(tg) != len(tk):
            raise ValueError(f"position file has {len(tg)} timesteps, kinematics has "
                             f"{len(tk)}; they must match.")
        if not np.allclose(tg, tk, rtol=0, atol=tol["t_atol"]):
            bad = int(np.argmax(np.abs(tg - tk)))
            raise ValueError(
                f"position and kinematics t columns differ at row {bad}: "
                f"{tg[bad]} vs {tk[bad]} (atol={tol['t_atol']}).")
    if n_particle_files != len(tg):
        raise ValueError(
            f"{n_particle_files} particle files but {len(tg)} pose timesteps; "
            "one particle file per timestep is required.")

    # 3. header units match the parameter file
    if geom["units"] != conv["length_unit"].lower():
        raise ValueError(
            f"geometry units '{geom['units']}' != parameter length_unit "
            f"'{conv['length_unit']}'.")
    if kin is not None:
        vunit = kin["units"].get("velocity")
        exp_v = f"{conv['length_unit'].lower()}/{conv['time_unit'].lower()}"
        if vunit is not None and vunit != exp_v:
            raise ValueError(
                f"kinematics velocity unit '{vunit}' != expected '{exp_v}' "
                "(length_unit/time_unit).")
        ounit = kin["units"].get("omega")
        if ounit is not None and ounit != "rad/s":
            raise ValueError(f"kinematics omega unit '{ounit}' != 'rad/s'.")

    # 3b. particle-file units, when the optional header declares them. A track
    #     file carrying positions in mm and velocities in m/s is numerically
    #     indistinguishable from a correct file -- the declared header is the
    #     only thing that can catch it, so if it is present it is enforced with
    #     exactly the same strictness as the geometry and kinematics headers.
    exp_len = conv["length_unit"].lower()
    exp_vel = f"{exp_len}/{conv['time_unit'].lower()}"
    for pu in particle_units or []:
        path, units = pu["path"], pu["units"]
        lunit = units.get("length")
        if lunit is not None and lunit != exp_len:
            raise ValueError(
                f"{path}: particle length unit '{lunit}' != parameter "
                f"length_unit '{exp_len}'.")
        vunit = units.get("velocity")
        if vunit is not None and vunit != exp_vel:
            raise ValueError(
                f"{path}: particle velocity unit '{vunit}' != expected "
                f"'{exp_vel}' (length_unit/time_unit). All files in a run share "
                f"ONE coherent unit system; convert the file or change "
                f"length_unit/time_unit.")

    # 4. physical consistency: d/dt(center) ≈ U (only when kinematics is supplied).
    #    There is deliberately NO Euler-angle-vs-omega check: the position-file
    #    orientation and the kinematics omega are independent inputs (omega is not
    #    necessarily d(theta)/dt), so we never differentiate the Euler angles here.
    if kin is not None and len(tg) >= 2:
        dcdt = _finite_diff(geom["centers"], tg)
        U = kin["U"]
        err_U = np.abs(dcdt - U)
        scaleU = np.abs(U).max() + tol["v_atol"]
        if np.max(err_U) > tol["rtol"] * scaleU + tol["v_atol"]:
            i = int(np.unravel_index(np.argmax(err_U), err_U.shape)[0])
            raise ValueError(
                "physical consistency failed: d/dt(center) disagrees with U at "
                f"row {i}: d/dt={dcdt[i]}, U={U[i]} "
                f"(max err {np.max(err_U):.4g} > tol). Check units / frame.")
    return notes


# --------------------------------------------------------------------------- #
# Assembly: four files -> (grid, config, frames)
# --------------------------------------------------------------------------- #
class AssembledRun:
    def __init__(self, grid, config, frames, meta):
        self.grid = grid
        self.config = config
        self.frames = frames        # list[FrameData], one per timestep
        self.meta = meta            # dict: t, notes, geometry type, etc.


def _conventions(params: dict) -> dict:
    return {
        "dx": float(_get(params, "dx", "delta_mm", default=1.0)),
        "length_unit": str(_get(params, "length_unit", default="mm")),
        "time_unit": str(_get(params, "time_unit", default="s")),
        "angle_unit": str(_get(params, "angle_unit", default="deg")),
        "euler_seq": str(_get(params, "euler_seq", default="ZYX")),
        "handedness": str(_get(params, "handedness", default="right")).lower(),
        "omega_frame": str(_get(params, "omega_frame", default="world")).lower(),
    }


def _resolve_boundary_constraints(params: dict) -> bool:
    """Read ``boundary_constraints``, honouring the deprecated ``enable_lema``.

    ``boundary_constraints = on|off`` is the current key. ``enable_lema`` is
    accepted for existing configs but warns; if both appear,
    ``boundary_constraints`` wins.
    """
    new = _get(params, "boundary_constraints")
    old = _get(params, "enable_lema")
    if old is not None:
        warnings.warn(
            "The 'enable_lema' parameter is deprecated; rename it to "
            "'boundary_constraints = on|off'. Note that 'off' removes the "
            "no-slip constraint rows but still masks interpolation and "
            "smoothing by the body classification -- it is not an all-fluid "
            "baseline."
            + (" 'boundary_constraints' is also set and takes precedence."
               if new is not None else ""),
            DeprecationWarning, stacklevel=3,
        )
    if new is not None:
        return bool(new)
    if old is not None:
        return bool(old)
    return True


def _config_from_params(params: dict, domain_min, domain_max) -> Config:
    """Map spec / ccmplus parameter names onto the ccmplus Config."""
    return Config(
        domain_min=domain_min, domain_max=domain_max,
        delta=float(_get(params, "dx", "delta_mm", default=1.0)),
        kappa=float(_get(params, "kappa", default=1.0)),
        solver_rtol=float(_get(params, "minres_tol", "solver_rtol", default=1e-6)),
        solver_maxiter=int(_get(params, "minres_maxit", "solver_maxiter", default=2000)),
        boundary_constraints=_resolve_boundary_constraints(params),
        constraint_div_tol=float(_get(params, "constraint_div_tol", default=1e-3)),
        constraint_body_tol=float(_get(params, "constraint_body_tol", default=1e-3)),
        use_jacobi_precond=bool(_get(params, "use_jacobi_precond", default=False)),
        interp_kernel=str(_get(params, "interp_kernel", default="wide")),
        lambda_coverage=float(_get(params, "lambda_c", "lambda_coverage", default=0.0)),
        coverage_ref_count=float(_get(params, "coverage_ref_count", default=1.0)),
        enable_proximity_reweight=bool(_get(params, "enable_proximity_reweight",
                                            default=False)),
    )


def _pose_rotation(geom: dict, i: int, conv: dict) -> np.ndarray:
    if geom["angles"] is not None:
        return build_rotation(geom["angles"][i], conv["euler_seq"],
                              conv["angle_unit"], conv["handedness"])
    q = geom["quats"][i]
    return _quat_to_R(q)


def _quat_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# --------------------------------------------------------------------------- #
# Region of interest (ROI) grid bounds + guards
# --------------------------------------------------------------------------- #
def _grid_node_count(domain_min, domain_max, dx) -> int:
    """Node count of the RectGrid these bounds + dx would build (must match
    RectGrid's Nx/Ny/Nz formula exactly, so the guard sees the true size)."""
    dmin = np.asarray(domain_min, dtype=float)
    dmax = np.asarray(domain_max, dtype=float)
    N = np.round((dmax - dmin) / dx).astype(np.int64) + 1
    return int(N[0]) * int(N[1]) * int(N[2])


def _kernel_reach_cells(kernel: str, radius_cells: float) -> int:
    """How many cells beyond a particle's own cell the interpolation footprint can
    reach. A particle within this distance OUTSIDE the grid box still informs
    boundary nodes and must be kept; beyond it, it cannot touch any node."""
    k = str(kernel).strip().lower()
    if k == "trilinear":
        return 1
    if k in ("wide", "bspline"):
        return 2                       # cubic B-spline touches offsets -1..+2
    if k == "gaussian":
        return int(np.ceil(radius_cells))
    return 2


def _body_roi_bounds(params: dict, geom: dict, conv: dict, geometry, dx: float):
    """roi_mode = body: union of the body AABB over the [t_start, t_end] window,
    expanded by ``roi_pad`` body characteristic lengths in every direction."""
    roi_pad = float(_get(params, "roi_pad", default=2.0))
    char_len = float(geometry.characteristic_length())
    tg = np.asarray(geom["t"], dtype=float)
    sel = np.ones(len(tg), dtype=bool)
    t_start = _get(params, "t_start")
    t_end = _get(params, "t_end")
    if t_start is not None:
        sel &= tg >= float(t_start)
    if t_end is not None:
        sel &= tg <= float(t_end)
    idxs = np.where(sel)[0]
    if idxs.size == 0:
        raise ValueError(
            f"roi_mode = body: no geometry timesteps fall in the window "
            f"[t_start={t_start}, t_end={t_end}] (t spans {tg.min()}..{tg.max()}).")
    bmin = np.full(3, np.inf)
    bmax = np.full(3, -np.inf)
    for i in idxs:
        c = np.asarray(geom["centers"][i], dtype=float)
        R = _pose_rotation(geom, i, conv)
        half = np.asarray(geometry.aabb_half_extents(R), dtype=float)
        bmin = np.minimum(bmin, c - half)
        bmax = np.maximum(bmax, c + half)
    pad = roi_pad * char_len
    return tuple((bmin - pad).tolist()), tuple((bmax + pad).tolist())


def _resolve_domain(params, conv, geom, geometry, pfiles, dx):
    """Return (domain_min, domain_max, roi_mode). roi_mode = auto reproduces the
    original grid-extent behaviour byte-for-byte."""
    roi_mode = str(_get(params, "roi_mode", default="auto")).strip().lower()
    if roi_mode not in ("auto", "box", "body"):
        raise ValueError(f"roi_mode must be one of auto|box|body, got '{roi_mode}'.")

    if roi_mode == "box":
        roi_box = _get(params, "roi_box")
        if roi_box is None:
            raise ValueError("roi_mode = box requires "
                             "'roi_box = xmin xmax ymin ymax zmin zmax'.")
        xs = [float(v) for v in _split(str(roi_box))]
        if len(xs) != 6:
            raise ValueError(f"roi_box needs 6 values "
                             f"(xmin xmax ymin ymax zmin zmax); got {len(xs)}: {xs}.")
        if not (xs[0] < xs[1] and xs[2] < xs[3] and xs[4] < xs[5]):
            raise ValueError(f"roi_box must satisfy xmin<xmax, ymin<ymax, zmin<zmax; "
                             f"got {xs}.")
        margin = float(_get(params, "grid_margin", default=3)) * dx
        domain_min = (xs[0] - margin, xs[2] - margin, xs[4] - margin)
        domain_max = (xs[1] + margin, xs[3] + margin, xs[5] + margin)
        return domain_min, domain_max, roi_mode

    if roi_mode == "body":
        domain_min, domain_max = _body_roi_bounds(params, geom, conv, geometry, dx)
        return domain_min, domain_max, roi_mode

    # roi_mode == auto : EXACTLY the original behaviour (do not alter numerics)
    extent = _get(params, "grid_extent", default="auto")
    if isinstance(extent, str) and extent.strip().lower() != "auto":
        xs = [float(v) for v in _split(extent)]
        domain_min = (xs[0], xs[2], xs[4])
        domain_max = (xs[1], xs[3], xs[5])
    else:
        allpos = np.vstack([read_particle_file(f)["positions"] for f in pfiles])
        margin = float(_get(params, "grid_margin", default=3)) * dx
        domain_min = tuple((allpos.min(axis=0) - margin).tolist())
        domain_max = tuple((allpos.max(axis=0) + margin).tolist())
    return domain_min, domain_max, roi_mode


def assemble(param_path) -> AssembledRun:
    """Read + validate the four inputs and build the ccmplus run objects."""
    params = read_param_file(param_path)
    base = Path(param_path).parent
    conv = _conventions(params)

    def _resolve(key, default=None):
        val = _get(params, key, default=default)
        if val is None:
            raise KeyError(f"parameter file must set '{key}'.")
        p = Path(val)
        return p if p.is_absolute() else (base / p)

    # Position file (a.k.a. geometry file): shape header + pose rows.
    geom_val = _get(params, "position_file", "geometry_file")
    if geom_val is None:
        raise KeyError("parameter file must set 'position_file' (or 'geometry_file').")
    gp = Path(geom_val)
    geom = read_geometry_file(gp if gp.is_absolute() else base / gp)

    # Kinematics file is OPTIONAL (two-file interface). When absent, velocity comes
    # from differentiating the position trajectory -- allowed for a sphere only.
    kin = None
    kin_val = _get(params, "kinematics_file")
    if kin_val is not None:
        kp = Path(kin_val)
        kp = kp if kp.is_absolute() else base / kp
        if not kp.exists():
            raise FileNotFoundError(f"kinematics_file is set but not found: {kp}")
        kin = read_kinematics_file(kp)

    pdir = _resolve("particles_dir")
    ppat = str(_get(params, "particles_pattern", default="particles_*.dat"))
    pfiles = discover_particle_files(pdir, ppat)

    tol = {
        "t_atol": float(_get(params, "consistency_t_atol", default=1e-6)),
        "rtol": float(_get(params, "consistency_rtol", default=5e-2)),
        "v_atol": float(_get(params, "consistency_v_atol", default=1e-6)),
        "w_atol": float(_get(params, "consistency_w_atol", default=1e-6)),
    }
    # Collect any declared particle-file units so the unit contract is checked
    # across ALL inputs, not just geometry + kinematics.
    particle_units = []
    for pf in pfiles:
        u = _header_kv(pf).get("units")
        if u:
            particle_units.append({"path": str(pf), "units": _parse_units_field(u)})

    notes = validate_run(geom, kin, len(pfiles), conv, tol, particle_units)

    # Velocity source (the two-file rule):
    #   kinematics present    -> kinematics wins; never differentiate position.
    #   position-only sphere  -> differentiate the trajectory, omega = 0
    #                            (reproduces the single-file sphere behaviour).
    #   position-only non-sphere -> hard error (do not differentiate a non-sphere).
    if kin is not None:
        U_all, omega_all = kin["U"], kin["omega"]
        omega_frame = conv["omega_frame"]
        velocity_source = "kinematics"
    else:
        if geom["type"] != "sphere":
            raise ValueError(
                f"position-only input is allowed for a sphere only; the geometry "
                f"type is '{geom['type']}'. Provide a kinematics_file with columns "
                "t u v w omega_x omega_y omega_z.")
        U_all = _finite_diff(geom["centers"], geom["t"])
        omega_all = np.zeros((len(geom["t"]), 3))
        omega_frame = "world"
        velocity_source = "position_diff"

    # Body shape is needed before the grid for roi_mode = body.
    geometry = GEOMETRY_REGISTRY[geom["type"]].from_params(geom["params"])

    # Grid bounds: roi_mode auto (original behaviour) | box | body.
    dx = conv["dx"]
    domain_min, domain_max, roi_mode = _resolve_domain(
        params, conv, geom, geometry, pfiles, dx)

    # Node-count guard BEFORE building RectGrid (which itself allocates a node +
    # neighbour table): fail here with a readable message instead of an OOM later.
    max_nodes = int(_get(params, "max_grid_nodes", default=2_000_000))
    n_nodes = _grid_node_count(domain_min, domain_max, dx)
    if n_nodes > max_nodes:
        raise ValueError(
            f"reconstruction grid would have {n_nodes:,} nodes, exceeding "
            f"max_grid_nodes = {max_nodes:,}.\n"
            f"  roi_mode = {roi_mode}\n"
            f"  bounds   = min {tuple(round(float(v), 6) for v in domain_min)} "
            f"max {tuple(round(float(v), 6) for v in domain_max)}\n"
            f"  dx       = {dx}\n"
            f"Restrict the grid (set roi_mode = body, or roi_mode = box with an "
            f"explicit roi_box) or increase dx. Refusing to allocate; the full "
            f"auto extent would otherwise OOM inside solve_saddle_point.")

    grid = RectGrid(domain_min, domain_max, dx)
    config = _config_from_params(params, domain_min, domain_max)

    sigma_gamma = float(_get(params, "sigma_gamma", "sigma_s_mm", default=0.5))
    sigma_u = float(_get(params, "sigma_u", "sigma_i_ms", default=0.01))
    void_margin = float(_get(params, "void_margin", default=0.0))

    # Particles beyond the grid box plus one interpolation-kernel reach cannot
    # inform any node; drop them at load time (a no-op for roi_mode = auto, whose
    # box already encloses every particle).
    dmin = np.asarray(domain_min, dtype=float)
    dmax = np.asarray(domain_max, dtype=float)
    reach = _kernel_reach_cells(config.interp_kernel,
                                config.kernel_radius_cells) * dx
    n_kept = 0
    n_dropped_roi = 0

    frames = []
    for i, pf in enumerate(pfiles):
        part = read_particle_file(pf)
        pos, vel = part["positions"], part["velocities"]
        c = geom["centers"][i]
        R = _pose_rotation(geom, i, conv)
        omega = omega_all[i]
        if omega_frame == "body":                  # u_gamma needs world-frame omega
            omega = R @ omega
        body = make_body(geometry, c, R, U_all[i], omega, sigma_gamma)

        # Natural void: cull ghost tracks inside the solid (phi < -void_margin).
        phi_p = body.sdf_fn(pos, body)
        keep = phi_p >= -void_margin
        pos, vel = pos[keep], vel[keep]
        if part["sigma"] is not None:
            unc = np.linalg.norm(part["sigma"][keep], axis=1) / np.sqrt(3.0)
        else:
            unc = np.full(len(pos), sigma_u)

        # ROI cull: drop tracks outside the grid box + kernel reach.
        in_box = np.all((pos >= dmin - reach) & (pos <= dmax + reach), axis=1)
        n_dropped_roi += int((~in_box).sum())
        pos, vel, unc = pos[in_box], vel[in_box], unc[in_box]
        n_kept += int(len(pos))

        frames.append(FrameData(positions=pos, velocities=vel, uncertainties=unc,
                                body=body, t=float(geom["t"][i])))

    meta = {"t": geom["t"], "type": geom["type"], "notes": notes,
            "velocity_source": velocity_source,
            "units": {
                "length": conv["length_unit"],
                "time": conv["time_unit"],
                "velocity": f"{conv['length_unit']}/{conv['time_unit']}",
                "angle": conv["angle_unit"],
                "angular_velocity": "rad/s",
            },
            "geometry_params": dict(geom["params"]),
            "omega_frame": omega_frame,
            "euler_seq": conv["euler_seq"],
            "handedness": conv["handedness"],
            "n_particle_units_declared": len(particle_units),
            "grid_shape": (grid.Nx, grid.Ny, grid.Nz),
            "n_timesteps": len(frames), "domain_min": domain_min,
            "domain_max": domain_max, "roi_mode": roi_mode, "n_nodes": n_nodes,
            "n_particles_kept": n_kept, "n_particles_dropped_roi": n_dropped_roi,
            "warm_start": bool(_get(params, "warm_start", default=True)),
            "output_dir": str(_get(params, "output_dir", default="outputs/ledm")),
            "case_name": str(_get(params, "case_name", default="ledm"))}
    return AssembledRun(grid, config, frames, meta)


def run_banner(run: "AssembledRun") -> list[str]:
    """Human-readable summary of everything the loader resolved.

    Printed before the first solve. The point is that a run's UNITS are a
    declaration, not something the code can infer from the numbers: a file of
    millimetre positions with metre-per-second velocities parses perfectly and
    produces silently wrong physics. Echoing the resolved units next to the
    geometry, grid and track count makes such a mistake visible in the first
    lines of output instead of in the results.
    """
    m = run.meta
    u = m["units"]
    gshape = m["grid_shape"]
    dmin = tuple(round(float(v), 4) for v in m["domain_min"])
    dmax = tuple(round(float(v), 4) for v in m["domain_max"])
    cfg = run.config

    geom_params = ", ".join(
        f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in m["geometry_params"].items()
    )
    declared = m["n_particle_units_declared"]
    unit_note = (
        f"({declared} particle file(s) declare units; checked)" if declared
        else "(particle files declare no units; not checkable -- "
             "add '# units: length=..., velocity=...' to enable the check)"
    )
    boundary_on = bool(getattr(cfg, "boundary_constraints", True))

    return [
        "[LE-DM] ================ run configuration ================",
        f"[LE-DM] units      : length={u['length']}  time={u['time']}  "
        f"velocity={u['velocity']}  angle={u['angle']}  "
        f"angular_velocity={u['angular_velocity']}",
        f"[LE-DM]              {unit_note}",
        f"[LE-DM] geometry   : {m['type']} ({geom_params})  "
        f"euler_seq={m['euler_seq']}  handedness={m['handedness']}  "
        f"omega_frame={m['omega_frame']}",
        f"[LE-DM] kinematics : velocity_source={m['velocity_source']}  "
        f"n_timesteps={m['n_timesteps']}",
        f"[LE-DM] grid       : {gshape[0]}x{gshape[1]}x{gshape[2]} = "
        f"{m['n_nodes']:,} nodes  dx={run.grid.delta:g} {u['length']}  "
        f"roi_mode={m['roi_mode']}",
        f"[LE-DM]              bounds min {dmin} max {dmax} [{u['length']}]",
        f"[LE-DM] tracks     : {m['n_particles_kept']:,} kept, "
        f"{m['n_particles_dropped_roi']:,} dropped (outside ROI)",
        f"[LE-DM] LE-DM opts : boundary_constraints={'on' if boundary_on else 'off'}  "
        f"kernel={cfg.interp_kernel}  kappa={cfg.kappa:g}  "
        f"lambda_c={cfg.lambda_coverage:g}  c_0={cfg.coverage_ref_count:g}",
        f"[LE-DM]              proximity_reweight="
        f"{'on' if cfg.enable_proximity_reweight else 'off'}  "
        f"minres_tol={cfg.solver_rtol:g}  minres_maxit={cfg.solver_maxiter}  "
        f"div_tol={cfg.constraint_div_tol:g}  body_tol={cfg.constraint_body_tol:g}",
        "[LE-DM] ===================================================",
    ]


# --------------------------------------------------------------------------- #
# Wire the run: assembled inputs -> UNCHANGED ccmplus solver -> outputs
# --------------------------------------------------------------------------- #
def run_four_file(param_path, first=None, last=None, write=True, log=None):
    """Consume the four-file contract and drive the frozen ccmplus solver.

    Returns ``(run, results)`` where ``run`` is the ``AssembledRun`` and
    ``results`` is a list of ``ReconstructionResult`` (one per timestep). The
    solver core is called exactly through the INTERFACE.md contract; the geometry
    layer only supplies each ``BodyState.sdf_fn`` and rigid kinematics.
    """
    from ccmplus.reconstruct import CCMPlus   # local import: keep module light

    def _say(msg):
        if log is not None:
            log(msg)

    run = assemble(param_path)
    m = run.meta
    for line in run_banner(run):
        _say(line)
    steps = range(len(run.frames))
    if first is not None or last is not None:
        lo = 0 if first is None else max(0, first)
        hi = len(run.frames) if last is None else min(len(run.frames), last + 1)
        steps = range(lo, hi)

    solver = CCMPlus(run.config, run.grid)
    out_dir = Path(run.meta["output_dir"])
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
    case = run.meta["case_name"]

    # Output format: npz (default, unchanged) | dat | both. The .dat writer lives
    # in io_output; npz stays the source of truth for warm_start / downstream code.
    params = read_param_file(param_path)
    conv = _conventions(params)
    out_fmt = str(_get(params, "output_format", default="npz")).strip().lower()
    if out_fmt not in ("npz", "dat", "both"):
        raise ValueError(f"output_format must be npz|dat|both, got '{out_fmt}'.")
    write_npz = out_fmt in ("npz", "both")
    write_dat_out = out_fmt in ("dat", "both")
    dat_flavor = str(_get(params, "dat_flavor", default="tecplot")).strip().lower()
    dat_precision = int(_get(params, "dat_precision", default=9))
    dat_order = str(_get(params, "dat_order", default="C")).strip().upper()

    # Output reference frame (post-solve transform in this layer only):
    #   lab  (default) -> write v_lab and grid nodes exactly as solved (unchanged).
    #   body           -> write v_rel = v_lab - u_body, u_body from the SAME body
    #                     velocity field (u_gamma) the solver pinned into the solid.
    out_frame = str(_get(params, "output_frame", default="lab")).strip().lower()
    if out_frame not in ("lab", "body"):
        raise ValueError(f"output_frame must be lab|body, got '{out_frame}'.")
    comoving = bool(_get(params, "comoving_coords", default=False))
    from ccmplus.io_output import body_frame_velocity  # v_rel = v_lab - u_gamma
    X_s0 = None                                       # first-processed body centre

    results = []
    for i in steps:
        if not run.meta["warm_start"]:
            solver.reset()
        frame = run.frames[i]
        res = solver.reconstruct(frame)
        results.append(res)
        _say(f"step {i}: t={frame.t:g}  n_tracks={len(frame.positions)}  "
             f"iters={res.iterations}  converged={res.converged}  "
             f"resid={res.residual:.3e}")
        if not write:
            continue

        # Resolve the output velocity / coordinates for this snapshot. In lab
        # frame these are exactly res.velocity and grid.nodes (bytes unchanged).
        body = frame.body
        if X_s0 is None:
            X_s0 = np.asarray(body.X_s, dtype=float).copy()
        vel_out = res.velocity
        nodes_out = run.grid.nodes
        frame_meta: dict = {"output_frame": out_frame}
        if out_frame == "body":
            vel_out = body_frame_velocity(res.velocity, run.grid.nodes, body)
            frame_meta.update(
                body_U=tuple(round(float(v), 9) for v in np.asarray(body.U_s)),
                body_omega=tuple(round(float(v), 9) for v in np.asarray(body.omega_s)),
                body_X_s=tuple(round(float(v), 9) for v in np.asarray(body.X_s)))
        if comoving:
            shift = np.asarray(body.X_s, dtype=float) - X_s0
            nodes_out = run.grid.nodes - shift
            frame_meta["comoving_coords"] = True
            frame_meta["body_X_s0"] = tuple(round(float(v), 9) for v in X_s0)

        if write_npz:
            np.savez_compressed(
                out_dir / f"{case}_step{i:05d}.npz",
                nodes=nodes_out, velocity=vel_out,
                classification=res.classification, t=frame.t, **_npz_frame(frame_meta))
        if write_dat_out:
            from ccmplus.io_output import (snapshot_from_arrays, write_dat,
                                           fields_from_result, LEDM_VERSION)
            fields = fields_from_result(res)           # u,v,w (+ classification, ...)
            fields["u"], fields["v"], fields["w"] = (
                vel_out[:, 0], vel_out[:, 1], vel_out[:, 2])
            coords, fields, dims = snapshot_from_arrays(
                nodes_out, fields, run.grid.shape, dat_order)
            dat_meta = {
                "case": case, "time": frame.t, "time_unit": conv["time_unit"],
                "dx": run.grid.delta, "length_unit": conv["length_unit"],
                "bounds_min": tuple(round(float(v), 6) for v in m["domain_min"]),
                "bounds_max": tuple(round(float(v), 6) for v in m["domain_max"]),
                "roi_mode": m["roi_mode"], "n_nodes": m["n_nodes"],
                "grid_dims": dims, "version": LEDM_VERSION, **frame_meta,
            }
            dat_path = out_dir / f"{case}_step{i:05d}.dat"
            write_dat(dat_path, coords, fields, dims, dat_meta,
                      flavor=dat_flavor, precision=dat_precision, order=dat_order)
    return run, results


def _npz_frame(frame_meta: dict) -> dict:
    """Extra npz keys recording an output-frame transform. Empty for lab frame so
    the default archive keeps exactly its original key set (regression contract)."""
    if frame_meta.get("output_frame", "lab") == "lab" and not frame_meta.get(
            "comoving_coords"):
        return {}
    out = {"output_frame": np.array(frame_meta["output_frame"])}
    for k in ("body_U", "body_omega", "body_X_s", "body_X_s0"):
        if k in frame_meta:
            out[k] = np.asarray(frame_meta[k], dtype=float)
    if frame_meta.get("comoving_coords"):
        out["comoving_coords"] = np.array(True)
    return out
