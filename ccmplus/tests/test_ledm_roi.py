"""ROI option for the four-file input path (io_ledm): box / body grid bounds,
the node-count guard, and the auto-mode regression.

The solver, reconstruct and sphere driver are untouched; every assertion here is
about which grid box the frozen solver is handed and which tracks reach it.
"""
import numpy as np
import pytest

from ccmplus.io_ledm import assemble, run_four_file


# --------------------------------------------------------------------------- #
# Shared sphere case authoring (mirrors test_ledm_regression._write_case)
# --------------------------------------------------------------------------- #
R = 2.0
C0 = np.array([0.0, 0.0, 0.0])
U = np.array([0.5, 0.0, 0.0])
TIMES = np.array([0.0, 0.1, 0.2])
DX = 1.5
SOLVER_TOL = 1e-8


def _fluid_field(x):
    return np.stack([0.3 - 0.1 * x[:, 1], 0.1 * x[:, 0], np.zeros(len(x))], axis=1)


def _write_case(tmp, extra="", grid_line="grid_extent = -6 6 -6 6 -6 6\n", dx=DX):
    """Author the three data files + parameter file. `extra` is appended verbatim
    (ROI keys etc.); `grid_line` sets the grid stanza."""
    tmp.mkdir(parents=True, exist_ok=True)
    pdir = tmp / "particles"
    pdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    for k, t in enumerate(TIMES, start=1):
        center = C0 + U * t
        pts = rng.uniform(-6, 6, size=(1200, 3))
        pts = pts[np.linalg.norm(pts - center, axis=1) > R + 1e-6]
        arr = np.hstack([pts, _fluid_field(pts)])
        with open(pdir / f"particles_{k:05d}.dat", "w", encoding="utf-8") as fh:
            fh.write("x y z u v w\n")
            for row in arr:
                fh.write(" ".join(f"{v:.17g}" for v in row) + "\n")

    with open(tmp / "geometry.dat", "w", encoding="utf-8") as fh:
        fh.write("# type: sphere\n")
        fh.write(f"# params: r={R}\n# units: mm\n")
        fh.write("# columns: t x y z alpha beta gamma\n")
        for t in TIMES:
            c = C0 + U * t
            fh.write(f"{t:.17g} {c[0]:.17g} {c[1]:.17g} {c[2]:.17g} 0 0 0\n")

    with open(tmp / "kinematics.dat", "w", encoding="utf-8") as fh:
        fh.write("# units: velocity=mm/s, omega=rad/s\n")
        fh.write("# columns: t u v w omega_x omega_y omega_z\n")
        for t in TIMES:
            fh.write(f"{t:.17g} {U[0]:.17g} {U[1]:.17g} {U[2]:.17g} 0 0 0\n")

    param = tmp / "params.txt"
    param.write_text(
        "geometry_file   = geometry.dat\n"
        "kinematics_file = kinematics.dat\n"
        "particles_dir   = particles\n"
        "particles_pattern = particles_*.dat\n"
        + grid_line
        + f"dx = {dx}\n"
        "kappa = 1.0\n"
        f"minres_tol = {SOLVER_TOL}\n"
        # constraint_div_tol is relaxed on purpose: this case exercises the
        # input contract / wiring, not constraint convergence, so it uses a
        # cheap solve. Constraint quality is asserted with tight solves and
        # normalized tolerances in test_solver_onefluid.py.
        "minres_maxit = 4000\nconstraint_div_tol = 1.0\n"
        "boundary_constraints = on\n"
        "interp_kernel = wide\n"
        "lambda_c = 0.0\n"
        "sigma_u = 0.01\n"
        "sigma_gamma = 0.5\n"
        "length_unit = mm\ntime_unit = s\nangle_unit = deg\n"
        "euler_seq = ZYX\nhandedness = right\nomega_frame = world\n"
        "warm_start = true\ncase_name = roi\n"
        + extra,
        encoding="utf-8",
    )
    return param


# --------------------------------------------------------------------------- #
# 1. roi_mode = box -> exact grid dimensions
# --------------------------------------------------------------------------- #
def test_roi_box_exact_grid_dimensions(tmp_path):
    # grid_margin = 0 so the grid spans roi_box exactly. Non-cubic on purpose.
    param = _write_case(
        tmp_path,
        grid_line="grid_extent = auto\n",
        extra="roi_mode = box\nroi_box = -3 3 -3 4.5 -1.5 1.5\ngrid_margin = 0\n",
    )
    run = assemble(param)
    assert run.meta["roi_mode"] == "box"
    # Nx = round((xmax-xmin)/dx)+1 : (6/1.5)+1=5, (7.5/1.5)+1=6, (3/1.5)+1=3
    assert run.grid.shape == (5, 6, 3)
    np.testing.assert_allclose(run.grid.domain_min, [-3, -3, -1.5])
    np.testing.assert_allclose(run.grid.domain_max, [3, 4.5, 1.5])
    # A track far outside the box is dropped; tracks inside are kept.
    assert run.meta["n_particles_dropped_roi"] > 0


def test_roi_box_requires_and_validates_roi_box(tmp_path):
    p = _write_case(tmp_path, grid_line="grid_extent = auto\n", extra="roi_mode = box\n")
    with pytest.raises(ValueError, match="roi_box"):
        assemble(p)
    p2 = _write_case(tmp_path / "b", grid_line="grid_extent = auto\n",
                     extra="roi_mode = box\nroi_box = 1 2 3\n")
    with pytest.raises(ValueError, match="6 values"):
        assemble(p2)


# --------------------------------------------------------------------------- #
# 2. roi_mode = body -> grid contains the sphere at every timestep, with padding
# --------------------------------------------------------------------------- #
def test_roi_body_contains_sphere_with_padding(tmp_path):
    roi_pad = 2.0
    param = _write_case(
        tmp_path, grid_line="grid_extent = auto\n",
        extra=f"roi_mode = body\nroi_pad = {roi_pad}\n",
    )
    run = assemble(param)
    assert run.meta["roi_mode"] == "body"
    dmin = np.asarray(run.grid.domain_min)
    dmax = np.asarray(run.grid.domain_max)

    # the sphere bbox at every timestep is strictly inside the grid box
    for t in TIMES:
        c = C0 + U * t
        assert np.all(c - R >= dmin - 1e-9)
        assert np.all(c + R <= dmax + 1e-9)

    # padding: grid extends by exactly roi_pad*characteristic_length (=2r) beyond
    # the union of the body bbox over the window.
    char_len = 2 * R
    body_min = np.min([C0 + U * t for t in TIMES], axis=0) - R
    body_max = np.max([C0 + U * t for t in TIMES], axis=0) + R
    np.testing.assert_allclose(dmin, body_min - roi_pad * char_len)
    np.testing.assert_allclose(dmax, body_max + roi_pad * char_len)


# --------------------------------------------------------------------------- #
# 3. node-count guard raises ValueError (not MemoryError) on an oversized auto case
# --------------------------------------------------------------------------- #
def test_node_guard_raises_valueerror_not_memoryerror(tmp_path):
    # dx tiny -> ~1e9 nodes; must be refused BEFORE RectGrid allocates anything.
    param = _write_case(tmp_path, grid_line="grid_extent = auto\n", dx=0.02)
    with pytest.raises(ValueError) as ei:
        assemble(param)
    msg = str(ei.value)
    assert "max_grid_nodes" in msg and "roi_mode" in msg and "dx" in msg

    # a generous ceiling lets the same case through
    param_ok = _write_case(tmp_path / "ok", grid_line="grid_extent = auto\n",
                           extra="max_grid_nodes = 5000\n")
    run = assemble(param_ok)          # ~15^3 = 3375 nodes < 5000
    assert run.meta["n_nodes"] <= 5000


def test_guard_never_reaches_allocation(tmp_path):
    """The oversized case must not raise MemoryError (guard precedes RectGrid)."""
    param = _write_case(tmp_path, grid_line="grid_extent = auto\n", dx=0.01)
    with pytest.raises(ValueError):
        assemble(param)


# --------------------------------------------------------------------------- #
# 4. REGRESSION: roi_mode = auto reproduces the no-ROI field to EXACT zero diff
# --------------------------------------------------------------------------- #
def test_roi_auto_is_exact_noop(tmp_path):
    # Reference: the four-file path with no ROI keys at all (current behaviour).
    p_ref = _write_case(tmp_path / "ref", grid_line="grid_extent = auto\n")
    run_ref, res_ref = run_four_file(p_ref, write=False)

    # Same inputs, but roi_mode = auto explicitly set.
    p_roi = _write_case(tmp_path / "roi", grid_line="grid_extent = auto\n",
                        extra="roi_mode = auto\n")
    run_roi, res_roi = run_four_file(p_roi, write=False)

    # Identical resolved grid, no tracks dropped, and byte-identical fields.
    np.testing.assert_array_equal(run_ref.grid.domain_min, run_roi.grid.domain_min)
    np.testing.assert_array_equal(run_ref.grid.domain_max, run_roi.grid.domain_max)
    assert run_roi.meta["n_particles_dropped_roi"] == 0
    assert run_ref.meta["n_particles_dropped_roi"] == 0
    assert len(res_ref) == len(res_roi) == len(TIMES)

    max_diff = 0.0
    for a, b in zip(res_ref, res_roi):
        assert np.array_equal(a.classification, b.classification)
        d = float(np.max(np.abs(a.velocity - b.velocity)))
        max_diff = max(max_diff, d)
        assert np.max(np.abs(a.velocity)) > 1e-3      # real physics ran
    print(f"\nroi_mode=auto regression: max field |diff| = {max_diff:.3e} (must be 0)")
    assert max_diff == 0.0
