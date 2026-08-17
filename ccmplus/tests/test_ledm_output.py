"""ASCII .dat output (io_output) + its wiring into the four-file path.

The load-bearing test is the round trip with an ASYMMETRIC field: u = 100*i +
10*j + k makes every node unique, so a wrong I/J/K reversal (which loads without
error but silently scrambles) is caught. It runs for both dat_order = C and F.
"""
import re

import numpy as np
import pytest

from ccmplus.config import ReconstructionResult, BodyState
from ccmplus.io_output import (write_dat, fields_from_result, snapshot_from_grid,
                               body_frame_velocity)
from ccmplus.io_ledm import run_four_file

# Reuse the sphere four-file case author from the ROI test suite (DRY).
from ccmplus.tests.test_ledm_roi import _write_case, TIMES


# --------------------------------------------------------------------------- #
# A minimal Tecplot POINT parser (header + numeric block)
# --------------------------------------------------------------------------- #
def _parse_tecplot(path):
    I = J = K = None
    varnames = None
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            up = s.upper()
            if up.startswith("TITLE"):
                continue
            if up.startswith("VARIABLES"):
                varnames = re.findall(r'"([^"]*)"', s)
                continue
            if up.startswith("ZONE"):
                I = int(re.search(r"\bI\s*=\s*(\d+)", s).group(1))
                J = int(re.search(r"\bJ\s*=\s*(\d+)", s).group(1))
                K = int(re.search(r"\bK\s*=\s*(\d+)", s).group(1))
                continue
            rows.append([float(v) for v in s.split()])
    return I, J, K, varnames, np.array(rows)


def _tecplot_field3d(col, I, J, K, order):
    """Recover the original (Nx,Ny,Nz) field from a Tecplot POINT column.

    Tecplot POINT fills I fastest -> the flat column is the Fortran ravel of an
    (I,J,K) array. For order='F' that (I,J,K)=(Nx,Ny,Nz) is the field directly;
    for order='C' the zone dims are reversed, so un-transpose to (Nx,Ny,Nz).
    """
    arr_tec = col.reshape((I, J, K), order="F")
    return arr_tec if order == "F" else arr_tec.transpose(2, 1, 0)


def _meta(dims):
    Nx, Ny, Nz = dims
    return {
        "case": "synth", "time": 0.25, "time_unit": "s", "dx": 1.0,
        "length_unit": "mm", "bounds_min": (0.0, 0.0, 0.0),
        "bounds_max": (float(Nx - 1), float(Ny - 1), float(Nz - 1)),
        "roi_mode": "box", "n_nodes": Nx * Ny * Nz, "grid_dims": dims,
        "version": "LE-DM v2",
    }


# --------------------------------------------------------------------------- #
# 1. ROUND TRIP (most important): asymmetric field, both orders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", ["C", "F"])
def test_tecplot_roundtrip_asymmetric(tmp_path, order):
    Nx, Ny, Nz = 3, 4, 5          # deliberately all different
    ii, jj, kk = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz),
                             indexing="ij")
    u = (100 * ii + 10 * jj + kk).astype(float)   # unique per node
    v = 1000.0 + u
    w = -u
    x, y, z = ii * 1.0, jj * 2.0, kk * 3.0        # unique coords too

    def rav(a):
        return a.ravel(order=order)

    dims = (Nx, Ny, Nz)
    coords = np.column_stack([rav(x), rav(y), rav(z)])
    fields = {"u": rav(u), "v": rav(v), "w": rav(w)}
    path = tmp_path / f"synth_{order}.dat"
    write_dat(path, coords, fields, dims, _meta(dims),
              flavor="tecplot", precision=12, order=order)

    I, J, K, varnames, data = _parse_tecplot(path)
    # The reversal must be correct or the file is silently scrambled.
    if order == "F":
        assert (I, J, K) == (Nx, Ny, Nz)
    else:
        assert (I, J, K) == (Nz, Ny, Nx)
    assert varnames == ["x", "y", "z", "u", "v", "w"]

    cols = {n: data[:, idx] for idx, n in enumerate(varnames)}
    np.testing.assert_array_equal(_tecplot_field3d(cols["u"], I, J, K, order), u)
    np.testing.assert_array_equal(_tecplot_field3d(cols["v"], I, J, K, order), v)
    np.testing.assert_array_equal(_tecplot_field3d(cols["w"], I, J, K, order), w)
    np.testing.assert_array_equal(_tecplot_field3d(cols["x"], I, J, K, order), x)
    np.testing.assert_array_equal(_tecplot_field3d(cols["y"], I, J, K, order), y)
    np.testing.assert_array_equal(_tecplot_field3d(cols["z"], I, J, K, order), z)
    print(f"\n[roundtrip order={order}] I,J,K={I},{J},{K}  "
          f"u[0,0,0..2]={u[0,0,:3]}  recovered exactly")


# --------------------------------------------------------------------------- #
# 2. Column set is derived from the result dict, in a stable order
# --------------------------------------------------------------------------- #
def test_columns_match_result_fields():
    ng = 24
    res = ReconstructionResult(
        velocity=np.arange(3 * ng, dtype=float).reshape(ng, 3),
        classification=np.tile(np.array([-1, 0, 1], np.int8), ng // 3),
        residual=0.0, iterations=1, converged=True,
    )
    fields = fields_from_result(res)
    # velocity split into u,v,w first, then the 1-D per-node classification.
    assert list(fields.keys()) == ["u", "v", "w", "classification"]
    np.testing.assert_array_equal(fields["u"], res.velocity[:, 0])
    np.testing.assert_array_equal(fields["classification"],
                                  res.classification.astype(float))


# --------------------------------------------------------------------------- #
# 3. plain flavor parses cleanly with np.loadtxt(comments='#')
# --------------------------------------------------------------------------- #
def test_plain_flavor_loadtxt(tmp_path):
    dims = (2, 3, 4)
    ng = dims[0] * dims[1] * dims[2]
    coords = np.random.default_rng(1).uniform(-1, 1, size=(ng, 3))
    fields = {"u": np.arange(ng, dtype=float),
              "classification": np.zeros(ng)}
    path = tmp_path / "plain.dat"
    write_dat(path, coords, fields, dims, _meta(dims),
              flavor="plain", precision=9, order="C")
    arr = np.loadtxt(path, comments="#")
    assert arr.shape == (ng, 5)           # x y z u classification
    np.testing.assert_allclose(arr[:, 3], fields["u"])
    # The one non-'#' header line is the column names, commented.
    first_noncomment = next(l for l in open(path, encoding="utf-8")
                            if not l.startswith("#"))
    assert first_noncomment.split()[0].replace("-", "").split(".")[0].isdigit()


# --------------------------------------------------------------------------- #
# 4. Provenance header carries time, dx, units, bounds
# --------------------------------------------------------------------------- #
def test_provenance_header(tmp_path):
    dims = (2, 2, 2)
    ng = 8
    coords = np.zeros((ng, 3))
    fields = {"u": np.zeros(ng), "v": np.zeros(ng), "w": np.zeros(ng)}
    path = tmp_path / "prov.dat"
    write_dat(path, coords, fields, dims, _meta(dims),
              flavor="tecplot", precision=9, order="C")
    text = path.read_text(encoding="utf-8")
    for token in ("# time: 0.25 [s]", "# dx: 1.0", "# length_unit: mm",
                  "# bounds_min:", "# bounds_max:", "# roi_mode: box",
                  "# LE-DM v2"):
        assert token in text, f"missing provenance token: {token!r}"


# --------------------------------------------------------------------------- #
# 5. REGRESSION: default (no new keys) -> identical .npz, no .dat produced
# --------------------------------------------------------------------------- #
def test_default_is_npz_only_and_unchanged(tmp_path):
    # Default run (no output_* keys): npz only.
    out_def = tmp_path / "def_out"
    p_def = _write_case(tmp_path / "def", grid_line="grid_extent = auto\n",
                        extra=f"output_dir = {out_def.as_posix()}\ncase_name = c\n")
    run_four_file(p_def, write=True)
    npz_files = sorted(out_def.glob("*.npz"))
    dat_files = sorted(out_def.glob("*.dat"))
    assert len(npz_files) == len(TIMES)
    assert dat_files == []                       # no .dat at default

    # output_format = both: npz must be byte-identical to the default run.
    out_both = tmp_path / "both_out"
    p_both = _write_case(tmp_path / "both", grid_line="grid_extent = auto\n",
                         extra=f"output_dir = {out_both.as_posix()}\ncase_name = c\n"
                               "output_format = both\n")
    run_four_file(p_both, write=True)
    npz_both = sorted(out_both.glob("*.npz"))
    dat_both = sorted(out_both.glob("*.dat"))
    assert len(npz_both) == len(TIMES)
    assert len(dat_both) == len(TIMES)           # .dat now present
    for a, b in zip(npz_files, npz_both):
        da, db = np.load(a), np.load(b)
        for key in ("nodes", "velocity", "classification", "t"):
            assert np.array_equal(da[key], db[key]), f"{key} differs in {a.name}"
    print(f"\n[npz regression] {len(npz_files)} snapshots byte-identical npz "
          f"with output_format default vs both; default produced 0 .dat")


# --------------------------------------------------------------------------- #
# 6. End-to-end: four-file .dat ordering recovers the grid field (both orders)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", ["C", "F"])
def test_four_file_dat_roundtrip_matches_grid(tmp_path, order):
    out = tmp_path / f"out_{order}"
    p = _write_case(
        tmp_path / f"case_{order}", grid_line="grid_extent = auto\n",
        extra=f"output_dir = {out.as_posix()}\ncase_name = s\n"
              f"output_format = dat\ndat_flavor = tecplot\ndat_order = {order}\n"
              "dat_precision = 12\n")
    run, results = run_four_file(p, write=True)

    dims = tuple(int(d) for d in run.grid.shape)
    # canonical (i-fastest) reference field for step 0
    res0 = results[0]
    u_ref = res0.velocity[:, 0].reshape(dims, order="F")
    x_ref = run.grid.nodes[:, 0].reshape(dims, order="F")

    dat0 = out / "s_step00000.dat"
    I, J, K, varnames, data = _parse_tecplot(dat0)
    cols = {n: data[:, idx] for idx, n in enumerate(varnames)}
    u_dat = _tecplot_field3d(cols["u"], I, J, K, order)
    x_dat = _tecplot_field3d(cols["x"], I, J, K, order)
    np.testing.assert_allclose(u_dat, u_ref, rtol=0, atol=1e-9)
    np.testing.assert_allclose(x_dat, x_ref, rtol=0, atol=1e-9)
    # classification column present and derived (not hard-coded away)
    assert "classification" in varnames


# --------------------------------------------------------------------------- #
# 7. Body-frame transform: pure translation subtracts the constant U exactly
# --------------------------------------------------------------------------- #
def test_body_frame_pure_translation():
    body = BodyState(X_s=np.array([1.0, 2.0, 3.0]), U_s=np.array([3.0, -2.0, 5.0]),
                     omega_s=np.zeros(3), radius=1.0, sigma_s=0.1)
    rng = np.random.default_rng(0)
    nodes = rng.uniform(-5, 5, size=(64, 3))
    v_lab = rng.uniform(-1, 1, size=(64, 3))
    v_rel = body_frame_velocity(v_lab, nodes, body)
    # omega = 0 -> u_body is the constant U_s everywhere, position-independent.
    np.testing.assert_allclose(v_rel, v_lab - body.U_s, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# 8. Body-frame transform: nonzero omega removes omega x r at an off-centre node
# --------------------------------------------------------------------------- #
def test_body_frame_omega_removes_rotation():
    Xs = np.array([1.0, 2.0, 3.0])
    omega = np.array([0.0, 0.0, 2.0])
    body = BodyState(X_s=Xs, U_s=np.zeros(3), omega_s=omega, radius=1.0, sigma_s=0.1)
    x = np.array([[3.0, 2.0, 3.0]])          # r = x - Xs = (2, 0, 0)
    # omega x r = (0,0,2) x (2,0,0) = (0, 4, 0)  -> a co-rotating solid node (v_lab
    # = u_body) must map to exactly zero in the body frame.
    u_body_expected = np.cross(omega, x[0] - Xs)
    np.testing.assert_allclose(u_body_expected, [0.0, 4.0, 0.0], atol=0)
    v_lab = np.array([[10.0, 4.0, 0.0]])
    v_rel = body_frame_velocity(v_lab, x, body)
    np.testing.assert_allclose(v_rel[0], [10.0, 0.0, 0.0], rtol=0, atol=1e-12)
    # a node moving exactly as the body -> zero relative velocity
    np.testing.assert_allclose(
        body_frame_velocity(u_body_expected[None, :], x, body)[0],
        [0.0, 0.0, 0.0], atol=1e-12)


# --------------------------------------------------------------------------- #
# 9. CORRECTNESS on the sphere case: solid+shell |v_rel| below solver tolerance
# --------------------------------------------------------------------------- #
def test_body_frame_solid_shell_below_tol(tmp_path):
    out = tmp_path / "bf"
    p = _write_case(
        tmp_path / "case", grid_line="grid_extent = -6 6 -6 6 -6 6\n",
        extra=f"output_dir = {out.as_posix()}\ncase_name = s\n"
              "output_format = npz\noutput_frame = body\n")
    run, _ = run_four_file(p, write=True)
    U0 = max(float(np.linalg.norm(f.body.U_s)) for f in run.frames)
    worst = 0.0
    for i in range(len(run.frames)):
        d = np.load(out / f"s_step{i:05d}.npz")
        v_rel, cls = d["velocity"], d["classification"]
        pinned = cls <= 0                     # solid (-1) + shell (0)
        assert pinned.any()
        worst = max(worst, float(np.max(np.abs(v_rel[pinned]))))
        # the body-frame archive is self-describing
        assert str(d["output_frame"]) == "body"
        np.testing.assert_allclose(d["body_U"], run.frames[i].body.U_s, atol=0)
    print(f"\n[body-frame correctness] max |v_rel| over solid+shell nodes = "
          f"{worst:.3e}  (U0={U0:g}, ratio {worst/U0:.2e}, tol ~1e-6*U0)")
    assert worst < 1e-6 * U0


# --------------------------------------------------------------------------- #
# 10. REGRESSION: output_frame = lab (default) reproduces npz byte-for-byte
# --------------------------------------------------------------------------- #
def test_output_frame_lab_matches_default(tmp_path):
    out_def = tmp_path / "def"
    out_lab = tmp_path / "lab"
    p_def = _write_case(tmp_path / "d", grid_line="grid_extent = auto\n",
                        extra=f"output_dir = {out_def.as_posix()}\ncase_name = c\n")
    p_lab = _write_case(tmp_path / "l", grid_line="grid_extent = auto\n",
                        extra=f"output_dir = {out_lab.as_posix()}\ncase_name = c\n"
                              "output_frame = lab\n")
    run_four_file(p_def, write=True)
    run_four_file(p_lab, write=True)
    files = sorted(out_def.glob("*.npz"))
    assert len(files) == len(TIMES)
    for a in files:
        da, db = np.load(a), np.load(out_lab / a.name)
        assert set(da.files) == {"nodes", "velocity", "classification", "t"}
        assert set(db.files) == set(da.files)          # lab adds no frame keys
        for key in da.files:
            assert np.array_equal(da[key], db[key]), f"{key} differs in {a.name}"


# --------------------------------------------------------------------------- #
# 11. comoving_coords shifts nodes by -(X_s - X_s0), velocity unchanged
# --------------------------------------------------------------------------- #
def test_comoving_coords_shifts_nodes(tmp_path):
    grid_line = "grid_extent = -6 6 -6 6 -6 6\n"
    common = "output_format = npz\noutput_frame = body\n"
    out_b, out_c = tmp_path / "b", tmp_path / "c"
    p_b = _write_case(tmp_path / "cb", grid_line=grid_line,
                      extra=f"output_dir = {out_b.as_posix()}\ncase_name = s\n" + common)
    p_c = _write_case(tmp_path / "cc", grid_line=grid_line,
                      extra=f"output_dir = {out_c.as_posix()}\ncase_name = s\n"
                            + common + "comoving_coords = true\n")
    run_b, _ = run_four_file(p_b, write=True)
    run_four_file(p_c, write=True)
    X_s0 = np.asarray(run_b.frames[0].body.X_s, dtype=float)
    for i, f in enumerate(run_b.frames):
        db = np.load(out_b / f"s_step{i:05d}.npz")
        dc = np.load(out_c / f"s_step{i:05d}.npz")
        shift = np.asarray(f.body.X_s, dtype=float) - X_s0
        np.testing.assert_allclose(dc["nodes"], db["nodes"] - shift, rtol=0, atol=1e-12)
        # velocity is identical to the (non-comoving) body-frame result
        np.testing.assert_array_equal(dc["velocity"], db["velocity"])
    # step 0: zero shift -> nodes identical to the fixed grid
    d0b = np.load(out_b / "s_step00000.npz")
    d0c = np.load(out_c / "s_step00000.npz")
    np.testing.assert_array_equal(d0c["nodes"], d0b["nodes"])
