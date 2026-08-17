"""Two-file (position + kinematics) input interface: sphere + ellipsoid.

Exercises only the IO/geometry layer (io_ledm, geometry). The frozen solver core
is untouched; these tests assert the layer feeds it the right BodyState.
"""
import numpy as np
import pytest

from ccmplus.io_ledm import assemble, run_four_file, _finite_diff
from ccmplus.geometry import Sphere, Ellipsoid, build_rotation, make_body

R = 2.0
C0 = np.array([0.0, 0.0, 0.0])
U = np.array([0.5, 0.0, 0.0])
TIMES = np.array([0.0, 0.1, 0.2])
DX = 1.5
TOL = 1e-8


def _fluid(x):
    return np.stack([0.3 - 0.1 * x[:, 1], 0.1 * x[:, 0], np.zeros(len(x))], axis=1)


def _write(tmp, shape="sphere", params="r=2", with_kin=True, kin_U=None, kin_om=None,
           angle_unit="deg", extra=""):
    """Author a two-file case. Omit the kinematics file when with_kin=False."""
    tmp.mkdir(parents=True, exist_ok=True)
    pdir = tmp / "particles"; pdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)
    for k, t in enumerate(TIMES, 1):
        c = C0 + U * t
        pts = rng.uniform(-6, 6, size=(1400, 3))
        pts = pts[np.linalg.norm(pts - c, axis=1) > R + 1e-6]
        arr = np.hstack([pts, _fluid(pts)])
        with open(pdir / f"particles_{k:05d}.dat", "w", encoding="utf-8") as fh:
            fh.write("x y z u v w\n")
            for row in arr:
                fh.write(" ".join(f"{v:.17g}" for v in row) + "\n")

    with open(tmp / "position.dat", "w", encoding="utf-8") as fh:
        fh.write(f"# type: {shape}\n# params: {params}\n# units: mm\n")
        fh.write("# columns: t x y z alpha beta gamma\n")
        for t in TIMES:
            c = C0 + U * t
            fh.write(f"{t:.17g} {c[0]:.17g} {c[1]:.17g} {c[2]:.17g} 0 0 0\n")

    kin_line = ""
    if with_kin:
        Us = kin_U if kin_U is not None else [U] * len(TIMES)
        oms = kin_om if kin_om is not None else [[0, 0, 0]] * len(TIMES)
        with open(tmp / "kinematics.dat", "w", encoding="utf-8") as fh:
            fh.write("# units: velocity=mm/s, omega=rad/s\n")
            fh.write("# columns: t u v w omega_x omega_y omega_z\n")
            for i, t in enumerate(TIMES):
                u_, o_ = Us[i], oms[i]
                fh.write(f"{t:.17g} {u_[0]:.17g} {u_[1]:.17g} {u_[2]:.17g} "
                         f"{o_[0]:.17g} {o_[1]:.17g} {o_[2]:.17g}\n")
        kin_line = "kinematics_file = kinematics.dat\n"

    (tmp / "params.txt").write_text(
        "geometry_file = position.dat\n" + kin_line +
        "particles_dir = particles\nparticles_pattern = particles_*.dat\n"
        "grid_extent = -6 6 -6 6 -6 6\n"
        # constraint_div_tol is relaxed on purpose: this case exercises the
        # input contract / wiring, not constraint convergence, so it uses a
        # cheap solve. Constraint quality is asserted with tight solves and
        # normalized tolerances in test_solver_onefluid.py.
        f"dx = {DX}\nkappa = 1.0\nminres_tol = {TOL}\nminres_maxit = 4000\nconstraint_div_tol = 1.0\n"
        "boundary_constraints = on\ninterp_kernel = wide\nlambda_c = 0.0\n"
        "sigma_u = 0.01\nsigma_gamma = 0.5\n"
        f"length_unit = mm\ntime_unit = s\nangle_unit = {angle_unit}\n"
        "euler_seq = ZYX\nhandedness = right\nomega_frame = world\n"
        "warm_start = true\ncase_name = tf\n" + extra, encoding="utf-8")
    return tmp / "params.txt"


# --------------------------------------------------------------------------- #
# 1. position-only sphere == both-files sphere (kinematics = differenced pos)
# --------------------------------------------------------------------------- #
def test_position_only_sphere_regression(tmp_path):
    p1 = _write(tmp_path / "po", shape="sphere", with_kin=False)
    run1, res1 = run_four_file(p1, write=False)
    assert run1.meta["velocity_source"] == "position_diff"

    centers = np.array([C0 + U * t for t in TIMES])
    Ud = _finite_diff(centers, TIMES)                     # what the single-file path uses
    p2 = _write(tmp_path / "bf", shape="sphere", with_kin=True,
                kin_U=[Ud[i] for i in range(len(TIMES))],
                kin_om=[[0, 0, 0]] * len(TIMES))
    run2, res2 = run_four_file(p2, write=False)
    assert run2.meta["velocity_source"] == "kinematics"
    maxd = 0.0
    for a, b in zip(res1, res2):
        assert np.array_equal(a.classification, b.classification)
        assert np.array_equal(a.velocity, b.velocity)
        maxd = max(maxd, float(np.max(np.abs(a.velocity - b.velocity))))
    print(f"\nposition-only vs single-file sphere: max |dv| = {maxd:.3e} (bit-for-bit)")


# --------------------------------------------------------------------------- #
# 2. non-sphere position-only -> clear error
# --------------------------------------------------------------------------- #
def test_non_sphere_position_only_errors(tmp_path):
    p = _write(tmp_path, shape="ellipsoid", params="a=2 b=2 c=2", with_kin=False)
    with pytest.raises(ValueError, match="sphere only.*kinematics_file|kinematics_file"):
        assemble(p)


# --------------------------------------------------------------------------- #
# 3. both files -> kinematics wins (omega from kinematics, not differencing)
# --------------------------------------------------------------------------- #
def test_both_files_uses_kinematics(tmp_path):
    kin_om = [[0.0, 0.0, 0.5]] * len(TIMES)      # nonzero; position-diff would give 0
    p = _write(tmp_path, shape="sphere", with_kin=True, kin_om=kin_om)
    run = assemble(p)
    assert run.meta["velocity_source"] == "kinematics"
    for fr in run.frames:
        np.testing.assert_allclose(fr.body.omega_s, [0, 0, 0.5])   # from kinematics
        np.testing.assert_allclose(fr.body.U_s, U)                 # kinematics velocity


# --------------------------------------------------------------------------- #
# 4. ellipsoid SDF analytic spot-checks
# --------------------------------------------------------------------------- #
def test_ellipsoid_sdf_spotchecks():
    a, b, c = 1.0, 2.0, 3.0
    el = Ellipsoid(a, b, c)
    surf = np.array([[a, 0, 0], [-a, 0, 0], [0, b, 0], [0, -b, 0], [0, 0, c], [0, 0, -c]])
    np.testing.assert_allclose(el.signed_distance(surf), np.zeros(6), atol=1e-7)      # on surface -> 0
    np.testing.assert_allclose(el.signed_distance(np.array([[0.0, 0, 0]])),
                               [-min(a, b, c)], atol=1e-6)                            # center -> -min axis
    np.testing.assert_allclose(el.signed_distance(np.array([[2 * a, 0, 0]])), [a], atol=1e-6)  # outside on-axis
    assert el.signed_distance(np.array([[0.5 * a, 0, 0]]))[0] < 0                     # inside
    assert el.signed_distance(np.array([[3 * a, 0, 0]]))[0] > 0                       # outside


# --------------------------------------------------------------------------- #
# 5. degenerate ellipsoid a=b=c=R reproduces the sphere, bit-for-bit
# --------------------------------------------------------------------------- #
def test_degenerate_ellipsoid_equals_sphere(tmp_path):
    rng = np.random.default_rng(3)
    pts = rng.uniform(-5, 5, size=(600, 3))
    np.testing.assert_allclose(Ellipsoid(R, R, R).signed_distance(pts),
                               Sphere(R).signed_distance(pts), atol=1e-8)
    p_s = _write(tmp_path / "s", shape="sphere", params=f"r={R}", with_kin=True)
    p_e = _write(tmp_path / "e", shape="ellipsoid", params=f"a={R} b={R} c={R}", with_kin=True)
    _, rs = run_four_file(p_s, write=False)
    _, re = run_four_file(p_e, write=False)
    maxd = 0.0
    for a, b in zip(rs, re):
        assert np.array_equal(a.classification, b.classification)
        assert np.array_equal(a.velocity, b.velocity)
        maxd = max(maxd, float(np.max(np.abs(a.velocity - b.velocity))))
    print(f"\ndegenerate ellipsoid a=b=c=R vs sphere: max |dv| = {maxd:.3e} (bit-for-bit)")


# --------------------------------------------------------------------------- #
# 6. Euler unit (deg vs rad) rotates the body as expected
# --------------------------------------------------------------------------- #
def test_euler_unit_switch():
    # In seq "ZYX" the FIRST angle is the Z rotation. deg 90 == rad pi/2, x -> y.
    np.testing.assert_allclose(build_rotation([90, 0, 0], "ZYX", "deg"),
                               build_rotation([np.pi / 2, 0, 0], "ZYX", "rad"), atol=1e-12)
    np.testing.assert_allclose(build_rotation([90, 0, 0], "ZYX", "deg") @ [1.0, 0, 0],
                               [0, 1, 0], atol=1e-12)
    # same angle VALUE, deg vs rad -> different body rotation (unit is honoured)
    assert not np.allclose(build_rotation([90, 0, 0], "ZYX", "deg"),
                           build_rotation([90, 0, 0], "ZYX", "rad"))
    # through make_body: ellipsoid a=1,b=3 rotated +90deg about z puts the b-axis on x
    el = Ellipsoid(1.0, 3.0, 2.0)
    bod = make_body(el, [0, 0, 0], build_rotation([90, 0, 0], "ZYX", "deg"),
                    [0, 0, 0], [0, 0, 0], 0.5)
    assert abs(bod.sdf_fn(np.array([[3.0, 0, 0]]), bod)[0]) < 1e-6      # on surface now
    bod0 = make_body(el, [0, 0, 0], np.eye(3), [0, 0, 0], [0, 0, 0], 0.5)
    assert bod0.sdf_fn(np.array([[3.0, 0, 0]]), bod0)[0] > 1.5          # far outside unrotated


# --------------------------------------------------------------------------- #
# 7. cylinder header -> clear "not yet implemented"
# --------------------------------------------------------------------------- #
def test_cylinder_rejected(tmp_path):
    p = _write(tmp_path, shape="cylinder", params="r=2 d=4", with_kin=True)
    with pytest.raises(ValueError, match="not yet implemented"):
        assemble(p)


# --------------------------------------------------------------------------- #
# Deprecated enable_lema alias
# --------------------------------------------------------------------------- #
def test_enable_lema_alias_still_works_but_warns(tmp_path):
    """Old configs keep working; the rename is announced, not enforced."""
    from ccmplus.config import Config

    with pytest.warns(DeprecationWarning, match="enable_lema"):
        cfg = Config(domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
                     delta=1.0, enable_lema=False)
    assert cfg.boundary_constraints is False
    # The resolved value is mirrored back onto the old attribute so code that
    # still reads config.enable_lema sees the truth rather than None.
    assert cfg.enable_lema is False

    with pytest.warns(DeprecationWarning):
        cfg_on = Config(domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
                        delta=1.0, enable_lema=True)
    assert cfg_on.boundary_constraints is True


def test_boundary_constraints_is_the_default_and_does_not_warn(recwarn):
    from ccmplus.config import Config

    cfg = Config(domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0), delta=1.0)
    assert cfg.boundary_constraints is True
    assert cfg.enable_lema is True          # mirrored, for backwards compatibility
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_param_file_enable_lema_maps_to_boundary_constraints():
    """The deprecated PARAMETER-FILE key maps across too, with a warning."""
    from ccmplus.io_ledm import _resolve_boundary_constraints

    with pytest.warns(DeprecationWarning, match="enable_lema"):
        assert _resolve_boundary_constraints({"enable_lema": False}) is False

    # New key wins when both are present.
    with pytest.warns(DeprecationWarning, match="takes precedence"):
        assert _resolve_boundary_constraints(
            {"enable_lema": False, "boundary_constraints": True}) is True

    # Neither key set -> default on, no warning.
    assert _resolve_boundary_constraints({}) is True
