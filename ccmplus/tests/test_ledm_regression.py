"""Phase 5 regression: the four-file input path must reproduce the sphere
reconstruction of the original sphere driver to solver tolerance.

Both paths are driven through the UNCHANGED ccmplus solver on identical grid,
config, particles and kinematics. The ONLY difference is how occupancy is built:

  * reference  -> BodyState with sdf_fn=None  = the analytic sphere distance the
                  existing driver uses (drivers/sphere.py / run_ledm.py:
                  BodyState(X_s, U_s, omega_s, radius, sigma_s)).
  * four-file  -> BodyState.sdf_fn from the geometry layer (Sphere + pose),
                  produced by reading the three data files + parameter file.

Because the geometry-layer Sphere SDF is byte-for-byte the analytic sphere under
identity pose, the reconstructed fields must agree to machine precision, well
inside the MINRES tolerance. If they do not, the new layer perturbs the physics.
"""
import numpy as np
import pytest

from ccmplus.config import BodyState
from ccmplus.reconstruct import CCMPlus
from ccmplus.io_ledm import run_four_file, assemble


# --------------------------------------------------------------------------- #
# Synthetic sphere case (source data shared by both paths)
# --------------------------------------------------------------------------- #
R = 2.0
C0 = np.array([0.0, 0.0, 0.0])
U = np.array([0.5, 0.0, 0.0])          # constant -> d/dt(center) == U exactly
TIMES = np.array([0.0, 0.1, 0.2])      # geometry & kinematics share these
GRID_EXTENT = "-6 6 -6 6 -6 6"
DX = 1.5
SOLVER_TOL = 1e-8                       # MINRES rtol used in the param file


def _fluid_field(x):
    """Smooth, divergence-free fluid velocity: uniform + rigid swirl about z."""
    return np.stack([0.3 - 0.1 * x[:, 1], 0.1 * x[:, 0], np.zeros(len(x))], axis=1)


def _write_case(tmp):
    """Author the three data files + parameter file from the source data."""
    tmp.mkdir(parents=True, exist_ok=True)
    pdir = tmp / "particles"
    pdir.mkdir(exist_ok=True)
    rng = np.random.default_rng(7)

    for k, t in enumerate(TIMES, start=1):
        center = C0 + U * t
        pts = rng.uniform(-6, 6, size=(1200, 3))
        outside = np.linalg.norm(pts - center, axis=1) > R + 1e-6   # natural void
        pts = pts[outside]
        vel = _fluid_field(pts)
        arr = np.hstack([pts, vel])
        with open(pdir / f"particles_{k:05d}.dat", "w", encoding="utf-8") as fh:
            fh.write("x y z u v w\n")
            for row in arr:
                fh.write(" ".join(f"{v:.17g}" for v in row) + "\n")

    with open(tmp / "geometry.dat", "w", encoding="utf-8") as fh:
        fh.write("# type: sphere\n")
        fh.write(f"# params: r={R}\n")
        fh.write("# units: mm\n")
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
        f"grid_extent = {GRID_EXTENT}\n"
        f"dx = {DX}\n"
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
        "length_unit = mm\n"
        "time_unit = s\n"
        "angle_unit = deg\n"
        "euler_seq = ZYX\n"
        "handedness = right\n"
        "omega_frame = world\n"
        "warm_start = true\n"
        "case_name = regr\n",
        encoding="utf-8",
    )
    return param


def test_four_file_path_matches_sphere_driver(tmp_path):
    param = _write_case(tmp_path)

    # NEW: four-file path (readers + geometry layer) through the frozen solver.
    run, results_new = run_four_file(param, write=False)
    assert run.meta["type"] == "sphere"
    assert len(results_new) == len(TIMES)

    # REFERENCE: identical inputs, but occupancy = the analytic sphere the
    # original sphere driver uses (sdf_fn=None). Same grid/config/particles.
    ref_solver = CCMPlus(run.config, run.grid)
    results_ref = []
    for frame in run.frames:
        b = frame.body
        ref_body = BodyState(X_s=b.X_s.copy(), U_s=b.U_s.copy(),
                             omega_s=b.omega_s.copy(), radius=b.radius,
                             sigma_s=b.sigma_s)          # sdf_fn=None -> analytic sphere
        ref_frame = type(frame)(positions=frame.positions, velocities=frame.velocities,
                                uncertainties=frame.uncertainties, body=ref_body,
                                t=frame.t)
        results_ref.append(ref_solver.reconstruct(ref_frame))

    # Occupancy identical, and reconstruction identical to machine precision.
    max_diff = 0.0
    worst = None
    for i, (rn, rf) in enumerate(zip(results_new, results_ref)):
        assert np.array_equal(rn.classification, rf.classification)
        d = float(np.max(np.abs(rn.velocity - rf.velocity)))
        if d > max_diff:
            max_diff, worst = d, i
        # the reconstruction must be non-trivial (real physics ran)
        assert np.max(np.abs(rf.velocity)) > 1e-3

    print(f"\nregression max field |Δ| = {max_diff:.3e} at step {worst}  "
          f"(solver tol {SOLVER_TOL:.0e})")
    assert max_diff <= SOLVER_TOL


def test_validation_accepts_independent_omega(tmp_path):
    """Position-file Euler angles and kinematics omega are INDEPENDENT inputs
    (omega is not d(theta)/dt), so the Euler-vs-omega consistency check was removed
    for the two-file interface. A spinning pose with omega = 0 is now ACCEPTED, and
    the body uses the kinematics omega, never a differenced angle rate."""
    _write_case(tmp_path)
    with open(tmp_path / "geometry.dat", "w", encoding="utf-8") as fh:
        fh.write("# type: sphere\n# params: r=2.0\n# units: mm\n")
        fh.write("# columns: t x y z alpha beta gamma\n")
        for t in TIMES:
            c = C0 + U * t
            fh.write(f"{t:.17g} {c[0]:.17g} {c[1]:.17g} {c[2]:.17g} "
                     f"{30.0 * t:.17g} 0 0\n")     # spinning pose; omega stays 0 (independent)
    run = assemble(tmp_path / "params.txt")          # no error: check removed by design
    for fr in run.frames:
        np.testing.assert_allclose(fr.body.omega_s, [0.0, 0.0, 0.0])
