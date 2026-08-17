"""Generate the tiny self-contained synthetic-sphere example (two-file interface).

A rigid sphere (r = 5 mm) translates at a constant U = (0, 20, 0) mm/s through a
quiescent fluid; the analytic lab-frame Stokes field around it is sampled as
scattered tracks. Produces, next to this script:

    geometry.dat                 # shape header + pose rows  (position file)
    kinematics.dat               # body U, omega per timestep
    particles/particles_0000{1,2,3}.dat

These are read by the public two-file path:  python run_ledm.py --four-file
configs/example_synthetic_sphere.txt   (from the package root).

The files are already shipped in the ZIP; re-run this only to regenerate them.
Deterministic (fixed RNG seed) so the shipped files are reproducible byte-for-byte.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # package root (has ccmplus/)
sys.path.insert(0, str(ROOT))
from ccmplus.synth.stokes_sphere import stokes_sphere_lab_frame

R_MM = 5.0                                  # sphere radius
U = np.array([0.0, 20.0, 0.0])              # mm/s, constant rise along +y
TS = np.array([0.0, 0.05, 0.10])            # s  (3 snapshots; dy = U*dt = 0,1,2 mm)
C0 = np.array([0.0, 0.0, 0.0])              # center at t=0
NP = 2000                                   # tracks per snapshot
HALF = 16.0                                 # sample box half-width about the center (mm)
SEED = 12345


def sample_outside(center, n, rng):
    pts = np.empty((0, 3))
    while len(pts) < n:
        cand = center + rng.uniform(-HALF, HALF, size=(2 * n, 3))
        cand = cand[np.linalg.norm(cand - center, axis=1) > R_MM + 1e-6]
        pts = np.vstack([pts, cand])
    return pts[:n]


def main():
    rng = np.random.default_rng(SEED)
    (HERE / "particles").mkdir(exist_ok=True)

    centers = C0 + np.outer(TS, U / np.array([1, 1, 1]))    # linear translation
    # geometry file (sphere; identity orientation -> alpha=beta=gamma=0)
    with open(HERE / "geometry.dat", "w") as f:
        f.write("# type: sphere\n# params: r=%.1f\n# units: mm\n"
                "# columns: t x y z alpha beta gamma\n" % R_MM)
        for t, c in zip(TS, centers):
            f.write("%.6f %.6f %.6f %.6f 0.0 0.0 0.0\n" % (t, c[0], c[1], c[2]))
    # kinematics file (constant U, omega = 0)
    with open(HERE / "kinematics.dat", "w") as f:
        f.write("# units: velocity=mm/s, omega=rad/s\n"
                "# columns: t u v w omega_x omega_y omega_z\n")
        for t in TS:
            f.write("%.6f %.6f %.6f %.6f 0.0 0.0 0.0\n" % (t, U[0], U[1], U[2]))
    # particle files (lab-frame Stokes field of the translating sphere)
    for k, (t, c) in enumerate(zip(TS, centers), start=1):
        pos = sample_outside(c, NP, rng)
        vel = stokes_sphere_lab_frame(pos, U, R_MM, c)      # (N,3) mm/s
        with open(HERE / "particles" / f"particles_{k:05d}.dat", "w") as f:
            # Declaring units is optional but strongly recommended: mm positions
            # with m/s velocities parse fine and are numerically indistinguishable
            # from a correct file, so this header is the only thing that can catch
            # that mistake at load time.
            f.write("# units: length=mm, velocity=mm/s\n")
            f.write("x y z u v w\n")
            for p, v in zip(pos, vel):
                f.write("%.6f %.6f %.6f %.6f %.6f %.6f\n" % (p[0], p[1], p[2], v[0], v[1], v[2]))
    print(f"wrote geometry.dat, kinematics.dat, particles/particles_00001..{len(TS):05d}.dat "
          f"({NP} tracks/frame) in {HERE}")


if __name__ == "__main__":
    main()
