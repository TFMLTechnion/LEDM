# Synthetic-sphere example — expected output

A rigid sphere (radius 5 mm) translates at a constant **U = (0, 20, 0) mm/s** through
quiescent fluid; the analytic lab-frame Stokes field around it is sampled as 2000 tracks
per snapshot over 3 snapshots. This example runs from the ZIP alone (no Zenodo data) and
verifies your install end-to-end.

## Run

From the package root:

```
python run_ledm.py --four-file configs/example_synthetic_sphere.txt
```

Writes `examples/synthetic_sphere/output/synthetic_sphere_step0000{0,1,2}.npz`
(lab-frame `nodes`, `velocity`, `classification`, `t`).

**Runtime: about 1.5 minutes** for the 3 snapshots. Most of that is MINRES. The
shipped config deliberately uses a tight solve (`minres_tol = 1e-11`,
`use_jacobi_precond = on`) because the saddle system reports "converged" on the
scaled system long before the divergence constraint is actually satisfied — at
`minres_tol = 1e-6` this case stops after ~12 iterations with a **9% divergence
error**. Speed here would be bought directly out of the physics.

## What you should see in the log

The run opens with a banner echoing every resolved setting — units first, since
a unit mix-up is the one input error the numbers alone cannot reveal:

```
[LE-DM] ================ run configuration ================
[LE-DM] units      : length=mm  time=s  velocity=mm/s  angle=rad  angular_velocity=rad/s
[LE-DM] geometry   : sphere (r=5)  euler_seq=ZYX  handedness=right  omega_frame=world
[LE-DM] kinematics : velocity_source=kinematics  n_timesteps=3
[LE-DM] grid       : 31x33x31 = 31,713 nodes  dx=1 mm  roi_mode=body
[LE-DM] tracks     : 6,000 kept, 0 dropped (outside ROI)
[LE-DM] LE-DM opts : boundary_constraints=on  kernel=wide  kappa=100  lambda_c=0.05  c_0=1
[LE-DM] ===================================================
step 0: t=0     n_tracks=2000  iters=1343  converged=True  resid=5.18e-06
step 1: t=0.05  n_tracks=2000  iters=1676  converged=True  resid=6.72e-07
step 2: t=0.1   n_tracks=2000  iters=1705  converged=True  resid=3.62e-07
four-file run: 3 timesteps, geometry=sphere -> ...
```

- **3/3 converged = True and NO `RuntimeWarning`** is the pass condition. A
  warning naming `div_rms_norm` or the no-slip residual means the solve did not
  actually meet the constraints, whatever `converged` says.
- Grid = **31,713 nodes** (roi_mode=body, roi_pad=1.0, dx=1.0), classes at step 0:
  solid (C=−1) = 485, shell (C=0) = 254, fluid (C=+1) = 30,974.
- Iteration counts depend on BLAS/LAPACK and numpy version; the constraint
  residuals below are the meaningful check.

## Sanity numbers (step 0, from the npz)

| quantity | value | meaning |
|---|---|---|
| solid+shell mean \|V\| | **20.000 mm/s** | body carries the rigid-body velocity \|U\| = 20 (no-slip) |
| shell max deviation from \|U\| | **< 1e-6 mm/s** | the no-slip rows hold to solver tolerance |
| `Delta*rms(div u)/U_ref` | **~1.5e-4** | dimensionless divergence error per cell (tolerance 1e-3) |
| fluid mean \|V\| | ≈ 3.8 mm/s | small vs 20 (Stokes disturbance decays ~1/r in a small box) |
| global max \|V\| | ≈ 32.4 mm/s | near-wall speed-up |

Quick check:

```python
import numpy as np
d = np.load("examples/synthetic_sphere/output/synthetic_sphere_step00000.npz")
V = np.linalg.norm(d["velocity"], axis=1); c = d["classification"]
print("solid+shell mean|V| =", V[c <= 0].mean())          # 20.000
print("shell max deviation =", np.abs(V[c == 0] - 20).max())  # < 1e-6
```

Exact digits depend on numpy/scipy versions and on BLAS; the tolerances quoted
above hold across them. The input files are deterministic
(fixed RNG seed) and can be regenerated with
`python examples/synthetic_sphere/make_synthetic_sphere.py`.
