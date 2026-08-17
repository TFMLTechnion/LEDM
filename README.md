# LE-DM : Lagrangian-to-Eulerian reconstruction with Dynamic Masking

**LE-DM is an open-source reference implementation of the boundary-aware
Lagrangian-to-Eulerian reconstruction framework introduced by Jose et al.** The
repository is a research tool and development base for reconstructing volumetric
particle-tracking data in the presence of moving solid boundaries. It implements the core
signed-distance dynamic mask, rigid-body boundary constraints, fluid-domain
incompressibility, temporal treatment, and spatial regularization on a fixed Cartesian
grid.

The bundled example demonstrates the input format and reconstruction workflow. It is **not**
a complete reproduction package for every validation case in the accompanying paper — see
[Relationship to the paper](#relationship-to-the-accompanying-paper).

**v1.0 geometry support:** rigid **spheres** and **ellipsoids**. The signed-distance
formulation can in principle be extended to additional geometries, multiple bodies, and
deforming interfaces, but those interfaces are **not** included in this public release.

---

## What LE-DM does

Given scattered particle tracks (positions + velocities) and a prescribed rigid-body
geometry and motion per timestep, LE-DM:

1. Builds a signed-distance field around the body and classifies every grid node as
   **open fluid**, **boundary shell**, or **solid interior**.
2. Fits the open-fluid nodes to the track data while enforcing incompressibility in the
   fluid and the rigid-body velocity on the shell and interior, in one constrained
   saddle-point solve.
3. Returns a divergence-free lab-frame velocity field (optionally re-expressed in the body
   frame) on the fixed grid.

The body region is a natural void in the track field; no separate mask is read. The body
(sphere or ellipsoid) is the single source of truth for occupancy.

---

## Install

Python 3.11+ (developed on CPython 3.11.9). Dependencies: numpy, scipy, matplotlib
(+ pytest for the tests). No compiled extensions.

```
python -m venv venv
# Windows:  venv\Scripts\activate      Linux/macOS:  source venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart — verify your install in seconds

A tiny synthetic translating sphere ships with the code and runs with no external data:

```
python run_ledm.py --four-file configs/example_synthetic_sphere.txt
```

Expect 3 snapshots, all `converged=True`, ~31k grid nodes, about 1.5 minutes of runtime
(the default configs use a tight solver tolerance — see
[Solver convergence](#solver-convergence)). The run prints the resolved units, geometry,
grid size, track count, and active options at startup, so you can confirm the inputs were
read as intended. Expected log and sanity
numbers are in
[`examples/synthetic_sphere/EXPECTED_OUTPUT.md`](examples/synthetic_sphere/EXPECTED_OUTPUT.md)
(body carries `|U|` ≈ 20 mm/s, far-field ≈ 0). Regenerate the inputs any time with
`python examples/synthetic_sphere/make_synthetic_sphere.py`.

Run the tests:

```
python -m pytest tests -q
```

---

## Units — read this before using your own data

**A run uses one coherent unit system.** All velocities are expressed in
`length_unit / time_unit`. With the common choice

```
length_unit = mm
time_unit   = s
```

**every velocity — particle track velocities and body kinematics — must be in mm/s.**
Mixing `mm` positions with `m/s` velocities is a silent factor-of-1000 error and the most
likely way to get a wrong reconstruction. The loader validates the declared units on the
geometry and kinematics files and prints the resolved unit system at startup; check that
line on every new dataset.

---

## The input interface (`--four-file`)

A run is driven by one parameter file that points at the data files. The flag is
`--four-file` because a full run references four kinds of input: geometry/position,
kinematics, particles, and parameters. Full contract:
[`LEDM_input_spec.md`](LEDM_input_spec.md); solver call contract:
[`INTERFACE.md`](INTERFACE.md).

1. **Geometry / position file** — shape header (`# type: sphere|ellipsoid`,
   `# params: r=...` or `a=.. b=.. c=..`, `# units:`,
   `# columns: t x y z alpha beta gamma`) plus one pose row per timestep. Only **sphere**
   and **ellipsoid** are implemented; other headers are rejected with a clear error.
2. **Kinematics file** *(optional for a sphere, required otherwise)* —
   `# columns: t u v w omega_x omega_y omega_z`. When present, it sets the shell velocity
   `U_s(x) = U + omega × (x − X_s)` directly; the trajectory is never differentiated.
3. **Particle files** — one per timestep, `particles_<k>.dat`, first line a column-name
   header `x y z u v w` (optional `su sv sw` uncertainties). `ax ay az` acceleration
   columns are accepted but **reserved/ignored** in v1.0.
4. **Parameter file** — `key = value`; grid/ROI, solver, output, and one declared
   coordinate/rotation convention. See the annotated `configs/*.txt`.

### Body-velocity modes

| Body | `kinematics_file` | Shell velocity source | `omega` |
|---|---|---|---|
| Sphere | omitted | position trajectory differencing | 0 |
| Sphere | provided | kinematics file | from file |
| Ellipsoid / rotating | **required** | kinematics file | from file |

Position-only mode is permitted for a **non-rotating sphere** when you want the body
velocity from finite-difference of the trajectory. A kinematics file may still be supplied
for a sphere (for example, to use a smoothed body velocity) and is **required** for any
non-spherical or rotating body, because `omega` cannot be recovered from the centre
trajectory. If `kinematics_file` is set but the file is missing, the run stops at load time
— there is no silent fallback.

### Boundary-detection uncertainty

`sigma_gamma` is currently a **run-level scalar** proximity-weighting length. Time-varying
body-detection uncertainty is supported by the formulation but is not yet exposed by the
public input interface.

---

## Outputs

Per snapshot, under the config's `output_dir`:

- **`.npz`** (default) — arrays `nodes`, `velocity`, `classification`, `t`. Source of truth
  for `warm_start` and downstream readers.
- **Tecplot `.dat`** (`output_format = dat` or `both`) — `TITLE`/`VARIABLES`/
  `ZONE ... DATAPACKING=POINT`, columns `x y z u v w` + `classification`.

`output_frame = lab` (default) writes the lab-frame velocity as solved; `output_frame =
body` writes `v_rel = v_lab − u_body` (solid/shell → 0) for the flow-around picture. See
`LEDM_input_spec.md` for `dat_flavor`, `dat_order`, `comoving_coords`.

---

## Relationship to the accompanying paper

The paper is the **methodological reference**. This repository provides the core open
implementation and example interfaces. The bundled example demonstrates usage rather than
reproducing every published validation result.

- **Method / citation:** Jose et al., *Dynamic masking for boundary-aware velocity
  reconstruction in volumetric particle tracking with moving solids*, arXiv:2606.25748
  (2026), <https://doi.org/10.48550/arXiv.2606.25748>.
- **Validation datasets:** archived separately on Zenodo,
  <https://doi.org/10.5281/zenodo.21965844> (CC-BY-4.0). Download and unpack under `data/`
  to run the paper-case templates.
- **Paper-case templates:** `paper_templates/` holds illustrative starting configs for the
  CFD-sphere, spheroid, and experiment cases. These are **starting points, not exact
  reproduction recipes** — set `kappa`, `lambda_c`, `sigma_u`, `sigma_gamma`, the ROI, and
  the uncertainty handling from the values reported in the paper for your run.

Manuscript validation scripts (figure/table generation, the Monte Carlo study, and the
parameter sweeps) are not part of this release and may be archived separately.

---

## Solver convergence

The saddle-point system is solved with MINRES. MINRES reports convergence against the
scaled global system, which can be reached **before** the incompressibility constraint is
actually satisfied: a loose tolerance can return `converged=True` while the divergence
error is still large. The shipped configs therefore use a tight setting
(`minres_tol = 1e-11`, `minres_maxit = 20000`, `use_jacobi_precond = on`) that drives the
normalized divergence residual to `~1e-4`. This is why the synthetic example takes about
1.5 minutes rather than seconds. If you loosen `minres_tol` for speed, judge the result by
the reported normalized divergence and constraint residuals, not by the `converged` flag
alone.

## Testing

```
python -m pytest tests -q
```

The suite covers the public input/geometry/ROI/regression path and the core solver
behavior: node classification, mask-aware interpolation, fluid-side divergence stencils,
and the hard-constraint residuals. The solver reports normalized divergence and
constraint-residual diagnostics in its output, so "divergence-free to solver tolerance" has
a concrete, checkable meaning. A clean checkout should be green; any failure is a real
regression.

---

## Repository layout

```
run_ledm.py                     entry point (--four-file interface)
requirements.txt  LICENSE  CITATION.cff  CHANGELOG.md  VERSION
README.md  INTERFACE.md  LEDM_input_spec.md
ccmplus/                        core reconstruction package
  sdf.py  geometry.py           body geometry + signed distance
  classify.py                   solid/shell/fluid classification + transition flags
  interp.py                     mask-aware cubic B-spline particle-to-grid interpolation
  constraints.py                fluid-side divergence + rigid-body kinematic constraints
  prior.py                      temporal prior + newly exposed-node treatment
  operators.py                  spatial / coverage-adaptive regularization
  reconstruct.py                one-snapshot orchestration
  solver.py                     sparse saddle-point assembly + MINRES
  io_ledm.py                    generic public input layer
  synth/                        analytic Stokes-sphere field + track sampler
tests/                          unit-test suite
configs/                        example_synthetic_sphere.txt
examples/synthetic_sphere/      runnable example: inputs, generator, EXPECTED_OUTPUT.md
paper_templates/                illustrative paper-case configs (need Zenodo data)
```

---

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

> Jibu Tom Jose, Arieh Jacobson, Dhanush Vittal Shenoy, Steven H. Frankel, and Omri Ram,
> *Dynamic masking for boundary-aware velocity reconstruction in volumetric particle
> tracking with moving solids*, arXiv:2606.25748 (2026).
> https://doi.org/10.48550/arXiv.2606.25748

If you use the validation datasets, please also cite the Zenodo record:

> LE-DM validation datasets, Zenodo. https://doi.org/10.5281/zenodo.21965844

License: **GPL-3.0** (see [`LICENSE`](LICENSE)). The separate Zenodo **data** is released
under CC-BY-4.0.
