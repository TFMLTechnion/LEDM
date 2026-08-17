# LE-DM — Lagrangian-to-Eulerian reconstruction with Dynamic Masking

LE-DM reconstructs a divergence-free Eulerian velocity field on a regular Cartesian grid
from scattered Lagrangian particle tracks (positions + velocities) around a moving rigid
body, enforcing the no-slip boundary condition on the body surface through a constrained
saddle-point solve. The body (sphere or ellipsoid) is the single source of truth for
occupancy: the grid, the fluid / shell / solid node classification, and the shell velocity
are all derived internally from the geometry and the body kinematics. No separate mask is
ever read — the body region is a natural void in the track field.

This package is self-contained: the solver, a single entry point (`run_ledm.py`),
annotated worked-example configs, a runnable synthetic example, and the unit-test suite.
**No experimental data ships with the code.** The paper datasets live on Zenodo (see
[Reproducing the paper](#reproducing-the-paper)), and no machine-specific paths are baked
into any config or document.

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)):

> Jibu Tom Jose, Arieh Jacobson, Dhanush Vittal Shenoy, Steven H. Frankel, and Omri Ram,
> *Dynamic masking for boundary-aware velocity reconstruction in volumetric particle
> tracking with moving solids*, arXiv:2606.25748 (2026).
> https://doi.org/10.48550/arXiv.2606.25748

If you use the datasets, please also cite the Zenodo record:

> LE-DM validation datasets, Zenodo. https://doi.org/10.5281/zenodo.21965844

License: **GPL-3.0** (see [`LICENSE`](LICENSE)). The separate Zenodo **data** is released
under CC-BY-4.0.

---

## What LE-DM does

- Takes scattered particle tracks and a prescribed rigid-body geometry + motion per timestep.
- Classifies every grid node as **open fluid**, **boundary shell**, or **solid interior**
  from a signed-distance field around the body.
- Fits the fluid nodes to the track data while enforcing incompressibility in the fluid and
  the rigid-body velocity on the shell / interior, all in one constrained solve.
- Returns a divergence-free lab-frame velocity field (optionally re-expressed in the body
  frame) on the fixed grid.

Supported bodies: **sphere** and **ellipsoid**. `cylinder` / `mesh` / `voxel` headers are
rejected with a clear error rather than silently mishandled.

---

## Install

Python 3.11+ (developed on CPython 3.11.9). Dependencies: numpy, scipy, matplotlib
(+ pytest for the tests). No compiled extensions.

```
python -m venv venv
# Windows:  venv\Scripts\activate      Linux/macOS:  source venv/bin/activate
pip install -r requirements.txt
```

The pinned versions **numpy 1.26.4 / scipy 1.12.0** are the reproducibility toolchain; see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Newer numpy/scipy will generally run, but the
step-0 reference numbers are validated against the pinned stack.

---

## Quickstart — verify your install in seconds

A tiny synthetic translating sphere ships in the ZIP and runs with no external data:

```
python run_ledm.py --four-file configs/example_synthetic_sphere.txt
```

Expect 3 snapshots, all `converged=True`, ~31k grid nodes, a few seconds of runtime. The
expected log and sanity numbers are in
[`examples/synthetic_sphere/EXPECTED_OUTPUT.md`](examples/synthetic_sphere/EXPECTED_OUTPUT.md).
As a quick check, step 0 should reproduce:

- 31,713 grid nodes, class split 485 solid / 254 shell / 30,974 fluid
- solid + shell mean `|V|` = 20.0 (body carries `|U|` = 20 mm/s)
- far-field (r > 12) mean `|V|` ≈ 2.2, global max `|V|` ≈ 31.6

Regenerate the inputs any time with
`python examples/synthetic_sphere/make_synthetic_sphere.py`.

Run the tests (optional):

```
python -m pytest ccmplus/tests -q
```

Baseline: **217 passed, 10 skipped, 8 failed**. The 8 failures are known and
pre-existing — they are documented in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). A clean
checkout should match this exactly; any *new* failure is a real regression.

---

## The input interface (`--four-file`)

A run is driven by one parameter file that points at the data files. The CLI flag is
`--four-file` because a full run references four kinds of input: geometry/position,
kinematics, particles, and parameters. Full contract:
[`LEDM_input_spec.md`](LEDM_input_spec.md); solver call contract:
[`INTERFACE.md`](INTERFACE.md). In brief:

1. **Geometry / position file** — shape header (`# type: sphere|ellipsoid`,
   `# params: r=...` or `a=.. b=.. c=..`, `# units:`,
   `# columns: t x y z alpha beta gamma`) plus one pose row per timestep. Only **sphere**
   and **ellipsoid** are implemented.
2. **Kinematics file** *(optional for a sphere, required otherwise)* —
   `# columns: t u v w omega_x omega_y omega_z`; body linear and angular velocity per
   timestep. When present, kinematics sets the shell velocity
   `U_s(x) = U + omega × (x − X_s)` directly (the trajectory is never differentiated).
3. **Particle files** — one per timestep, `particles_<k>.dat`, first line a column-name
   header `x y z u v w` (optional `su sv sw` uncertainties, `ax ay az` acceleration). The
   body region is a natural void; no mask is read.
4. **Parameter file** — `key = value`; grid / ROI, solver, output, and one declared
   coordinate / rotation convention. See the annotated `configs/*.txt`.

All files in a run share one length unit, one time unit, and one Euler convention, declared
once in the parameter file. Load-time checks fail loudly on any mismatch.

### Body-velocity modes — the input-mode rule

There are two ways the body velocity reaches the solver, selected by whether a
`kinematics_file` is set:

| Body | `kinematics_file` | Shell velocity source | `omega` |
|---|---|---|---|
| **Sphere** | omitted | position trajectory differencing | forced to 0 |
| **Sphere** | provided | kinematics file | from file |
| **Ellipsoid / rotating** | **required** | kinematics file | from file |

For a **sphere** you may omit `kinematics_file`: the body velocity is obtained by
differentiating the position trajectory with `omega = 0`. This reproduces the original
single-file sphere behaviour, and it is why `paper_cfd_sphere.txt` sets no
`kinematics_file` and `data/cfd_sphere/` needs no `kinematics.dat`.

For a **non-spherical or rotating body**, `kinematics_file` is **required**: `omega` cannot
be recovered from the centre trajectory alone, and the reader refuses to differentiate a
non-sphere.

**There is no silent fallback.** If `kinematics_file` is set but points at a path that does
not exist, the run stops at load time with a `FileNotFoundError` — it does not fall back to
differencing. See [Troubleshooting](#troubleshooting).

---

## Parameter file and conventions

The parameter file declares the grid / region of interest, the solver settings, the output
format and frame, and exactly one coordinate / rotation convention that every input in the
run is interpreted against. Each shipped `configs/*.txt` is annotated key by key. The full
list of keys and their defaults is in [`LEDM_input_spec.md`](LEDM_input_spec.md).

---

## Outputs

Per snapshot, written under the config's `output_dir`:

- **`.npz`** (default, `output_format = npz`) — arrays `nodes`, `velocity`,
  `classification`, `t`. This is the source of truth for `warm_start` and for downstream
  readers.
- **Tecplot `.dat`** (`output_format = dat` or `both`) — `TITLE` / `VARIABLES` /
  `ZONE ... DATAPACKING=POINT`, columns `x y z u v w` + `classification`.

Frame control:

- `output_frame = lab` (default) writes the reconstructed lab-frame velocity as solved.
- `output_frame = body` writes `v_rel = v_lab − u_body` (solid / shell → 0), giving the
  flow-around-the-body picture.

See `LEDM_input_spec.md` for `dat_flavor`, `dat_order`, and `comoving_coords`.

---

## Reproducing the paper

The paper datasets are **not** bundled. Download them from Zenodo
(<https://doi.org/10.5281/zenodo.21965844>) and unpack under `data/` so the paper configs
resolve. Each case and its input mode:

| Case | Config | Data folder | Body | Input mode | Velocity source | Proximity reweight |
|---|---|---|---|---|---|---|
| CFD rising sphere | `configs/paper_cfd_sphere.txt` | `data/cfd_sphere/` | sphere | position-only | trajectory differencing, `omega = 0` | **ON** (`sigma_gamma = 0.5 mm`) |
| CFD tumbling spheroid | `configs/paper_cfd_spheroid.txt` | `data/cfd_spheroid/` | ellipsoid | two-file | kinematics file (linear + angular) | **OFF** |
| Lab PTV experiment | `configs/paper_experiment.txt` | `data/experiment/` | sphere | two-file | kinematics file | **ON** (`sigma_gamma = 0.5 mm`) |

**Critical setting — proximity reweighting.** The two-file path defaults
`enable_proximity_reweight = false`. The CFD-sphere and experiment cases used it **ON**
(`sigma_gamma = 0.5 mm`); the spheroid used it **OFF**. Every shipped config sets this
explicitly, so the default never decides it for you. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the per-case table and the reasoning.

The paper configs cannot run until the Zenodo data is in place; every other document and
config in the release is data-free.

---

## Troubleshooting

**`FileNotFoundError: kinematics_file is set but not found: ...`**
The parameter file sets `kinematics_file`, but the file is missing. This is intentional —
there is no fallback to trajectory differencing when the key is set. Either place the
kinematics file where the config points, or, for a **sphere only**, remove the
`kinematics_file` line to use position-only mode (`omega = 0`).

**A non-sphere run refuses to start without kinematics.**
Ellipsoid (and any rotating body) requires a `kinematics_file`; `omega` cannot be
differentiated from the centre trajectory. Supply the kinematics file.

**Unit or Euler-convention mismatch at load time.**
All inputs in a run share one length unit, one time unit, and one Euler convention,
declared in the parameter file. Fix the offending file's header to match the declared
convention.

**Unsupported geometry.**
Only `sphere` and `ellipsoid` are implemented. `cylinder` / `mesh` / `voxel` headers are
rejected by design.

**Test failures.**
217 pass / 10 skip / 8 fail is the expected baseline; those 8 are pre-existing (see
`REPRODUCIBILITY.md`). Investigate only failures beyond that set.

---

## Repository layout

```
run_ledm.py                     entry point  (--four-file interface)
requirements.txt  LICENSE  CITATION.cff  CHANGELOG.md  VERSION
README.md  REPRODUCIBILITY.md  INTERFACE.md  LEDM_input_spec.md
ccmplus/                        solver package (frozen core + io / geometry layer)
  synth/                        analytic Stokes-sphere field + track sampler
  tests/                        unit-test suite
configs/                        example_synthetic_sphere + paper_* configs
examples/synthetic_sphere/      runnable example: inputs, generator, EXPECTED_OUTPUT.md
```

---

## Documentation index

| Document | Purpose |
|---|---|
| [`README.md`](README.md) | This overview: install, quickstart, interface, paper reproduction. |
| [`LEDM_input_spec.md`](LEDM_input_spec.md) | Full input contract: file formats, all parameter keys, conventions. |
| [`INTERFACE.md`](INTERFACE.md) | Solver call contract for programmatic use. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Pinned toolchain, per-case settings, known test baseline. |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history. |
| [`CITATION.cff`](CITATION.cff) | Citation metadata and DOI. |
| `examples/synthetic_sphere/EXPECTED_OUTPUT.md` | Reference log and sanity numbers for the bundled example. |
