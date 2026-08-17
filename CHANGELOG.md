# Changelog

All notable changes to LE-DM. Format loosely follows Keep a Changelog; this project
uses semantic versioning.

## [1.0.0] — 2026-08-16

First public release, as an **open research reference implementation** of LE-DM:
meant to be run, read, and extended. It is not a paper-reproduction archive, and it
ships no scripts that regenerate specific manuscript figures, tables, or sweeps.

### Changed — core algorithm

These change results. Any numbers produced with an earlier build should be
regenerated.

- **Divergence stencils are now strictly fluid-side.** The incompressibility
  operator can no longer reach into a shell (`C == 0`) or solid (`C == -1`) node.
  Each fluid divergence row now selects, per axis, the first admissible option in a
  strict hierarchy: centered 2nd order iff *both* opposite neighbours are open
  fluid; else one-sided 2nd order iff the first two nodes on the fluid side are open
  fluid; else one-sided 1st order iff the first fluid-side node is open fluid; else
  the row is dropped and the node reported as `insufficiently_resolved`.
  Previously the choice depended only on index availability, and two fallback paths
  could pull a non-fluid node into a stencil. There is now **no** fallback that
  crosses the interface, not even to rescue a row.
  (`ccmplus/constraints.py`; `build_constraints(..., return_diagnostics=True)`
  returns per-rule counts and the dropped-node list.)
- **Coverage-adaptive smoothness now implements Eqs. 13-14 literally**
  (`ccmplus/operators.py`, `ccmplus/reconstruct.py`):
  - `c_j` is a genuine spatial query — the number of *tracks* within radius `Delta`
    of open-fluid node `j`, via `scipy.spatial.cKDTree`. It was previously
    approximated by the column count of the interpolation matrix `A`, which measures
    kernel reach rather than track density and silently changed meaning whenever the
    kernel changed.
  - the edge weight is the **average of the two endpoint weights**,
    `w_bar_jn = 0.5*(w_j + w_n)`, so an edge's penalty no longer depends on which
    endpoint it is attributed to. It was `sqrt(w)` of the lower-index endpoint.
  - **spacing convention, stated once and applied consistently:** Eq. 13 is an
    *unscaled* neighbour difference, so the `1/Delta` factor is removed and
    `lambda_c` is dimensionless relative to the data term and does not need
    recalibrating on grid refinement. The `spacing_scaled` argument is gone from this
    operator (the separate Laplacian/gradient smoothers keep their own switch).
    Documented in the source and in `LEDM_input_spec.md`.
  - unchanged: an edge is emitted only when both endpoints are open fluid.
- **No stored regression baseline needed regenerating.**
  `ccmplus/tests/test_ledm_regression.py` compares the four-file path against the
  direct-`BodyState` path on identical inputs rather than against saved numbers, so
  it validates the two paths' agreement at whatever the current numerics are. No
  golden files exist anywhere in the tree. The new output was checked by hand
  instead: shell nodes carry the body velocity to < 1e-6 mm/s, `Delta*rms(div u)/U_ref`
  is ~1.5e-4, and the far field is unchanged in character.

### Added — solver diagnostics and constraint tolerances

- `SolverInfo.constraints` (`ConstraintDiagnostics`) reports **normalized** constraint
  residuals after every solve: `Delta*rms(div u)/U_ref` and `Delta*max|div u|/U_ref`
  for the fluid rows, and a relative residual for the no-slip identity rows, each with
  its own pass/fail flag. The two families are judged separately and differently on
  purpose — a divergence row has units of velocity/length, so only the normalized form
  is a meaningful threshold, while the body rows are exact Dirichlet conditions with an
  O(1) right-hand side.
- New config keys `constraint_div_tol` (default 1e-3) and `constraint_body_tol`
  (default 1e-3). A solve that misses either raises a `RuntimeWarning` naming the
  number. `div_rms_norm` is also in the per-solve log line.
- **This exposed a real problem in the shipped example.** MINRES reports `converged`
  against the *scaled* saddle system well before the divergence constraint is
  satisfied: at `minres_tol = 1e-6` the synthetic sphere stopped after ~12 iterations
  with a **9% divergence error**. `configs/example_synthetic_sphere.txt` and the paper
  templates now use `minres_tol = 1e-11`, `minres_maxit = 20000`,
  `use_jacobi_precond = on`, which reaches ~1.5e-4. The example takes ~1.5 min instead
  of a few seconds; that is the honest cost of an actually divergence-free field, and
  the physics was not relaxed to avoid it.

### Added — input contract

- **Startup banner.** Every run now opens by echoing what the loader resolved: units
  (coordinate, velocity, angular velocity, time), geometry type and parameters, Euler
  convention, grid dimensions and bounds, track counts, and the active LE-DM options
  and tolerances. Units come first because a unit mix-up is the one input error the
  numbers alone cannot reveal.
- **Particle files may declare units** via an optional `# units: length=…, velocity=…`
  header, checked against the parameter file exactly as the geometry and kinematics
  headers are. A file with mm positions and m/s velocities parses perfectly and is
  numerically indistinguishable from a correct one, so this header is the only thing
  that can catch it. The header is optional for backwards compatibility; the banner
  states whether the check was possible. The bundled example now declares its units.
  (v1.0 keeps ONE coherent unit system; no separate `velocity_unit` was added.)

### Changed — naming

- `enable_lema` is renamed **`boundary_constraints = on|off`** in parameter files and
  `Config`. `enable_lema` still works in both places but emits a `DeprecationWarning`;
  when both are set, `boundary_constraints` wins.
- The `off` mode is **no longer described as "base CCM"** anywhere public. It removes
  the shell/solid constraint rows and enforces divergence uniformly over all interior
  nodes, but interpolation and smoothing remain masked by the body classification, so
  tracks still never write into the body interior. It is an ablation of the boundary
  conditions, not an all-fluid reconstruction, and it is not equivalent to the paper's
  all-fluid base-CCM comparison. Documented in `Config`, `run_ledm.py`,
  `LEDM_input_spec.md`, and `REPRODUCIBILITY.md`.

### Changed — docs and layout

- `configs/paper_*.txt` moved to **`paper_templates/`**, each with a header stating it
  is an *illustrative starting config, not an exact paper-reproduction recipe*, and
  that its numerical values are placeholders to be set from the published values.
  `configs/example_synthetic_sphere.txt` stays in `configs/` as the runnable,
  self-contained example. Added `paper_templates/README.md`.
- `REPRODUCIBILITY.md` rewritten as **"Relationship to the paper"**: method reference
  and Zenodo data pointer, plus the settings that genuinely change results (solver
  tolerance, proximity reweighting, boundary constraints, `lambda_c`). It no longer
  promises exact reproduction.
- `CITATION.cff`: real authors, title, arXiv DOI `10.48550/arXiv.2606.25748`, Zenodo
  data DOI `10.5281/zenodo.21965844`, repository URL. No placeholders remain. (No
  software Zenodo DOI is claimed, since none is minted.)
- Corrected stale claims:
  - `ccmplus/interp.py` and `Config` described the default `wide` kernel as a *radial
    Gaussian*. It is the tensor-product **cubic B-spline** (4 nodes/axis, 4x4x4 =
    64-node support, linear precision). `radius_cells`/`sigma_cells` apply to the
    separate `gaussian` kernel only and are ignored by `wide`.
  - `LEDM_input_spec.md` claimed the `ax ay az` acceleration columns are "used for
    pressure and for the VIC# comparison". They are parsed and never consumed; there
    is no pressure or VIC# path in this release. Relabelled **reserved / ignored
    (future extension)**.
  - `LEDM_input_spec.md` promised a load-time check of the Euler angle rates against
    `omega`. The loader deliberately does not do this — pose and angular velocity are
    independent rigid-body inputs and `omega` is not `d(theta)/dt` — so the check
    would reject correct input. Removed, with the reasoning.
  - Removed references to `ZENODO_UPLOAD.md`, which is not part of this tree.
  - `requirements.txt` no longer advertises 8 known-failing tests (the suite is green)
    and no longer pins upper bounds for "reproducibility of the published figures".

### Fixed — test suite (now green)

`python -m pytest ccmplus/tests -q` → **256 passed, 10 skipped, 0 failed, 0 warnings**.
Previously 8 tests failed. None were fixed by weakening an assertion about the physics:

- `test_solver_onefluid` x3 demanded `1e-5` *absolute* agreement on `B @ x - g`, which
  is not a physically meaningful threshold for a divergence row. They now assert the
  documented normalized tolerances (`div_rms_norm < 1e-4`, `div_max_norm < 1e-3`) and
  tighten the solve to reach them. Added tests that the no-slip rows hold to solver
  tolerance with a body present, and that an under-converged solve is *flagged*.
- `test_noisy_data_bounded_error` assumed `A ≈ I`, which holds only for the compact
  trilinear kernel; under the default cubic B-spline, recovering nodal values from
  track values is a deconvolution and amplifies high-wavenumber noise. The test now
  pins the kernel to match its own premise, and a new companion test documents the real
  behaviour of the default kernel: the error is amplified but **unbiased**, and is
  damped by stronger `kappa`.
- `test_constraints` x2 were stale: they asserted the 8-node/24-nonzero trilinear
  stencil, and sampled linear-precision points inside the one-cell boundary margin
  where the B-spline's stencil is clipped. Updated to the 64-node support, with a
  separate test retaining the 24-nonzero check for the `trilinear` kernel.
- `test_io_tecplot` x3 were stale: the writer emits **nine** BLOCK variables (a derived
  `Vmag` column was added) and quotes the zone title, but the test helper assumed eight
  and an unquoted title. The helper now returns variables by name so adding a column
  cannot silently invalidate the tests again.

New test files: `test_coverage_smoothing.py` (Eqs. 13-14: radius-`Delta` count,
endpoint-average edge weight, unscaled spacing, no interface-crossing edges, SPSD) and
`test_ledm_units.py` (particle unit enforcement and the startup banner). Added a test
that no divergence stencil references a non-fluid node, checked column-by-column on the
assembled matrix across several body radii.

Pipeline smoke tests that exercise wiring rather than constraint convergence set
`constraint_div_tol` loose deliberately, with a comment pointing at
`test_solver_onefluid.py` for the constraint assertions — so the constraint warning
stays meaningful instead of firing everywhere.

### Unchanged

`README.md` (supplied separately). Per-snapshot `sigma_gamma` remains out of scope:
it is a run-level scalar. No mesh / multiple-body / stationary-wall / deforming-interface
geometry. No true all-fluid base-CCM conversion — the change above is the rename and the
honest description, not the conversion.

### Added
- Public **two-file input interface** (geometry/position + kinematics + per-timestep
  particles + parameter file), driven by `run_ledm.py --four-file`, with a documented
  contract (`LEDM_input_spec.md`) and solver call contract (`INTERFACE.md`).
- **Sphere** and **ellipsoid** geometries. `cylinder`/`mesh`/`voxel` headers are rejected
  with a clear "not yet implemented" error rather than silently falling back.
- Region-of-interest options (`roi_mode = auto | box | body`, `roi_pad`, `max_grid_nodes`)
  and output options (`output_format = npz | dat | both`, `output_frame = lab | body`,
  Tecplot `.dat` writer).
- Runnable, self-contained **synthetic-sphere example** (`configs/example_synthetic_sphere.txt`,
  `examples/synthetic_sphere/`) with a generator and `EXPECTED_OUTPUT.md`.
- Illustrative **paper-case templates** (`paper_templates/`) for the CFD sphere, CFD
  spheroid, and lab experiment, each setting `enable_proximity_reweight` and
  `sigma_gamma` explicitly. The CFD-sphere template uses the **position-only** path
  (no `kinematics_file`); the spheroid and experiment templates use the two-file path.
- `REPRODUCIBILITY.md`, `CITATION.cff`, pinned `requirements.txt`, GPL-3.0 `LICENSE`.

### Reproducibility notes
- The two-file path defaults `enable_proximity_reweight = false`. Paper cases: CFD sphere
  and experiment used it **ON** (`sigma_gamma = 0.5 mm`), spheroid **OFF**. Shipped configs
  set it explicitly — see `REPRODUCIBILITY.md`.
- Input mode per body shape: a **sphere** omits `kinematics_file` and its velocity comes
  from differentiating the position trajectory (`omega = 0`); a **non-spherical or
  rotating body requires** `kinematics_file`. A `kinematics_file` pointing at a missing
  path is a load-time error, not a fallback. See `REPRODUCIBILITY.md` /
  `LEDM_input_spec.md` §3.
- Developed on numpy 1.26.4 / scipy 1.12.0 / CPython 3.11.9; the suite is also green on
  numpy 2.4 / scipy 1.17. Exact digits vary with numpy/scipy and BLAS, so the suite
  asserts tolerances rather than digit-for-digit agreement.

### Packaging
- Removed internal-only material: build/handoff notes, a machine-specific diagnostic
  (`diag_sigma_gamma_sweep.py`), a dead dataset driver (`drivers/r12.py`), and an empty
  stub (`synth/validate.py`). All absolute personal paths and identifiers scrubbed.
