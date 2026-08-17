# Changelog

All notable changes to LE-DM. Format loosely follows Keep a Changelog; this project
uses semantic versioning.

## [1.0.0] — 2026-08-16

First public release.

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
- Worked **paper-case configs** for the CFD sphere, CFD spheroid, and lab experiment,
  each setting `enable_proximity_reweight` and `sigma_gamma` explicitly. The CFD-sphere
  config uses the **position-only** path (no `kinematics_file`); the spheroid and
  experiment configs use the two-file path.
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
- Toolchain pinned to numpy 1.26.4 / scipy 1.12.0 / CPython 3.11.9.

### Known issues (pre-existing, not packaging-induced)
- 8 strict-accuracy unit tests fail on the pinned toolchain (and identically on
  numpy 2.4 / scipy 1.17): `test_constraints` ×2, `test_io_tecplot` ×3,
  `test_solver_onefluid` ×3. They reflect divergence-free / interpolation tolerances the
  frozen solver build does not fully meet; the reconstruction pipeline, the bundled
  example, and the paper cases all converge. See `REPRODUCIBILITY.md`.

### Packaging
- Frozen solver core (`ccmplus/solver.py`, `reconstruct.py`, `constraints.py`, `interp.py`,
  `sdf.py`, `kinematics.py`, `classify.py`, `config.py`, `grid.py`, `operators.py`,
  `prior.py`) is byte-for-byte unchanged from the internal build.
- Removed internal-only material: build/handoff notes, a machine-specific diagnostic
  (`diag_sigma_gamma_sweep.py`), a dead dataset driver (`drivers/r12.py`), and an empty
  stub (`synth/validate.py`). All absolute personal paths and identifiers scrubbed.
