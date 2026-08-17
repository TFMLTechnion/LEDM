# Reproducibility

What to set to reproduce each result in the paper, and the toolchain used.

## Proximity reweighting — the one non-obvious flag

LE-DM can down-weight tracks within ~`sigma_gamma` of the body surface
(`W_ii = (1/sigma_i^2)·[1 − exp(−max(0,phi)/sigma_gamma)]`). This is controlled by
`enable_proximity_reweight`.

**The two-file / four-file input path (`run_ledm.py --four-file`) defaults
`enable_proximity_reweight = false`.** A run therefore does NOT apply near-wall
down-weighting unless you set the flag `true`. Every shipped config sets it
**explicitly** (no silent default), so what you run is what the paper ran.

`sigma_gamma` is a length in **mm**, compared directly to the signed distance `phi`
(no `dx` scaling). When `enable_proximity_reweight = false`, `sigma_gamma` is carried
but inert.

### Per-case settings (as used in the paper)

| Paper case | config | input mode | `enable_proximity_reweight` | `sigma_gamma` |
|---|---|---|---|---|
| CFD rising sphere | `configs/paper_cfd_sphere.txt` | position-only | **true (ON)** | 0.5 mm |
| CFD tumbling spheroid | `configs/paper_cfd_spheroid.txt` | two-file | **false (OFF)** | 0.5 mm (inert) |
| Laboratory PTV experiment | `configs/paper_experiment.txt` | two-file | **true (ON)** | 0.5 mm |
| Bundled synthetic sphere (install check) | `configs/example_synthetic_sphere.txt` | two-file | false (OFF) | 0.5 mm (inert) |

So: to reproduce the **CFD-sphere** and **experiment** numbers you must run the
four-file path with `enable_proximity_reweight = true`; the shipped configs already do.
The **spheroid** used the default OFF.

## Input mode — which cases need a kinematics file

- **Sphere → omit `kinematics_file`.** The body velocity is obtained by differentiating
  the position trajectory, with `omega = 0`. This is the position-only path and it
  reproduces the original single-file sphere behaviour, which is what the paper's
  CFD-sphere case used — so `paper_cfd_sphere.txt` ships with **no** `kinematics_file`
  and `data/cfd_sphere/` needs no `kinematics.dat`.
- **Non-spherical or rotating body → `kinematics_file` is required.** A tumbling
  spheroid's `omega` cannot be recovered from its centre trajectory, so the reader
  refuses to differentiate a non-sphere and raises a hard error asking for the file.
  `paper_cfd_spheroid.txt` and `paper_experiment.txt` therefore keep theirs.

Setting `kinematics_file` to a path that does not exist is a load-time
`FileNotFoundError`, not a silent fallback to differentiation — so an unused key must be
removed, not left dangling. See `LEDM_input_spec.md` §3 for the full contract.

### Path caveat (why the flag only works on the four-file path)

- `python run_ledm.py --four-file <config>` (the two-file interface) reads
  `enable_proximity_reweight` from the config and honors it. **Use this path.**
- The historical sphere *research* driver (`ccmplus/drivers/sphere.py`), which
  generated the original CFD-sphere figures, applies the reweighting
  *unconditionally*. The four-file path with `enable_proximity_reweight = true`,
  `sigma_gamma = 0.5` reproduces that behavior through the public interface.

## Toolchain

Verified-working combo, pinned in `requirements.txt`:

| package | version |
|---|---|
| Python | CPython 3.11.9 |
| numpy | 1.26.4 |
| scipy | 1.12.0 |
| matplotlib | 3.11.1 |
| pytest | 8.x |

Pin the upper bounds (numpy < 2.0, scipy < 1.13) for numerical reproducibility of the
published figures; on numpy ≥ 2.4 / scipy ≥ 1.17 the float results drift.

## Test-suite status (honest)

`python -m pytest ccmplus/tests -q` → **217 passed, 10 skipped, 8 failed** on the
pinned toolchain.

The 8 failures are **pre-existing limitations of the frozen v2 solver build**, present
identically on the pinned combo AND on numpy 2.4 / scipy 1.17 — they are **not**
introduced by this packaging and **not** fixed by the pin:

- `test_constraints`: `test_interpolates_linear_field_exactly`, `test_nonzeros_per_particle`
- `test_io_tecplot`: `test_header_fields`, `test_variable_values`, `test_classification_integers`
- `test_solver_onefluid`: `test_noisy_data_bounded_error`,
  `test_reconstructed_field_divergence_free_at_interior`, `test_constraint_satisfaction_general`

They reflect strict divergence-free / interpolation-accuracy tolerances the frozen build
does not fully meet (a MINRES preconditioner + benchmark re-tuning are the documented
follow-ups). **The reconstruction pipeline itself is unaffected**: every
reconstruct / geometry / two-file / classify / grid test passes, the bundled synthetic
example converges (3/3), and the paper cases converge. Treat the 8 as a known baseline,
not a packaging regression.
