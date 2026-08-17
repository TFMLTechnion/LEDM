# Relationship to the paper

This package is an **open research reference implementation of LE-DM**: a readable,
runnable, extensible implementation of the method. It is **not** a paper-reproduction
archive, and it does not ship scripts that regenerate specific manuscript figures,
tables, or sweeps.

## Method reference

> Jibu Tom Jose, Arieh Jacobson, Dhanush Vittal Shenoy, Steven H. Frankel, Omri Ram,
> *Dynamic masking for boundary-aware velocity reconstruction in volumetric particle
> tracking with moving solids*, arXiv:2606.25748,
> DOI [10.48550/arXiv.2606.25748](https://doi.org/10.48550/arXiv.2606.25748).

Read the paper for the derivation, the notation, and the validation study. This
repository implements the method described there; equation numbers in the source
comments (e.g. Eqs. 13-14 for the coverage-adaptive smoothness) refer to it.

## Data

The datasets used in the paper are archived separately:

> Zenodo data record, DOI [10.5281/zenodo.21965844](https://doi.org/10.5281/zenodo.21965844).

Download the case you want and unpack it under a `data/` folder next to the code, so
the relative paths in `paper_templates/*.txt` resolve.

## What is and is not claimed

**Provided.** A faithful implementation of the method; a self-contained synthetic
example that runs from the repository with no external data
(`configs/example_synthetic_sphere.txt`); illustrative starting configs for each
published case (`paper_templates/`); and a unit-test suite covering the geometry
layer, the input contract, the constraint assembly, and the solver.

**Not provided, and not claimed.** Bit-for-bit reproduction of published numbers.
The configs in `paper_templates/` are starting points whose numerical parameters
(`dx`, `kappa`, `lambda_c`, `sigma_u`, `sigma_gamma`, `roi_pad`, solver tolerances)
are **placeholders**, not the manuscript's values. Set them from the published
values and from your own data. Results also depend on BLAS, numpy/scipy versions,
and the MINRES tolerance you choose.

## Things that will change your results

These are the settings most likely to matter, and the ones worth stating explicitly
in any write-up that uses this code.

### 1. The MINRES tolerance is a physics setting, not a performance knob

The saddle-point solve reports `converged` against the **scaled** system, which
happens well before the divergence constraint is actually satisfied. On the bundled
synthetic example, `minres_tol = 1e-6` stops after ~12 iterations with a **9%
divergence error** while still reporting `converged=True`.

Every run therefore reports two normalized constraint residuals after the solve
(`SolverInfo.constraints`, and in the log line):

| quantity | meaning | key |
|---|---|---|
| `dx * rms(div u) / U_ref` | dimensionless divergence error per cell | `constraint_div_tol` |
| relative residual of the no-slip rows | how well `u = u_Gamma` holds on the body | `constraint_body_tol` |

A run that misses either tolerance raises a `RuntimeWarning` naming the number.
**Treat that warning as a failed run.** The fix is to tighten the solve
(`minres_tol`, `minres_maxit`, `use_jacobi_precond = on`), not to relax the
tolerance.

### 2. Proximity reweighting

`enable_proximity_reweight` controls whether tracks within ~`sigma_gamma` of the body
surface are down-weighted:

```
W_ii = (1/sigma_i^2) * [1 - exp(-max(0, phi)/sigma_gamma)]
```

**It defaults to `false`**, so a run does not apply near-wall down-weighting unless
you ask for it. `sigma_gamma` is a length in the run's length unit, compared directly
to the signed distance `phi` (no `dx` scaling); when reweighting is off it is carried
but inert. Every shipped config sets the flag explicitly rather than relying on the
default. It is a run-level scalar, not per-snapshot.

### 3. Boundary constraints on/off

`boundary_constraints = off` removes the shell/solid no-slip constraint rows and
enforces divergence uniformly over all interior nodes. **It does not turn the run
into an all-fluid reconstruction:** interpolation and smoothing are still masked by
the body classification, so tracks never write into the body interior. Report it as
an ablation of the boundary conditions, and do not describe it as a body-agnostic
baseline. (`enable_lema` is the deprecated spelling of this flag.)

### 4. Coverage-adaptive smoothness

`lambda_c` weights the Eq. 13-14 penalty, with `coverage_ref_count` (`c_0`) setting
the track count at which the smoothing weight halves. The neighbour difference is
**unscaled** (no `1/dx` factor), so `lambda_c` is dimensionless relative to the data
term and does not need recalibrating when the grid is refined. See
`LEDM_input_spec.md` and `ccmplus/operators.py`.

## Toolchain

Developed on CPython 3.11.9 with numpy 1.26.4 / scipy 1.12.0; `requirements.txt`
pins compatible ranges. Exact floating-point digits vary with numpy/scipy and BLAS.
The tolerances asserted by the test suite and quoted in
`examples/synthetic_sphere/EXPECTED_OUTPUT.md` are chosen to hold across those
variations — judge a run by those, not by digit-for-digit agreement.

Run the tests with:

```
python -m pytest ccmplus/tests -q
```
