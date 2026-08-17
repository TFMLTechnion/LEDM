# Paper-case templates

**These are illustrative starting configurations, not exact paper-reproduction
recipes.** They show how to wire each published case into the two-file input
contract: which geometry type, which input mode (position-only vs. two-file),
and which keys matter. They are a place to start from, not a script that
regenerates a figure.

Numerical parameters here (`dx`, `kappa`, `lambda_c`, `sigma_u`, `roi_pad`,
`sigma_gamma`, solver tolerances) are **placeholders chosen to be reasonable and
runnable**. Set them from the values published in the paper for the case you are
reproducing, and from your own data's resolution and noise level. Do not assume
a number in these files is the number used in the manuscript.

The data is not bundled. See the Zenodo data record (DOI in `CITATION.cff`) and
unpack it under a `data/` folder next to the code so the relative paths resolve.

| Template | Body | Input mode |
|---|---|---|
| `paper_cfd_sphere.txt` | sphere | position-only (no `kinematics_file`) |
| `paper_cfd_spheroid.txt` | prolate spheroid, tumbling | two-file (position + kinematics) |
| `paper_experiment.txt` | lab PTV | two-file (position + kinematics) |

Run one with:

```
python run_ledm.py --four-file paper_templates/paper_cfd_sphere.txt
```

For a config that runs out of the box with no external data, use
[`../configs/example_synthetic_sphere.txt`](../configs/example_synthetic_sphere.txt).

Input-mode rule: a **sphere** may omit `kinematics_file`, in which case the body
velocity is differentiated from the centre trajectory with `omega = 0`. A
**non-spherical or rotating body requires** `kinematics_file`. See
[`../LEDM_input_spec.md`](../LEDM_input_spec.md) §3.
