# LE-DM Input Contract (v0.1, draft)

Frozen input specification for the open-source LE-DM code (Lagrangian-to-Eulerian
reconstruction with Dynamic Masking). This is the contract the code reads and that
users write against. It is deliberately minimal so the released code stays compact
and so the published results remain reproducible.

## Scope and invariants

- Formulation is fixed and identical to the paper: incompressible flow, rigid body.
  The solver core is not changed by this contract; only the input/output layer and
  the geometry abstraction are defined here.
- One world coordinate frame, one length unit, and one rotation convention apply to
  every file in a run. All conventions are declared once, in the parameter file.
- The body is the single source of truth for occupancy. The reconstruction grid,
  the node classification (fluid / shell / solid), and the shell velocity are all
  derived internally from the geometry and the grid. They are not user inputs.

## The four inputs

A run consists of three data inputs plus one configuration file:

1. Particle field, one file per timestep.
2. Geometry, one file per run.
3. Kinematics, one file per run.
4. Parameter file, one file per run.

---

### 1. Particle field (one file per timestep)

Scattered Lagrangian tracks at a single time. The body region appears as a natural
void (no tracks inside the solid); the code does not read a mask.

- Naming: `particles_<k>.<ext>` with a zero-padded integer index `k` that matches the
  snapshot order, e.g. `particles_00001.dat`. The index-to-time map comes from the
  geometry and kinematics `t` columns (same ordering).
- Format: plain text, whitespace or comma separated, with a one-line column header.
- Columns (fixed order):

  | column        | required | meaning                                   |
  |---------------|----------|-------------------------------------------|
  | x y z         | yes      | track position, world frame, length unit  |
  | u v w         | yes      | track velocity, length/time unit          |
  | ax ay az      | no       | **reserved / ignored** (see note)         |
  | su sv sw      | no       | per-track velocity uncertainty (1 sigma)  |

- Acceleration columns are **parsed but not consumed**: they are read into
  `part["accel"]` and go no further. There is no pressure solver and no VIC#
  comparison path in this release. The columns are reserved for a future
  extension; supplying or omitting them changes nothing in the reconstruction.
  If you do supply them, use the Lagrangian material derivative Du/Dt following
  the particle (not the Eulerian dv/dt at a point) so the data stays correct for
  whenever the extension lands.
- Uncertainty, if present, feeds the per-track data weight. If absent, the scalar
  default `sigma_u` from the parameter file is used for all tracks.
- Ghost tracks inside the solid are removed by the code using the signed-distance
  field, with a small inward margin. Tracks just outside the surface are kept but
  down-weighted by the proximity weight `sigma_gamma`; they are not culled.

---

### 2. Geometry (one file per run)

Shape plus rigid-body pose at every timestep. The header declares the shape; the rows
give the pose.

Header (comment lines, `key: value`):

```
# type: ellipsoid
# params: a=5.0 b=5.0 c=8.0
# units: mm
# columns: t x y z alpha beta gamma
```

- `type`: the geometry. **Implemented in this release: `sphere` and `ellipsoid` only.**
  A `cylinder` header (or any unregistered type) is rejected at load time with a clear
  "not yet implemented" error — never a silent fallback. `mesh`/`voxel` are design
  placeholders described below, not implemented.
- `params`: the shape parameters the type needs. `sphere` needs `r`; `ellipsoid` needs
  `a b c` (the semi-axes).
- `units`: length unit for `params` and for the pose columns. Must equal the parameter
  file length unit.

Rows (one per timestep):

| column        | meaning                                             |
|---------------|-----------------------------------------------------|
| t             | time stamp, time unit (matches kinematics `t`)      |
| x y z         | body center in the world frame                      |
| alpha beta gamma | orientation as Euler angles, per the parameter-file convention |

Pose semantics: a point `p_body` in the body frame maps to the world frame as
`p_world = c + R(alpha, beta, gamma) * p_body`, where `c = (x, y, z)` and `R` is built
from the declared Euler convention. The shape parameters are defined in the body frame.

---

### 3. Kinematics (one file per run)

Body linear and angular velocity at every timestep. **Optional** — the two-file
interface (position + kinematics):

- **Both files present** → kinematics wins for the shell velocity
  `U_s(x) = (u,v,w) + omega × (x − X_s)`; the position trajectory is never
  differentiated. The Euler angles in the position file and `omega` here are treated
  as **independent** (no consistency check between them — `omega` is not `d(theta)/dt`).
- **Position file only** → allowed **for a sphere only**: velocity is obtained by
  differentiating the position trajectory and `omega = 0` (reproduces the original
  single-file sphere behaviour, bit-for-bit). For any non-sphere shape a missing
  kinematics file is a hard error asking for it — a non-sphere is never differentiated.

Shapes implemented: `sphere` and `ellipsoid`. A header naming `cylinder` (or any other
shape) is rejected with a clear "not yet implemented" error rather than a silent
fallback.

Header:

```
# units: velocity=mm/s, omega=rad/s
# columns: t u v w omega_x omega_y omega_z
```

Rows (one per timestep):

| column              | meaning                                        |
|---------------------|------------------------------------------------|
| t                   | time stamp (matches geometry `t`)              |
| u v w               | body center velocity, world frame              |
| omega_x omega_y omega_z | body angular velocity, world frame, rad/s  |

Surface velocity used by LE-DM on the boundary shell:
`u_Gamma(x) = U + omega x (x - c)`, with `U = (u, v, w)`, `c` the center from the
geometry file, and both `omega` and `(x - c)` in the world frame.

Important: `omega` is NOT the time derivative of the Euler angles. It relates to the
angle rates through the convention-dependent Euler kinematic map, and equals the raw
rates only for single-axis rotation. Take `omega` from the flow solver or the measured
body motion; do not build it by differencing `alpha, beta, gamma`.

---

### 4. Parameter file (one file per run)

Human-readable, commented `key = value`. Groups: time, grid, solver, convention. This
is the only file the user hand-edits to control a run.

```
# --- time ---
time_source   = file        # use the t columns; or "range"
t_start       = 0.0
t_end         = 0.55
time_unit     = s

# --- grid ---
grid_extent   = auto        # or: xmin xmax ymin ymax zmin zmax
grid_margin   = 3           # cells of padding when extent = auto / roi_mode = box
dx            = 1.0         # grid spacing (isotropic); length_unit below
                            # (shell half-width is fixed internally at dx/2)

# --- region of interest (optional; restricts the grid to a box) --------
roi_mode      = auto        # auto | box | body   (default auto = grid_extent above)
#   auto : original behaviour (full particle bounding box, or grid_extent).
#   box  : grid spans roi_box exactly (+/- grid_margin cells).
#   body : grid spans the union of the body bounding box over [t_start, t_end],
#          expanded by roi_pad body characteristic-lengths in every direction.
roi_box       = xmin xmax ymin ymax zmin zmax   # required when roi_mode = box (length_unit)
roi_pad       = 2.0         # body diameters of padding when roi_mode = body (default 2.0)
max_grid_nodes = 2000000    # hard node-count ceiling: assembly raises ValueError with a
                            # readable message if Nx*Ny*Nz exceeds this, instead of OOM-ing
                            # inside the solver on a full-cloud auto grid (default 2_000_000)

# --- LE-DM solver (names must match ccmplus) ---
boundary_constraints = on   # on | off. ON assembles the no-slip shell/solid
                            # Dirichlet rows, seeds transition nodes from the body
                            # velocity, and uses body-aware (fluid-side) divergence
                            # stencils. OFF drops those constraint rows and enforces
                            # divergence uniformly over all interior nodes.
                            # OFF is an ABLATION of the boundary conditions, NOT an
                            # all-fluid reconstruction: interpolation and smoothing
                            # remain masked by the body classification either way.
                            # (`enable_lema` is the deprecated spelling; it still
                            # works but warns.)
sigma_gamma   = 0.25        # surface-proximity down-weighting length [mm]
enable_proximity_reweight = false  # apply the sigma_gamma near-wall down-weighting?
kappa         = 10          # temporal-prior weight
sigma_u       = 0.005       # default per-track velocity uncertainty
warm_start    = true

# --- coverage-adaptive smoothness (Eqs. 13-14) ---
lambda_c           = 0.05   # weight of the coverage-adaptive smoothness penalty
coverage_ref_count = 1.0    # c_0: local track count giving half smoothing weight
#   c_j  = number of tracks within a radius of dx of open-fluid node j
#          (an actual cKDTree radius query over the track positions).
#   w_j  = 1 / (1 + c_j / c_0)
#   edge weight w_bar_jn = 0.5 * (w_j + w_n)   -- the AVERAGE of the two endpoint
#          weights, so the penalty on an edge does not depend on which endpoint
#          it is attributed to. An edge is emitted only when BOTH endpoints are
#          open fluid, so smoothing never couples across the body interface.
#
#   SPACING CONVENTION: Eq. 13 is an UNSCALED neighbour difference. The operator
#   penalises sum_edges w_bar_jn * ||u_n - u_j||^2 with NO 1/dx factor, so
#   lambda_c is dimensionless relative to the velocity-squared data term and its
#   calibration does not shift when the grid is refined. This is the single,
#   consistent choice throughout; see ccmplus/operators.py.

# --- MINRES solve ---
minres_tol    = 1e-11
minres_maxit  = 20000
use_jacobi_precond = on     # block-Jacobi (diag of H) preconditioner; roughly
                            # halves the iteration count on stiff systems.
#   WARNING: MINRES reports "converged" on the SCALED saddle system well before
#   div(u) is actually small. On the bundled synthetic example, minres_tol = 1e-6
#   stops after ~12 iterations with a 9% divergence error. Judge the solve by the
#   reported div_rms_norm, not by `converged`.

# --- constraint tolerances (checked after the solve; a miss raises a warning) ---
constraint_div_tol  = 1e-3  # dimensionless dx*rms(div u)/U_ref over the fluid rows
constraint_body_tol = 1e-3  # relative residual of the no-slip identity rows
#   The two families are judged differently on purpose. A divergence row has units
#   of velocity/length, so only the normalised form is a physically meaningful
#   threshold. The body rows are exact algebraic Dirichlet conditions with an O(1)
#   right-hand side, so a plain relative residual is the right measure and it should
#   hold to roughly the MINRES tolerance.

# --- output (optional; default reproduces the .npz-only behaviour) -----
output_format = npz         # npz | dat | both (default npz)
#   npz  : one compressed .npz per snapshot (nodes, velocity, classification, t).
#          Source of truth for warm_start and any downstream reader.
#   dat  : one ASCII .dat per snapshot (Tecplot/ParaView/MATLAB-loadable).
#   both : write .npz AND .dat (npz stays authoritative).
dat_flavor    = tecplot     # tecplot | plain (default tecplot)
#   tecplot : TITLE / VARIABLES / ZONE ... DATAPACKING=POINT, I fastest.
#   plain   : one '#'-commented column-name line, then whitespace columns.
dat_precision = 9           # significant digits per column (default 9)
dat_order     = C           # C | F ravel order of the structured grid (default C)
#   The zone I,J,K are set from dat_order so Tecplot always reads I fastest
#   (C -> I,J,K = Nz,Ny,Nx; F -> Nx,Ny,Nz). Both are exact round trips.
output_frame  = lab         # lab | body (default lab)
#   lab  : write the reconstructed lab-frame velocity as solved (default).
#   body : write v_rel = v_lab - u_body, where u_body = U + omega x (x - X_s) is the
#          body's rigid-body field (the SAME u_Gamma the solver pinned). Solid/shell
#          nodes go to ~0; the flow-around picture appears. Applies to npz and dat.
comoving_coords = false     # false | true (default false)
#   true : also shift node coordinates by -(X_s - X_s0) so the body stays fixed
#          across snapshots (for animation). Velocity is unaffected. Off by default.

# --- coordinate convention (governs ALL files) ---
length_unit   = mm
angle_unit    = deg
euler_seq     = ZYX         # intrinsic; see convention section
handedness    = right
omega_frame   = world
```

The parameter names in the solver group must match the existing `ccmplus`
`parameters.txt` keys exactly, so the released code and the paper runs share one
vocabulary.

---

## Coordinate and rotation convention

Ambiguity here is the single most common silent error, so it is centralized and
validated.

- World frame: right-handed Cartesian (X, Y, Z). Same axes for particles, geometry
  centers, velocities, and angular velocity.
- Length: one unit (`length_unit`) throughout every file.
- Time: one unit (`time_unit`); `t` values agree across geometry and kinematics, and
  the particle-file index follows the same ordering.
- Rotation `R(alpha, beta, gamma)`: built from a single declared Euler sequence
  (`euler_seq`), interpreted as intrinsic rotations, with angles in `angle_unit`.
  Recommended default: intrinsic ZYX (yaw, pitch, roll). A `quaternion` option
  (columns `qw qx qy qz` in the geometry file) may be offered to avoid Euler ambiguity
  entirely.
- Angular velocity `omega`: `rad/s`, in the frame set by `omega_frame` (default
  `world`), so that `omega x (x - c)` is directly the world-frame surface velocity.

Load-time consistency checks (fail with a clear message, do not default silently):

1. `dx > 0`; `euler_seq`, `handedness`, `omega_frame` are in the allowed sets.
2. Geometry and kinematics `t` columns match (values, count, order); particle-file
   count matches.
3. Units declared in file headers match the parameter file. This covers the geometry
   header (`# units:`), the kinematics header (`velocity=`, `omega=`) and, when the
   optional `# units:` line is present, each particle file. A particle file that
   declares no units cannot be checked — positions in `mm` with velocities in `m/s`
   parse perfectly and are numerically indistinguishable from a correct file — so
   declaring them is strongly recommended. The resolved units are echoed in the
   startup banner.
4. Physical consistency: `d/dt(center)` from the geometry file agrees with `U` from the
   kinematics file within a tolerance. This catches a naively differentiated `U`, a
   frame mismatch, and a unit error in a single pass.

   There is deliberately **no** check of the Euler angle rates against `omega`. Pose
   and angular velocity are independent rigid-body inputs: `omega` is not
   `d(theta)/dt` under any Euler sequence, so comparing them would reject correct
   input. The loader does not differentiate the angle columns.
5. Grid size: after the ROI bounds are resolved (`roi_mode` = auto | box | body) the node
   count `Nx*Ny*Nz` is checked against `max_grid_nodes`. If it exceeds the ceiling, assembly
   raises `ValueError` naming the node count, `dx`, the resolved bounds and `roi_mode`, and
   suggesting `roi_mode = body`/`box` or a larger `dx` — rather than letting the full-cloud
   grid reach an out-of-memory allocation inside the solver.

Tracks that fall outside the resolved grid box by more than one interpolation-kernel width
cannot inform any node and are dropped at load time; the kept/dropped counts are reported.
For `roi_mode = auto` the box already encloses every track, so nothing is dropped and the
reconstruction is identical to before this option existed.

---

## Geometry interface

Geometry is a small pluggable abstraction. The header `type` selects a class from a
registry. Every geometry implements exactly one required method; the framework does
everything else generically.

```python
class Geometry:
    """Body shape in its own body frame."""

    def signed_distance(self, pts_body):
        """Signed distance at points given in the BODY frame.
        Negative inside the solid, zero on the surface, positive outside.
        pts_body: (N, 3) array. Returns: (N,) array."""
        raise NotImplementedError

    def bounding_radius(self):
        """Optional: radius of a bounding sphere, for fast culling."""
        ...

GEOMETRY_REGISTRY = {}   # "sphere" -> Sphere, "ellipsoid" -> Ellipsoid, ...

def register(name):
    def deco(cls):
        GEOMETRY_REGISTRY[name] = cls
        return cls
    return deco
```

The framework, for each timestep, does the shape-independent work:

1. Build `R` from `(alpha, beta, gamma)` and `c` from `(x, y, z)`.
2. Transform grid nodes into the body frame: `X_body = R^T (X - c)`.
3. `phi = geometry.signed_distance(X_body)`.
4. Classify nodes and build the shell from `phi` and `dx` (internal, per the paper).
5. Set `u_Gamma = U + omega x (X_shell - c)` from the kinematics file.

Only step 3 is shape-specific. Steps 4 and 5 are identical for every body, because
`u_Gamma` never needs per-shape code.

Concrete backends **implemented in this release**:

- `sphere`: `phi = norm(p) - r`. Exact and trivial.
- `ellipsoid`: the sign is exact from `(x/a)^2 + (y/b)^2 + (z/c)^2 - 1`; the near-surface
  distance needed for the shell band uses a point-to-ellipsoid projection (Eberly /
  Geometric Tools).

**Not implemented** (rejected with a clear error; listed as future work):

- `cylinder`: analytic distance to a finite capped cylinder.
- `mesh`: signed distance from a triangle mesh (e.g. trimesh / libigl).
- `voxel`: an occupancy grid → distance via a signed distance transform (a non-analytic
  body, replacing a separate mask input).

Adding a shape is: subclass `Geometry`, implement `signed_distance`, and `@register` it.

---

## Consistency with the paper and validation

- The solver core (`ccmplus`) is the frozen, canonical build and is not modified. This
  contract feeds it; it does not change the method.
- The release ships a runnable, self-contained example (`configs/example_synthetic_sphere.txt`,
  under `examples/synthetic_sphere/`) and worked paper-case configs
  (`configs/paper_cfd_sphere.txt`, `paper_cfd_spheroid.txt`, `paper_experiment.txt`) whose
  data is distributed via Zenodo, so "run and validate the paper" is reproducible from a
  fresh checkout plus the Zenodo dataset.
- A regression test asserts that the new four-file input path produces the same sphere
  reconstruction (to solver tolerance) as the original sphere driver, so the added I/O
  and geometry layer provably do not perturb the physics.

## Outputs (for completeness)

The reconstructed Eulerian field on the fixed grid per timestep (velocity, and node
classification), plus a run log with the solver statistics (grid size, MINRES
iterations, per-snapshot time).

The native format is one compressed `.npz` per snapshot (`nodes`, `velocity`,
`classification`, `t`) — the default (`output_format = npz`) and the source of truth
for `warm_start` and any downstream reader. For Tecplot/ParaView or MATLAB pre/post
scripts, `output_format = dat` (or `both`) additionally writes one ASCII `.dat` per
snapshot via `ccmplus/io_output.py`, named like the `.npz` with the extension swapped:

- `dat_flavor = tecplot` writes a `TITLE` / `VARIABLES` / `ZONE ... DATAPACKING=POINT`
  file; `dat_flavor = plain` writes a single `#`-commented column-name line then
  whitespace columns. Both start with a `#` provenance block (case, time+unit, `dx`,
  `length_unit`, resolved bounds, `roi_mode`, `n_nodes`, version) — Tecplot ignores
  `#` lines, so the block is safe in both flavors.
- Columns are `x y z u v w` plus every per-node field the reconstruction produced
  (currently `classification`); the list is derived from the result, not hard-coded.
- `dat_precision` sets significant digits; `dat_order` (`C`/`F`) sets the ravel order.
  Because Tecplot POINT requires `I` to vary fastest, the zone `I,J,K` are the reversed
  grid dims for `C` and the grid dims for `F`; a round-trip test proves exact recovery
  for both. The default (no `output_*` keys) writes the `.npz` exactly as before and no
  `.dat`.

Output reference frame (`output_frame`, default `lab`) selects what velocity is written,
as a post-solve transform in the output layer only (the solver is untouched):

- `lab` writes the reconstructed lab-frame velocity exactly as solved.
- `body` writes the relative velocity `v_rel(x) = v_lab(x) - u_body(x)`, where
  `u_body(x) = U + omega x (x - X_s)` is evaluated from the *same* body velocity field
  the solver pinned into the shell/solid for that snapshot. Solid and shell nodes
  (`classification <= 0`) therefore go to zero to solver tolerance, giving the
  stationary-body flow-around picture method-comparison figures need. It applies
  identically to `.npz` and `.dat`, and the body-frame files record `output_frame`, the
  subtracted `U`/`omega`, and `X_s` (npz keys / `#` header lines) so they are
  self-describing. Because it is the full rigid-body field (not a translation-only
  subtraction) it stays correct for rotating and non-spherical bodies.
- `comoving_coords` (default `false`) additionally shifts node coordinates by
  `-(X_s - X_s0)` so the body stays fixed across snapshots for animation; velocity is
  unaffected. Coordinates are otherwise never modified.
