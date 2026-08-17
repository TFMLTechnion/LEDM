# INTERFACE.md — the ccmplus solver call contract (Phase 0)

Read-only study of `ccmplus` and `drivers/sphere.py`. This is the boundary the new
input/geometry layer must match so the reconstruction physics stays byte-for-byte
identical. **Nothing in this file is changed by the new layer; it is only called.**

## The one call the driver makes

```python
grid   = RectGrid(domain_min, domain_max, delta)          # uniform Cartesian
config = Config(domain_min, domain_max, delta, kappa=…, solver_rtol=…,
                solver_maxiter=…, boundary_constraints=True, interp_kernel=…, …)
solver = CCMPlus(config, grid)
for step in order:                                        # warm-start chain
    body  = BodyState(X_s, U_s, omega_s, radius, sigma_s,
                      sdf_fn=None, velocity_fn=None)
    frame = FrameData(positions, velocities, uncertainties, body, t)
    res   = solver.reconstruct(frame)                     # -> ReconstructionResult
```

`res.velocity` is `(Ng, 3)`, `res.classification` is `(Ng,)` int8 in {-1,0,+1}, plus
`residual`, `iterations`, `converged`. The solve additionally returns normalized
constraint diagnostics on `SolverInfo.constraints` (`div_rms_norm`, `body_rel`, and
their pass/fail flags) — check those rather than `converged` alone, which only
reflects the scaled saddle system. Node order is `idx = i + Nx*(j + Ny*k)`, DOFs
`3*idx (+0,+1,+2)`. Reference: [drivers/sphere.py:1438-1521](ccmplus/drivers/sphere.py#L1438-L1521)
and [run_ledm.py:184-211](run_ledm.py#L184-L211).

## The three things a body must supply

`CCMPlus.reconstruct` ([reconstruct.py:33-185](ccmplus/reconstruct.py#L33)) derives the
grid, the fluid/shell/solid classification, and the shell velocity **internally** from
two body-supplied quantities. The geometry layer only needs to provide these two:

### 1. Signed-distance field (occupancy) — `BodyState.sdf_fn`
- Signature: `sdf_fn(points_world: (N,3), body: BodyState) -> phi: (N,)`.
- **World-frame points in, negative inside the solid.** Called at every grid node
  (`signed_distance_body`, [sdf.py:21-43](ccmplus/sdf.py#L21)) and at every particle
  (`signed_distance_body_points`).
- If `sdf_fn is None`, ccmplus falls back to the analytic sphere
  `‖x − X_s‖ − radius` (`signed_distance_sphere_points`, [sdf.py:46-54](ccmplus/sdf.py#L46)).
  This is the exact distance the existing sphere driver uses.
- From `phi`, classification is fixed by the paper ([classify.py:8-21](ccmplus/classify.py#L8)):
  `C=-1` if `phi<0` (solid), `C=0` if `0≤phi≤δ/2` (shell), `C=+1` if `phi>δ/2` (fluid).
  Shell half-width is `δ/2`, internal. **Not a user input.**

### 2. Surface velocity `u_Γ` — via `X_s, U_s, omega_s` (or `velocity_fn`)
- `u_gamma(x, body)` ([kinematics.py:8-29](ccmplus/kinematics.py#L8)) returns
  `U_s + omega_s × (x − X_s)` for world-frame points `x` — the rigid-body law, computed
  **once for all shapes**. If `velocity_fn` is given it overrides; otherwise the rigid
  formula is used.
- `u_Γ` is consumed as the Dirichlet target `g` on shell rows (`C==0`) and solid rows
  (`C==-1`) in `build_constraints` ([constraints.py:164-191](ccmplus/constraints.py#L164)),
  and to seed newly exposed nodes in the temporal prior
  ([prior.py:10-36](ccmplus/prior.py#L10)). All three call the same `u_gamma`.

## What this means for the new layer (physics untouched)

- The geometry layer supplies **only** `sdf_fn` (a pose-applying wrapper around a body-frame
  `signed_distance`) and fills `X_s=center`, `U_s=U`, `omega_s=ω(world)`, `radius=bounding_radius`.
  It sets **no** `velocity_fn`, so `u_Γ = U + ω×(x−c)` is exactly the frozen rigid-body law.
- The generic framework step `X_body = Rᵀ(X − c)` → `signed_distance(X_body)` reproduces the
  analytic sphere **bit-for-bit** when the shape is a sphere and the pose is identity (matmul
  by the exact identity matrix and translation are exact), so a sphere driven through the new
  layer and through the old analytic path agree to machine precision.
- Nothing in `ccmplus/` (solver, constraints, classify, prior, kinematics, sdf, grid,
  reconstruct) needs modification: `BodyState` already exposes `sdf_fn`/`velocity_fn` hooks.
- The region-of-interest options (`roi_mode = auto | box | body`, `roi_box`, `roi_pad`,
  `max_grid_nodes`) only choose the `domain_min/domain_max` passed to `RectGrid`/`Config` and
  cull far-field tracks in the input layer (`io_ledm.py`). They change **which** box the frozen
  solver reconstructs, never **how** it reconstructs. `roi_mode = auto` reproduces the previous
  bounds exactly (regression: zero field difference). The node-count guard runs **before**
  `RectGrid(...)` is constructed, so an oversized case fails with a `ValueError` rather than an
  allocation error inside `solve_saddle_point`. Body characteristic length for `body` padding
  comes from `Geometry.characteristic_length()` (`2r` sphere, `2·max(a,b,c)` ellipsoid), never
  a hard-coded sphere assumption.
- The output options (`output_format = npz | dat | both`, `dat_flavor`, `dat_precision`,
  `dat_order`) live entirely in the output layer (`ccmplus/io_output.py`, wired from
  `run_four_file`). They read `res.velocity`/`res.classification` **after** the solver returns
  and serialise them; they never call into or alter the solver. `.npz` stays the source of
  truth (default, and for `warm_start`); `.dat` is an added view. The one correctness invariant
  is index ordering: grid nodes are `idx = i + Nx*(j + Ny*k)` (Fortran ravel, `i` fastest), and
  Tecplot POINT needs `I` fastest, so the zone `I,J,K` are `Nx,Ny,Nz` for `dat_order = F` and
  the **reversed** `Nz,Ny,Nx` for `dat_order = C` — proven by an asymmetric round-trip test.
- The output reference frame (`output_frame = lab | body`, `comoving_coords`) is also a
  post-solve transform in this layer. For `body`, each node's velocity is written as
  `v_rel = v_lab − u_body`, with `u_body` evaluated by the **same** `u_gamma(nodes, body)`
  (`config.py` `BodyState` → `kinematics.u_gamma`) that produced the Dirichlet `u_Γ` the solver
  pinned — so the transform is exactly consistent with the solve and solid/shell nodes vanish to
  solver tolerance (correctness test: max `|v_rel|` over `classification ≤ 0` ≈ `1e-8·U₀`). It is
  the full rigid field `U + ω×(x−X_s)`, valid for rotating / non-spherical bodies. `lab` (default)
  writes `res.velocity` and `grid.nodes` untouched, so the byte-identical npz regression holds.

## `BodyState` / `FrameData` fields (from [config.py:89-107](ccmplus/config.py#L89))

| struct | field | meaning |
|--------|-------|---------|
| BodyState | `X_s (3,)` | body center, world frame |
| | `U_s (3,)` | translational velocity, world frame |
| | `omega_s (3,)` | angular velocity, world frame (for `ω×(x−c)`) |
| | `radius` | bounding radius (proximity length / culling) |
| | `sigma_s` | near-wall proximity weighting length |
| | `sdf_fn` | occupancy hook (world pts → signed distance) |
| | `velocity_fn` | optional surface-velocity override (unused here) |
| FrameData | `positions (Nt,3)` | track positions, world frame |
| | `velocities (Nt,3)` | track velocities |
| | `uncertainties (Nt,)` | per-track 1σ (data weight) |
| | `body` | the `BodyState` above |
| | `t` | time stamp |
