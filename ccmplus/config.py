"""Configuration and data structure dataclasses."""

from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Callable
from typing import Any
import numpy as np

SdfFunction = Callable[[np.ndarray, "BodyState"], np.ndarray]
VelocityFunction = Callable[[np.ndarray, "BodyState"], np.ndarray]


@dataclass
class Config:
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    delta: float

    kappa: float = 1.0
    solver_rtol: float = 1e-6
    solver_maxiter: int = 2000

    # ---- constraint-satisfaction tolerances (physics-level, not MINRES) ----
    # MINRES can converge on the scaled saddle system while the constraints are
    # still out of tolerance, so these are checked separately after the solve
    # and reported in SolverInfo.constraints.
    #   constraint_div_tol  : dimensionless Delta*rms(div u)/U_ref over fluid rows
    #   constraint_body_tol : relative residual of the no-slip identity rows
    constraint_div_tol: float = 1e-3
    constraint_body_tol: float = 1e-3
    output_dir: str = "./outputs"
    enable_field_smoothing: bool = False
    field_smoothing_type: str = "laplacian"
    lambda_laplacian: float = 0.0
    lambda_gradient: float = 0.0
    smoothing_exclude_mask: bool = True
    smoothing_exclude_shell: bool = True
    smoothing_no_cross_mask: bool = True
    smoothing_spacing_scaled: bool = True
    smoothing_componentwise: bool = True
    laplacian_taper: bool = False
    lap_taper_mm: float = 2.0

    enable_irls: bool = False
    irls_loss: str = "huber"
    irls_threshold_sigma: float = 2.5
    irls_max_outer: int = 5
    irls_tol: float = 1e-3
    irls_min_weight: float = 1e-4

    # Boundary (no-slip) constraints on the body.
    #
    #   True  -- the shell/solid Dirichlet rows u_j = u_Gamma(x_j) are assembled,
    #            transition nodes are seeded from the body velocity, and the
    #            divergence stencils are body-aware (fluid-side only).
    #   False -- those constraint rows are NOT assembled and divergence is
    #            enforced uniformly across every interior node, body included.
    #
    # IMPORTANT -- what False is NOT. It is *not* an all-fluid reconstruction.
    # The body classification is still computed and still governs interpolation
    # and smoothing: the interpolation matrix is masked to C == 1, so tracks
    # never write into the body interior, and the coverage-adaptive smoothing
    # still refuses to emit edges that touch shell or solid nodes. Turning this
    # off removes the *constraints*, not the mask. Treat it as an ablation of
    # the boundary conditions, and do not describe it as a body-agnostic
    # baseline.
    boundary_constraints: bool = True

    # DEPRECATED alias for boundary_constraints, kept so existing configs and
    # scripts keep working. Leave as None to use boundary_constraints; setting
    # it to a bool overrides boundary_constraints and emits a DeprecationWarning.
    enable_lema: bool | None = None

    # ---- interpolation kernel ----
    # "wide" = tensor-product cubic B-spline (4x4x4 = 64-node support, linear
    # precision); "gaussian" = Shepard-Gaussian footprint; "trilinear" = compact
    # 8-corner stencil.
    interp_kernel: str = "wide"
    kernel_radius_cells: float = 2.0   # "gaussian" kernel ONLY (ignored by "wide")
    kernel_sigma_cells: float = 1.0    # "gaussian" kernel ONLY (ignored by "wide")

    # ---- coverage-adaptive smoothness (Eqs. 13-14) ----
    # ONE global weight, identical whether boundary_constraints is on or off and
    # across datasets. Negligible where tracks are dense (w -> 0 protects covered
    # cells, so a large value does no harm there), dominant in empty cells where
    # it must beat kappa to kill the frozen warm-start mode. The neighbour
    # difference is UNSCALED (no 1/Delta), so lambda_coverage is dimensionless
    # relative to the data term -- see ccmplus/operators.py.
    lambda_coverage: float = 0.0       # 'lambda_c' in parameter files
    coverage_ref_count: float = 1.0    # c_0: track count giving half smoothing weight

    # Proximity reweighting (v1 near-wall particle downweighting). Decoupled from
    # boundary_constraints, so both modes share data weights (wake identity).
    enable_proximity_reweight: bool = False

    # Coverage-adaptive Tikhonov (per-node kappa -> ~0 at empty nodes). Experimental;
    # OFF in the conservative preset, which uses uniform config.kappa everywhere.
    enable_coverage_kappa: bool = False

    # ---- v2: time pipeline (Problem 1) ----
    expected_rise_speed_ms: float = 0.0   # 0 disables the rise-speed sanity check

    # ---- v2 final-run: coverage-gated temporal prior ----
    # When True, the prior is gated by local track support: exposed nodes are
    # seeded with u_Gamma only where supported (else 0), and unsupported fluid
    # nodes decay their inherited prior by prior_decay_factor each snapshot.
    # Prevents the prior-dominated / frozen wake. Off = v1 prior behaviour.
    enable_coverage_gated_prior: bool = False
    prior_decay_factor: float = 0.5        # multiply unsupported-fluid prior / snapshot
    coverage_support_mm: float = 1.0       # local-support radius (= 1 grid spacing)

    # Block-Jacobi (diagonal-of-H) MINRES preconditioner for the saddle system.
    # Helps when strong coverage smoothing makes the upper-left block stiff.
    use_jacobi_precond: bool = False

    def __post_init__(self) -> None:
        # Resolve the deprecated enable_lema alias onto boundary_constraints,
        # then mirror the resolved value back so that code still reading
        # ``config.enable_lema`` keeps seeing the truth rather than None.
        if self.enable_lema is not None:
            warnings.warn(
                "Config(enable_lema=...) is deprecated and will be removed in a "
                "future release; use Config(boundary_constraints=...) instead. "
                "Note that boundary_constraints=False removes the no-slip "
                "constraint rows but still masks interpolation and smoothing by "
                "the body classification -- it is not an all-fluid baseline.",
                DeprecationWarning, stacklevel=2,
            )
            self.boundary_constraints = bool(self.enable_lema)
        self.enable_lema = bool(self.boundary_constraints)


@dataclass
class BodyState:
    X_s: np.ndarray       # (3,) center position
    U_s: np.ndarray       # (3,) translational velocity
    omega_s: np.ndarray   # (3,) angular velocity
    radius: float
    sigma_s: float        # positional uncertainty for proximity reweighting
    sdf_fn: SdfFunction | None = None
    velocity_fn: VelocityFunction | None = None


@dataclass
class FrameData:
    positions: np.ndarray     # (Nt, 3)
    velocities: np.ndarray    # (Nt, 3)
    uncertainties: np.ndarray # (Nt,)
    body: BodyState
    t: float


@dataclass
class ReconstructionResult:
    velocity: np.ndarray       # (Ng, 3)
    classification: np.ndarray # (Ng,) int8 in {-1, 0, +1}
    residual: float
    iterations: int
    converged: bool = False
    smoothing_stats: dict[str, Any] | None = None
