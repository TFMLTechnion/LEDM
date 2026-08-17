"""Top-level per-timestep reconstruction driver."""

from __future__ import annotations
import numpy as np

from ccmplus.config import Config, FrameData, ReconstructionResult
from ccmplus.grid import RectGrid
from ccmplus.sdf import signed_distance_body, signed_distance_body_points
from ccmplus.classify import classify, near_wall_fluid_set, transition_flags
from ccmplus.interp import build_interpolation_matrix
from ccmplus.constraints import build_constraints
from ccmplus.solver import build_weight_matrix, solve_saddle_point, solve_with_irls
from ccmplus.prior import apply_prior_correction, apply_coverage_gated_prior
from ccmplus.operators import (
    build_field_smoothing_operators,
    build_coverage_adaptive_smoothing_operator,
    coverage_count_by_radius,
)


class CCMPlus:
    """Constrained Cost Minimisation solver with dynamic body masking.

    Maintains per-timestep state (previous solution, previous classification)
    to enable warm-starting and transition-flag computation.
    """

    def __init__(self, config: Config, grid: RectGrid) -> None:
        self.config = config
        self.grid = grid
        self._x_prev: np.ndarray | None = None
        self._C_prev: np.ndarray | None = None

    def reconstruct(self, frame: FrameData) -> ReconstructionResult:
        """Run one reconstruction timestep.

        Steps (``boundary_constraints`` on / off):
          1. Signed distance → ternary classification → near-wall set
          2. Transition flags from previous classification
          3. Prior correction (on: seed newly exposed nodes with the body
             velocity; off: carry x_prev forward unchanged, no seeding)
          4. Interpolation matrix A -- masked to open fluid in BOTH cases
          5. Diagonal weight matrix W (optional proximity reweighting)
          6. Constraint matrix B and targets g (on: fluid-side divergence +
             body identity rows; off: uniform divergence over all interior
             nodes, no body rows)
          7. MINRES saddle-point solve
          8. Cache solution and classification for next timestep
        """
        # Body boundary (no-slip) constraints. Note this gates the CONSTRAINT
        # rows and the prior seeding only: interpolation and smoothing are
        # masked by the body classification either way (see Config).
        boundary_constraints = bool(getattr(self.config, "boundary_constraints",
                                            True))

        # 1. SDF, classification, near-wall set
        phi = signed_distance_body(self.grid, frame.body)
        C = classify(phi, self.grid.delta)
        N_mask = near_wall_fluid_set(C, self.grid)

        # 2. Transition flags
        tau = transition_flags(C, self._C_prev)

        # 3. Prior
        if getattr(self.config, "enable_coverage_gated_prior", False):
            # v2 final-run: gate the prior by local track support (both modes share
            # the decay rule; only LE-DM seeds u_Gamma at supported exposed nodes).
            from scipy.spatial import cKDTree
            r_sup = float(getattr(self.config, "coverage_support_mm", 1.0))
            if frame.positions.shape[0] > 0:
                tree = cKDTree(frame.positions)
                support = np.asarray(
                    tree.query_ball_point(self.grid.nodes, r=r_sup, return_length=True),
                    dtype=np.int32)
            else:
                support = np.zeros(self.grid.size, dtype=np.int32)
            x_prior = apply_coverage_gated_prior(
                self._x_prev, tau, C, support, frame.body, self.grid,
                enable_lema=boundary_constraints,
                decay_factor=float(getattr(self.config, "prior_decay_factor", 0.5)),
            )
        elif boundary_constraints:
            x_prior = apply_prior_correction(self._x_prev, tau, frame.body, self.grid)
        else:
            # Boundary constraints off: carry the previous solution forward
            # unchanged, with no body-velocity seeding at transition nodes.
            x_prior = (self._x_prev.copy() if self._x_prev is not None
                       else np.zeros(3 * self.grid.size))

        # 4. Interpolation matrix. NOTE: masked to open fluid (C == 1)
        #    regardless of boundary_constraints -- the body interior is excluded
        #    from interpolation in both cases, which is why the off mode is an
        #    ablation of the boundary conditions and not an all-fluid baseline.
        #    Wide cubic-B-spline kernel so a sparse track informs a whole
        #    neighbourhood of nodes.
        A = build_interpolation_matrix(
            frame.positions, self.grid, allowed_nodes=(C == 1),
            kernel=getattr(self.config, "interp_kernel", "wide"),
            radius_cells=getattr(self.config, "kernel_radius_cells", 2.0),
            sigma_cells=getattr(self.config, "kernel_sigma_cells", 1.0),
        )

        # 5. Weight matrix.
        # Proximity reweighting is DECOUPLED from boundary_constraints and is
        # off by default, so both modes use identical data weights and agree
        # outside a 2-cell shell by construction. The near-wall no-slip
        # behaviour comes from the body Dirichlet rows + coverage smoothing, not
        # from downweighting near-wall tracks. Set enable_proximity_reweight to
        # re-enable the near-wall downweighting.
        phi_particles = signed_distance_body_points(frame.positions, frame.body)
        use_prox = boundary_constraints and getattr(self.config, "enable_proximity_reweight", False)
        sigma_w = frame.body.sigma_s if use_prox else 0.0
        W = build_weight_matrix(phi_particles, frame.uncertainties, sigma_w)

        # 6. Constraint matrix
        B, g = build_constraints(C, N_mask, phi, frame.body, self.grid,
                                  enable_body=boundary_constraints)

        # 7. Solve saddle-point system
        y = frame.velocities.ravel()
        smoothing_operators, smoothing_stats = build_field_smoothing_operators(
            self.grid, C, self.config, phi=phi
        )

        # 7b. Per-node local track support c_j (Eq. 13): the number of TRACKS
        #     within a radius of Delta (one grid spacing) of the node. This is a
        #     direct spatial query via cKDTree -- deliberately NOT the column
        #     count of A, which measures the interpolation footprint (how many
        #     particles can reach the node through the kernel) rather than how
        #     many tracks are physically near it, and which would change meaning
        #     whenever the kernel changed. Only computed when a coverage-aware
        #     feature is enabled.
        ref_count = getattr(self.config, "coverage_ref_count", 1.0)
        lambda_cov = getattr(self.config, "lambda_coverage", 0.0)
        use_cov_kappa = getattr(self.config, "enable_coverage_kappa", False)
        if lambda_cov > 0.0 or use_cov_kappa:
            coverage_count = coverage_count_by_radius(
                frame.positions, self.grid, radius=self.grid.delta,
                node_mask=(C == 1),
            )
        else:
            coverage_count = None

        # Coverage-adaptive Tikhonov (Problem 2b, the frozen-mode fix): the prior
        # pull kappa is scaled down toward ~0 at zero-coverage nodes, so those
        # nodes are determined by spatial smoothing + divergence + data-boundary
        # coupling, NOT by the temporal prior. OFF in the conservative preset
        # (enable_coverage_kappa=False) -> uniform kappa*I as in v1 production.
        if use_cov_kappa:
            kappa = float(self.config.kappa)
            cov_frac = coverage_count / (coverage_count + ref_count)
            KAPPA_FLOOR_FRAC = 0.01
            kappa_node = kappa * (KAPPA_FLOOR_FRAC + (1.0 - KAPPA_FLOOR_FRAC) * cov_frac)
            kappa_diag = np.repeat(kappa_node, 3)             # (3*Ng,)
        else:
            kappa_diag = None                                  # uniform config.kappa

        # 7c. v2: coverage-adaptive smoothness (Problem 2b). Weak Laplacian,
        #     inversely weighted by support, body-masked. ONE frozen global weight
        #     lambda_c, shared by both modes. Negligible where dense
        #     (w_cov->0), dominant in empty cells where it anchors nodes to their
        #     neighbours instead of to the temporal prior.
        if lambda_cov > 0.0:
            cov_res = build_coverage_adaptive_smoothing_operator(
                self.grid, C, coverage_count, ref_count,
            )
            if cov_res.matrix.shape[0] > 0:
                smoothing_operators = list(smoothing_operators) + [
                    (float(lambda_cov), cov_res.matrix)
                ]
            smoothing_stats = dict(smoothing_stats)
            smoothing_stats["coverage_adaptive"] = cov_res.stats

        if self.config.enable_irls:
            x, lam, info = solve_with_irls(
                A, W, y, B, g, x_prior, self.config, smoothing_operators,
                frame.uncertainties,
            )
        else:
            x, lam, info = solve_saddle_point(
                A, W, y, B, g, x_prior, self.config, smoothing_operators,
                kappa_diag=kappa_diag,
            )

        # 8. Cache for next timestep
        self._x_prev = x
        self._C_prev = C

        return ReconstructionResult(
            velocity=x.reshape(-1, 3),
            classification=C,
            residual=info.residual,
            iterations=info.iterations,
            converged=info.converged,
            smoothing_stats=smoothing_stats,
        )

    def reset(self) -> None:
        """Clear cached state (use between unrelated sequences of frames)."""
        self._x_prev = None
        self._C_prev = None
