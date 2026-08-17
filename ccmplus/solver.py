"""Saddle-point assembly and MINRES solve."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ccmplus.config import Config

_log = logging.getLogger(__name__)

_RESTART_MAXITER = 20_000


@dataclass
class SolverInfo:
    residual: float
    iterations: int
    converged: bool
    block_residuals: dict | None = None
    constraints: "ConstraintDiagnostics | None" = None


@dataclass
class ConstraintDiagnostics:
    """How well the solved field actually satisfies each constraint family.

    The two families have genuinely different natural scales and must not share
    a tolerance:

    * **Fluid divergence rows.** A row evaluates a finite-difference divergence,
      units velocity/length, so its raw residual shrinks or grows with the grid
      and the flow speed and an absolute threshold is meaningless. The reported
      quantity is the DIMENSIONLESS

          div_rms_norm = Delta * rms_j(div u_j) / U_ref
          div_max_norm = Delta * max_j|div u_j| / U_ref

      i.e. the divergence error expressed as a fraction of the velocity scale
      over one cell. ``U_ref`` is the peak reconstructed speed.

    * **Body identity rows.** These are Dirichlet rows ``u_j = u_Gamma(x_j)``
      with an O(1) right-hand side, so a plain relative norm is the right
      measure and it should hold to roughly the MINRES tolerance.
    """
    div_rms_norm: float
    div_max_norm: float
    div_rms_abs: float
    div_max_abs: float
    body_rel: float
    body_max_abs: float
    u_ref: float
    n_div_rows: int
    n_body_rows: int
    div_tol: float
    body_tol: float
    div_ok: bool
    body_ok: bool

    @property
    def ok(self) -> bool:
        return self.div_ok and self.body_ok

    def as_dict(self) -> dict:
        return {
            "div_rms_norm": self.div_rms_norm,
            "div_max_norm": self.div_max_norm,
            "div_rms_abs": self.div_rms_abs,
            "div_max_abs": self.div_max_abs,
            "body_rel": self.body_rel,
            "body_max_abs": self.body_max_abs,
            "u_ref": self.u_ref,
            "n_div_rows": self.n_div_rows,
            "n_body_rows": self.n_body_rows,
            "div_tol": self.div_tol,
            "body_tol": self.body_tol,
            "div_ok": self.div_ok,
            "body_ok": self.body_ok,
        }


def constraint_diagnostics(
    B_csr: sp.csr_matrix,
    g: np.ndarray,
    x_sol: np.ndarray,
    body_mask: np.ndarray,
    delta: float,
    *,
    div_tol: float = 1e-3,
    body_tol: float = 1e-3,
    u_ref: float | None = None,
) -> ConstraintDiagnostics:
    """Normalized constraint residuals for the solved field.

    body_mask : (m,) bool, True on Dirichlet (identity) rows of B.
    u_ref     : velocity scale; defaults to the peak reconstructed speed.
    """
    Bx_g = B_csr @ x_sol - g
    div_res = Bx_g[~body_mask]
    body_res = Bx_g[body_mask]

    if u_ref is None:
        speeds = np.linalg.norm(np.asarray(x_sol, dtype=float).reshape(-1, 3), axis=1)
        u_ref = float(speeds.max()) if speeds.size else 0.0
    u_ref = float(u_ref)
    scale = u_ref if u_ref > 1e-300 else 1.0

    if div_res.size:
        div_rms_abs = float(np.sqrt(np.mean(div_res ** 2)))
        div_max_abs = float(np.max(np.abs(div_res)))
    else:
        div_rms_abs = div_max_abs = 0.0

    if body_res.size:
        body_max_abs = float(np.max(np.abs(body_res)))
        body_rel = float(
            np.linalg.norm(body_res) / (np.linalg.norm(g[body_mask]) + 1e-300)
        )
    else:
        body_max_abs = 0.0
        body_rel = 0.0

    div_rms_norm = float(delta) * div_rms_abs / scale
    div_max_norm = float(delta) * div_max_abs / scale

    return ConstraintDiagnostics(
        div_rms_norm=div_rms_norm,
        div_max_norm=div_max_norm,
        div_rms_abs=div_rms_abs,
        div_max_abs=div_max_abs,
        body_rel=body_rel,
        body_max_abs=body_max_abs,
        u_ref=u_ref,
        n_div_rows=int(div_res.size),
        n_body_rows=int(body_res.size),
        div_tol=float(div_tol),
        body_tol=float(body_tol),
        div_ok=bool(div_rms_norm <= div_tol),
        body_ok=bool(body_rel <= body_tol),
    )


def build_weight_matrix(
    phi_at_particles: np.ndarray,
    uncertainties: np.ndarray,
    sigma_s: float,
) -> sp.csr_matrix:
    """Diagonal weight matrix W of shape (3*n_p, 3*n_p).

    W_ii = (1/sigma_i^2) * [1 - exp(-max(0, phi_i) / sigma_s)]   (Eq. 9)

    phi_at_particles: (n_p,) signed distance at each particle location.
    uncertainties: (n_p,) per-particle velocity uncertainty sigma_i.
    sigma_s: body-detection positional uncertainty (from BodyState).
    """
    n_p = len(uncertainties)
    phi_pos = np.maximum(0.0, phi_at_particles)
    if sigma_s > 1e-12:
        proximity_factor = 1.0 - np.exp(-phi_pos / sigma_s)
    else:
        proximity_factor = np.ones(n_p)
    w = proximity_factor / (uncertainties ** 2)
    return sp.diags(np.repeat(w, 3), format="csr")


def _block_residuals(
    B_csr: sp.csr_matrix,
    g: np.ndarray,
    x_sol: np.ndarray,
    body_mask: np.ndarray,
) -> dict:
    """Per-block residuals in the original (unscaled) constraint space."""
    Bx_g = B_csr @ x_sol - g
    if body_mask.any():
        body_rel = float(
            np.linalg.norm(Bx_g[body_mask])
            / (np.linalg.norm(g[body_mask]) + 1e-300)
        )
    else:
        body_rel = 0.0
    div_abs = float(np.linalg.norm(Bx_g[~body_mask]))
    return {"body_rel": body_rel, "div_abs": div_abs}


def solve_saddle_point(
    A: sp.csr_matrix,
    W: sp.csr_matrix,
    y: np.ndarray,
    B: sp.csr_matrix,
    g: np.ndarray,
    x_prior: np.ndarray,
    config: Config,
    smoothing_operators: list[tuple[float, sp.csr_matrix]] | None = None,
    kappa_diag: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, SolverInfo]:
    """Assemble and solve the saddle-point system via MINRES.

    System (scaled):
      [H     B_eff^T] [x  ]   [r    ]
      [B_eff  0     ] [lam] = [g_eff]

    where B_eff = D @ B, g_eff = D @ g, and D is a diagonal scaling that
    equalises the body-constraint (identity) row norms with the fluid RHS
    norm, preventing MINRES from ignoring the body constraint block.

    kappa_diag : optional (3*Ng,) per-DOF Tikhonov weights. v2 uses this to make
    the prior pull coverage-adaptive -- ~0 at zero-coverage nodes so they are
    determined by spatial smoothing and data-boundary coupling rather than by the
    temporal prior (kills the frozen warm-start mode). When None, the scalar
    config.kappa * I is used (v1 behaviour).

    Returns (x, lam_unscaled, info).
    """
    Ng3 = A.shape[1]
    m = B.shape[0]

    # H = A^T W A + K_tikhonov (symmetric PSD), K_tikhonov = diag(kappa_diag) or kappa*I
    WA = W @ A
    if kappa_diag is not None:
        kvec = np.asarray(kappa_diag, dtype=float)
        if kvec.shape != (Ng3,):
            raise ValueError(f"kappa_diag must have shape ({Ng3},), got {kvec.shape}")
        H = A.T @ WA + sp.diags(kvec, format="csr")
        r = A.T @ (W @ y) + kvec * x_prior
    else:
        kappa = config.kappa
        H = A.T @ WA + kappa * sp.eye(Ng3, format="csr")
        r = A.T @ (W @ y) + kappa * x_prior
    if smoothing_operators:
        for weight, operator in smoothing_operators:
            if weight > 0.0 and operator.shape[0] > 0:
                H = H + float(weight) * (operator.T @ operator)

    # ------------------------------------------------------------------
    # Body-row scaling: identity rows (nnz == 1) in B are Dirichlet
    # constraints.  Scale so their RHS norm equals ||r||, forcing MINRES
    # to satisfy them at the requested rtol.
    # ------------------------------------------------------------------
    B_csr = B.tocsr()
    nnz_per_row = np.diff(B_csr.indptr)
    body_mask = nnz_per_row == 1
    n_body_rows = int(body_mask.sum())

    r_norm = float(np.linalg.norm(r))
    row_scales = np.ones(m, dtype=float)

    if n_body_rows > 0:
        g_body_norm = float(np.linalg.norm(g[body_mask]))
        if g_body_norm > 1e-300 and r_norm > 1e-300:
            s_body = r_norm / g_body_norm
            row_scales[body_mask] = s_body
            _log.debug(
                "body-constraint scaling: n_body=%d s_body=%.3e "
                "||r||=%.3e ||g_body||=%.3e",
                n_body_rows, s_body, r_norm, g_body_norm,
            )

    D = sp.diags(row_scales, format="csr")
    B_eff = D @ B_csr
    g_eff = row_scales * g

    # Saddle-point block matrix (symmetric indefinite)
    K = sp.bmat([[H, B_eff.T], [B_eff, None]], format="csr")
    rhs = np.concatenate([r, g_eff])

    # Optional block-Jacobi preconditioner: diag(H)^-1 on the velocity block,
    # identity on the lambda block. SPD, cheap; helps when strong coverage
    # smoothing makes H stiff. (Constructed only when requested.)
    M = None
    if getattr(config, "use_jacobi_precond", False):
        Hdiag = np.asarray(H.diagonal(), dtype=float)
        Hdiag = np.where(Hdiag > 1e-12, Hdiag, 1.0)
        m_inv = np.concatenate([1.0 / Hdiag, np.ones(m)])
        M = spla.LinearOperator((Ng3 + m, Ng3 + m), matvec=lambda v: m_inv * v)

    # Initial guess: warm-start velocity block with x_prior, zero lambda
    x0 = np.concatenate([x_prior, np.zeros(m)])

    iters = [0]

    def _cb(xk):
        iters[0] += 1

    sol, flag = spla.minres(
        K,
        rhs,
        x0=x0,
        rtol=config.solver_rtol,
        maxiter=config.solver_maxiter,
        M=M,
        callback=_cb,
    )

    res_norm = float(np.linalg.norm(K @ sol - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    relative_residual = res_norm / (rhs_norm + 1e-300)

    # Per-block residuals in original (unscaled) constraint space
    blk = _block_residuals(B_csr, g, sol[:Ng3], body_mask)
    blk["full_rel"] = relative_residual

    # Safety-net restart if body constraint not satisfied
    if blk["body_rel"] > 1e-3 and iters[0] < _RESTART_MAXITER:
        warnings.warn(
            f"MINRES body_rel={blk['body_rel']:.2e} after {iters[0]} iters. "
            f"Restarting (maxiter={_RESTART_MAXITER}).",
            RuntimeWarning, stacklevel=2,
        )
        sol, flag = spla.minres(
            K, rhs, x0=sol,
            rtol=config.solver_rtol,
            maxiter=_RESTART_MAXITER,
            M=M,
            callback=_cb,
        )
        res_norm = float(np.linalg.norm(K @ sol - rhs))
        relative_residual = res_norm / (rhs_norm + 1e-300)
        blk = _block_residuals(B_csr, g, sol[:Ng3], body_mask)
        blk["full_rel"] = relative_residual

    # Normalized constraint diagnostics: the quantities a user should judge the
    # solve by, and the ones the tests assert on.
    cdiag = constraint_diagnostics(
        B_csr, g, sol[:Ng3], body_mask, config.delta,
        div_tol=float(getattr(config, "constraint_div_tol", 1e-3)),
        body_tol=float(getattr(config, "constraint_body_tol", 1e-3)),
    )
    blk["div_rms_norm"] = cdiag.div_rms_norm
    blk["div_max_norm"] = cdiag.div_max_norm

    _log.info(
        "solve_saddle_point: iters=%d converged=%s body_rel=%.2e "
        "div_abs=%.2e div_rms_norm=%.2e full_rel=%.2e",
        iters[0], flag == 0, blk["body_rel"], blk["div_abs"],
        cdiag.div_rms_norm, relative_residual,
    )

    # Surface a physics-level miss loudly: MINRES can report "converged" on the
    # scaled saddle system while the divergence or no-slip constraint is still
    # out of tolerance.
    if not cdiag.div_ok:
        warnings.warn(
            f"divergence constraint not met: Delta*rms(div u)/U_ref = "
            f"{cdiag.div_rms_norm:.2e} > constraint_div_tol = {cdiag.div_tol:.2e} "
            f"(over {cdiag.n_div_rows} fluid rows, U_ref={cdiag.u_ref:.3e}). "
            f"Tighten minres_tol / raise minres_maxit, or enable "
            f"use_jacobi_precond.",
            RuntimeWarning, stacklevel=2,
        )
    if not cdiag.body_ok:
        warnings.warn(
            f"no-slip (body identity) constraint not met: relative residual "
            f"{cdiag.body_rel:.2e} > constraint_body_tol = {cdiag.body_tol:.2e} "
            f"over {cdiag.n_body_rows} rows.",
            RuntimeWarning, stacklevel=2,
        )

    info = SolverInfo(
        residual=relative_residual,
        iterations=iters[0],
        converged=(flag == 0),
        block_residuals=blk,
        constraints=cdiag,
    )

    # Unscale lambda before returning (lambda unused downstream, but kept correct)
    lam_orig = sol[Ng3:] / row_scales

    return sol[:Ng3], lam_orig, info


def _robust_weight(r: np.ndarray, s: np.ndarray, loss: str) -> np.ndarray:
    """Per-particle IRLS weight for a given robust loss function.

    r : (n_p,) per-particle residual norms
    s : (n_p,) per-particle scale thresholds (irls_threshold_sigma * sigma_i)
    loss : one of "huber", "tukey", "cauchy"

    Returns omega in (0, 1] (or 0 for Tukey outliers).
    """
    eps = 1e-300
    if loss == "huber":
        omega = np.where(r <= s, np.ones_like(r), s / np.maximum(r, eps))
    elif loss == "tukey":
        t = r / np.maximum(s, eps)
        omega = np.where(t < 1.0, (1.0 - t * t) ** 2, np.zeros_like(r))
    elif loss == "cauchy":
        t = r / np.maximum(s, eps)
        omega = 1.0 / (1.0 + t * t)
    else:
        raise ValueError(f"Unknown irls_loss {loss!r}; use 'huber', 'tukey', or 'cauchy'.")
    return omega


def solve_with_irls(
    A: sp.csr_matrix,
    W: sp.csr_matrix,
    y: np.ndarray,
    B: sp.csr_matrix,
    g: np.ndarray,
    x_prior: np.ndarray,
    config: Config,
    smoothing_operators: list[tuple[float, sp.csr_matrix]] | None,
    uncertainties: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, SolverInfo]:
    """IRLS outer loop wrapping solve_saddle_point.

    Modifies only the data-fidelity weight W between iterations.
    Hard constraints (B, g) and the Tikhonov/smoothing terms are fixed.

    uncertainties: (n_p,) per-particle measurement sigma_i [same units as y].
    """
    n_p = len(uncertainties)
    w_base = W.diagonal()[::3]       # per-particle base weights, shape (n_p,)
    s = config.irls_threshold_sigma * uncertainties   # (n_p,) scale thresholds

    omega = np.ones(n_p)             # initial: all inliers, no reweighting
    x_cur = x_prior.copy()
    lam_final = np.zeros(B.shape[0])
    info_final = SolverInfo(residual=float("nan"), iterations=0, converged=False)

    for _ in range(config.irls_max_outer):
        w_eff = w_base * np.maximum(omega, config.irls_min_weight)
        W_eff = sp.diags(np.repeat(w_eff, 3), format="csr")

        x_new, lam_final, info_final = solve_saddle_point(
            A, W_eff, y, B, g, x_cur, config, smoothing_operators
        )

        # Per-particle velocity residual
        r_vec = y - A @ x_new          # (3*n_p,)
        r_p = np.linalg.norm(r_vec.reshape(n_p, 3), axis=1)  # (n_p,)

        omega = _robust_weight(r_p, s, config.irls_loss)

        delta_rel = np.linalg.norm(x_new - x_cur) / (np.linalg.norm(x_cur) + 1e-300)
        x_cur = x_new
        if delta_rel < config.irls_tol:
            break

    return x_cur, lam_final, info_final
