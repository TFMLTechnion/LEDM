"""Constraint matrix B assembly (divergence-free + no-slip conditions)."""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from ccmplus.grid import RectGrid
from ccmplus.config import BodyState
from ccmplus.kinematics import u_gamma  # re-exported; also imported by tests


def _phi_gradient(phi: np.ndarray, grid: RectGrid) -> np.ndarray:
    """Second-order finite-difference gradient of phi at all nodes. Shape (Ng, 3)."""
    Ng = grid.size
    grad = np.zeros((Ng, 3))
    nb = grid.neighbors
    all_idx = np.arange(Ng, dtype=np.intp)
    d = grid.delta

    for ax in range(3):
        neg_nb = nb[:, 2 * ax]
        pos_nb = nb[:, 2 * ax + 1]

        both = (neg_nb >= 0) & (pos_nb >= 0)
        j = all_idx[both]
        grad[j, ax] = (phi[pos_nb[j]] - phi[neg_nb[j]]) / (2.0 * d)

        fwd = (neg_nb < 0) & (pos_nb >= 0)
        j = all_idx[fwd]
        grad[j, ax] = (phi[pos_nb[j]] - phi[j]) / d

        bwd = (neg_nb >= 0) & (pos_nb < 0)
        j = all_idx[bwd]
        grad[j, ax] = (phi[j] - phi[neg_nb[j]]) / d

    return grad


def _fluid_divergence_stencils(
    C: np.ndarray,
    cand: np.ndarray,
    grid: RectGrid,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray, dict]:
    """Select a strictly fluid-side divergence stencil for each candidate node.

    For every candidate node and every axis, exactly one of the following is
    used (first admissible rule wins).  A node is "open fluid" iff ``C == 1``;
    shell (``C == 0``) and solid (``C == -1``) nodes are never admissible, and
    a missing neighbour (index ``-1``, i.e. the domain boundary) is likewise
    never admissible:

      1. centered 2nd order   -- iff BOTH opposite neighbours are open fluid;
      2. one-sided 2nd order  -- else, iff the first TWO nodes on the fluid side
                                 are open fluid;
      3. one-sided 1st order  -- else, iff the FIRST node on the fluid side is
                                 open fluid;
      4. none                 -- else: the axis has no admissible stencil.

    If any axis falls through to rule 4 the whole divergence row is dropped and
    the node is reported as ``insufficiently_resolved``.  There is deliberately
    NO fallback that reaches across the interface: a shell or solid node must
    never appear in a fluid divergence stencil, not even to rescue a row.

    Note that rules 1-3 are mutually exclusive by construction: if both first
    neighbours are open fluid, rule 1 fires, so when it does not fire at most
    one side can have an open-fluid first node.  The choice of "fluid side" is
    therefore unambiguous and needs no surface-normal information.

    Returns ``(per_axis, keep, stats)`` where ``per_axis[comp]`` is a triple of
    ``(rows_local, cols, vals)`` index arrays relative to the kept-node list.
    """
    nb = grid.neighbors
    d = grid.delta
    J = np.where(cand)[0]
    nJ = len(J)

    def _is_fluid(idx: np.ndarray) -> np.ndarray:
        """Elementwise 'is an open-fluid node', safe for -1 (missing) entries."""
        ok = np.zeros(idx.shape, dtype=bool)
        present = idx >= 0
        ok[present] = C[idx[present]] == 1
        return ok

    axis_plan = []
    ok_all = np.ones(nJ, dtype=bool)
    n_centered = n_second = n_first = 0

    for comp in range(3):
        neg_n = nb[J, 2 * comp]
        pos_n = nb[J, 2 * comp + 1]
        neg_ok = _is_fluid(neg_n)
        pos_ok = _is_fluid(pos_n)

        centered = neg_ok & pos_ok                     # rule 1
        use_pos = pos_ok & ~centered                   # fluid side = +axis
        use_neg = neg_ok & ~centered                   # fluid side = -axis
        one_sided = use_pos | use_neg

        # First node on the fluid side, and the axis column to step further along.
        n1 = np.where(use_pos, pos_n, np.where(use_neg, neg_n, -1))
        step = np.where(use_pos, 2 * comp + 1, 2 * comp)
        n2 = np.full(nJ, -1, dtype=nb.dtype)
        m1 = one_sided & (n1 >= 0)
        n2[m1] = nb[n1[m1], step[m1]]

        second = one_sided & _is_fluid(n2)             # rule 2
        first = one_sided & ~second                    # rule 3
        ok_all &= centered | one_sided                 # else rule 4

        s = np.where(use_pos, 1.0, -1.0)
        axis_plan.append((comp, neg_n, pos_n, n1, n2, centered, second, first, s))
        n_centered += int(centered.sum())
        n_second += int(second.sum())
        n_first += int(first.sum())

    keep = J[ok_all]
    local = np.full(len(J), -1, dtype=np.intp)
    local[ok_all] = np.arange(len(keep))

    per_axis: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for comp, neg_n, pos_n, n1, n2, centered, second, first, s in axis_plan:
        rows_l: list[np.ndarray] = []
        cols_l: list[np.ndarray] = []
        vals_l: list[np.ndarray] = []

        # rule 1: (u[+] - u[-]) / (2d)
        m = centered & ok_all
        if m.any():
            r = local[m]
            rows_l += [r, r]
            cols_l += [3 * pos_n[m] + comp, 3 * neg_n[m] + comp]
            vals_l += [np.full(m.sum(), 1.0 / (2.0 * d)),
                       np.full(m.sum(), -1.0 / (2.0 * d))]

        # rule 2: (-3 u[j] + 4 u[n1] - u[n2]) / (2 s d)
        m = second & ok_all
        if m.any():
            r = local[m]
            sc = 1.0 / (2.0 * s[m] * d)
            rows_l += [r, r, r]
            cols_l += [3 * keep[r] + comp, 3 * n1[m] + comp, 3 * n2[m] + comp]
            vals_l += [-3.0 * sc, 4.0 * sc, -1.0 * sc]

        # rule 3: (u[n1] - u[j]) / (s d)
        m = first & ok_all
        if m.any():
            r = local[m]
            sc = 1.0 / (s[m] * d)
            rows_l += [r, r]
            cols_l += [3 * n1[m] + comp, 3 * keep[r] + comp]
            vals_l += [sc, -sc]

        if rows_l:
            per_axis.append((np.concatenate(rows_l),
                             np.concatenate(cols_l),
                             np.concatenate(vals_l)))

    stats = {
        "divergence_candidates": int(nJ),
        "divergence_rows": int(len(keep)),
        "axis_centered": n_centered,
        "axis_one_sided_2nd": n_second,
        "axis_one_sided_1st": n_first,
        "insufficiently_resolved": J[~ok_all],
        "n_insufficiently_resolved": int((~ok_all).sum()),
    }
    return per_axis, keep, stats


def build_constraints(
    C: np.ndarray,
    N_mask: np.ndarray,
    phi: np.ndarray,
    body: BodyState,
    grid: RectGrid,
    *,
    enable_body: bool = True,
    return_diagnostics: bool = False,
):
    """Assemble constraint matrix B (m × 3*Ng) and target vector g (m,).

    Row blocks in order (boundary constraints ON, ``enable_body=True``):
      1. Divergence at open-fluid nodes, using strictly fluid-side stencils
         (see :func:`_fluid_divergence_stencils`).  Near-wall and interior
         fluid nodes share one block; the stencil rule, not the block, decides
         whether a row is centered or one-sided.
      2. Shell identity rows (C==0), three per node
      3. Solid-interior identity rows (C==-1), three per node

    Boundary constraints OFF (``enable_body=False``):
      1. Centered divergence at ALL interior nodes (uniform, body-agnostic).
      Blocks 2-3 are omitted and divergence is enforced through the body
      region as if it were fluid, so the fluid-side stencil hierarchy does not
      apply -- there is no interface to respect in this mode.

    Returns ``(B, g)``, or ``(B, g, diagnostics)`` when ``return_diagnostics``.
    """
    nb = grid.neighbors
    d = grid.delta
    Ng = grid.size
    Ndof = 3 * Ng

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    g_list: list[float] = []
    row = 0

    # ------------------------------------------------------------------ #
    # Block 1: divergence rows                                            #
    # ------------------------------------------------------------------ #
    if enable_body:
        # Candidate set = exactly the nodes that carried a divergence row
        # before the fluid-side rewrite: interior open-fluid nodes plus every
        # near-wall open-fluid node.  Only the *stencil* changed, not which
        # nodes are constrained (minus any node the hierarchy has to drop).
        interior_all = np.all(nb >= 0, axis=1)
        cand = (C == 1) & (interior_all | N_mask)
        per_axis, div_nodes, div_stats = _fluid_divergence_stencils(C, cand, grid)
        n_div = len(div_nodes)
        for r_loc, c_loc, v_loc in per_axis:
            rows += (row + r_loc).tolist()
            cols += c_loc.tolist()
            vals += v_loc.tolist()
        g_list += [0.0] * n_div
        row += n_div
    else:
        # Body-agnostic mode: one uniform centered stencil at every interior
        # node.  No classification is consulted, so no interface can be
        # crossed by definition.
        F_nodes = np.where(np.all(nb >= 0, axis=1))[0]
        n_div = len(F_nodes)
        div_nodes = F_nodes
        div_stats = {
            "divergence_candidates": int(n_div),
            "divergence_rows": int(n_div),
            "axis_centered": int(3 * n_div),
            "axis_one_sided_2nd": 0,
            "axis_one_sided_1st": 0,
            "insufficiently_resolved": np.empty(0, dtype=np.intp),
            "n_insufficiently_resolved": 0,
        }
        if n_div > 0:
            row_base = row + np.arange(n_div)
            for comp in range(3):
                neg_n = nb[F_nodes, 2 * comp]
                pos_n = nb[F_nodes, 2 * comp + 1]
                rows += np.concatenate([row_base, row_base]).tolist()
                cols += np.concatenate([3 * pos_n + comp, 3 * neg_n + comp]).tolist()
                vals += [1.0 / (2.0 * d)] * n_div + [-1.0 / (2.0 * d)] * n_div
            g_list += [0.0] * n_div
            row += n_div

    # ------------------------------------------------------------------ #
    # Blocks 2 & 3: body identity rows (boundary constraints ON only)     #
    # ------------------------------------------------------------------ #
    if enable_body:
        # Block 2: shell identity rows (C==0)
        shell_nodes = np.where(C == 0)[0]
        if len(shell_nodes) > 0:
            n_sh = len(shell_nodes)
            pos_sh = grid.nodes[shell_nodes]
            ug_sh = u_gamma(pos_sh, body)
            row_sh = row + np.arange(3 * n_sh)
            col_sh = 3 * np.repeat(shell_nodes, 3) + np.tile(np.arange(3), n_sh)
            rows += row_sh.tolist()
            cols += col_sh.tolist()
            vals += [1.0] * (3 * n_sh)
            g_list += ug_sh.ravel().tolist()
            row += 3 * n_sh

        # Block 3: solid-interior identity rows (C==-1)
        solid_nodes = np.where(C == -1)[0]
        if len(solid_nodes) > 0:
            n_so = len(solid_nodes)
            pos_so = grid.nodes[solid_nodes]
            ug_so = u_gamma(pos_so, body)
            row_so = row + np.arange(3 * n_so)
            col_so = 3 * np.repeat(solid_nodes, 3) + np.tile(np.arange(3), n_so)
            rows += row_so.tolist()
            cols += col_so.tolist()
            vals += [1.0] * (3 * n_so)
            g_list += ug_so.ravel().tolist()
            row += 3 * n_so

    m = row
    B = sp.coo_matrix((vals, (rows, cols)), shape=(m, Ndof)).tocsr()
    g = np.array(g_list)

    if not return_diagnostics:
        return B, g

    diagnostics = dict(div_stats)
    diagnostics["divergence_row_nodes"] = np.asarray(div_nodes)
    diagnostics["n_body_rows"] = int(m - n_div)
    diagnostics["n_rows"] = int(m)
    return B, g, diagnostics
