"""Sparse Eulerian regularization operators for CCM+."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ccmplus.config import Config
from ccmplus.grid import RectGrid


@dataclass
class OperatorBuildResult:
    matrix: sp.csr_matrix
    stats: dict[str, int | float | bool | str]


def fluid_smoothing_nodes(C: np.ndarray, config: Config) -> np.ndarray:
    """Return nodes allowed to participate in ordinary fluid smoothing."""
    C = np.asarray(C)
    if config.smoothing_exclude_mask:
        allowed = C == 1
    elif config.smoothing_exclude_shell:
        allowed = C != 0
    else:
        allowed = np.ones_like(C, dtype=bool)
    if config.smoothing_exclude_shell:
        allowed &= C != 0
    return allowed


def build_laplacian_smoothing_operator(
    grid: RectGrid,
    C: np.ndarray,
    config: Config,
    phi: np.ndarray | None = None,
) -> OperatorBuildResult:
    """Build componentwise rows for the spacing-scaled vector Laplacian.

    Rows are emitted only for fluid nodes whose full 6-neighbour stencil is
    available and, when requested, entirely inside the allowed fluid smoothing
    region. This skips interface-crossing stencils instead of smoothing through
    the dynamic sphere mask.

    When config.laplacian_taper is True and phi is supplied, each row is scaled
    by sqrt(w_lap) where w_lap(j) = 1 - exp(-(phi_j / lap_taper_mm)^2). This
    applies a proximity taper that reduces smoothing near the body surface
    without a hard interface cut.
    """
    allowed = fluid_smoothing_nodes(C, config)
    d = float(grid.delta) if config.smoothing_spacing_scaled else 1.0
    inv_d2 = 1.0 / (d * d)

    use_taper = config.laplacian_taper and phi is not None
    if use_taper:
        taper_mm = max(float(config.lap_taper_mm), 1e-8)
        # sqrt-scale so that L^T L has diagonal w_lap
        w_taper = np.sqrt(np.maximum(
            1.0 - np.exp(-(np.asarray(phi) / taper_mm) ** 2), 0.0
        ))
    else:
        w_taper = None

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    row = 0
    skipped_boundary = 0
    skipped_mask = 0
    candidate_nodes = int(np.sum(allowed))

    for node in np.where(allowed)[0]:
        nb = grid.neighbors[node]
        if np.any(nb < 0):
            skipped_boundary += 1
            continue
        if config.smoothing_no_cross_mask and not np.all(allowed[nb]):
            skipped_mask += 1
            continue
        if not config.smoothing_componentwise:
            raise NotImplementedError("Only componentwise smoothing is implemented.")
        scale = float(w_taper[node]) if use_taper else 1.0
        for comp in range(3):
            rows.extend([row] * 7)
            cols.extend(
                [3 * node + comp]
                + [3 * int(neighbor) + comp for neighbor in nb]
            )
            vals.extend([scale * (-6.0 * inv_d2)] + [scale * inv_d2] * 6)
            row += 1

    L = sp.coo_matrix((vals, (rows, cols)), shape=(row, 3 * grid.size)).tocsr()
    L.eliminate_zeros()
    stats = {
        "type": "laplacian",
        "rows": int(L.shape[0]),
        "candidate_nodes": candidate_nodes,
        "skipped_boundary": int(skipped_boundary),
        "skipped_mask": int(skipped_mask),
        "skipped_boundary_rows": int(3 * skipped_boundary),
        "skipped_mask_rows": int(3 * skipped_mask),
        "crosses_mask": False,
        "masked_nodes_included": bool(np.any((C != 1) & allowed)),
        "spacing_scaled": bool(config.smoothing_spacing_scaled),
        "componentwise": bool(config.smoothing_componentwise),
        "laplacian_taper": use_taper,
    }
    return OperatorBuildResult(L, stats)


def build_gradient_smoothing_operator(
    grid: RectGrid,
    C: np.ndarray,
    config: Config,
) -> OperatorBuildResult:
    """Build componentwise first-gradient difference rows.

    Each row penalizes a spacing-scaled neighbour difference. Rows are added
    only for pairs of allowed smoothing nodes, so no row crosses the mask
    interface when ``smoothing_no_cross_mask`` is true.
    """
    allowed = fluid_smoothing_nodes(C, config)
    d = float(grid.delta) if config.smoothing_spacing_scaled else 1.0
    inv_d = 1.0 / d
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    row = 0
    skipped_mask = 0
    skipped_boundary = 0

    for node in np.where(allowed)[0]:
        for ax in range(3):
            neighbor = int(grid.neighbors[node, 2 * ax + 1])
            if neighbor < 0:
                skipped_boundary += 1
                continue
            if config.smoothing_no_cross_mask and not allowed[neighbor]:
                skipped_mask += 1
                continue
            if not config.smoothing_componentwise:
                raise NotImplementedError("Only componentwise smoothing is implemented.")
            for comp in range(3):
                rows.extend([row, row])
                cols.extend([3 * neighbor + comp, 3 * node + comp])
                vals.extend([inv_d, -inv_d])
                row += 1

    G = sp.coo_matrix((vals, (rows, cols)), shape=(row, 3 * grid.size)).tocsr()
    G.eliminate_zeros()
    stats = {
        "type": "gradient",
        "rows": int(G.shape[0]),
        "candidate_nodes": int(np.sum(allowed)),
        "skipped_boundary": int(skipped_boundary),
        "skipped_mask": int(skipped_mask),
        "skipped_boundary_rows": int(3 * skipped_boundary),
        "skipped_mask_rows": int(3 * skipped_mask),
        "crosses_mask": False,
        "masked_nodes_included": bool(np.any((C != 1) & allowed)),
        "spacing_scaled": bool(config.smoothing_spacing_scaled),
        "componentwise": bool(config.smoothing_componentwise),
    }
    return OperatorBuildResult(G, stats)


def build_field_smoothing_operators(
    grid: RectGrid,
    C: np.ndarray,
    config: Config,
    phi: np.ndarray | None = None,
) -> tuple[list[tuple[float, sp.csr_matrix]], dict[str, int | float | bool | str]]:
    """Build all enabled field-level smoothing operators.

    phi: (Ng,) signed distance at grid nodes, required for near-wall taper.
    """
    operators: list[tuple[float, sp.csr_matrix]] = []
    stats: dict[str, int | float | bool | str] = {
        "enabled": bool(config.enable_field_smoothing),
        "type": config.field_smoothing_type,
        "lambda_laplacian": float(config.lambda_laplacian),
        "lambda_gradient": float(config.lambda_gradient),
        "laplacian_rows": 0,
        "gradient_rows": 0,
        "smoothing_rows": 0,
        "smoothing_rows_skipped_mask": 0,
        "smoothing_rows_skipped_boundary": 0,
        "smoothing_crosses_mask": False,
        "masked_nodes_included": False,
        "spacing_scaled": bool(config.smoothing_spacing_scaled),
        "laplacian_taper": bool(config.laplacian_taper),
    }
    if not config.enable_field_smoothing:
        return operators, stats

    smoothing_type = config.field_smoothing_type.strip().lower()
    if config.lambda_laplacian > 0.0 and smoothing_type in ("laplacian", "both"):
        result = build_laplacian_smoothing_operator(grid, C, config, phi=phi)
        operators.append((float(config.lambda_laplacian), result.matrix))
        stats["laplacian_rows"] = int(result.stats["rows"])
        stats["smoothing_rows_skipped_mask"] = int(stats["smoothing_rows_skipped_mask"]) + int(
            result.stats["skipped_mask_rows"]
        )
        stats["smoothing_rows_skipped_boundary"] = int(
            stats["smoothing_rows_skipped_boundary"]
        ) + int(result.stats["skipped_boundary_rows"])
        stats["masked_nodes_included"] = bool(stats["masked_nodes_included"]) or bool(
            result.stats["masked_nodes_included"]
        )

    if config.lambda_gradient > 0.0 and smoothing_type in ("gradient", "both"):
        result = build_gradient_smoothing_operator(grid, C, config)
        operators.append((float(config.lambda_gradient), result.matrix))
        stats["gradient_rows"] = int(result.stats["rows"])
        stats["smoothing_rows_skipped_mask"] = int(stats["smoothing_rows_skipped_mask"]) + int(
            result.stats["skipped_mask_rows"]
        )
        stats["smoothing_rows_skipped_boundary"] = int(
            stats["smoothing_rows_skipped_boundary"]
        ) + int(result.stats["skipped_boundary_rows"])
        stats["masked_nodes_included"] = bool(stats["masked_nodes_included"]) or bool(
            result.stats["masked_nodes_included"]
        )

    stats["smoothing_rows"] = int(stats["laplacian_rows"]) + int(stats["gradient_rows"])
    return operators, stats


def operator_rms(operator: sp.csr_matrix, velocity: np.ndarray) -> float:
    """RMS norm of a sparse operator applied to a flattened velocity field."""
    if operator.shape[0] == 0:
        return float("nan")
    flat = np.asarray(velocity, dtype=float).reshape(-1)
    vals = operator @ flat
    return float(np.sqrt(np.mean(vals * vals)))


# ---------------------------------------------------------------------------
# Coverage-adaptive smoothing (LE-DM v2, Problem 2b)
# ---------------------------------------------------------------------------

def coverage_count_by_radius(
    positions: np.ndarray,
    grid: RectGrid,
    radius: float,
    node_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Local track support c_j: tracks within ``radius`` of each grid node (Eq. 13).

    This is a genuine spatial query (``scipy.spatial.cKDTree``), not a proxy
    derived from the interpolation footprint: c_j counts the *tracks* that lie
    within a ball of radius ``radius`` centred on node j, independent of which
    kernel A happens to use. ``radius`` is the grid spacing Delta.

    positions : (n_p, 3) track positions.
    node_mask : optional (Ng,) bool. Nodes outside the mask are left at 0; the
                count is only ever consumed at open-fluid nodes, so restricting
                the query there keeps the tree lookup cheap.

    Returns (Ng,) integer counts.
    """
    from scipy.spatial import cKDTree

    Ng = grid.size
    counts = np.zeros(Ng, dtype=np.int64)
    positions = np.asarray(positions, dtype=float)
    if positions.shape[0] == 0:
        return counts

    if node_mask is None:
        query_idx = np.arange(Ng)
    else:
        query_idx = np.where(np.asarray(node_mask, dtype=bool))[0]
    if query_idx.size == 0:
        return counts

    tree = cKDTree(positions)
    counts[query_idx] = np.asarray(
        tree.query_ball_point(grid.nodes[query_idx], r=float(radius),
                              return_length=True),
        dtype=np.int64,
    )
    return counts


def coverage_weight(coverage_count: np.ndarray, ref_count: float) -> np.ndarray:
    """Per-node smoothing weight, inverse to local track support (Eq. 14).

        w_j = 1 / (1 + c_j / c_0)

    where c_j is the radius-Delta track count at node j and c_0 = ``ref_count``.

    Empty cell (c_j = 0)  -> w_j = 1   (full smoothness penalty)
    c_j = c_0             -> w_j = 1/2
    Data-rich cell        -> w_j -> 0  (penalty negligible; data dominates)
    """
    ref = max(float(ref_count), 1e-12)
    return 1.0 / (1.0 + np.asarray(coverage_count, dtype=float) / ref)


def build_coverage_adaptive_smoothing_operator(
    grid: RectGrid,
    C: np.ndarray,
    coverage_count: np.ndarray,
    ref_count: float,
) -> OperatorBuildResult:
    """Coverage-adaptive smoothness operator (Eqs. 13-14).

    The penalty assembled here is

        S(u) = sum over fluid-fluid edges (j,n) of  w_bar_jn * ||u_n - u_j||^2 ,
        w_bar_jn = 0.5 * (w_j + w_n),   w_j = 1 / (1 + c_j / c_0),

    with c_j the number of tracks within radius Delta of node j. It enters the
    normal equations as ``H += lambda_c * (G^T G)``, so each emitted row is
    ``sqrt(w_bar_jn) * (u_n - u_j)`` for one velocity component and G^T G is the
    weighted graph Laplacian: a "look like your neighbours" penalty that is
    negligible where tracks are dense (w -> 0) and dominant in empty cells
    (w -> 1), where it anchors nodes to their neighbours instead of to the
    temporal prior.

    SPACING CONVENTION (single, explicit choice): Eq. 13 is written as an
    UNSCALED neighbour difference, so this operator emits ``u_n - u_j`` with no
    1/Delta factor. ``lambda_c`` is therefore dimensionless relative to the
    velocity-squared data term and its calibration does not move when the grid
    is refined. Do not reintroduce a spacing scaling here without redefining
    lambda_c to match -- the two must stay consistent. (The separate Laplacian
    and gradient smoothers in this module keep their own
    ``smoothing_spacing_scaled`` switch; they are different operators with a
    different weight.)

    Why first differences and not a vector Laplacian: G^T G (graph Laplacian) has
    spectrum O(1/d^2), whereas L^T L for a discrete Laplacian is biharmonic with
    spectrum O(1/d^4). The biharmonic term makes the saddle system far too
    ill-conditioned for unpreconditioned MINRES to converge; the gradient form is
    ~10x better conditioned and converges, while regularising empty cells just as
    effectively.

    Masking: an edge is emitted only when BOTH endpoints are open fluid (C==1),
    so fluid nodes never couple to shell (C==0) or solid (C==-1) nodes --
    identical masking to the divergence stencils. The operator is SPSD and
    leaves the saddle-point / MINRES structure unchanged.
    """
    C = np.asarray(C)
    allowed = (C == 1)
    w_cov = coverage_weight(coverage_count, ref_count)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    row = 0
    n_edges = 0
    nb_table = grid.neighbors
    node_idx = np.where(allowed)[0]

    # One edge per (node, +axis neighbour) pair with BOTH endpoints open fluid.
    # Scanning only the +x/+y/+z neighbour visits each undirected edge once.
    for ax in range(3):
        nbr = nb_table[node_idx, 2 * ax + 1]
        ok = nbr >= 0
        ok[ok] &= allowed[nbr[ok]]
        j = node_idx[ok]
        n = nbr[ok]
        if j.size == 0:
            continue

        # Eq. 14 edge weight: the AVERAGE of the two endpoint weights, not the
        # weight of one arbitrarily chosen endpoint. This makes the operator
        # symmetric in (j, n) -- the penalty on an edge cannot depend on which
        # end happens to have the lower node index.
        w_bar = 0.5 * (w_cov[j] + w_cov[n])
        scale = np.sqrt(np.maximum(w_bar, 0.0))

        keep = scale > 0.0
        j, n, scale = j[keep], n[keep], scale[keep]
        if j.size == 0:
            continue
        n_edges += int(j.size)

        for comp in range(3):
            r = row + np.arange(j.size)
            rows += [r, r]
            cols += [3 * n + comp, 3 * j + comp]
            vals += [scale, -scale]
            row += int(j.size)

    if rows:
        L = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(row, 3 * grid.size),
        ).tocsr()
    else:
        L = sp.csr_matrix((0, 3 * grid.size))
    L.eliminate_zeros()
    stats = {
        "type": "coverage_adaptive_gradient",
        "rows": int(L.shape[0]),
        "edges": int(n_edges),
        "ref_count": float(ref_count),
        "mean_w_cov_fluid": float(np.mean(w_cov[allowed])) if np.any(allowed) else 0.0,
        "mean_coverage_count_fluid": (
            float(np.mean(np.asarray(coverage_count)[allowed]))
            if np.any(allowed) else 0.0
        ),
        "spacing_scaled": False,      # Eq. 13 is an unscaled neighbour difference
        "crosses_mask": False,
    }
    return OperatorBuildResult(L, stats)
