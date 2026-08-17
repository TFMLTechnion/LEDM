"""Coverage-adaptive smoothness operator: Eqs. 13-14.

Checks the three things that define the operator and were previously only
approximated: the support count is a real radius-Delta track count (not an
interpolation-footprint proxy), the edge weight is the AVERAGE of the two
endpoint weights, and the neighbour difference is UNSCALED (no 1/Delta).
"""
import numpy as np
import pytest

from ccmplus.grid import RectGrid
from ccmplus.operators import (
    build_coverage_adaptive_smoothing_operator,
    coverage_count_by_radius,
    coverage_weight,
)


def make_grid(delta=1.0):
    return RectGrid((0.0, 0.0, 0.0), (3.0, 3.0, 3.0), delta=delta)


# --------------------------------------------------------------------------- #
# c_j : radius-Delta track count
# --------------------------------------------------------------------------- #
class TestCoverageCount:
    def test_counts_tracks_within_radius(self):
        g = make_grid()
        node = g.idx_from_ijk(1, 1, 1)
        centre = g.nodes[node]
        # 3 tracks inside the ball of radius 1.0, 2 well outside.
        pts = np.array([
            centre + [0.10, 0.0, 0.0],
            centre + [0.0, -0.50, 0.0],
            centre + [0.30, 0.30, 0.30],
            centre + [2.50, 0.0, 0.0],
            centre + [0.0, 0.0, 2.50],
        ])
        counts = coverage_count_by_radius(pts, g, radius=1.0)
        assert counts[node] == 3

    def test_is_not_the_interpolation_footprint(self):
        """The count must not change when the interpolation kernel changes.

        The old implementation derived coverage from the column nnz of A, so it
        measured kernel reach rather than physical track density. A single track
        must count as exactly 1 for the one node it sits on, no matter that the
        cubic B-spline would spread it over 64 nodes.
        """
        g = make_grid()
        node = g.idx_from_ijk(1, 1, 1)
        pts = g.nodes[node][None, :].copy()
        counts = coverage_count_by_radius(pts, g, radius=1e-6)
        assert counts[node] == 1
        assert counts.sum() == 1

    def test_node_mask_restricts_the_query(self):
        g = make_grid()
        pts = g.nodes.copy()
        mask = np.zeros(g.size, dtype=bool)
        keep = g.idx_from_ijk(1, 1, 1)
        mask[keep] = True
        counts = coverage_count_by_radius(pts, g, radius=1.0, node_mask=mask)
        assert counts[keep] > 0
        assert counts.sum() == counts[keep]

    def test_no_tracks_gives_zero(self):
        g = make_grid()
        counts = coverage_count_by_radius(np.empty((0, 3)), g, radius=1.0)
        assert counts.shape == (g.size,)
        assert not counts.any()


# --------------------------------------------------------------------------- #
# w_j = 1 / (1 + c_j / c_0)
# --------------------------------------------------------------------------- #
class TestCoverageWeight:
    def test_matches_equation_14(self):
        c = np.array([0.0, 1.0, 3.0])
        np.testing.assert_allclose(
            coverage_weight(c, 1.0), [1.0, 0.5, 0.25]
        )

    def test_ref_count_sets_the_half_weight_point(self):
        assert coverage_weight(np.array([4.0]), 4.0)[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# The operator
# --------------------------------------------------------------------------- #
class TestCoverageOperator:
    def _all_fluid(self, g):
        return np.ones(g.size, dtype=np.int8)

    def test_edge_weight_is_the_endpoint_average(self):
        """w_bar_jn = 0.5*(w_j + w_n), not sqrt(w) of one chosen endpoint."""
        g = make_grid()
        C = self._all_fluid(g)
        j = g.idx_from_ijk(1, 1, 1)
        n = g.idx_from_ijk(2, 1, 1)          # +x neighbour of j

        # Give j and n very different support; everything else matches j.
        counts = np.zeros(g.size)
        counts[j] = 0.0                      # w_j = 1
        counts[n] = 3.0                      # w_n = 1/4
        res = build_coverage_adaptive_smoothing_operator(g, C, counts, 1.0)

        L = res.matrix.tocsr()
        # Find the row coupling DOF 3*j and 3*n (the u-component of this edge).
        target = None
        for r in range(L.shape[0]):
            cols = set(L.indices[L.indptr[r]:L.indptr[r + 1]])
            if cols == {3 * j, 3 * n}:
                target = r
                break
        assert target is not None, "no row found for the (j, n) u-component edge"

        row = L[target].toarray().ravel()
        expected = np.sqrt(0.5 * (1.0 + 0.25))
        assert abs(row[3 * n]) == pytest.approx(expected)
        assert abs(row[3 * j]) == pytest.approx(expected)
        # Opposite signs: the row is a difference, not a sum.
        assert row[3 * n] * row[3 * j] < 0

    def test_edge_weight_is_symmetric_in_its_endpoints(self):
        """Swapping which endpoint is data-rich leaves the edge weight alone."""
        g = make_grid()
        C = self._all_fluid(g)
        j = g.idx_from_ijk(1, 1, 1)
        n = g.idx_from_ijk(2, 1, 1)

        def edge_scale(cj, cn):
            counts = np.zeros(g.size)
            counts[j], counts[n] = cj, cn
            L = build_coverage_adaptive_smoothing_operator(
                g, C, counts, 1.0).matrix.tocsr()
            for r in range(L.shape[0]):
                cols = set(L.indices[L.indptr[r]:L.indptr[r + 1]])
                if cols == {3 * j, 3 * n}:
                    return abs(L[r].toarray().ravel()[3 * n])
            raise AssertionError("edge row not found")

        assert edge_scale(0.0, 3.0) == pytest.approx(edge_scale(3.0, 0.0))

    def test_neighbour_difference_is_unscaled_by_spacing(self):
        """Eq. 13 is an unscaled difference: no 1/Delta factor.

        Halving the grid spacing must not rescale the operator entries, which is
        what makes lambda_c independent of grid refinement.
        """
        entries = []
        for delta in (1.0, 0.5):
            g = RectGrid((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), delta=delta)
            C = np.ones(g.size, dtype=np.int8)
            counts = np.zeros(g.size)           # w = 1 everywhere
            L = build_coverage_adaptive_smoothing_operator(g, C, counts, 1.0).matrix
            entries.append(np.abs(L.data).max())
        assert entries[0] == pytest.approx(entries[1])
        assert entries[0] == pytest.approx(1.0)

    def test_no_edge_crosses_the_body_interface(self):
        """Edges are emitted only when BOTH endpoints are open fluid."""
        g = RectGrid((-4.0, -4.0, -4.0), (4.0, 4.0, 4.0), delta=1.0)
        r = np.linalg.norm(g.nodes, axis=1)
        C = np.where(r > 2.5, 1, np.where(r > 1.5, 0, -1)).astype(np.int8)
        assert (C == 0).any() and (C == -1).any()

        counts = np.zeros(g.size)
        L = build_coverage_adaptive_smoothing_operator(g, C, counts, 1.0).matrix
        touched = np.unique(L.tocoo().col // 3)
        assert np.all(C[touched] == 1)

    def test_operator_is_symmetric_positive_semidefinite(self):
        g = make_grid()
        C = self._all_fluid(g)
        counts = np.arange(g.size, dtype=float) % 4
        L = build_coverage_adaptive_smoothing_operator(g, C, counts, 1.0).matrix
        H = (L.T @ L).toarray()
        np.testing.assert_allclose(H, H.T, atol=1e-12)
        assert np.linalg.eigvalsh(H).min() > -1e-10

    def test_constant_field_is_in_the_nullspace(self):
        """A uniform velocity costs nothing: the penalty is on differences."""
        g = make_grid()
        C = self._all_fluid(g)
        counts = np.zeros(g.size)
        L = build_coverage_adaptive_smoothing_operator(g, C, counts, 1.0).matrix
        x = np.zeros(3 * g.size)
        x[0::3], x[1::3], x[2::3] = 2.0, -1.0, 0.5
        np.testing.assert_allclose(L @ x, 0.0, atol=1e-12)

    def test_stats_report_the_spacing_convention(self):
        g = make_grid()
        C = self._all_fluid(g)
        res = build_coverage_adaptive_smoothing_operator(
            g, C, np.zeros(g.size), 1.0)
        assert res.stats["spacing_scaled"] is False
        assert res.stats["crosses_mask"] is False
        assert res.stats["edges"] > 0
