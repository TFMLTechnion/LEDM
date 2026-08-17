"""Tests for ccmplus.io_tecplot: round-trip write/read and fixture-based parsing."""

from __future__ import annotations

import os
import re
import tempfile

import numpy as np
import pytest

from ccmplus.io_tecplot import read_trajectory, read_tracks_zone, write_tecplot_volume

# ---------------------------------------------------------------------------
# Fixture file paths
# ---------------------------------------------------------------------------

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TRAJ_FIXTURE = os.path.join(FIXTURE_DIR, "sphereTrajectory_fixture.dat")
TRACKS_FIXTURE = os.path.join(FIXTURE_DIR, "timeStep_fixture.dat")


# ---------------------------------------------------------------------------
# Helper: minimal BLOCK reader for round-trip verification
# ---------------------------------------------------------------------------

# write_tecplot_volume emits nine BLOCK variables, in this order. Vmag is a
# derived convenience column for Tecplot users; the tests below index by these
# names rather than by hard-coded positions so adding a column cannot silently
# invalidate them again.
TECPLOT_VARIABLES = ("x", "y", "z", "u", "v", "w", "Vmag", "phi", "C")


def _read_block_file(path: str):
    """Parse a write_tecplot_volume output and return all numeric values.

    Returns ``(header, {name: array})`` with one entry per variable in
    ``TECPLOT_VARIABLES``.
    """
    # Collect header info and BLOCK data values
    header = {}
    values: list[float] = []
    in_data = False

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue

            if not in_data:
                # I=Nx, J=Ny, K=Nz
                m = re.search(r"\bI\s*=\s*(\d+)", stripped)
                if m:
                    header["Nx"] = int(m.group(1))
                m = re.search(r"\bJ\s*=\s*(\d+)", stripped)
                if m:
                    header["Ny"] = int(m.group(1))
                m = re.search(r"\bK\s*=\s*(\d+)", stripped)
                if m:
                    header["Nz"] = int(m.group(1))
                # ZONE T="1200"  (the title is quoted by the writer)
                m = re.match(r'ZONE\s+T\s*=\s*"?(\d+)"?', stripped, re.IGNORECASE)
                if m:
                    header["zone_time"] = int(m.group(1))
                # Detect start of BLOCK data
                if stripped.upper().startswith("DATAPACKING"):
                    in_data = True
                continue

            # Numeric lines
            parts = stripped.split()
            try:
                values.extend(float(p) for p in parts)
            except ValueError:
                pass

    Ng = header["Nx"] * header["Ny"] * header["Nz"]
    nvar = len(TECPLOT_VARIABLES)
    arr = np.array(values)
    assert len(arr) == nvar * Ng, f"Expected {nvar * Ng} values, got {len(arr)}"
    blocks = arr.reshape(nvar, Ng)
    return header, dict(zip(TECPLOT_VARIABLES, blocks))


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Write a synthetic volume then read it back and verify values."""

    def _make_arrays(self, Nx=3, Ny=4, Nz=2):
        Ng = Nx * Ny * Nz
        rng = np.random.default_rng(0)
        x   = rng.uniform(0.0, 10.0, Ng)
        y   = rng.uniform(0.0, 10.0, Ng)
        z   = rng.uniform(0.0, 10.0, Ng)
        u   = rng.standard_normal(Ng)
        v   = rng.standard_normal(Ng)
        w   = rng.standard_normal(Ng)
        phi = rng.uniform(-5.0, 5.0, Ng)
        C   = rng.integers(-1, 2, size=Ng).astype(np.int8)
        return x, y, z, u, v, w, phi, C

    def test_header_fields(self):
        Nx, Ny, Nz = 3, 4, 2
        zone_time = 1200
        x, y, z, u, v, w, phi, C = self._make_arrays(Nx, Ny, Nz)
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
            path = tmp.name
        try:
            write_tecplot_volume(
                path, x, y, z, u, v, w, phi, C,
                title="test", zone_time=zone_time,
                Nx=Nx, Ny=Ny, Nz=Nz,
            )
            header, _ = _read_block_file(path)
            assert header["Nx"] == Nx
            assert header["Ny"] == Ny
            assert header["Nz"] == Nz
            assert header["zone_time"] == zone_time
        finally:
            os.unlink(path)

    def test_variable_values(self):
        Nx, Ny, Nz = 3, 4, 2
        x, y, z, u, v, w, phi, C = self._make_arrays(Nx, Ny, Nz)
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
            path = tmp.name
        try:
            write_tecplot_volume(
                path, x, y, z, u, v, w, phi, C,
                title="test", zone_time=500,
                Nx=Nx, Ny=Ny, Nz=Nz,
            )
            _, data = _read_block_file(path)
            # 6 significant figures in output → 1e-5 relative tolerance
            np.testing.assert_allclose(data["x"],   x,   rtol=1e-5)
            np.testing.assert_allclose(data["y"],   y,   rtol=1e-5)
            np.testing.assert_allclose(data["z"],   z,   rtol=1e-5)
            np.testing.assert_allclose(data["u"],   u,   rtol=1e-5)
            np.testing.assert_allclose(data["v"],   v,   rtol=1e-5)
            np.testing.assert_allclose(data["w"],   w,   rtol=1e-5)
            np.testing.assert_allclose(data["phi"], phi, rtol=1e-5)
            np.testing.assert_array_equal(data["C"].astype(np.int8), C)
            # Vmag is derived, not stored: check it is consistent.
            np.testing.assert_allclose(
                data["Vmag"], np.sqrt(u ** 2 + v ** 2 + w ** 2), rtol=1e-5
            )
        finally:
            os.unlink(path)

    def test_windows_line_endings(self):
        Nx, Ny, Nz = 2, 2, 2
        x, y, z, u, v, w, phi, C = self._make_arrays(Nx, Ny, Nz)
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
            path = tmp.name
        try:
            write_tecplot_volume(
                path, x, y, z, u, v, w, phi, C,
                title="test", zone_time=100,
                Nx=Nx, Ny=Ny, Nz=Nz,
            )
            with open(path, "rb") as fh:
                raw = fh.read()
            assert b"\r\n" in raw, "Expected Windows (CRLF) line endings"
        finally:
            os.unlink(path)

    def test_classification_integers(self):
        """C values should round-trip as exact integers {-1, 0, 1}."""
        Nx, Ny, Nz = 2, 3, 2
        Ng = Nx * Ny * Nz
        C = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int8)
        x = y = z = u = v = w = phi = np.zeros(Ng)
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as tmp:
            path = tmp.name
        try:
            write_tecplot_volume(
                path, x, y, z, u, v, w, phi, C,
                title="test", zone_time=0,
                Nx=Nx, Ny=Ny, Nz=Nz,
            )
            _, data = _read_block_file(path)
            np.testing.assert_array_equal(data["C"].astype(np.int8), C)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Fixture-based tests: read_trajectory
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.isfile(TRAJ_FIXTURE),
    reason=f"Fixture not found: {TRAJ_FIXTURE}",
)
class TestReadTrajectory:
    def test_diameter(self):
        result = read_trajectory(TRAJ_FIXTURE)
        assert result["diameter_mm"] == pytest.approx(24.0)

    def test_first_position(self):
        result = read_trajectory(TRAJ_FIXTURE)
        np.testing.assert_allclose(
            result["positions"][0], [74.5, 36.5, 74.5], rtol=1e-10
        )

    def test_times_are_numeric(self):
        result = read_trajectory(TRAJ_FIXTURE)
        assert result["times"].dtype.kind in ("f", "i")

    def test_first_time(self):
        result = read_trajectory(TRAJ_FIXTURE)
        assert result["times"][0] == 100

    def test_positions_shape(self):
        result = read_trajectory(TRAJ_FIXTURE)
        n = len(result["times"])
        assert result["positions"].shape == (n, 3)


# ---------------------------------------------------------------------------
# Fixture-based tests: read_tracks_zone
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.isfile(TRACKS_FIXTURE),
    reason=f"Fixture not found: {TRACKS_FIXTURE}",
)
class TestReadTracksZone:
    def test_zone_time(self):
        result = read_tracks_zone(TRACKS_FIXTURE)
        assert result["zone_time"] == 10800

    def test_n_points(self):
        result = read_tracks_zone(TRACKS_FIXTURE)
        assert result["n_points"] == 299009

    def test_first_position(self):
        result = read_tracks_zone(TRACKS_FIXTURE)
        np.testing.assert_allclose(
            result["positions_mm"][0], [67.86, 194.08, 77.48], rtol=1e-5
        )

    def test_column_count(self):
        result = read_tracks_zone(TRACKS_FIXTURE)
        n = len(result["positions_mm"])
        assert result["velocities_ms"].shape == (n, 3)
        assert result["track_ids"].shape == (n,)

    def test_track_ids_are_ints(self):
        result = read_tracks_zone(TRACKS_FIXTURE)
        assert result["track_ids"].dtype == np.int64


class TestSnapshotFormat:
    def test_float_trajectory_times_and_nan_rows(self):
        text = """TITLE = "sphereTrajectory.dat"
VARIABLES = "Time" "x[mm]" "y[mm]" "z[mm]"
DIAMETER=11.11[mm]
0.000000 -3.0 -25.0 0.7
4.166667 -2.0 -24.0 0.8
8.333333 NaN NaN NaN
"""
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False, encoding="utf-8") as tmp:
            tmp.write(text)
            path = tmp.name
        try:
            result = read_trajectory(path)
            assert result["diameter_mm"] == pytest.approx(11.11)
            np.testing.assert_allclose(result["times"], [0.0, 4.166667])
            assert result["positions"].shape == (2, 3)
        finally:
            os.unlink(path)

    def test_snapshot_track_columns_are_detected_from_variables(self):
        text = """TITLE = "B00001"
VARIABLES = "x[mm]" "y[mm]" "z[mm]" "I" "Vx[m/s]" "Vy[m/s]" "Vz[m/s]" "|V|[m/s]" "trackID" "Ax[m/s^2]"
ZONE T="Snapshot 0000"
STRANDID=1, SOLUTIONTIME=43.124719773
I=2, J=1, K=1, ZONETYPE = Ordered
DATAPACKING = POINT
50.1 12.6 3.75 0.0525 -0.086 -0.373 0.0491 0.386 7 19.9
50 32.4 6.16 0.0807 -0.00634 0.244 0.0363 0.246 8 -6.32
"""
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False, encoding="utf-8") as tmp:
            tmp.write(text)
            path = tmp.name
        try:
            result = read_tracks_zone(path)
            assert result["zone_time_is_numeric"] is False
            assert result["zone_label"] == "Snapshot 0000"
            np.testing.assert_allclose(result["positions_mm"][0], [50.1, 12.6, 3.75])
            np.testing.assert_allclose(result["velocities_ms"][0], [-0.086, -0.373, 0.0491])
            np.testing.assert_allclose(result["vmag_ms"], [0.386, 0.246])
            np.testing.assert_array_equal(result["track_ids"], [7, 8])
        finally:
            os.unlink(path)
