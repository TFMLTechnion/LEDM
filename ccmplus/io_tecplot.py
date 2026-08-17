"""Tecplot ASCII I/O for CCM+ — reader for sphere trajectory and track files,
writer for reconstructed velocity volumes."""

from __future__ import annotations

import logging
import re
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reader: sphere trajectory
# ---------------------------------------------------------------------------

def read_trajectory(path: str) -> dict:
    """Read a sphere trajectory Tecplot file.

    Expected header lines (any order before the numeric block):
      TITLE = "..."
      VARIABLES = ...
      DIAMETER=<value>[mm]
      SOLID/FLUID DENSITY RATIO=...

    Returns
    -------
    dict with keys:
      diameter_mm : float
      times       : (N,) float64 array  - trajectory Time column
      positions   : (N, 3) float64 array  - (x, y, z) in mm

    Rows containing NaN are skipped. This lets the reader handle trajectory
    files whose detector lost the sphere for some frames.
    """
    diameter_mm = None
    times: list[float] = []
    positions: list[list[float]] = []
    skipped_nan = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue

            # Capture DIAMETER before attempting numeric parse
            if re.search(r"DIAMETER", stripped, re.IGNORECASE):
                m = re.search(r"=\s*([\d.]+)", stripped)
                if m:
                    diameter_mm = float(m.group(1))
                continue

            # Try numeric row: Time x y z. Times may be floats.
            parts = stripped.split()
            if len(parts) != 4:
                continue
            try:
                t, x, y, z = (float(parts[0]), float(parts[1]),
                              float(parts[2]), float(parts[3]))
            except ValueError:
                continue
            if np.isnan([t, x, y, z]).any():
                skipped_nan += 1
                continue

            times.append(t)
            positions.append([x, y, z])

    if skipped_nan:
        log.warning("read_trajectory(%s): skipped %d NaN row(s).", path, skipped_nan)

    if diameter_mm is None:
        raise ValueError(f"DIAMETER not found in {path}")
    if not times:
        raise ValueError(f"No numeric data rows found in {path}")

    return {
        "diameter_mm": diameter_mm,
        "times": np.array(times, dtype=np.float64),
        "positions": np.array(positions, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Reader: single-zone track file
# ---------------------------------------------------------------------------

_REQUIRED_TRACK_COLUMNS = ["x", "y", "z", "Vx", "Vy", "Vz", "trackID"]


def _strip_units(name: str) -> str:
    """Return a Tecplot variable name without bracketed units."""
    return re.sub(r"\[.*?\]", "", name).strip()


def _parse_variables_line(line: str) -> dict[str, int]:
    """Parse a VARIABLES line into a stripped-name to column-index map."""
    _, _, rest = line.partition("=")
    return {_strip_units(name): idx for idx, name in enumerate(re.findall(r'"([^"]*)"', rest))}


def read_tracks_zone(path: str) -> dict:
    """Read a single-zone Tecplot POINT-format track file.

    The column layout is read from the VARIABLES header line. This supports
    both the older 8-column track files and Snapshot files with intensity and
    acceleration columns.

    Returns
    -------
    dict with keys:
      zone_time_is_numeric : bool
      zone_time            : int, present when ZONE T is numeric
      zone_label           : str, present when ZONE T is a label such as
                             "Snapshot 0000"
      n_points             : int, parsed from ``I=<value>``
      positions_mm : (n, 3) float64
      velocities_ms: (n, 3) float64
      vmag_ms      : (n,) float64
      track_ids    : (n,) int64
    """
    col_map: dict[str, int] | None = None
    zone_time_raw: str | None = None
    zone_time_is_numeric = False
    zone_time: int | None = None
    zone_label: str | None = None
    n_points: int | None = None
    n_header = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()

            if re.match(r"VARIABLES\b", stripped, re.IGNORECASE):
                col_map = _parse_variables_line(stripped)
                n_header += 1
                continue

            # Parse zone time from ZONE T=<number> or ZONE T="<label>".
            if re.match(r"ZONE\b", stripped, re.IGNORECASE):
                m_quoted = re.search(r'\bT\s*=\s*"([^"]*)"', stripped)
                m_bare = re.search(r'\bT\s*=\s*([^\s,]+)', stripped)
                if m_quoted:
                    zone_time_raw = m_quoted.group(1)
                elif m_bare:
                    zone_time_raw = m_bare.group(1)
                if zone_time_raw is not None:
                    try:
                        zone_time = int(float(zone_time_raw))
                        zone_time_is_numeric = True
                    except ValueError:
                        zone_label = zone_time_raw
                        zone_time_is_numeric = False
                n_header += 1
                continue

            # Parse n_points from  I=<int>, J=1, K=1, ZONETYPE = Ordered
            if re.search(r"\bI\s*=\s*\d+", stripped) and re.search(
                r"ZONETYPE", stripped, re.IGNORECASE
            ):
                m = re.search(r"\bI\s*=\s*(\d+)", stripped)
                if m:
                    n_points = int(m.group(1))
                n_header += 1
                continue

            # Detect first numeric data row
            parts = stripped.split()
            if parts:
                try:
                    float(parts[0])
                    break  # stop counting header rows
                except ValueError:
                    pass
            n_header += 1

    if zone_time_raw is None:
        raise ValueError(f"'ZONE T=' not found in {path}")
    if n_points is None:
        raise ValueError(f"'I=' dimension line not found in {path}")
    if col_map is None:
        raise ValueError(f"'VARIABLES' line not found in {path}")

    missing = [name for name in _REQUIRED_TRACK_COLUMNS if name not in col_map]
    if missing:
        raise ValueError(
            f"Required columns {missing} not found in {path}. "
            f"Available columns: {sorted(col_map)}"
        )

    # Fast numeric read: split on whitespace (avoids slow np.loadtxt line-by-line parsing)
    with open(path, "r", encoding="utf-8") as fh:
        for _ in range(n_header):
            fh.readline()
        text_data = fh.read()

    values = np.array(text_data.split(), dtype=np.float64)
    n_cols = len(col_map)
    n_rows = len(values) // n_cols
    data = values[: n_rows * n_cols].reshape(n_rows, n_cols)

    positions_mm = data[:, [col_map["x"], col_map["y"], col_map["z"]]]
    velocities_ms = data[:, [col_map["Vx"], col_map["Vy"], col_map["Vz"]]]
    vmag_ms = data[:, col_map["|V|"]] if "|V|" in col_map else np.linalg.norm(velocities_ms, axis=1)
    track_ids = data[:, col_map["trackID"]].astype(np.int64)

    result: dict = {
        "zone_time_is_numeric": zone_time_is_numeric,
        "n_points": n_points,
        "positions_mm": positions_mm,
        "velocities_ms": velocities_ms,
        "vmag_ms": vmag_ms,
        "track_ids": track_ids,
    }
    if zone_time_is_numeric:
        result["zone_time"] = zone_time
    else:
        result["zone_label"] = zone_label
    return result


# ---------------------------------------------------------------------------
# Writer: reconstructed velocity volume
# ---------------------------------------------------------------------------

_VALS_PER_LINE = 8


def _write_block(fh, arr: np.ndarray, fmt: str = "{:.6g}") -> None:
    """Write a flat array in BLOCK format, 8 values per line, CRLF endings."""
    n = len(arr)
    for start in range(0, n, _VALS_PER_LINE):
        chunk = arr[start : start + _VALS_PER_LINE]
        fh.write(" ".join(fmt.format(v) for v in chunk) + "\r\n")


def write_tecplot_volume(
    path: str,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    phi: np.ndarray,
    C: np.ndarray,
    title: str,
    zone_time: int,
    Nx: int,
    Ny: int,
    Nz: int,
) -> None:
    """Write a reconstructed velocity field as Tecplot ASCII BLOCK format.

    All array arguments are (Ng,) flat arrays with node ordering
    idx = i + Nx*(j + Ny*k)  (i fastest, then j, then k).

    x, y, z    : node positions [mm]
    u, v, w    : velocity components [m/s]
    phi        : signed distance [mm]
    C          : classification {-1, 0, +1}
    Nx, Ny, Nz : grid node counts along each axis

    Output uses Windows line endings (CRLF).
    """
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f'TITLE = "CCM+ reconstruction t={zone_time}"\r\n')
        fh.write(
            'VARIABLES = "x[mm]" "y[mm]" "z[mm]"'
            ' "Vx[m/s]" "Vy[m/s]" "Vz[m/s]" "Vmag" "phi[mm]" "C"\r\n'
        )
        fh.write(f'ZONE T="{zone_time}"\r\n')
        fh.write(f"STRANDID=1, SOLUTIONTIME={zone_time}\r\n")
        fh.write(f"I={Nx}, J={Ny}, K={Nz}, ZONETYPE = Ordered\r\n")
        fh.write("DATAPACKING = BLOCK\r\n")
        _write_block(fh, x)
        _write_block(fh, y)
        _write_block(fh, z)
        _write_block(fh, u)
        _write_block(fh, v)
        _write_block(fh, w)
        # velocity magnitude (3D): sqrt(u^2 + v^2 + w^2)
        vmag = np.sqrt(np.asarray(u) ** 2 + np.asarray(v) ** 2 + np.asarray(w) ** 2)
        _write_block(fh, vmag)
        _write_block(fh, phi)
        _write_block(fh, C.astype(np.float64), fmt="{:.0f}")
