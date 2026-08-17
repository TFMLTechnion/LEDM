"""ASCII .dat output for the four-file path (Tecplot POINT or plain columns).

A small, dependency-free writer for the structured reconstruction grid produced
by ``io_ledm``. It is deliberately independent of ``io_tecplot`` (whose writer is
``DATAPACKING=BLOCK`` with a hard-coded variable list and no provenance/precision
controls); nothing here modifies the frozen solver.

The one subtlety worth stating up front is index ordering. The frozen grid stores
nodes as ``idx = i + Nx*(j + Ny*k)`` — i.e. the Fortran-order ravel of a
``(Nx, Ny, Nz)`` field, so ``i`` (x) varies fastest. Tecplot POINT packing also
requires the first zone index ``I`` to vary fastest. Therefore:

    * ``order = "F"``  rows vary axis-0 fastest -> ``I,J,K = Nx,Ny,Nz``.
    * ``order = "C"``  rows vary axis-2 fastest -> ``I,J,K = Nz,Ny,Nx`` (REVERSED).

Getting that reversal wrong yields a file that loads without error and is silently
scrambled, so :func:`write_dat` derives ``I,J,K`` from ``order`` and the round-trip
test proves it for both orders with an asymmetric field.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

# The codebase names itself "LE-DM v2" (see run_ledm.py); stamped into provenance.
LEDM_VERSION = "LE-DM v2"

_ALLOWED_FLAVORS = ("tecplot", "plain")
_ALLOWED_ORDERS = ("C", "F")


# --------------------------------------------------------------------------- #
# Field / ordering helpers (used by the io_ledm wiring)
# --------------------------------------------------------------------------- #
def fields_from_result(res) -> dict[str, np.ndarray]:
    """Derive the per-node scalar fields actually present in a reconstruction
    result, in a stable order: u, v, w from ``velocity``, then every other 1-D
    per-node array on the result (``classification``, and any future coverage /
    pressure fields). The list is NOT hard-coded — it follows the result object.
    """
    vel = np.asarray(res.velocity, dtype=float)
    ng = vel.shape[0]
    fields: dict[str, np.ndarray] = {
        "u": vel[:, 0], "v": vel[:, 1], "w": vel[:, 2],
    }
    # dataclasses.fields keeps declaration order -> stable column order.
    field_names = ([f.name for f in dataclasses.fields(res)]
                   if dataclasses.is_dataclass(res) else [])
    for name in field_names:
        if name == "velocity":
            continue
        val = getattr(res, name, None)
        arr = np.asarray(val) if isinstance(val, np.ndarray) else None
        if arr is not None and arr.ndim == 1 and arr.shape[0] == ng:
            fields[name] = arr.astype(float)
    return fields


def body_frame_velocity(v_lab, nodes, body) -> np.ndarray:
    """Body-frame (relative) velocity: ``v_rel = v_lab - u_body``.

    ``u_body`` is the body's full rigid-body field ``U_s + omega_s x (x - X_s)``
    evaluated by the SAME :func:`ccmplus.kinematics.u_gamma` the solver used to pin
    the shell/solid, so the transform is exactly consistent with what was solved
    (and stays correct for rotating / non-spherical bodies via ``velocity_fn``).
    """
    from ccmplus.kinematics import u_gamma
    v_lab = np.asarray(v_lab, dtype=float)
    return v_lab - u_gamma(np.asarray(nodes, dtype=float), body)


def _to_order(a: np.ndarray, dims, order: str) -> np.ndarray:
    """Re-ravel a canonical (i-fastest / Fortran-order) flat array into ``order``.

    ``a`` is either ``(Ng,)`` or ``(Ng, k)`` in the grid's native node order
    (``idx = i + Nx*(j + Ny*k)``). For ``order = "F"`` that is already the target
    layout and ``a`` is returned unchanged; for ``order = "C"`` each column is
    reshaped to ``dims`` (Fortran) and re-raveled C-order.
    """
    order = str(order).upper()
    a = np.asarray(a)
    if order == "F":
        return a
    if a.ndim == 1:
        return np.ascontiguousarray(a.reshape(dims, order="F").ravel(order="C"))
    cols = [a[:, j].reshape(dims, order="F").ravel(order="C")
            for j in range(a.shape[1])]
    return np.column_stack(cols)


def snapshot_from_arrays(nodes, fields, dims, order: str):
    """Build ``(coords, fields, dims)`` for :func:`write_dat` from explicit node
    coordinates + a per-node field mapping, re-raveled into ``order`` so the row
    sequence matches the zone header :func:`write_dat` will emit.

    Used by the wiring so that an output-layer transform (e.g. body-frame velocity
    subtraction or co-moving coordinates) can be applied to ``nodes``/``fields``
    before serialisation without going through the raw result object.
    """
    dims = tuple(int(d) for d in dims)
    coords = _to_order(np.asarray(nodes, dtype=float), dims, order)
    out = {name: _to_order(arr, dims, order) for name, arr in fields.items()}
    return coords, out, dims


def snapshot_from_grid(grid, res, order: str):
    """Convenience wrapper: canonical grid + result -> ``(coords, fields, dims)``,
    with the field list derived from the result (see :func:`fields_from_result`)."""
    return snapshot_from_arrays(grid.nodes, fields_from_result(res),
                                grid.shape, order)


# --------------------------------------------------------------------------- #
# The writer
# --------------------------------------------------------------------------- #
def _provenance_lines(meta: dict, colnames, order, ijk) -> list[str]:
    """'#'-prefixed provenance block. Tecplot treats '#' lines as comments, so it
    is safe in both flavors. A .dat with no units/frame info is unusable later."""
    I, J, K = ijk

    def g(key, default="?"):
        return meta.get(key, default)

    lines = [
        f"# {g('version', LEDM_VERSION)} reconstruction output",
        f"# case: {g('case')}",
        f"# time: {g('time')} [{g('time_unit')}]",
        f"# dx: {g('dx')} [{g('length_unit')}]",
        f"# length_unit: {g('length_unit')}",
        f"# bounds_min: {g('bounds_min')} [{g('length_unit')}]",
        f"# bounds_max: {g('bounds_max')} [{g('length_unit')}]",
        f"# roi_mode: {g('roi_mode')}",
        f"# n_nodes: {g('n_nodes')}",
        f"# grid_dims (Nx,Ny,Nz): {g('grid_dims')}",
        f"# dat_order: {str(order).upper()}   zone I,J,K = {I},{J},{K}",
    ]
    # Output reference frame (only stamped when a non-lab transform was applied).
    frame = meta.get("output_frame", "lab")
    lines.append(f"# output_frame: {frame}")
    if frame == "body":
        lines.append(f"# body_U (subtracted): {g('body_U')} [{g('length_unit')}/"
                     f"{g('time_unit')}]")
        lines.append(f"# body_omega (subtracted): {g('body_omega')} [rad/{g('time_unit')}]")
        lines.append(f"# body_X_s: {g('body_X_s')} [{g('length_unit')}]")
        lines.append(f"# v_rel(x) = v_lab(x) - [U + omega x (x - X_s)]")
    if meta.get("comoving_coords"):
        lines.append(f"# comoving_coords: true  (nodes shifted by -(X_s - X_s0))")
        lines.append(f"# body_X_s0: {g('body_X_s0')} [{g('length_unit')}]")
    lines.append(f"# columns: {' '.join(colnames)}")
    return lines


def write_dat(path, coords, fields, dims, meta,
              flavor: str = "tecplot", precision: int = 9, order: str = "C") -> str:
    """Write one structured snapshot as ASCII .dat.

    Parameters
    ----------
    path : output file path (.dat).
    coords : (Ng, 3) node positions x y z.
    fields : ordered mapping ``name -> (Ng,)`` of the extra columns (u v w, then
        classification / any others). Column order follows the mapping.
    dims : (Nx, Ny, Nz) grid node counts.
    meta : provenance dict (case, time, time_unit, dx, length_unit, bounds_min,
        bounds_max, roi_mode, n_nodes, grid_dims, version). Missing keys print '?'.
    flavor : 'tecplot' (TITLE/VARIABLES/ZONE ... DATAPACKING=POINT) or 'plain'
        (single '#'-commented column header, then whitespace columns).
    precision : significant digits for every numeric column (``%.<p>g``).
    order : 'C' or 'F'. Describes the ravel order of ``coords``/``fields`` w.r.t.
        ``dims`` and fixes the zone I,J,K so Tecplot reads I fastest either way.

    ``coords``/``fields`` rows are written in the order given; only the zone I,J,K
    header depends on ``order``. Streaming is via ``np.savetxt`` on the open handle
    (no whole-file string is built).
    """
    flavor = str(flavor).strip().lower()
    if flavor not in _ALLOWED_FLAVORS:
        raise ValueError(f"dat_flavor must be one of {_ALLOWED_FLAVORS}, got '{flavor}'.")
    order = str(order).strip().upper()
    if order not in _ALLOWED_ORDERS:
        raise ValueError(f"dat_order must be one of {_ALLOWED_ORDERS}, got '{order}'.")
    precision = int(precision)
    if precision < 1:
        raise ValueError(f"dat_precision must be >= 1, got {precision}.")

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (Ng, 3), got {coords.shape}.")
    ng = coords.shape[0]

    names = list(fields.keys())
    colnames = ["x", "y", "z"] + names
    columns = [coords[:, 0], coords[:, 1], coords[:, 2]]
    for nm in names:
        col = np.asarray(fields[nm], dtype=float).ravel()
        if col.shape[0] != ng:
            raise ValueError(
                f"field '{nm}' has {col.shape[0]} rows but coords has {ng}.")
        columns.append(col)
    data = np.column_stack(columns)

    Nx, Ny, Nz = (int(d) for d in dims)
    if Nx * Ny * Nz != ng:
        raise ValueError(
            f"dims {(Nx, Ny, Nz)} imply {Nx*Ny*Nz} nodes but got {ng} rows.")
    # Tecplot POINT needs I to vary fastest. Fortran ravel -> axis 0 (Nx) fastest;
    # C ravel -> axis 2 (Nz) fastest, so I,J,K are the REVERSED dims.
    ijk = (Nx, Ny, Nz) if order == "F" else (Nz, Ny, Nx)

    fmt = f"%.{precision}g"
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for line in _provenance_lines(meta, colnames, order, ijk):
            fh.write(line + "\n")
        if flavor == "tecplot":
            case = meta.get("case", "reconstruction")
            tval = meta.get("time", "?")
            fh.write(f'TITLE = "{case} t={tval}"\n')
            fh.write("VARIABLES = " + " ".join(f'"{c}"' for c in colnames) + "\n")
            fh.write(f'ZONE T="{case}_t{tval}", '
                     f'I={ijk[0]}, J={ijk[1]}, K={ijk[2]}, DATAPACKING=POINT\n')
        else:  # plain
            fh.write("# " + " ".join(colnames) + "\n")
        np.savetxt(fh, data, fmt=fmt, delimiter=" ")
    return str(path)
