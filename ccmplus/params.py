"""Parameter-file parser for CCMPlus drivers.

Stdlib only — no third-party dependencies.
"""
from __future__ import annotations

from pathlib import Path


def read_parameters(path) -> dict:
    """Parse a ``key = value`` parameter file into a dict.

    Rules:
    - Lines whose first non-whitespace character is ``#`` are skipped.
    - Inline comments (``# ...`` after a value) are stripped.
    - Values are auto-coerced: bool > int > float > str.
    - Strings may be surrounded by single or double quotes (stripped).
    - Raises ``ValueError`` with file path and line number on malformed lines.
    """
    path = Path(path)
    params: dict = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("#")[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(
                    f"{path}:{lineno}: expected 'key = value', got {raw.rstrip()!r}"
                )
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            if not key:
                raise ValueError(f"{path}:{lineno}: empty key in {raw.rstrip()!r}")
            params[key] = _coerce(val)
    return params


def _coerce(val: str):
    lower = val.lower()
    if lower in ("true", "yes", "on"):
        return True
    if lower in ("false", "no", "off"):
        return False
    if (val.startswith('"') and val.endswith('"')) or \
       (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def require(params: dict, keys: list) -> None:
    """Raise ``KeyError`` listing every missing key at once (not one at a time)."""
    missing = [k for k in keys if k not in params]
    if missing:
        raise KeyError(
            "Missing required parameter(s): " + ", ".join(missing)
        )


def apply_defaults(params: dict, defaults: dict) -> None:
    """Fill cosmetic/optional keys absent from ``params`` with ``defaults``."""
    for key, val in defaults.items():
        params.setdefault(key, val)


def resolve_domain_truncation(params: dict) -> tuple[bool, dict | None]:
    """Read and validate the ``domain_truncation`` option.

    ``domain_truncation`` is a boolean (accepts yes/no, true/false, on/off, 1/0,
    case-insensitive). Absent -> treated as ``no``.

    When enabled, the four rectangle limits ``x_min, x_max, y_min, y_max`` are
    REQUIRED and must parse as floats. Any missing or unparseable limit raises a
    ``ValueError`` naming exactly which one, and the limits are not silently
    defaulted. When disabled, the limits are neither read nor required.

    Returns ``(truncate, limits)`` where ``limits`` is
    ``{"x_min","x_max","y_min","y_max"}`` (floats) when enabled, else ``None``.
    """
    raw = params.get("domain_truncation", False)
    if isinstance(raw, bool):
        truncate = raw
    else:
        truncate = str(raw).strip().lower() in ("yes", "true", "on", "1")
    if not truncate:
        return False, None

    limits: dict = {}
    missing: list[str] = []
    bad: list[str] = []
    for key in ("x_min", "x_max", "y_min", "y_max"):
        if key not in params:
            missing.append(key)
            continue
        try:
            limits[key] = float(params[key])
        except (TypeError, ValueError):
            bad.append(f"{key}={params[key]!r}")
    if missing:
        raise ValueError(
            "domain_truncation=yes requires the limit(s) "
            + ", ".join(missing) + " in the parameter file (missing)."
        )
    if bad:
        raise ValueError(
            "domain_truncation=yes: limit(s) not parseable as float: "
            + ", ".join(bad) + "."
        )
    if limits["x_min"] >= limits["x_max"]:
        raise ValueError(
            f"domain_truncation: x_min ({limits['x_min']}) must be < "
            f"x_max ({limits['x_max']})."
        )
    if limits["y_min"] >= limits["y_max"]:
        raise ValueError(
            f"domain_truncation: y_min ({limits['y_min']}) must be < "
            f"y_max ({limits['y_max']})."
        )
    return True, limits
