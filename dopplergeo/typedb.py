"""Aircraft type -> engine class, from wiedehopf/tar1090-db (ICAO Doc 8643 descriptors).

Descriptor e.g. "L1P": L=landplane/H=helicopter/G=gyro..., engine count, P=piston,
T=turboprop/turboshaft, J=jet, E=electric. Tonal flyover lines come from
propellers and rotors, so piston/turboprop/helicopter types are the candidates.
"""

from __future__ import annotations

import gzip
import json

from .cache import OfflineMiss, atomic_write_bytes, cache_dir
from .http import get_bytes

TYPES_URL = "https://raw.githubusercontent.com/wiedehopf/tar1090-db/master/db/icao_aircraft_types2.js"
_TYPES: dict | None = None


def load_types(offline: bool = False) -> dict:
    global _TYPES
    if _TYPES is not None:
        return _TYPES
    path = cache_dir("typedb") / "icao_aircraft_types2.json"
    if not path.exists():
        if offline:
            raise OfflineMiss("aircraft type table not cached")
        raw = get_bytes(TYPES_URL)
        try:
            raw = gzip.decompress(raw)
        except OSError:
            pass
        atomic_write_bytes(path, raw)
    _TYPES = json.loads(path.read_text())
    return _TYPES


def engine_class(
    icao_type: str | None, desc: str | None = None, types: dict | None = None
) -> str:
    """Return piston | turboprop | jet | helicopter | electric | unknown."""
    code = None
    if icao_type and types and icao_type in types:
        entry = types[icao_type]
        code = entry[1] if isinstance(entry, list) else entry.get("desc")
    if not code:
        return "unknown"
    if code[0] in "HG":
        return "helicopter"
    return {"P": "piston", "T": "turboprop", "J": "jet", "E": "electric"}.get(
        code[-1], "unknown"
    )


TONAL_CLASSES = {"piston", "turboprop", "helicopter", "unknown"}
