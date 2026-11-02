"""Local cache directory (override with DOPPLER_GEO_CACHE)."""

from __future__ import annotations

import os
from pathlib import Path


def cache_dir(*parts: str) -> Path:
    root = Path(
        os.environ.get("DOPPLER_GEO_CACHE") or Path.home() / ".cache" / "doppler-geo"
    )
    p = root.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


class OfflineMiss(RuntimeError):
    """Raised when offline mode needs data that is not cached."""
