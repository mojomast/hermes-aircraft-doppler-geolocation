"""Content-addressed cache of detected events, shared by detect, locate and eval.

Key = sha256 over (audio file bytes, decode sample rate, detector settings,
detector implementation). The implementation identity is a hash of the source
of the modules that produce events (detect.py, broadband.py, physics.py), so
any detector change invalidates old entries without a manual version bump;
SCHEMA covers the stored layout. Region, date and scoring options are not part
of the key: re-locating a clip with another hint reuses its detection.

Entries are JSON written atomically (temp file + rename). A corrupt, truncated
or mismatched entry is treated as a miss and removed. Entries unused for
MAX_AGE_DAYS, and the oldest beyond MAX_ENTRIES, are pruned on write.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .cache import atomic_write_bytes, cache_dir

SCHEMA = 1
MAX_ENTRIES = 500
MAX_AGE_DAYS = 90
_DETECTOR_MODULES = ("detect.py", "broadband.py", "physics.py")
_impl = None


def detector_impl_hash() -> str:
    """Hash of the detector source code (cached per process)."""
    global _impl
    if _impl is None:
        h = hashlib.sha256()
        here = Path(__file__).resolve().parent
        for name in _DETECTOR_MODULES:
            h.update(name.encode() + b"\0" + (here / name).read_bytes())
        _impl = h.hexdigest()[:16]
    return _impl


def file_sha256(path, bufsize=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def _canon(v):
    if isinstance(v, dict):
        return {str(k): _canon(v[k]) for k in sorted(v)}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, float):
        return repr(v)
    return v


def make_key(audio_sha256: str, sr: int, det_kw: dict) -> tuple[str, dict]:
    material = {
        "schema": SCHEMA,
        "audio_sha256": audio_sha256,
        "sr": int(sr),
        "detector": _canon(det_kw or {}),
        "impl": detector_impl_hash(),
    }
    blob = json.dumps(material, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:32], material


def _dir() -> Path:
    return cache_dir("events")


def load(key: str, material: dict):
    """Cached entry dict or None (miss, corrupt or mismatched entries are removed)."""
    p = _dir() / f"{key}.json"
    try:
        raw = p.read_bytes()
    except OSError:
        return None
    try:
        d = json.loads(raw)
        if d.get("key_material") != material or not isinstance(d.get("events"), list):
            raise ValueError("mismatched entry")
    except (ValueError, TypeError, AttributeError):
        try:
            p.unlink()
        except OSError:
            pass
        return None
    try:
        os.utime(p)  # LRU: mark used
    except OSError:
        pass
    return d


def store(key: str, material: dict, payload: dict) -> None:
    d = _dir()
    atomic_write_bytes(
        d / f"{key}.json",
        json.dumps({**payload, "key_material": material}).encode(),
    )
    prune(d)


def prune(d: Path | None = None, max_entries=MAX_ENTRIES, max_age_days=MAX_AGE_DAYS):
    d = d or _dir()
    now = time.time()
    files = []
    for p in d.glob("*.json"):
        try:
            files.append((p.stat().st_mtime, p))
        except OSError:
            continue
    files.sort(reverse=True)
    for i, (mt, p) in enumerate(files):
        if i >= max_entries or now - mt > max_age_days * 86400:
            try:
                p.unlink()
            except OSError:
                pass
