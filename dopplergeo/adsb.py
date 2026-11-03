"""ADS-B history from adsb.lol daily releases (ODbL 1.0, no account).

Each day is published as a GitHub release in adsblol/globe_history_<year>, tag
v<YYYY>.<MM>.<DD>-planes-readsb-prod-0, as a split tar (.tar.aa, .tar.ab, ...)
of readsb trace files traces/<xx>/trace_full_<hex>.json (gzip JSON). We stream
the parts in order through tarfile without saving them, keep only points inside
a bounding box and time window, and cache that crop. A day is ~3 GB to stream.

Trace point layout (readsb globe_history): [dt_s, lat, lon, alt_baro_ft|"ground",
gs_kt, track_deg, flags, vrate_fpm, extra|None, source, alt_geom_ft, ...];
absolute time = file "timestamp" + dt_s.

Data: (c) adsb.lol contributors, Open Database License (ODbL) 1.0.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import tarfile
from collections.abc import Iterable
from pathlib import Path

from .cache import OfflineMiss, cache_dir
from .http import get_json, open_url
from .typedb import engine_class, load_types

FT = 0.3048
KT = 0.514444
ATTRIBUTION = (
    "ADS-B data: adsb.lol (https://adsb.lol), Open Database License (ODbL) 1.0"
)

try:  # optional speed-up
    import orjson as _json

    _loads = _json.loads
except ImportError:  # pragma: no cover
    _loads = json.loads


def release_tag(day: dt.date, variant: str = "prod-0") -> tuple[str, str]:
    return (
        f"adsblol/globe_history_{day.year}",
        f"v{day:%Y.%m.%d}-planes-readsb-{variant}",
    )


def release_assets(day: dt.date, variant: str = "prod-0") -> list[str]:
    """Download URLs of the split tar parts, in order (checked live via the GitHub API)."""
    repo, tag = release_tag(day, variant)
    rel = get_json(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
    urls = sorted(
        (a["name"], a["browser_download_url"])
        for a in rel.get("assets", [])
        if ".tar." in a["name"]
    )
    if not urls:
        raise RuntimeError(f"no tar parts in {repo} {tag}")
    return [u for _, u in urls]


class _Chain(io.RawIOBase):
    """Read several streams back to back as one (the split tar parts)."""

    def __init__(self, openers: Iterable):
        self._openers = iter(openers)
        self._cur = None

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self._cur is None:
                try:
                    self._cur = next(self._openers)()
                except StopIteration:
                    return 0
            n = self._cur.readinto(b)
            if n:
                return n
            self._cur.close()
            self._cur = None


def _crop_trace(d: dict, bbox, t0: float, t1: float, types: dict | None):
    la0, lo0, la1, lo1 = bbox
    base = float(d.get("timestamp", 0))
    T, LA, LO, ALT, GS = [], [], [], [], []
    for p in d.get("trace", ()):
        t = base + p[0]
        if t < t0 or t > t1:
            continue
        lat, lon = p[1], p[2]
        if lat is None or not (la0 <= lat <= la1 and lo0 <= lon <= lo1):
            continue
        alt = p[10] if len(p) > 10 and isinstance(p[10], (int, float)) else p[3]
        if not isinstance(alt, (int, float)):
            continue  # "ground" or missing
        T.append(t)
        LA.append(lat)
        LO.append(lon)
        ALT.append(alt * FT)
        GS.append((p[4] or 0.0) * KT if isinstance(p[4], (int, float)) else None)
    if len(T) < 2:
        return None
    icao_type = d.get("t")
    return {
        "hex": d.get("icao"),
        "type": icao_type,
        "desc": d.get("desc"),
        "reg": d.get("r"),
        "class": engine_class(icao_type, d.get("desc"), types),
        "t": T,
        "lat": LA,
        "lon": LO,
        "alt_m": ALT,
        "gs_mps": GS,
    }


def crop_tar_stream(fileobj, bbox, t0, t1, types=None, progress=None) -> list[dict]:
    out = []
    n = 0
    with tarfile.open(fileobj=fileobj, mode="r|") as tf:
        for m in tf:
            if not m.isfile() or "trace_full_" not in m.name:
                continue
            n += 1
            raw = tf.extractfile(m).read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            try:
                d = _loads(raw)
            except ValueError:
                continue
            a = _crop_trace(d, bbox, t0, t1, types)
            if a:
                out.append(a)
            if progress and n % 20000 == 0:
                progress(n, len(out))
    return out


def _key(day, bbox, t0, t1):
    s = json.dumps([str(day), [round(v, 4) for v in bbox], int(t0), int(t1)])
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _find_cached(day, bbox, t0, t1) -> Path | None:
    d = cache_dir("adsb", str(day))
    for f in sorted(d.glob("*.json.gz")):
        try:
            with gzip.open(f, "rt") as fh:
                meta = json.loads(fh.readline())
        except (OSError, ValueError):
            continue
        b = meta["bbox"]
        if (
            b[0] <= bbox[0]
            and b[1] <= bbox[1]
            and b[2] >= bbox[2]
            and b[3] >= bbox[3]
            and meta["t0"] <= t0
            and meta["t1"] >= t1
        ):
            return f
    return None


def _read_cache(f: Path, bbox, t0, t1) -> list[dict]:
    with gzip.open(f, "rt") as fh:
        fh.readline()
        planes = [json.loads(line) for line in fh]
    la0, lo0, la1, lo1 = bbox
    out = []
    for a in planes:
        keep = [
            i
            for i, (t, la, lo) in enumerate(zip(a["t"], a["lat"], a["lon"]))
            if t0 <= t <= t1 and la0 <= la <= la1 and lo0 <= lo <= lo1
        ]
        if len(keep) >= 2:
            out.append(
                {
                    **a,
                    **{
                        k: [a[k][i] for i in keep]
                        for k in ("t", "lat", "lon", "alt_m", "gs_mps")
                    },
                }
            )
    return out


def fetch_tracks(
    day: dt.date,
    bbox: tuple[float, float, float, float],
    t0: float | None = None,
    t1: float | None = None,
    offline: bool = False,
    tar_parts: list[str] | None = None,
    variant: str = "prod-0",
    progress=None,
) -> list[dict]:
    """Aircraft tracks inside bbox=(lat_min, lon_min, lat_max, lon_max) during [t0, t1] (unix s).

    Uses the local cache when a crop covers the request. Otherwise streams the
    day's release (or local tar_parts) and caches the crop. offline=True never
    touches the network.
    """
    day_start = dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp()
    t0 = day_start if t0 is None else t0
    t1 = day_start + 86400 if t1 is None else t1
    hit = _find_cached(day, bbox, t0, t1)
    if hit:
        return _read_cache(hit, bbox, t0, t1)
    if offline and not tar_parts:
        raise OfflineMiss(
            f"no cached ADS-B crop for {day} covering "
            + "lat {0:.4f}..{2:.4f}, lon {1:.4f}..{3:.4f}".format(*map(float, bbox))
        )
    try:
        types = load_types(offline=offline)
    except (OfflineMiss, OSError):
        types = None
    if tar_parts:
        openers = [lambda p=p: open(p, "rb") for p in tar_parts]
    else:
        openers = [
            lambda u=u: open_url(u, timeout=300) for u in release_assets(day, variant)
        ]
    stream = io.BufferedReader(_Chain(openers), buffer_size=1 << 20)
    planes = crop_tar_stream(stream, bbox, t0, t1, types, progress)
    f = cache_dir("adsb", str(day)) / f"{_key(day, bbox, t0, t1)}.json.gz"
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    with gzip.open(tmp, "wt") as fh:
        fh.write(
            json.dumps(
                {
                    "bbox": list(bbox),
                    "t0": t0,
                    "t1": t1,
                    "day": str(day),
                    "source": ATTRIBUTION,
                }
            )
            + "\n"
        )
        for a in planes:
            fh.write(json.dumps(a) + "\n")
    os.replace(tmp, f)  # atomic: readers never see a truncated cache file
    return planes
