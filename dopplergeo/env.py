"""Ground elevation (Copernicus GLO-30) and air temperature (Open-Meteo archive).

DEM: Copernicus DEM GLO-30 on AWS Open Data (s3://copernicus-dem-30m, anonymous
HTTPS), one 1x1 degree COG per tile. Tiles are cached. Decoding needs only
tifffile (tag parsing) + zlib: the floating-point predictor is undone here, so
imagecodecs/rasterio are not required. Without tifffile, offline misses, or
ocean tiles (no file), elevation falls back to flat terrain (0 m, or the value
passed as flat_m) and the result says so.

Temperature: Open-Meteo historical archive (no key, non-commercial use), hourly
2 m temperature -> speed of sound c = 331.3 * sqrt(1 + T/273.15).

Copernicus DEM: (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH
2014-2018, provided under COPERNICUS by the European Union and ESA.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import urllib.error
import zlib

import numpy as np

from .cache import OfflineMiss, cache_dir
from .http import get_bytes, get_json
from .physics import C_DEFAULT, speed_of_sound

DEM_ATTRIBUTION = "Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the EU and ESA"
WEATHER_ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"


def _tile_name(lat_i: int, lon_i: int) -> str:
    ns = f"N{lat_i:02d}" if lat_i >= 0 else f"S{-lat_i:02d}"
    ew = f"E{lon_i:03d}" if lon_i >= 0 else f"W{-lon_i:03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


class _Tile:
    def __init__(self, path):
        import tifffile

        self.path = path
        tf = tifffile.TiffFile(path)
        p = tf.pages[0]
        self.h, self.w = p.shape
        self.tw, self.tl = p.tilewidth, p.tilelength
        self.offsets, self.counts = p.dataoffsets, p.databytecounts
        tie = p.tags["ModelTiepointTag"].value
        sc = p.tags["ModelPixelScaleTag"].value
        self.lon0, self.lat0, self.dx, self.dy = tie[3], tie[4], sc[0], sc[1]
        self.predictor = p.tags["Predictor"].value if "Predictor" in p.tags else 1
        tf.close()
        self._cache = {}

    def _tile(self, i):
        if i not in self._cache:
            with open(self.path, "rb") as fh:
                fh.seek(self.offsets[i])
                raw = zlib.decompress(fh.read(self.counts[i]))
            if self.predictor == 3:
                b = np.frombuffer(raw, np.uint8).reshape(self.tl, self.tw * 4)
                b = (
                    np.cumsum(b, axis=1, dtype=np.uint8)
                    .reshape(self.tl, 4, self.tw)
                    .transpose(0, 2, 1)
                )
                a = np.ascontiguousarray(b).view(">f4").reshape(self.tl, self.tw)
            else:
                a = np.frombuffer(raw, "<f4").reshape(self.tl, self.tw)
            self._cache[i] = a.astype(np.float32)
        return self._cache[i]

    def sample(self, lat, lon):
        r = np.clip(((self.lat0 - lat) / self.dy).astype(int), 0, self.h - 1)
        c = np.clip(((lon - self.lon0) / self.dx).astype(int), 0, self.w - 1)
        ntx = (self.w + self.tw - 1) // self.tw
        out = np.empty(len(r), np.float32)
        ti = (r // self.tl) * ntx + (c // self.tw)
        for i in np.unique(ti):
            m = ti == i
            out[m] = self._tile(int(i))[r[m] % self.tl, c[m] % self.tw]
        return out


def _get_tile(lat_i, lon_i, offline):
    name = _tile_name(lat_i, lon_i)
    path = cache_dir("dem") / f"{name}.tif"
    missing = cache_dir("dem") / f"{name}.none"
    if path.exists():
        return _Tile(path)
    if missing.exists():
        return None
    if offline:
        raise OfflineMiss(f"DEM tile {name} not cached")
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
    try:
        data = get_bytes(url, timeout=300)
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):  # ocean / no tile
            missing.write_text("no tile")
            return None
        raise
    path.write_bytes(data)
    return _Tile(path)


def elevation(lat, lon, offline=False, flat_m=0.0) -> tuple[np.ndarray, str]:
    """Ground elevation in metres for arrays of lat/lon. Returns (elev, source)."""
    lat = np.atleast_1d(np.asarray(lat, float))
    lon = np.atleast_1d(np.asarray(lon, float))
    out = np.full(lat.shape, float(flat_m))
    try:
        import tifffile  # noqa: F401
    except ImportError:
        return out, "flat (tifffile not installed)"
    src = "copernicus-glo30"
    li, oi = np.floor(lat).astype(int), np.floor(lon).astype(int)
    for a, b in set(zip(li.tolist(), oi.tolist())):
        m = (li == a) & (oi == b)
        try:
            tile = _get_tile(a, b, offline)
        except (OfflineMiss, OSError, urllib.error.URLError) as e:
            src = f"flat (DEM unavailable: {e})"
            continue
        if tile is None:
            continue  # sea: 0 m is right
        out[m] = tile.sample(lat[m], lon[m])
    return out, src


def temperature_c(
    lat: float, lon: float, when: dt.datetime, offline=False
) -> float | None:
    """Hourly 2 m air temperature from the Open-Meteo archive (cached)."""
    day = when.date()
    key = f"{lat:.2f}_{lon:.2f}_{day}"
    f = cache_dir("weather") / f"{key}.json"
    if f.exists():
        data = json.loads(f.read_text())
    else:
        if offline:
            return None
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat:.4f}&longitude={lon:.4f}&start_date={day}&end_date={day}&hourly=temperature_2m&timezone=UTC"
        )
        try:
            data = get_json(url)
        except (OSError, urllib.error.URLError, ValueError):
            return None
        f.write_text(json.dumps(data))
    temps = data.get("hourly", {}).get("temperature_2m") or []
    if not temps:
        return None
    h = min(len(temps) - 1, when.hour)
    v = temps[h]
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def sound_speed(lat, lon, when, offline=False) -> tuple[float, str]:
    if when is None:
        return C_DEFAULT, "default 343 m/s (no time)"
    tc = temperature_c(lat, lon, when, offline)
    if tc is None:
        return C_DEFAULT, "default 343 m/s (temperature unavailable)"
    return speed_of_sound(tc), f"Open-Meteo {tc:.1f} C"
