"""Bounded DEM tile reuse and streamed ADS-B crop reads."""

import gzip
import json

import numpy as np

from dopplergeo import adsb, env


def test_dem_tiles_are_reused_and_bounded(tmp_path, monkeypatch):
    made = []

    class FakeTile:
        def __init__(self, path):
            made.append(path)

    monkeypatch.setattr(env, "_Tile", FakeTile)
    monkeypatch.setattr(env, "_open_tiles", env.OrderedDict())
    paths = []
    for i in range(env.MAX_DEM_FILES + 2):
        p = tmp_path / f"{i}.tif"
        p.write_bytes(b"x")
        paths.append(p)
    a = env._open_tile(paths[0])
    assert env._open_tile(paths[0]) is a and len(made) == 1
    for p in paths:
        env._open_tile(p)
    assert len(env._open_tiles) == env.MAX_DEM_FILES
    paths[-1].write_bytes(b"changed")  # rewritten file -> reparsed
    env._open_tile(paths[-1])
    assert made.count(paths[-1]) == 2


def _old_read_cache(f, bbox, t0, t1):  # v0.3.0 behaviour
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


def test_streamed_crop_read_matches_full_load(tmp_path):
    rng = np.random.default_rng(0)
    f = tmp_path / "c.json.gz"
    with gzip.open(f, "wt") as fh:
        fh.write(json.dumps({"bbox": [0, 0, 1, 1], "t0": 0, "t1": 5000}) + "\n")
        for i in range(200):
            n = int(rng.integers(0, 60))
            t0 = float(rng.uniform(0, 5000))
            a = {
                "hex": f"h{i}",
                "t": list(t0 + np.cumsum(rng.uniform(1, 20, n))),
                "lat": list(rng.uniform(0, 1, n)),
                "lon": list(rng.uniform(0, 1, n)),
                "alt_m": list(rng.uniform(0, 3000, n)),
                "gs_mps": list(rng.uniform(0, 200, n)),
            }
            fh.write(json.dumps(a) + "\n")
    for bbox, t0, t1 in (((0.2, 0.2, 0.8, 0.8), 1000, 3000), ((0, 0, 1, 1), 0, 9e9)):
        assert adsb._read_cache(f, bbox, t0, t1) == _old_read_cache(f, bbox, t0, t1)
    assert adsb._crop_meta(f)["t1"] == 5000
