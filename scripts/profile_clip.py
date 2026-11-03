#!/usr/bin/env python3
"""Stage timings + peak memory for one manifest clip (detect -> ADS-B -> search).

    python3 scripts/profile_clip.py manifest.json CLIP_ID --tol 30 --out /tmp/p.pkl \
        [--det broadband=true]

Prints one JSON line with stage seconds, peak RSS (MB) and a result digest; the
pickle (events, per-level logl/t0, best) is used for before/after equivalence.
ADS-B and DEM are read from the dopplergeo cache (DOPPLER_GEO_CACHE).
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_batch as eb  # noqa: E402
from dopplergeo import detect, env, score  # noqa: E402


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("clip")
    ap.add_argument("--tol", type=float, default=30.0)
    ap.add_argument("--det", action="append")
    ap.add_argument("--set", action="append")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    m = next(e for e in json.loads(Path(a.manifest).read_text()) if e["id"] == a.clip)
    det_kw, opt_kw = eb.parse_set(a.det), eb.parse_set(a.set)
    st = {}
    t = time.perf_counter()
    x, sr = detect.load_audio(m["audio"])
    st["load_audio"] = time.perf_counter() - t
    bb = det_kw.pop("broadband", False)
    t = time.perf_counter()
    ev, _diag = detect.detect_events(x, sr, **det_kw)
    st["detect_tonal"] = time.perf_counter() - t
    if bb:
        from dopplergeo.broadband import detect_broadband

        t = time.perf_counter()
        bev, _ = detect_broadband(x, sr)
        st["detect_broadband"] = time.perf_counter() - t
        bev = [b for b in bev if not any(abs(b.t_cpa - e.t_cpa) < 15.0 for e in ev)]
        ev = sorted(ev + bev, key=lambda e: e.t_cpa)
    del _diag
    dur = len(x) / sr
    del x
    st["peak_mb_after_detect"] = peak_mb()
    evs = [e for e in ev if not e.likely_ground_vehicle]
    T0 = float(m["t0_unix"])
    lat0, lon0 = eb.region_center(m, 5.0)
    o = score.Options(**opt_kw)
    t = time.perf_counter()
    tracks = eb.load_tracks(T0 - a.tol - 900, T0 + a.tol + dur + 900)
    need_jets = o.include_jets or any(score.is_broadband(e) for e in evs)
    pl = score.prepare_planes(tracks, lat0, lon0, include_jets=need_jets)
    st["adsb_load_prepare"] = time.perf_counter() - t
    lv_t = []
    last = [time.perf_counter()]

    def prog(li, n):
        now = time.perf_counter()
        lv_t.append((li, n, now - last[0]))
        last[0] = now

    res = None
    if evs and pl:
        t = time.perf_counter()
        last[0] = t
        res = score.search(
            evs, pl, lat0, lon0, 15000.0,
            lambda la, lo: env.elevation(la, lo, offline=True)[0],
            343.0, o, t0_window=(T0 - a.tol, T0 + a.tol), clip_dur=dur, progress=prog,
        )
        st["search"] = time.perf_counter() - t
    st["peak_mb"] = peak_mb()
    out = {"clip": a.clip, "events": len(evs), "planes": len(pl), "stages": {k: round(v, 3) for k, v in st.items()},
           "levels": [(li, n, round(s, 3)) for li, n, s in lv_t]}
    if res and res["best"]:
        b = res["best"]
        out["err_km"] = round(eb.km(b["lat"], b["lon"], m["truth_lat"], m["truth_lon"]), 4)
        cov = eb.coverage(res, m["truth_lat"], m["truth_lon"], m["truth_unc_m"] / 1000)
        out.update({k: cov[k] for k in ("in68u", "in95u", "a68_km2", "a95_km2")})
        out["n_consistent"] = score.n_consistent_matches(b["matches"] or [])
    if a.out:
        keep = None
        if res:
            keep = {"best": res["best"], "levels": [{k: L[k] for k in ("step", "x", "y", "logl", "t0", "temperature", "a68_m2", "a95_m2", "matches")} for L in res["levels"]]}
        Path(a.out).write_bytes(pickle.dumps({"events": [e.to_dict(series=True) for e in ev], "res": keep}))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
