#!/usr/bin/env python3
"""Batch real-world evaluation: detect + locate every manifest clip, score vs ground truth.

Manifest (JSON list) entries need: id, audio, t0_unix (UTC start of the audio),
truth_lat, truth_lon, truth_unc_m (how well the camera spot is known), split
('tune'|'test'). The search region is centred off the truth (offset_km at a
bearing derived from the id) so the grid is not aligned on the answer.

ADS-B comes from the local dopplergeo cache only (run a fetch first); every
crop cached for the day(s) is merged.

    python3 scripts/eval_batch.py manifest.json --out results --tol 30 600 \\
        --split tune --set sigma_model=0.006 --set detect_prob=0.2
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dopplergeo import detect, env, score
from dopplergeo.cache import cache_dir


def km(lat1, lon1, lat2, lon2):
    k = math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot((lat1 - lat2) * 111.2, (lon1 - lon2) * 111.2 * k)


def load_tracks(t_a, t_b):
    out, seen = [], set()
    d = dt.datetime.fromtimestamp(t_a, dt.UTC).date()
    while d <= dt.datetime.fromtimestamp(t_b, dt.UTC).date():
        p = cache_dir("adsb", str(d))
        for f in sorted(p.glob("*.json.gz")):
            with gzip.open(f, "rt") as fh:
                fh.readline()
                for line in fh:
                    a = json.loads(line)
                    t = a.get("t") or []
                    if t and (t[-1] < t_a or t[0] > t_b):
                        continue  # outside the window: keep memory bounded
                    key = (a["hex"], a["t"][0] if a["t"] else 0)
                    if key not in seen:
                        seen.add(key)
                        out.append(a)
        d += dt.timedelta(days=1)
    return out


def events_for(entry, det_kw, work):
    key = hashlib.md5(
        json.dumps([entry["audio"], det_kw], sort_keys=True).encode()
    ).hexdigest()[:10]
    f = work / f"ev_{entry['id']}_{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    x, sr = detect.load_audio(entry["audio"])
    ev, _ = detect.detect_events(x, sr, **det_kw)
    r = (ev, len(x) / sr)
    f.write_bytes(pickle.dumps(r))
    return r


def _detect_job(args):
    m, det_kw, work = args
    try:
        events_for(m, det_kw, Path(work))
    except Exception:  # noqa: BLE001 - surfaced again by run_one
        return False
    return True


def region_center(entry, offset_km):
    b = int(hashlib.md5(entry["id"].encode()).hexdigest()[:6], 16) % 360
    la, lo = entry["truth_lat"], entry["truth_lon"]
    dla = offset_km * math.cos(math.radians(b)) / 111.2
    dlo = offset_km * math.sin(math.radians(b)) / (111.2 * math.cos(math.radians(la)))
    return la + dla, lo + dlo


def run_one(args):
    entry, tol, opt_kw, det_kw, work, radius_km, offset_km, heat = args
    work = Path(work)
    t_start = time.time()
    row = {"id": entry["id"], "split": entry.get("split", ""), "tol_s": tol}
    try:
        ev, dur = events_for(entry, det_kw, work)
        ev = [e for e in ev if not e.likely_ground_vehicle]
        row["events"] = len(ev)
        T0 = float(entry["t0_unix"])
        lat0, lon0 = region_center(entry, offset_km)
        o = score.Options(**opt_kw)
        tracks = load_tracks(T0 - tol - 900, T0 + tol + dur + 900)
        pl = score.prepare_planes(tracks, lat0, lon0, include_jets=o.include_jets)
        row["planes"] = len(pl)
        if not ev or not pl:
            row["error"] = "no events" if not ev else "no planes"
            return row

        def ef(la, lo):
            return env.elevation(la, lo, offline=False)[0]

        res = score.search(
            ev,
            pl,
            lat0,
            lon0,
            radius_km * 1000,
            ef,
            343.0,
            o,
            t0_window=(T0 - tol, T0 + tol),
            clip_dur=dur,
        )
        b = res["best"]
        if b is None:
            row["error"] = "no finite likelihood"
            return row
        tla, tlo, unc = (
            entry["truth_lat"],
            entry["truth_lon"],
            entry["truth_unc_m"] / 1000,
        )
        row["err_km"] = round(km(b["lat"], b["lon"], tla, tlo), 3)
        row["t0_off_s"] = round(b["t0_unix"] - T0, 1)
        row["n_matched"] = len(b["matches"] or [])
        cov = coverage(res, tla, tlo, unc)
        row.update(cov)
        if heat:
            from dopplergeo import locate

            locate.plot_heatmap(
                res,
                pl,
                lat0,
                lon0,
                str(work / f"heat_{entry['id']}_{tol}.png"),
                b,
                truth=(tla, tlo),
            )
    except Exception as e:  # noqa: BLE001 - report, keep going
        row["error"] = f"{type(e).__name__}: {e}"
    finally:
        row["runtime_s"] = round(time.time() - t_start, 1)
    return row


def coverage(res, tla, tlo, unc_km):
    """Credible-region coverage on the finest level that has the truth inside its grid.

    in68/in95: truth's nearest cell lies in the region. in68u/in95u: some region
    cell lies within the truth uncertainty radius (truth is only known that well).
    """
    out = {}
    L = None
    for lev in res["levels"]:
        d = np.hypot(
            (lev["lat"] - tla) * 111.2,
            (lev["lon"] - tlo) * 111.2 * math.cos(math.radians(tla)),
        )
        if d.min() <= lev["step"] / 1000 * 0.75:
            L, dist = lev, d
    if L is None:
        L = res["levels"][0]
        dist = np.hypot(
            (L["lat"] - tla) * 111.2,
            (L["lon"] - tlo) * 111.2 * math.cos(math.radians(tla)),
        )
    T = L.get("temperature", 1.0)
    p, a68, a95 = score.posterior(L["logl"] / T, L["step"] ** 2)
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    rank = np.empty_like(cum)
    rank[order] = cum - p[order]  # mass strictly above each cell
    j = int(np.argmin(dist))
    out["level_m"] = L["step"]
    out["temperature"] = round(float(T), 2)
    out["truth_mass_above"] = round(float(rank[j]), 4)
    out["in68"] = bool(p[j] > 0 and rank[j] < 0.68)
    out["in95"] = bool(p[j] > 0 and rank[j] < 0.95)
    near = dist <= max(unc_km, L["step"] / 1000)
    out["in68u"] = bool(np.any(near & (rank < 0.68) & (p > 0)))
    out["in95u"] = bool(np.any(near & (rank < 0.95) & (p > 0)))
    out["a68_km2"] = round((a68 or 0) / 1e6, 3)
    out["a95_km2"] = round((a95 or 0) / 1e6, 3)
    return out


def parse_set(items):
    kw = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
        if isinstance(kw[k], list):
            kw[k] = tuple(kw[k])
    return kw


def summarize(rows):
    s = {}
    for split, tol in sorted({(r["split"], r["tol_s"]) for r in rows}):
        rr = [r for r in rows if r["split"] == split and r["tol_s"] == tol]
        ok = [r for r in rr if "err_km" in r]
        e = [r["err_km"] for r in ok]
        s[f"{split}_tol{int(tol)}"] = {
            "n": len(rr),
            "solved": len(ok),
            "median_err_km": round(float(np.median(e)), 2) if e else None,
            "frac_err_lt_2km": round(float(np.mean([x < 2 for x in e])), 2)
            if e
            else None,
            "in68": round(float(np.mean([r["in68"] for r in ok])), 2) if ok else None,
            "in95": round(float(np.mean([r["in95"] for r in ok])), 2) if ok else None,
            "in68u": round(float(np.mean([r["in68u"] for r in ok])), 2) if ok else None,
            "in95u": round(float(np.mean([r["in95u"] for r in ok])), 2) if ok else None,
            "runtime_s": round(float(sum(r["runtime_s"] for r in rr)), 1),
        }
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--tol", type=float, nargs="+", default=[30.0, 600.0])
    ap.add_argument("--split", default=None, help="tune|test (default: all)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--set", action="append", help="score.Options key=value")
    ap.add_argument("--det", action="append", help="detect_events key=value")
    ap.add_argument("--radius-km", type=float, default=15.0)
    ap.add_argument("--offset-km", type=float, default=5.0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--det-jobs", type=int, default=2)
    ap.add_argument("--heatmaps", action="store_true")
    ap.add_argument(
        "--resume", action="store_true", help="keep rows already in rows.jsonl"
    )
    a = ap.parse_args(argv)
    man = json.loads(Path(a.manifest).read_text())
    man = [m for m in man if not m.get("skip")]
    if a.split:
        man = [m for m in man if m.get("split") == a.split]
    if a.only:
        man = [m for m in man if m["id"] in a.only]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    opt_kw, det_kw = parse_set(a.set), parse_set(a.det)
    jobs = []
    for m in man:
        # entries without a known start time carry their own fixed window
        tols = [m["tol_s_fixed"]] if m.get("tol_s_fixed") else a.tol
        for t in tols:
            jobs.append(
                (m, t, opt_kw, det_kw, str(out), a.radius_km, a.offset_km, a.heatmaps)
            )
    # detection first (memory heavy: a 20 min clip spectrogram is ~2-3 GB), cached
    with ProcessPoolExecutor(a.det_jobs) as ex:
        list(ex.map(_detect_job, [(m, det_kw, str(out)) for m in man]))
    # resumable: (id, tol) rows already in rows.jsonl are kept, not recomputed;
    # results are written as they complete, and one OOM-killed worker only
    # loses its own job (re-run the same command to retry it)
    rows, done = [], set()
    rp = out / "rows.jsonl"
    if a.resume and rp.exists():
        for line in rp.read_text().splitlines():
            r = json.loads(line)
            if "BrokenProcessPool" in str(r.get("error", "")):
                continue
            rows.append(r)
            done.add((r["id"], float(r["tol_s"])))
    jobs = [j for j in jobs if (j[0]["id"], float(j[1])) not in done]
    with open(rp, "w") as log:
        for r in rows:
            log.write(json.dumps(r) + "\n")
        with ProcessPoolExecutor(a.jobs, max_tasks_per_child=1) as ex:
            futs = {ex.submit(run_one, j): j for j in jobs}
            for f in as_completed(futs):
                j = futs[f]
                try:
                    r = f.result()
                except Exception as e:  # noqa: BLE001 - e.g. worker OOM-killed
                    r = {
                        "id": j[0]["id"],
                        "split": j[0].get("split", ""),
                        "tol_s": j[1],
                        "error": f"{type(e).__name__}: {e}",
                    }
                rows.append(r)
                log.write(json.dumps(r) + "\n")
                log.flush()
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    summ = {"options": opt_kw, "detect": det_kw, "summary": summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ["summary"], indent=1))
    return rows


if __name__ == "__main__":
    main()
