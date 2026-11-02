#!/usr/bin/env python3
"""Self-test: synthesize a quiet prop flyover (curved track) buried in noise, recover it
with doppler_track.py, then locate the observer on a grid with score_candidates.py.
Offline, no network. Exit 0 on pass."""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_candidates import R_E, enu

C = 343.0


def run_selftest(workdir=None, grid_half_m=3000, grid_step_m=200, seed=0, verbose=True):
    d = workdir or tempfile.mkdtemp()
    sr = 4000
    rng = np.random.default_rng(seed)
    lat0, lon0 = 40.0, -75.0
    true = (
        40.0 + 600 / R_E * 180 / np.pi,
        -75.0,
        50.0,
    )  # observer 600 m north of origin
    # eastbound 80 m/s at 900 m MSL with a gentle turn (non-linear geometry)
    tt = np.arange(0, 200, 1.0)
    hd = np.radians(90 + 0.08 * (tt - 100))
    xs = np.cumsum(80 * np.sin(hd)) - 8000
    ys = np.cumsum(80 * np.cos(hd))
    ys -= ys[100]
    lat = lat0 + np.degrees(ys / R_E)
    lon = lon0 + np.degrees(xs / (R_E * np.cos(np.radians(lat0))))
    alt = np.full_like(tt, 900.0)
    np.savetxt(
        f"{d}/track.csv",
        np.c_[tt, lat, lon, alt],
        delimiter=",",
        header="time_s,lat,lon,alt_m",
        comments="",
    )
    # audio heard by the true observer (emission-time physics), recording starts at track t=30
    tau = np.arange(0, 200, 1 / sr)
    P = enu(
        np.interp(tau, tt, lat), np.interp(tau, tt, lon), np.interp(tau, tt, alt), *true
    )
    r = np.linalg.norm(P, axis=1)
    trec = tau + r / C
    rt = np.arange(30, 170, 1 / sr)
    phase_t = np.interp(rt, trec, tau)
    f0 = 110.0
    amp = 300 / np.interp(rt, trec, r)
    x = amp * (
        np.sin(2 * np.pi * f0 * phase_t) + 0.5 * np.sin(4 * np.pi * f0 * phase_t)
    )
    x = x + rng.normal(0, 1.0, x.size)
    x /= np.abs(x).max() * 1.1
    sf.write(f"{d}/rec.wav", x, sr)

    def run(*a):
        return subprocess.run(
            [sys.executable, *a], check=True, capture_output=True, text=True
        ).stdout

    run(
        f"{HERE}/doppler_track.py",
        f"{d}/rec.wav",
        "--fmin",
        "80",
        "--fmax",
        "150",
        "--out",
        f"{d}/obs",
    )
    with open(f"{d}/obs.json") as fh:
        fit = json.load(fh)
    g = np.arange(-grid_half_m, grid_half_m + 1, grid_step_m)
    with open(f"{d}/cands.csv", "w") as fh:
        fh.write("lat,lon,elev_m\n")
        for gy in g:
            fh.writelines(
                f"{lat0 + np.degrees(gy / R_E)},"
                f"{lon0 + np.degrees(gx / (R_E * np.cos(np.radians(lat0))))},50\n"
                for gx in g
            )
    # known recording start (true 30 s, given 31.5 s with +/-2 s clock error)
    run(
        f"{HERE}/score_candidates.py",
        f"{d}/track.csv",
        f"{d}/cands.csv",
        f"{d}/obs.csv",
        "--f0",
        str(round(fit["f0_hz"], 1)),
        "--rec-start",
        "31.5",
        "--dt-tol",
        "2",
        "--out",
        f"{d}/ranked.csv",
        "--plot",
        f"{d}/scores.png",
        "--true",
        f"{true[0]},{true[1]}",
    )
    best = np.loadtxt(f"{d}/ranked.csv", delimiter=",", skiprows=1)[0]
    dn = np.radians(best[1] - true[0]) * R_E
    de = np.radians(best[2] - true[1]) * R_E * np.cos(np.radians(lat0))
    res = {
        "f0_hz": fit["f0_hz"],
        "speed_mps": fit["speed_mps"],
        "cpa_slant_m": fit["cpa_slant_m"],
        "true_slant_m": float(np.hypot(600, 850)),
        "location_error_m": float(np.hypot(dn, de)),
        "workdir": d,
    }
    if verbose:
        print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = run_selftest()
    ok = (
        abs(r["f0_hz"] - 110) < 2
        and abs(r["speed_mps"] - 80) < 5
        and r["location_error_m"] <= 300
    )
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    sys.exit(0 if ok else 1)
