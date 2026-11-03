#!/usr/bin/env python3
"""Model the exact Doppler curve each candidate observer would hear from a REAL flight
track (ADS-B), emission-time corrected, and score it against an observed ridge.

track CSV: time_s,lat,lon,alt_m (alt above MSL; geometric preferred). Unix or relative s.
candidates CSV: lat,lon,elev_m (ground elevation from a DEM).
observed CSV: output of doppler_track.py (time_s,freq_hz[,power_db]) in recording time.
Score = RMS Hz misfit after sweeping f0 (+/- f0-tol) and time offset (+/- dt-tol) around
the measured CPA. Lower is better. --cpa-rec is the observed CPA receipt time (recording s).
Without --rec-start a single flyover has an along-track ambiguity (shown in the video:
use a known time OR a second event at a locked offset).
Usage: score_candidates.py track.csv candidates.csv observed.csv --f0 120 --cpa-rec 95 [--out ranked.csv]
"""

import argparse
import csv

import numpy as np

C = 343.0
R_E = 6371000.0


def enu(lat, lon, alt, lat0, lon0, h0):
    """Local flat-earth ENU (fine for <~30 km)."""
    x = np.radians(lon - lon0) * R_E * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R_E
    return np.c_[x, y, alt - h0]


def smooth_track(P, dt, smooth_s):
    """Centred moving average of positions. ADS-B position jitter of a few tens of metres
    becomes metres-per-second velocity noise under np.gradient, i.e. several Hz of Doppler.

    The window is forced to an odd number of samples so it is truly centred (an even
    window shifts the track by half a sample), and the ends are padded by odd reflection
    about the end points, which extends a straight track linearly so velocities near the
    ends are not biased towards zero."""
    n = round(smooth_s / dt)
    if n < 2 or len(P) < 2:
        return P
    h = n // 2  # odd window of 2h+1 samples
    h = min(h, len(P) - 1)
    if h < 1:
        return P
    k = np.ones(2 * h + 1) / (2 * h + 1)
    pad = np.pad(P, ((h, h), (0, 0)), mode="reflect", reflect_type="odd")
    return np.column_stack(
        [np.convolve(pad[:, i], k, "valid") for i in range(P.shape[1])]
    )


def received_curve(tr, cand, f0, c=C, dt=0.5, smooth_s=0.0):
    """Return (receive_times, freq, cpa_receive_time) for one observer."""
    t, lat, lon, alt = tr.T
    tau = np.arange(t.min(), t.max(), dt)
    P = enu(
        np.interp(tau, t, lat), np.interp(tau, t, lon), np.interp(tau, t, alt), *cand
    )
    P = smooth_track(P, dt, smooth_s)
    V = np.gradient(P, dt, axis=0)
    r = np.linalg.norm(P, axis=1)
    vr = -np.sum(P * V, axis=1) / r  # +ve approaching, AT EMISSION TIME tau
    trecv = tau + r / c
    k = np.argmin(r)
    return trecv, f0 * c / (c - vr), trecv[k], r[k]


def score(
    obs_t, obs_f, trecv, fmod, cpa_recv, cpa_obs, f0, f0_tol, dt_tol, rec_start=None
):
    best = (np.inf, 0, 1)
    for dts in np.arange(-dt_tol, dt_tol + 1e-9, 0.5):
        # known clock: anchor absolutely; unknown: align measured CPA to modelled CPA
        tt = (
            (obs_t + rec_start + dts)
            if rec_start is not None
            else (obs_t - cpa_obs + cpa_recv + dts)
        )
        m = (tt >= trecv.min()) & (tt <= trecv.max())
        if m.sum() < 5:
            continue
        base = np.interp(tt[m], trecv, fmod) / f0
        for fs in np.linspace(f0 - f0_tol, f0 + f0_tol, 21):
            e = np.sqrt(np.mean((base * fs - obs_f[m]) ** 2))
            if e < best[0]:
                best = (e, dts, fs)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("candidates")
    ap.add_argument("observed")
    ap.add_argument("--f0", type=float, required=True)
    ap.add_argument("--cpa-rec", type=float, default=0.0)
    ap.add_argument(
        "--rec-start",
        type=float,
        help="track-clock time of recording t=0 if known (then --dt-tol = clock error)",
    )
    ap.add_argument("--f0-tol", type=float, default=3.0)
    ap.add_argument("--dt-tol", type=float, default=5.0)
    ap.add_argument(
        "--smooth-s",
        type=float,
        default=0.0,
        help="moving-average window (s) on track positions before differentiating; "
        "~10-20 s helps with jittery 1 Hz ADS-B feeds (0 = off)",
    )
    ap.add_argument("--out", default="ranked.csv")
    ap.add_argument(
        "--plot", help="write a PNG heatmap of candidate misfit to this path"
    )
    ap.add_argument("--true", help="lat,lon of a known true point to mark on the plot")
    a = ap.parse_args()
    tr = np.loadtxt(a.track, delimiter=",", skiprows=1)
    cands = np.loadtxt(a.candidates, delimiter=",", skiprows=1, ndmin=2)
    ob = np.loadtxt(a.observed, delimiter=",", skiprows=1)
    rows = []
    for cand in cands:
        trecv, fm, cpa_r, dmin = received_curve(tr, cand, a.f0, smooth_s=a.smooth_s)
        e, dts, fs = score(
            ob[:, 0],
            ob[:, 1],
            trecv,
            fm,
            cpa_r,
            a.cpa_rec,
            a.f0,
            a.f0_tol,
            a.dt_tol,
            a.rec_start,
        )
        rows.append((e, *cand, dmin, cpa_r + dts, fs))
    rows.sort()
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "rms_hz",
                "lat",
                "lon",
                "elev_m",
                "cpa_slant_m",
                "cpa_receive_time_track",
                "f0_best",
            ]
        )
        w.writerows(rows)
    for r in rows[:5]:
        print(", ".join(f"{v:.6g}" for v in r))
    if a.plot:
        true = [float(v) for v in a.true.split(",")] if a.true else None
        plot_scores(np.array(rows), tr, a.plot, true)


def plot_scores(rows, track, path, true=None, top=5):
    """Heatmap of RMS misfit per candidate, flight track, top-ranked points, true point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        rows[:, 2],
        rows[:, 1],
        c=np.log10(rows[:, 0]),
        cmap="viridis_r",
        s=40,
        marker="s",
    )
    fig.colorbar(sc, ax=ax, label="log10 RMS misfit (Hz), lower = better")
    lo, la = track[:, 2], track[:, 1]
    m = (lo >= rows[:, 2].min() - 0.02) & (lo <= rows[:, 2].max() + 0.02)
    ax.plot(
        lo[m],
        la[m],
        "-",
        color="orangered",
        lw=3,
        zorder=3,
        label="ADS-B track (curving)",
    )
    ax.scatter(
        rows[:top, 2],
        rows[:top, 1],
        s=140,
        facecolors="none",
        edgecolors="red",
        lw=2,
        label=f"top {top}",
    )
    for i in range(min(top, len(rows))):
        ax.annotate(
            str(i + 1),
            (rows[i, 2], rows[i, 1]),
            color="red",
            xytext=(6, 6),
            textcoords="offset points",
        )
    if true:
        ax.scatter(
            [true[1]],
            [true[0]],
            marker="*",
            s=300,
            c="gold",
            edgecolors="k",
            label="true observer",
        )
    ax.set_xlim(rows[:, 2].min() - 0.003, rows[:, 2].max() + 0.003)
    ax.set_ylim(rows[:, 1].min() - 0.003, rows[:, 1].max() + 0.003)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Candidate scoring (emission-time Doppler model vs observed ridge)")
    ax.legend(loc="lower left", facecolor="#dddddd")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
