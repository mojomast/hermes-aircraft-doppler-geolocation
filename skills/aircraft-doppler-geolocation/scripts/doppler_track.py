#!/usr/bin/env python3
"""Extract a Doppler ridge (e.g. propeller-aircraft flyover) from audio/video and fit
a constant-velocity straight-line flyover model.

Outputs: <out>.csv (time_s, freq_hz, power_db), <out>.png (spectrogram + ridge + fit),
<out>.json (fit: f0_hz, speed_mps, cpa_slant_m, cpa_time_s in recording time).
Usage: doppler_track.py INPUT --fmin 60 --fmax 200 [--t0 S --t1 S] [--out PREFIX]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
from scipy import optimize, signal

C = 343.0  # speed of sound m/s (adjust for temperature: 331.3*sqrt(1+T/273.15))


def load_audio(path, sr=4000):
    """Decode any audio/video to mono float at sr via ffmpeg (keeps low freqs)."""
    tmp = os.path.join(tempfile.mkdtemp(), "a.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            tmp,
        ],
        check=True,
    )
    x, sr = sf.read(tmp)
    return x.astype(float), sr


def doppler_model(t, f0, v, d, tc, c=C):
    """Received frequency at receive times t for a source at constant speed v passing
    an observer at closest slant distance d. Emission-time corrected: the frequency heard
    at t is set by the radial velocity when the sound was EMITTED (tau), t = tau + r/c."""
    tau = np.linspace(t.min() - 60 - 3 * d / c, t.max() + 60, 4000)
    x = v * (tau - tc)  # along-track position relative to CPA
    r = np.sqrt(x**2 + d**2)
    vr = -v * x / r  # +ve = approaching
    f = f0 * c / (c - vr)
    trecv = tau + r / c
    # tc here = time the CPA *sound* arrives: shift so CPA receipt == tc
    trecv = trecv - d / c
    return np.interp(t, trecv, f)


def track_ridge(x, sr, fmin, fmax, nperseg=None, hop_s=0.25, max_slope_hz_s=8.0):
    nperseg = nperseg or int(sr * 2)  # 0.5 Hz bins
    hop = max(1, int(sr * hop_s))
    f, t, S = signal.spectrogram(
        x,
        sr,
        nperseg=nperseg,
        noverlap=nperseg - hop,
        window="hann",
        detrend="constant",
    )
    band = (f >= fmin) & (f <= fmax)
    fb, Sb = f[band], 10 * np.log10(S[band] + 1e-20)
    # global Viterbi ridge: maximise sum(normalised power) - penalty*|jump| (bins/frame)
    Z = (Sb - np.median(Sb, axis=0)) / (np.std(Sb, axis=0) + 1e-9)
    nb, nt = Z.shape
    # steepest allowed ridge slope; near-CPA passes at a few hundred metres exceed 2 Hz/s
    maxjump = max(2, round(max_slope_hz_s * hop_s / (fb[1] - fb[0])))
    pen = 0.3
    acc = Z[:, 0].copy()
    back = np.zeros((nb, nt), dtype=int)
    for i in range(1, nt):
        best = np.full(nb, -np.inf)
        arg = np.zeros(nb, dtype=int)
        for j in range(-maxjump, maxjump + 1):
            sh = np.roll(acc, j) - pen * abs(j)
            if j > 0:
                sh[:j] = -np.inf
            elif j < 0:
                sh[j:] = -np.inf
            m = sh > best
            best[m] = sh[m]
            arg[m] = (np.arange(nb) - j)[m]
        acc = best + Z[:, i]
        back[:, i] = arg
    idx = np.zeros(nt, dtype=int)
    idx[-1] = np.argmax(acc)
    for i in range(nt - 1, 0, -1):
        idx[i - 1] = back[idx[i], i]
    # parabolic interpolation
    fr = []
    for i, k in enumerate(idx):
        if 0 < k < len(fb) - 1:
            a, b, c = Sb[k - 1, i], Sb[k, i], Sb[k + 1, i]
            p = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        else:
            p = 0
        fr.append(fb[k] + p * (fb[1] - fb[0]))
    return t, np.array(fr), Sb[idx, np.arange(len(idx))], (f, t, S)


def ridge_snr(F, S, fmin, fmax, pw):
    """Median z-score of the ridge power against each frame's in-band distribution."""
    band = (F >= fmin) & (F <= fmax)
    Sb = 10 * np.log10(S[band] + 1e-20)
    return float(np.median((pw - np.median(Sb, axis=0)) / (np.std(Sb, axis=0) + 1e-9)))


def assess(res, min_snr=2.5):
    """Gate a fit: the optimiser always returns *something*, even on pure noise."""
    why = []
    if res["ridge_snr_z"] < min_snr:
        why.append(f"ridge SNR {res['ridge_snr_z']:.1f} < {min_snr} (no coherent tone)")
    if res["rms_resid_hz"] > 0.02 * res["f0_hz"]:
        why.append(
            "residual > 2% of f0 (ridge jumped between tones or not a straight pass)"
        )
    if res["speed_mps"] < 20:
        why.append("fitted speed < 20 m/s (flat ridge, no Doppler drop)")
    if res["stderr"]["cpa_slant_m"] > 0.5 * res["cpa_slant_m"]:
        why.append("CPA distance poorly constrained")
    return not why, why


def fit(t, fr, w=None):
    fhi, flo = np.percentile(fr, 95), np.percentile(fr, 5)
    f0g = np.sqrt(fhi * flo)
    vg = C * (fhi - flo) / (fhi + flo)
    # CPA ~ steepest descent of the (smoothed) ridge; also try the f0 crossing and window quarters
    sm = np.convolve(fr, np.ones(9) / 9, mode="same")
    tcs = {
        float(t[np.argmin(np.abs(fr - f0g))]),
        float(t[4:-4][np.argmin(np.gradient(sm, t)[4:-4])])
        if len(t) > 12
        else float(t.mean()),
        float(np.percentile(t, 33)),
        float(np.percentile(t, 66)),
    }
    best = None
    for dg, tcg in [(dg, tc) for dg in (300, 800, 1500, 3000) for tc in sorted(tcs)]:
        try:
            p, cov = optimize.curve_fit(
                doppler_model,
                t,
                fr,
                p0=[f0g, max(vg, 10), dg, tcg],
                bounds=([1, 1, 20, t.min() - 120], [5000, 340, 20000, t.max() + 120]),
                sigma=w,
                maxfev=20000,
            )
            res = np.sqrt(np.mean((doppler_model(t, *p) - fr) ** 2))
            if best is None or res < best[2]:
                best = (p, cov, res)
        except (
            RuntimeError,
            ValueError,
        ) as exc:  # fit did not converge from this start
            print(f"fit start d={dg} m skipped: {exc}", file=sys.stderr)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--fmin", type=float, default=50)
    ap.add_argument("--fmax", type=float, default=400)
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--sr", type=int, default=4000)
    ap.add_argument("--max-slope", type=float, default=8.0, help="max ridge slope Hz/s")
    ap.add_argument(
        "--min-snr",
        type=float,
        default=2.5,
        help="min median ridge SNR (z) to report a detection",
    )
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.input)[0] + "_doppler"
    x, sr = load_audio(a.input, a.sr)
    off = 0.0
    if a.t0 is not None or a.t1 is not None:
        i0 = int((a.t0 or 0) * sr)
        i1 = int(a.t1 * sr) if a.t1 else len(x)
        x, off = x[i0:i1], (a.t0 or 0)
    t, fr, pw, (F, T, S) = track_ridge(
        x, sr, a.fmin, a.fmax, max_slope_hz_s=a.max_slope
    )
    t = t + off
    snr = ridge_snr(F, S, a.fmin, a.fmax, pw)
    np.savetxt(
        out + ".csv",
        np.c_[t, fr, pw],
        delimiter=",",
        header="time_s,freq_hz,power_db",
        comments="",
    )
    keep = pw >= np.percentile(pw, 25)
    b = fit(t[keep], fr[keep])
    res = {}
    if b:
        p, cov, rms = b
        e = np.sqrt(np.diag(cov))
        res = {
            "f0_hz": p[0],
            "speed_mps": p[1],
            "cpa_slant_m": p[2],
            "cpa_receive_time_s": p[3],
            "stderr": {
                "f0_hz": e[0],
                "speed_mps": e[1],
                "cpa_slant_m": e[2],
                "cpa_receive_time_s": e[3],
            },
            "rms_resid_hz": rms,
            "speed_of_sound": C,
            "ridge_snr_z": snr,
        }
        res["detected"], res["rejected_because"] = assess(res, a.min_snr)
    else:
        res = {
            "detected": False,
            "rejected_because": ["fit did not converge"],
            "ridge_snr_z": snr,
        }
    with open(out + ".json", "w") as fh:
        json.dump(res, fh, indent=2)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        band = (F >= a.fmin) & (F <= a.fmax)
        plt.figure(figsize=(11, 5))
        plt.pcolormesh(
            T + off,
            F[band],
            10 * np.log10(S[band] + 1e-20),
            shading="auto",
            cmap="magma",
        )
        plt.plot(t, fr, "c.", ms=2, label="ridge")
        if b:
            plt.plot(t, doppler_model(t, *b[0]), "lime", lw=1, label="fit")
        plt.xlabel("time (s)")
        plt.ylabel("Hz")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out + ".png", dpi=110)
    except ImportError:
        print("matplotlib missing; skipped PNG", file=sys.stderr)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
