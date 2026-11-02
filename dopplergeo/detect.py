"""Automatic detection of aircraft flyover tones in a recording.

Pipeline: STFT -> per-frame spectral whitening (running median across
frequency, robust z-score) -> harmonic-sum over candidate f0 -> CA-CFAR mask ->
multi-ridge Viterbi with an "off" state (births and deaths) -> per-harmonic
refinement -> joint harmonic-family fit (all lines share v, CPA distance and
time; line k sits at k*f0) -> F-test of the Doppler model against a constant
tone. No hand-picked band or window is needed.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage, optimize, signal
from scipy import stats as sstats

from .physics import C_DEFAULT, straight_doppler

SR = 4000


def load_audio(path: str, sr: int = SR) -> tuple[np.ndarray, int]:
    """Decode any audio/video file to mono float at sr using ffmpeg (no shell)."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        tmp = os.path.join(d, "a.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(path),
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
    return np.asarray(x, float), int(sr)


@dataclass
class Event:
    t_start: float
    t_end: float
    t_cpa: float
    f0_hz: float
    speed_mps: float
    cpa_slant_m: float
    rms_resid_hz: float
    snr_z: float
    harmonics: list[float]
    f_test: float
    p_value: float
    straight_fit_ok: bool
    likely_ground_vehicle: bool = False
    times: list[float] = field(repr=False, default_factory=list)
    ratio: list[float] = field(
        repr=False, default_factory=list
    )  # observed f / f0 (Doppler ratio)
    sigma_ratio: float = 0.0

    def to_dict(self, series: bool = False) -> dict:
        d = asdict(self)
        if not series:
            d.pop("times")
            d.pop("ratio")
        return d


def whitened_spectrogram(x, sr=SR, win_s=2.0, hop_s=0.25, fmax=1000.0):
    nper = int(sr * win_s)
    hop = max(1, int(sr * hop_s))
    f, t, S = signal.spectrogram(
        x, sr, nperseg=nper, noverlap=nper - hop, window="hann", detrend="constant"
    )
    keep = (f >= 20) & (f <= fmax)
    f, L = f[keep], 10 * np.log10(S[keep] + 1e-20)
    df = f[1] - f[0]
    # remove the smooth spectral envelope (wind, roar, traffic) with a running median across frequency
    k = max(3, int(round(12.0 / df)) | 1)
    base = ndimage.median_filter(L, size=(k, 1), mode="nearest")
    W = L - base
    mad = np.median(np.abs(W - np.median(W, axis=0)), axis=0) * 1.4826 + 1e-6
    Z = (W - np.median(W, axis=0)) / mad
    return f, t, Z, L


def harmonic_sum(f, Z, f0_grid, kmax=6, fmax=None):
    """HS(f0,t) = sum_k max(Z near k*f0) / sqrt(K). Noise stays ~N(0,1)-scaled."""
    fmax = fmax or f[-1]
    df = f[1] - f[0]
    Zc = np.clip(Z, -3, 12)
    # tolerance grows with frequency: +/- ~1% of k*f0 (Doppler slope smears high harmonics)
    widths = (3, 5, 9, 15, 25)
    Zms = {w: ndimage.maximum_filter1d(Zc, size=w, axis=0) for w in widths}
    HS = np.zeros((len(f0_grid), Z.shape[1]))
    K = np.zeros(len(f0_grid))
    for k in range(1, kmax + 1):
        fk = k * f0_grid
        ok = fk <= fmax
        idx = np.clip(np.round((fk - f[0]) / df).astype(int), 0, len(f) - 1)
        want = 2 * np.round(0.01 * fk / df) + 1
        for w in widths:
            sel = ok & (np.searchsorted(widths, want.clip(3, 25)) == widths.index(w))
            HS[sel] += Zms[w][idx[sel]]
        K[ok] += 1
    HS /= np.sqrt(np.maximum(K, 1))[:, None]
    return HS, K


def cfar(HS, guard=2, train=12, scale=3.5):
    """Cell-averaging CFAR along f0 for each frame; returns the CFAR excess map."""
    n = HS.shape[0]
    kernel = np.ones(2 * (guard + train) + 1)
    kernel[train : train + 2 * guard + 1] = 0
    kernel /= kernel.sum()
    mean = ndimage.convolve1d(HS, kernel, axis=0, mode="nearest")
    sq = ndimage.convolve1d(HS**2, kernel, axis=0, mode="nearest")
    sd = np.sqrt(np.maximum(sq - mean**2, 1e-6))
    del n
    return (HS - mean) / sd - scale


def viterbi_segments(score, max_jump, birth=6.0, death=6.0, jump_pen=0.1):
    """Viterbi over states {off, f0 bins}. score>0 rewards being on a cell.

    Returns a list of (frame_start, frame_end, bins[]) segments.
    """
    nb, nt = score.shape
    NEG = -1e18
    on = np.full(nb, NEG)
    off = 0.0
    back_on = np.zeros((nb, nt), dtype=np.int32)  # -1 means came from off
    back_off = np.zeros(nt, dtype=np.int32)  # -1 stay off, else came from bin
    for i in range(nt):
        best = np.full(nb, NEG)
        arg = np.full(nb, -1, dtype=np.int32)
        for j in range(-max_jump, max_jump + 1):
            sh = np.roll(on, j) - jump_pen * abs(j)
            if j > 0:
                sh[:j] = NEG
            elif j < 0:
                sh[j:] = NEG
            m = sh > best
            best[m] = sh[m]
            arg[m] = (np.arange(nb) - j)[m]
        from_off = off - birth
        m = from_off > best
        best[m] = from_off
        arg[m] = -1
        k = int(np.argmax(on))
        if on[k] - death > off:
            new_off, back_off[i] = on[k] - death, k
        else:
            new_off, back_off[i] = off, -1
        on = best + score[:, i]
        back_on[:, i] = arg
        off = new_off
    # backtrack
    segs, cur = [], []
    k = int(np.argmax(on))
    state = k if on[k] > off else -1
    for i in range(nt - 1, -1, -1):
        if state >= 0:
            cur.append((i, state))
            prev = back_on[state, i]
            if prev < 0:
                segs.append(cur[::-1])
                cur = []
            state = prev
        else:
            prev = back_off[i]
            state = prev  # -1 stays off, else we were on that bin at frame i-1
    if cur:
        segs.append(cur[::-1])
    out = []
    for s in segs[::-1]:
        fr = np.array([a for a, _ in s])
        bins = np.array([b for _, b in s])
        out.append((int(fr[0]), int(fr[-1]), bins))
    return out


def _refine_lines(f, t, Z, f0_track, frames, kmax, fmax):
    """Measure each harmonic k*f0(t) on the whitened map; return per-k (t, f, z)."""
    df = f[1] - f[0]
    lines = {}
    for k in range(1, kmax + 1):
        tt, ff, zz = [], [], []
        for fi, f0 in zip(frames, f0_track):
            fk = k * f0
            if fk > fmax:
                break
            c = int(round((fk - f[0]) / df))
            w = max(2, int(round(0.012 * fk / df)))
            lo, hi = max(1, c - w), min(len(f) - 2, c + w)
            if hi <= lo:
                continue
            j = lo + int(np.argmax(Z[lo : hi + 1, fi]))
            a, b, cc = Z[j - 1, fi], Z[j, fi], Z[j + 1, fi]
            p = 0.5 * (a - cc) / (a - 2 * b + cc + 1e-12)
            tt.append(t[fi])
            ff.append(f[j] + np.clip(p, -0.5, 0.5) * df)
            zz.append(b)
        if len(tt) > 8:
            lines[k] = (np.array(tt), np.array(ff), np.array(zz))
    return lines


def _fix_subharmonic(lines):
    """Relabel a harmonic family picked at a sub-harmonic of the true f0.

    The harmonic sum can lock onto f0/2 (or f0/3) when only every 2nd (3rd) line
    of the candidate family has support, e.g. lines {2, 4, 6}: the true family is
    {1, 2, 3} at twice the frequency. Prefer the f0 whose multiples carry the
    support: divide the line indices by their gcd.
    """
    if not lines:
        return lines
    g = 0
    for k in lines:
        g = math.gcd(g, int(k))
    if g <= 1:
        return lines
    return {int(k) // g: v for k, v in lines.items()}


def _family_fit(lines, c):
    """Fit f_k(t) = k * D(t; f0, v, d, tc). Returns params, cov, rms (Hz at k=1)."""
    T = np.concatenate([v[0] for v in lines.values()])
    K = np.concatenate([np.full(len(v[0]), k) for k, v in lines.items()])
    F = np.concatenate([v[1] for v in lines.values()]) / K  # in f0 units
    fhi, flo = np.percentile(F, 95), np.percentile(F, 5)
    f0g = np.sqrt(fhi * flo)
    vg = max(c * (fhi - flo) / (fhi + flo), 10.0)
    sm = ndimage.uniform_filter1d(F[np.argsort(T)], 9)
    Ts = np.sort(T)
    tcs = {float(np.percentile(T, q)) for q in (30, 50, 70)}
    if len(Ts) > 12:
        tcs.add(
            float(
                Ts[4:-4][
                    np.argmin(np.gradient(sm, Ts + 1e-9 * np.arange(len(Ts)))[4:-4])
                ]
            )
        )
    best = None
    for dg in (200, 600, 1500, 3500):
        for tc in sorted(tcs):
            try:
                p, cov = optimize.curve_fit(
                    lambda tt, f0, v, d, tcc: straight_doppler(tt, f0, v, d, tcc, c),
                    T,
                    F,
                    p0=[f0g, vg, dg, tc],
                    bounds=(
                        [5, 1, 20, T.min() - 120],
                        [3000, 340, 20000, T.max() + 120],
                    ),
                    maxfev=4000,
                )
            except (RuntimeError, ValueError):
                continue
            rms = float(np.sqrt(np.mean((straight_doppler(T, *p, c) - F) ** 2)))
            if best is None or rms < best[2]:
                best = (p, cov, rms)
    return best, T, F


def _merge_same_pass(events, dt=10.0):
    """Several tone families (engine, propeller, aliases) of one pass give one event.

    Events that overlap in time with closest approaches within dt seconds are
    treated as one pass; keep the best supported fit (long and low residual).
    """

    def quality(e):
        return (e.t_end - e.t_start) / max(e.rms_resid_hz / e.f0_hz, 0.002)

    def same_curve(k, e):
        # a short tone inside a longer one whose Doppler ratio follows it is the same aircraft
        ov = min(k.t_end, e.t_end) - max(k.t_start, e.t_start)
        if ov < 0.8 * (e.t_end - e.t_start):
            return False
        te, re_ = np.array(e.times), np.array(e.ratio)
        rk = np.interp(te, np.array(k.times), np.array(k.ratio))
        a, b = re_ - np.median(re_), rk - np.median(rk)
        return float(np.median(np.abs(a - b))) < 0.01

    kept: list[Event] = []
    for e in sorted(events, key=quality, reverse=True):
        if any(
            (
                abs(k.t_cpa - e.t_cpa) < dt
                and min(k.t_end, e.t_end) - max(k.t_start, e.t_start) > 0
            )
            or same_curve(k, e)
            for k in kept
        ):
            continue
        kept.append(e)
    return kept


def _accept(cd, c, hop_s, win_s, p_max):
    fitres, T, F = _family_fit(cd["lines"], c)
    if fitres is None:
        return None, "fit did not converge"
    p, _cov, rms = fitres
    rss1 = float(np.sum((straight_doppler(T, *p, c) - F) ** 2))
    rss0 = float(np.sum((F - np.mean(F)) ** 2))
    n_eff = max(8.0, len(T) * hop_s / win_s)
    F_stat = ((rss0 - rss1) / 3.0) / max(rss1 / max(n_eff - 4, 1), 1e-12)
    pval = float(sstats.f.sf(F_stat, 3, max(n_eff - 4, 1)))
    drop = (np.percentile(F, 95) - np.percentile(F, 5)) / np.median(F)
    dur = cd["t_end"] - cd["t_start"]
    why = []
    if pval > p_max:
        why.append(f"Doppler model not better than a constant tone (p={pval:.1e})")
    if drop < 0.01:
        why.append("frequency drop < 1%")
    if not (15 <= p[1] <= 330):
        why.append(f"implausible speed {p[1]:.0f} m/s")
    if abs(p[2]) < 60:
        why.append(
            f"degenerate CPA distance {abs(p[2]):.0f} m (step-like fit, not a resolvable pass)"
        )
    if not (cd["t_start"] - 5 - 0.1 * dur <= p[3] <= cd["t_end"] + 5 + 0.1 * dur):
        why.append(
            "closest approach outside the tone (one-sided segment; cannot constrain)"
        )
    if why:
        return None, "; ".join(why)
    order = np.argsort(T)
    return (
        Event(
            t_start=cd["t_start"],
            t_end=cd["t_end"],
            t_cpa=float(p[3]),
            f0_hz=float(p[0]),
            speed_mps=float(p[1]),
            cpa_slant_m=float(p[2]),
            rms_resid_hz=rms,
            snr_z=cd["snr"],
            harmonics=sorted(round(float(k), 3) for k in cd["lines"]),
            f_test=float(F_stat),
            p_value=pval,
            straight_fit_ok=bool(rms < 0.01 * p[0]),
            # cars and trucks also make Doppler tones; aircraft seldom pass this slow
            likely_ground_vehicle=bool(p[1] < 35.0),
            times=[float(v) for v in T[order]],
            ratio=[float(v) for v in (F[order] / p[0])],
            sigma_ratio=float(max(rms, 0.3) / p[0]),
        ),
        "",
    )


_RATIOS = (1.0, 2.0, 0.5, 1.5, 2 / 3, 3.0, 1 / 3, 4 / 3, 0.75, 5 / 4, 0.8, 5 / 3, 0.6)


def _link_segments(cands, c, max_gap=8.0):
    """Join a segment with a later one across a short gap (dropout at CPA).

    The later segment's tone may be a different line of the same family, so try
    small rational ratios and keep the join that fits one straight pass best.
    """
    changed = True
    while changed:
        changed = False
        cands.sort(key=lambda d: d["t_start"])
        for i in range(len(cands)):
            for j in range(len(cands)):
                A, B = cands[i], cands[j]
                if i == j or not (0 <= B["t_start"] - A["t_end"] <= max_gap):
                    continue
                fa = np.median(A["lines"][min(A["lines"])][1][-8:]) / min(A["lines"])
                fb = np.median(B["lines"][min(B["lines"])][1][:8]) / min(B["lines"])
                best = None
                for q in _RATIOS:
                    # after the gap the tone must be lower (receding), by < 60%
                    if not (0.4 < (fb / q) / fa < 1.0):
                        continue
                    lines = {k: v for k, v in A["lines"].items()}
                    for k, (tt, ff, zz) in B["lines"].items():
                        key = k / q
                        if key in lines:
                            t0, f0_, z0 = lines[key]
                            lines[key] = (np.r_[t0, tt], np.r_[f0_, ff], np.r_[z0, zz])
                        else:
                            lines[key] = (tt, ff, zz)
                    fr, _, _ = _family_fit(lines, c)
                    if fr is None:
                        continue
                    if best is None or fr[2] / fr[0][0] < best[0]:
                        best = (fr[2] / fr[0][0], lines)
                if best and best[0] < 0.01:
                    cands[i] = {
                        "t_start": A["t_start"],
                        "t_end": B["t_end"],
                        "snr": max(A["snr"], B["snr"]),
                        "lines": best[1],
                    }
                    del cands[j]
                    changed = True
                    break
            if changed:
                break
    return cands


def detect_events(
    x,
    sr=SR,
    f0_min=40.0,
    f0_max=400.0,
    fmax=1000.0,
    kmax=6,
    max_slope_hz_s=10.0,
    min_duration_s=4.0,
    min_snr=3.0,
    p_max=1e-4,
    max_events=8,
    c=C_DEFAULT,
):
    """Find flyover tone events in a mono signal. Returns (events, diagnostics)."""
    hop_s, win_s = 0.25, 1.0
    f, t, Z, L = whitened_spectrogram(x, sr, win_s, hop_s, fmax)
    f0_grid = np.arange(f0_min, f0_max, 0.5)
    HS, _ = harmonic_sum(f, Z, f0_grid, kmax, fmax)
    # tones persist: a short running mean along time (0.75 s) lifts weak lines above noise
    HS = ndimage.uniform_filter1d(HS, size=3, axis=1) * np.sqrt(3)
    ex = cfar(HS)
    # reward cells above the CFAR threshold, mildly penalise the rest
    # floor near zero so a track survives a few faint seconds through the steep CPA part
    score = np.clip(ex, -0.35, 6.0)
    max_jump = max(2, round(max_slope_hz_s * hop_s / 0.5))  # f0 grid step 0.5 Hz
    rejected = []
    cands = []  # dicts: t_start, t_end, snr, lines{k: (t, f, z)}
    mask = np.ones_like(score)
    for _ in range(max_events * 3):
        segs = viterbi_segments(score * mask + (mask - 1) * 5, max_jump)
        segs = [sg for sg in segs if (sg[1] - sg[0]) * hop_s >= min_duration_s]
        if not segs:
            break
        strengths = [
            float(np.mean(HS[b_, np.arange(s0_, s1_ + 1)])) for s0_, s1_, b_ in segs
        ]
        s0, s1, bins = segs[int(np.argmax(strengths))]
        frames = np.arange(s0, s1 + 1)
        for mult in (0.5, 1.0, 2.0, 1.5, 3.0, 2 / 3, 1 / 3):
            mb = np.round((f0_grid[bins] * mult - f0_min) / 0.5).astype(int)
            for dlt in range(-6, 7):
                ok = (mb + dlt >= 0) & (mb + dlt < len(f0_grid))
                mask[(mb + dlt)[ok], frames[ok]] = 0
        snr = float(np.mean(HS[bins, frames]))
        if snr < min_snr:
            rejected.append(
                {
                    "t_start": float(t[s0]),
                    "t_end": float(t[s1]),
                    "why": f"harmonic-sum SNR {snr:.1f}",
                }
            )
            continue
        f0_track = ndimage.median_filter(f0_grid[bins], size=5)
        lines = _refine_lines(f, t, Z, f0_track, frames, kmax, fmax)
        lines = {k: v for k, v in lines.items() if np.median(v[2]) > 2.0}
        lines = _fix_subharmonic(lines)
        if not lines:
            rejected.append(
                {
                    "t_start": float(t[s0]),
                    "t_end": float(t[s1]),
                    "why": "no harmonic line above z=2",
                }
            )
            continue
        cands.append(
            {"t_start": float(t[s0]), "t_end": float(t[s1]), "snr": snr, "lines": lines}
        )
    cands = _link_segments(cands, c)
    events = []
    for cd in cands:
        ev, why = _accept(cd, c, hop_s, win_s, p_max)
        if ev is None:
            rejected.append(
                {"t_start": cd["t_start"], "t_end": cd["t_end"], "why": why}
            )
        else:
            events.append(ev)
    events = _merge_same_pass(events)
    events.sort(key=lambda e: e.t_cpa)
    diag = {
        "frames": len(t),
        "hop_s": hop_s,
        "rejected": rejected,
        "spectrogram": (f, t, L),
        "harmonic_sum": (f0_grid, HS),
    }
    return events, diag


def plot_detection(diag, events, path, title="", fmax=600):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, t, L = diag["spectrogram"]
    m = f <= fmax
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.pcolormesh(
        t,
        f[m],
        L[m],
        shading="auto",
        cmap="magma",
        vmin=np.percentile(L[m], 20),
        vmax=np.percentile(L[m], 99.7),
    )
    for i, e in enumerate(events):
        tt = np.linspace(e.t_start, e.t_end, 300)
        for k in e.harmonics:
            ax.plot(
                tt,
                k * straight_doppler(tt, e.f0_hz, e.speed_mps, e.cpa_slant_m, e.t_cpa),
                "c-",
                lw=0.9,
            )
        ax.text(
            e.t_cpa,
            min(fmax * 0.95, e.f0_hz * max(e.harmonics) * 1.08),
            f"#{i + 1}",
            color="cyan",
            fontsize=10,
        )
    for r in diag["rejected"]:
        ax.axvspan(r["t_start"], r["t_end"], ymin=0, ymax=0.03, color="grey")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz")
    ax.set_title(title or f"{len(events)} event(s) detected", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
