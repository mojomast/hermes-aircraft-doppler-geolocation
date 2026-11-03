"""Broadband (jet) pass detection from the ground-reflection multipath delay.

A microphone at height h above reflecting ground hears the direct sound and a
ground-reflected copy delayed by tau(t) ~= 2 h sin(elevation) / c (Lloyd's
mirror). The log spectrum is periodic in frequency with period 1/tau, so the
band-limited power cepstrum peaks at quefrency tau (Lo & Ferguson 2003, IEEE
TAES 39(1); cepstrum-track identification, IDC 2002). For a straight level
pass the delay follows

    tau(t) = tau_max / sqrt(1 + ((t - t_cpa) / T)^2),  T = S0 / v,

with S0 the slant range at closest approach and v the ground speed. The
microphone height is unknown, so the scorer uses the normalised shape
tau(t) / tau_max against each ADS-B candidate's predicted sin(elevation).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist

from .detect import Event


def cepstrogram(
    x,
    sr,
    win_s=0.5,
    hop_s=0.25,
    fband=(150.0, 2000.0),
    qmin=1.5e-3,
    qmax=0.06,
    dq=5e-5,
):
    """Band-limited power cepstrum per frame, z-scored per frame (CFAR-style).

    Returns (t_centres, q_grid, Z, level_db).
    """
    x = np.asarray(x, float)
    n = int(win_s * sr)
    hop = int(hop_s * sr)
    if len(x) < n:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0)), np.zeros(0)
    idx = np.arange(0, len(x) - n + 1, hop)
    win = np.hanning(n)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    band = (f >= fband[0]) & (f <= min(fband[1], sr / 2 * 0.95))
    fb = f[band]
    q = np.arange(qmin, qmax, dq)
    basis = np.cos(2 * np.pi * np.outer(fb, q))  # (nf, nq)
    # remove slow spectral trend (source spectrum) with a low-order polynomial
    u = (fb - fb.mean()) / (np.ptp(fb) / 2)
    V = np.vander(u, 5)
    # least-squares polynomial residual L - L V pinv(V)^T, applied through the
    # (nf, 5) factors instead of a dense (nf, nf) projection matrix
    pinv_t = np.linalg.pinv(V).T
    VT = np.ascontiguousarray(V.T)
    Z = np.empty((len(idx), len(q)), np.float32)
    lev = np.empty(len(idx))
    for i0 in range(0, len(idx), 256):
        blk = idx[i0 : i0 + 256]
        fr = np.stack([x[k : k + n] for k in blk]) * win
        P = np.abs(np.fft.rfft(fr, axis=1))[:, band] ** 2 + 1e-12
        L = np.log(P)
        lev[i0 : i0 + len(blk)] = 10 * np.log10(P.mean(1))
        L -= (L @ pinv_t) @ VT
        C = L @ basis / len(fb)
        med = np.median(C, axis=1, keepdims=True)
        mad = np.median(np.abs(C - med), axis=1, keepdims=True) * 1.4826 + 1e-12
        Z[i0 : i0 + len(blk)] = (C - med) / mad
    t = (idx + n / 2) / sr
    return t, q, Z, lev


def track_segments(Z, max_jump=20, z_thr=4.0, birth=8.0, min_frames=16):
    """Viterbi track of the cepstral peak with an 'off' state.

    Emission score z - z_thr for on-states; bounded jump per hop. Returns a list of
    (frame_indices, q_indices) runs where the path is 'on'.
    """
    nt, nq = Z.shape
    if nt == 0:
        return []
    S = np.clip(Z, -2, 12) - z_thr
    v_on = S[0].copy() - birth
    v_off = 0.0
    bp = np.zeros((nt, nq), np.int32)  # predecessor q (>=0) or -1 = from off
    off_from_on = np.full(nt, -1, np.int32)
    for t in range(1, nt):
        best = np.full(nq, -np.inf)
        arg = np.zeros(nq, np.int32)
        for d in range(-max_jump, max_jump + 1):
            src = np.roll(v_on, d)
            if d > 0:
                src[:d] = -np.inf
            elif d < 0:
                src[d:] = -np.inf
            upd = src > best
            best[upd] = src[upd]
            arg[upd] = (np.arange(nq) - d)[upd]
        from_off = v_off - birth
        use_off = from_off > best
        best[use_off] = from_off
        arg[use_off] = -1
        k = int(np.argmax(v_on))
        if v_on[k] > v_off:
            new_off, off_from_on[t] = float(v_on[k]), k
        else:
            new_off, off_from_on[t] = v_off, -1
        v_on = best + S[t]
        bp[t] = arg
        v_off = new_off
    # backtrack
    state = int(np.argmax(v_on)) if v_on.max() > v_off else -1
    path = np.full(nt, -1, np.int32)
    for t in range(nt - 1, -1, -1):
        path[t] = state
        if t == 0:
            break
        if state >= 0:
            state = int(bp[t, state])
        else:
            state = int(off_from_on[t])
    runs, cur = [], []
    for t in range(nt):
        if path[t] >= 0:
            cur.append(t)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [(np.array(r), path[np.array(r)]) for r in runs if len(r) >= min_frames]


def delay_model(t, a, tc, T):
    return a / np.sqrt(1.0 + ((t - tc) / T) ** 2)


def fit_pass(t, tau):
    """Least-squares fit of the straight-pass multipath-delay curve."""
    k = int(np.argmax(tau))
    span = t[-1] - t[0]
    p0 = (tau[k], t[k], max(span / 4, 2.0))
    lo = (tau.min() * 0.5, t[0] - 0.5 * span, 1.0)
    hi = (tau.max() * 2.0, t[-1] + 0.5 * span, 300.0)
    p, _ = curve_fit(delay_model, t, tau, p0=p0, bounds=(lo, hi), maxfev=4000)
    return p


def detect_broadband(
    x,
    sr,
    hop_s=0.25,
    z_thr=4.0,
    min_z=5.0,
    min_span_s=4.0,
    p_max=1e-4,
    min_contrast=1.15,
    max_rel_rms=0.08,
    T_bounds=(0.8, 150.0),  # T = S0/v: ~1 s for a 70 m/s jet 70 m overhead
    **cep_kw,
):
    """Broadband pass events with gating. Returns (events, diag)."""
    t, q, Z, _lev = cepstrogram(x, sr, hop_s=hop_s, **cep_kw)
    diag = {"t": t, "q": q, "Z": Z, "rejected": [], "tracks": []}
    if len(t) == 0:
        return [], diag
    dq = q[1] - q[0]
    runs = track_segments(Z, max_jump=max(1, int(1.5e-3 / dq)), z_thr=z_thr)
    events = []
    for fr, qi in runs:
        tt = t[fr]
        # parabolic quefrency refinement
        zc = Z[fr, qi]
        zl = Z[fr, np.clip(qi - 1, 0, len(q) - 1)]
        zr = Z[fr, np.clip(qi + 1, 0, len(q) - 1)]
        den = zl - 2 * zc + zr
        off = np.where(np.abs(den) > 1e-9, 0.5 * (zl - zr) / den, 0.0)
        tau = q[qi] + np.clip(off, -0.5, 0.5) * dq
        diag["tracks"].append((tt, tau))
        span = tt[-1] - tt[0]
        why = []
        if span < min_span_s:
            why.append(f"span {span:.1f} s")
        if np.median(zc) < min_z:
            why.append(f"cepstral z {np.median(zc):.1f} < {min_z}")
        try:
            a, tc, T = fit_pass(tt, tau)
        except (RuntimeError, ValueError):
            diag["rejected"].append(
                {"t_start": tt[0], "t_end": tt[-1], "why": "fit failed"}
            )
            continue
        m = delay_model(tt, a, tc, T)
        rss1 = float(((tau - m) ** 2).sum())
        rss0 = float(((tau - tau.mean()) ** 2).sum())
        n = len(tt)
        F = ((rss0 - rss1) / 2) / max(rss1 / max(n - 3, 1), 1e-30)
        p = float(f_dist.sf(F, 2, max(n - 3, 1)))
        rel = float(np.sqrt(rss1 / n) / a)
        contrast = float(m.max() / max(m.min(), 1e-12))
        if p > p_max:
            why.append(f"not better than constant delay (p={p:.1e})")
        if contrast < min_contrast:
            why.append(f"delay contrast {contrast:.2f}")
        if not (tt[0] <= tc <= tt[-1]):
            why.append("CPA outside track")
        if not (T_bounds[0] <= T <= T_bounds[1]):
            why.append(f"T={T:.1f} s implausible")
        if rel > max_rel_rms:
            why.append(f"rel rms {rel:.3f}")
        if why:
            diag["rejected"].append(
                {"t_start": tt[0], "t_end": tt[-1], "why": "; ".join(why)}
            )
            continue
        events.append(
            Event(
                t_start=float(tt[0]),
                t_end=float(tt[-1]),
                t_cpa=float(tc),
                f0_hz=0.0,
                speed_mps=float("nan"),
                cpa_slant_m=float("nan"),
                rms_resid_hz=rel,
                snr_z=float(np.median(zc)),
                harmonics=[],
                f_test=float(F),
                p_value=p,
                straight_fit_ok=True,
                times=[float(v) for v in tt],
                ratio=[float(v) for v in tau / a],
                sigma_ratio=rel,
                kind="broadband",
                tau_max_s=float(a),
                T_s=float(T),
            )
        )
    return events, diag
