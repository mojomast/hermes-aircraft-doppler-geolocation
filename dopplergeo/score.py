"""Score ground locations by predicting each event's Doppler curve from real ADS-B tracks.

For a ground point g and aircraft track A(tau) the received frequency ratio at
time t is D = c / (c - v_r(tau_e)), where tau_e solves tau_e + |A(tau_e) - g|/c = t
(emission time). Turns, climbs and speed changes are in the track, so nothing
assumes a straight line. Each event's observed ratio series is compared with D
after fitting one scale factor (the unknown rest frequency) and a small clock
offset; the residual gives a Gaussian log-likelihood. Events combine jointly with
one shared recording start time T0, and a silence penalty is charged for tonal
aircraft that should have been heard (close pass during the clip) but match no
event. Search is coarse-to-fine over a local ENU grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .physics import R_E
from .typedb import TONAL_CLASSES


@dataclass
class Plane:
    hex: str
    cls: str
    tau: np.ndarray  # 1 s samples, unix time
    pos: np.ndarray  # (n, 3) ENU metres (z = altitude MSL)
    label: str = ""


@dataclass
class Options:
    r_max_m: float = 8000.0  # beyond this a tonal pass is not used
    r_silence_m: float = (
        2500.0  # a tonal pass this close during the clip should be heard
    )
    silence_penalty: float = 6.0
    miss_penalty: float = (
        1.0  # x (cost of an all-outlier fit) for an event with no consistent pass
    )
    dt_tol_s: float = 6.0  # clock / CPA-time slack per event
    dt_step_s: float = 0.5
    sigma_floor: float = 0.003  # Doppler-ratio noise floor (0.3% ~ 0.4 Hz at 130 Hz)
    hop_over_win: float = 0.25  # detector hop / window: effective independent samples
    include_jets: bool = False
    scale_bounds: tuple = (0.85, 1.15)


@dataclass
class PointResult:
    logl: np.ndarray
    t0: np.ndarray
    matches: list = field(default_factory=list)


def to_enu(lat, lon, lat0, lon0):
    x = np.radians(np.asarray(lon) - lon0) * R_E * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * R_E
    return x, y


def from_enu(x, y, lat0, lon0):
    lat = lat0 + np.degrees(np.asarray(y) / R_E)
    lon = lon0 + np.degrees(np.asarray(x) / (R_E * np.cos(np.radians(lat0))))
    return lat, lon


def prepare_planes(
    tracks, lat0, lon0, include_jets=False, max_gap_s=60.0
) -> list[Plane]:
    """Resample each track to 1 s ENU samples, splitting at data gaps."""
    out = []
    for a in tracks:
        cls = a.get("class", "unknown")
        if not include_jets and cls not in TONAL_CLASSES:
            continue
        t = np.asarray(a["t"], float)
        if len(t) < 2:
            continue
        o = np.argsort(t)
        t = t[o]
        x, y = to_enu(np.asarray(a["lat"])[o], np.asarray(a["lon"])[o], lat0, lon0)
        z = np.asarray(a["alt_m"], float)[o]
        breaks = np.where(np.diff(t) > max_gap_s)[0]
        for seg in np.split(np.arange(len(t)), breaks + 1):
            if len(seg) < 2 or t[seg[-1]] - t[seg[0]] < 20:
                continue
            tau = np.arange(t[seg[0]], t[seg[-1]], 1.0)
            pos = np.stack(
                [
                    np.interp(tau, t[seg], x[seg]),
                    np.interp(tau, t[seg], y[seg]),
                    np.interp(tau, t[seg], z[seg]),
                ],
                1,
            )
            out.append(
                Plane(
                    a.get("hex", "?"),
                    cls,
                    tau,
                    pos,
                    f"{a.get('type') or '?'} {a.get('reg') or a.get('hex')}",
                )
            )
    return out


def find_passes(planes, gxyz, opts: Options, c: float):
    """For every (point, plane) list local minima of slant range below r_max.

    Returns a list per point of passes: (plane_idx, k_min, r_min, t_recv_cpa).
    """
    passes = [[] for _ in range(len(gxyz))]
    tree = cKDTree(gxyz[:, :2])
    for pi, pl in enumerate(planes):
        # candidate points within r_max horizontally of any track sample
        near = set()
        for idx in tree.query_ball_point(pl.pos[::5, :2], opts.r_max_m):
            near.update(idx)
        if not near:
            continue
        P = np.fromiter(near, int)
        for chunk in np.array_split(P, max(1, len(P) // 2000 + 1)):
            r = np.linalg.norm(pl.pos[None, :, :] - gxyz[chunk, None, :], axis=2)
            dr = np.diff(r, axis=1)
            is_min = np.zeros_like(r, bool)
            is_min[:, 1:-1] = (dr[:, :-1] < 0) & (dr[:, 1:] >= 0)
            is_min[:, 0] = dr[:, 0] >= 0
            is_min[:, -1] = dr[:, -1] < 0
            is_min &= r < opts.r_max_m
            for row, k in zip(*np.nonzero(is_min)):
                rm = r[row, k]
                passes[chunk[row]].append(
                    (pi, int(k), float(rm), float(pl.tau[k] + rm / c))
                )
    return passes


def predicted_ratio(pl: Plane, g, t_abs, c, k_cpa, half_window_s):
    """Doppler ratio heard at g at absolute times t_abs (any shape), emission-time solved."""
    lo = max(0, k_cpa - int(half_window_s) - 120)
    hi = min(len(pl.tau), k_cpa + int(half_window_s) + 120)
    if hi - lo < 3:
        return None
    pos = pl.pos[lo:hi]
    d = pos - g
    r = np.linalg.norm(d, axis=1)
    vel = np.gradient(pos, pl.tau[lo:hi], axis=0)
    vr = -np.sum(d * vel, axis=1) / np.maximum(r, 1.0)
    trecv = pl.tau[lo:hi] + r / c
    D = c / (c - vr)
    ok = (t_abs >= trecv[0]) & (t_abs <= trecv[-1])
    out = np.interp(t_abs, trecv, D)
    out[~ok] = np.nan
    return out


def event_pass_loglik(ev, pl, g, k_cpa, t_cpa_recv, c, opts: Options, t0_center=None):
    """Best log-likelihood of one event against one pass over the clock slack.

    Returns (logl, T0, scale). T0 = recording start (unix) implied by the alignment.
    """
    times = np.asarray(ev.times)
    obs = np.asarray(ev.ratio)
    if t0_center is None:
        t0_center = t_cpa_recv - ev.t_cpa
    shifts = np.arange(-opts.dt_tol_s, opts.dt_tol_s + 1e-9, opts.dt_step_s)
    T0s = t0_center + shifts
    t_abs = T0s[:, None] + times[None, :]
    half = (times.max() - times.min()) / 2 + opts.dt_tol_s + 60
    D = predicted_ratio(pl, g, t_abs, c, k_cpa, half)
    if D is None:
        return -np.inf, None, None
    sigma = max(ev.sigma_ratio * 0.5, opts.sigma_floor)
    valid = ~np.isnan(D)
    frac = valid.mean(axis=1)
    Dz = np.where(valid, D, 0.0)
    oz = np.where(valid, obs[None, :], 0.0)
    # robust scale (unknown rest frequency): median start, then two inlier refits
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.nanmedian(
            np.where(valid, oz / np.where(Dz == 0, np.nan, Dz), np.nan), axis=1
        )
    s = np.nan_to_num(s, nan=1.0)
    for _ in range(2):
        w = valid & (np.abs(oz - s[:, None] * Dz) < 3 * sigma)
        num = (w * oz * Dz).sum(1)
        den = (w * Dz * Dz).sum(1)
        s = np.where(den > 0, num / np.maximum(den, 1e-12), s)
    s = np.clip(s, *opts.scale_bounds)
    res = np.where(valid, oz - s[:, None] * Dz, 0.0)
    res = np.clip(
        res, -4 * sigma, 4 * sigma
    )  # robust: an outlier frame costs at most 8 sigma^2
    rss = (res**2).sum(1)
    n = valid.sum(1)
    # samples outside the modelled span count as full outliers
    rss = rss + (len(times) - n) * (4 * sigma) ** 2
    ll = -0.5 * rss / sigma**2 * opts.hop_over_win
    ll[frac < 0.5] = -np.inf
    j = int(np.argmax(ll))
    return float(ll[j]), float(T0s[j]), float(s[j])


def score_points(events, planes, gxyz, c, opts: Options, t0_window=None, clip_dur=None):
    """Joint log-likelihood per grid point. t0_window=(lo, hi) unix bounds on recording start."""
    passes = find_passes(planes, gxyz, opts, c)
    n = len(gxyz)
    logl = np.full(n, -np.inf)
    t0 = np.full(n, np.nan)
    matches = [None] * n
    clip_dur = clip_dur or (max(e.t_end for e in events) + 5 if events else 0)
    for i in range(n):
        P = passes[i]
        if not P:
            continue
        # per event, candidate (logl, T0, pass)
        per_ev = []
        for ev in events:
            cand = []
            for pi, k, rm, tr in P:
                T0c = tr - ev.t_cpa
                if t0_window and not (
                    t0_window[0] - opts.dt_tol_s <= T0c <= t0_window[1] + opts.dt_tol_s
                ):
                    continue
                ll, T0, sc = event_pass_loglik(ev, planes[pi], gxyz[i], k, tr, c, opts)
                if np.isfinite(ll):
                    cand.append((ll, T0, pi, rm, sc))
            per_ev.append(cand)
        hyps = sorted({round(cnd[1], 1) for cand in per_ev for cnd in cand})
        best = (-np.inf, None, None)
        for T0h in hyps:
            if t0_window and not (
                t0_window[0] - opts.dt_tol_s <= T0h <= t0_window[1] + opts.dt_tol_s
            ):
                continue
            tot, used = 0.0, []
            for e_i, cand in enumerate(per_ev):
                ok = [
                    cnd
                    for cnd in cand
                    if abs(cnd[1] - T0h) <= opts.dt_tol_s
                    and cnd[2] not in [u[1] for u in used]
                ]
                if ok:
                    b = max(ok, key=lambda z: z[0])
                    tot += b[0]
                    used.append((e_i, b[2], b[3], b[4]))
                else:
                    # as bad as a fit where every sample is an outlier (0.5 * 4^2 per effective sample)
                    tot -= (
                        opts.miss_penalty
                        * 8.0
                        * opts.hop_over_win
                        * len(events[e_i].times)
                    )
            # silence: close tonal passes inside the clip that no event explains
            used_pl = {u[1] for u in used}
            for pi, _k, rm, tr in P:
                if pi in used_pl or rm > opts.r_silence_m:
                    continue
                if T0h <= tr <= T0h + clip_dur:
                    tot -= opts.silence_penalty * (1.0 - rm / opts.r_silence_m)
            if tot > best[0]:
                best = (tot, T0h, used)
        logl[i], t0[i] = best[0], (best[1] if best[1] is not None else np.nan)
        if best[2] is not None:
            matches[i] = [
                {
                    "event": int(e_i),
                    "aircraft": planes[pi].label,
                    "hex": planes[pi].hex,
                    "class": planes[pi].cls,
                    "cpa_slant_m": rm,
                    "f0_scale": sc,
                }
                for e_i, pi, rm, sc in best[2]
            ]
    return PointResult(logl, t0, matches)


def grid(cx, cy, half, step):
    xs = np.arange(cx - half, cx + half + 1e-6, step)
    ys = np.arange(cy - half, cy + half + 1e-6, step)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()


def posterior(logl, cell_area_m2):
    ok = np.isfinite(logl)
    p = np.zeros_like(logl)
    if not ok.any():
        return p, None, None
    p[ok] = np.exp(logl[ok] - logl[ok].max())
    p /= p.sum()
    srt = np.sort(p)[::-1]
    cum = np.cumsum(srt)
    a68 = (np.searchsorted(cum, 0.68) + 1) * cell_area_m2
    a95 = (np.searchsorted(cum, 0.95) + 1) * cell_area_m2
    return p, a68, a95


def search(
    events,
    planes,
    lat0,
    lon0,
    radius_m,
    elev_fn,
    c,
    opts: Options,
    t0_window=None,
    clip_dur=None,
    steps=(1000.0, 200.0, 50.0),
    progress=None,
):
    """Coarse-to-fine search. elev_fn(lat, lon) -> metres. Returns dict with levels and best."""
    levels = []
    centers = [(0.0, 0.0, radius_m)]
    for li, step in enumerate(steps):
        xs, ys = [], []
        for cx, cy, half in centers:
            x, y = grid(cx, cy, half, step)
            keep = np.hypot(x, y) <= radius_m + step
            xs.append(x[keep])
            ys.append(y[keep])
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        xy = np.unique(np.round(np.c_[x, y] / step).astype(np.int64), axis=0) * step
        x, y = xy[:, 0].astype(float), xy[:, 1].astype(float)
        lat, lon = from_enu(x, y, lat0, lon0)
        z = elev_fn(lat, lon) + 1.5  # microphone ~1.5 m above ground
        g = np.c_[x, y, z]
        # coarse levels: temper the likelihood so a slightly-off grid point still ranks
        o = Options(
            **{**opts.__dict__, "sigma_floor": opts.sigma_floor * (step / 50.0) ** 0.5}
        )
        res = score_points(events, planes, g, c, o, t0_window, clip_dur)
        if progress:
            progress(li, len(g))
        p, a68, a95 = posterior(res.logl, step * step)
        levels.append(
            {
                "step": step,
                "x": x,
                "y": y,
                "lat": lat,
                "lon": lon,
                "logl": res.logl,
                "t0": res.t0,
                "matches": res.matches,
                "p": p,
                "a68_m2": a68,
                "a95_m2": a95,
            }
        )
        fin = np.isfinite(res.logl)
        if not fin.any():
            break
        order = np.argsort(-np.where(fin, res.logl, -np.inf))
        centers, picked = [], []
        for j in order:
            if not np.isfinite(res.logl[j]):
                break
            if all(np.hypot(x[j] - px, y[j] - py) > 3 * step for px, py in picked):
                picked.append((x[j], y[j]))
                centers.append((x[j], y[j], 3 * step if li + 1 < len(steps) else step))
            if len(picked) >= (5 if li == 0 else 3):
                break
        if li + 1 < len(steps):
            centers = [(cx, cy, 3 * step) for cx, cy, _ in centers]
    last = levels[-1]
    fin = np.isfinite(last["logl"])
    best = None
    if fin.any():
        j = int(np.nanargmax(np.where(fin, last["logl"], -np.inf)))
        best = {
            "lat": float(last["lat"][j]),
            "lon": float(last["lon"][j]),
            "logl": float(last["logl"][j]),
            "t0_unix": float(last["t0"][j]),
            "matches": last["matches"][j],
        }
    return {"levels": levels, "best": best}
