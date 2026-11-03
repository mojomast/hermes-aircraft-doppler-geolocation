# Frozen copy of dopplergeo/score.py at v0.3.0 (1d06ae6): reference for the
# prepared-scoring equivalence tests. Do not edit.
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

from dopplergeo.physics import R_E
from dopplergeo.typedb import TONAL_CLASSES


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
    miss_null: bool = False  # miss cost vs a no-Doppler null (flatter; see notes)
    miss_penalty: float = (
        1.0  # x (cost of an all-outlier fit) for an event with no consistent pass
    )
    dt_tol_s: float = 6.0  # clock / CPA-time slack per event
    dt_step_s: float = 0.5
    # extra posterior temper on top of the Birge ratio (calibrated on real clips so
    # 95% regions contain the truth ~95% of the time; 1.0 = uncalibrated)
    temper_extra: float = 4.0
    sigma_floor: float = 0.003  # Doppler-ratio noise floor (0.3% ~ 0.4 Hz at 130 Hz)
    # Model error (ADS-B position/time error, ~20 s track sampling, wind, altitude
    # reference) adds to the detector noise and is correlated over seconds, so
    # the likelihood uses sigma = hypot(detector, model) and counts one
    # independent sample per corr_s seconds of event, not one per frame.
    sigma_model: float = 0.008
    corr_s: float = 4.0
    n_eff_max: float = 30.0  # no single long event may dominate the joint fit
    hop_over_win: float = 0.25  # cap: never more than this fraction of frames
    include_jets: bool = False
    scale_bounds: tuple = (0.7, 1.3)
    scale_prior_sd: float = (
        0.08  # Gaussian prior on rest-frequency scale (type table error)
    )
    # P(detector picks up a tonal pass at 0 m); decays linearly to 0 at r_silence_m.
    # Silence is scored as a Bernoulli miss, -log(1 - pd), so a recording with
    # commentary/music (low recall) is not pushed away from busy traffic.
    detect_prob: float = 0.15
    marginalize: bool = True  # sum, not max, over candidate aircraft per event
    span_tol: float = 0.05  # prefilter: |log Doppler span(pred/obs)| allowed
    # CPA-distance consistency: the event's own fitted CPA slant vs the pass's
    # slant range at CPA, Gaussian in log-ratio (detector range estimates are
    # loose, d trades off against speed, so the tolerance is a factor ~e).
    cpa_log_sd: float = 1.0
    cpa_floor_m: float = 150.0  # below this both ranges are "overhead"
    cpa_weight: float = 1.0  # 0 disables the term
    # broadband (ground-reflection delay) events: observed tau/tau_max vs predicted
    # sin(elevation) times an unknown scale 1/sin(elev at CPA) (mic height unknown)
    bb_scale_bounds: tuple = (1.0, 60.0)
    bb_sigma_model: float = 0.06  # relative model error of the normalised delay curve
    n_seeds: int = 10  # coarse-level basins refined (flat likelihoods need more)


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
        # bound the (points x samples x 3) temporary to ~100 MB
        per = max(1, int(4e6 // max(len(pl.pos), 1)))
        for chunk in np.array_split(P, max(1, -(-len(P) // per))):
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


def predicted_sin_elev(pl: Plane, g, t_abs, c, k_cpa, half_window_s):
    """sin(elevation) of the aircraft seen from g at receive times t_abs (emission-time solved)."""
    lo = max(0, k_cpa - int(half_window_s) - 120)
    hi = min(len(pl.tau), k_cpa + int(half_window_s) + 120)
    if hi - lo < 3:
        return None
    d = pl.pos[lo:hi] - g
    r = np.linalg.norm(d, axis=1)
    trecv = pl.tau[lo:hi] + r / c
    se = np.clip(d[:, 2] / np.maximum(r, 1.0), 0.0, 1.0)
    ok = (t_abs >= trecv[0]) & (t_abs <= trecv[-1])
    out = np.interp(t_abs, trecv, se)
    out[~ok] = np.nan
    return out


def is_broadband(ev) -> bool:
    return getattr(ev, "kind", "tonal") == "broadband"


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
    bb = is_broadband(ev)
    pred = predicted_sin_elev if bb else predicted_ratio
    bounds = opts.bb_scale_bounds if bb else opts.scale_bounds
    D = pred(pl, g, t_abs, c, k_cpa, half)
    if D is None:
        return -np.inf, None, None
    sigma = event_sigma(ev, opts)
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
    s = np.clip(s, *bounds)
    hit_bound = (s <= bounds[0] + 1e-9) | (s >= bounds[1] - 1e-9)
    res = np.where(valid, oz - s[:, None] * Dz, 0.0)
    res = np.clip(
        res, -4 * sigma, 4 * sigma
    )  # robust: an outlier frame costs at most 8 sigma^2
    rss = (res**2).sum(1)
    n = valid.sum(1)
    # samples outside the modelled span count as full outliers
    rss = rss + (len(times) - n) * (4 * sigma) ** 2
    ll = -0.5 * rss / sigma**2 * event_weight(ev, opts)
    if not bb:  # broadband scale (1/sin elevation at CPA) is log-uniform
        ll -= 0.5 * ((s - 1.0) / opts.scale_prior_sd) ** 2
    ll[frac < 0.5] = -np.inf
    ll[hit_bound] = -np.inf  # scale outside the plausible range: not this aircraft
    j = int(np.argmax(ll))
    return float(ll[j]), float(T0s[j]), float(s[j])


def event_sigma(ev, opts: Options) -> float:
    if is_broadband(ev):
        return float(np.hypot(ev.sigma_ratio, opts.bb_sigma_model))
    return float(
        np.hypot(max(ev.sigma_ratio * 0.5, opts.sigma_floor), opts.sigma_model)
    )


def event_weight(ev, opts: Options) -> float:
    """Effective independent samples per frame (residuals are time-correlated)."""
    n = max(len(ev.times), 1)
    t = np.asarray(ev.times)
    span = float(t.max() - t.min()) if n > 1 else 0.0
    n_eff = min(opts.n_eff_max, max(3.0, span / opts.corr_s))
    return min(opts.hop_over_win, n_eff / n)


def cpa_consistency(ev, r_pass_m, opts: Options) -> float:
    """Log-likelihood term: event CPA-slant estimate vs pass slant at CPA."""
    if opts.cpa_weight <= 0:
        return 0.0
    d_ev = getattr(ev, "cpa_slant_m", None)
    if d_ev is None or not np.isfinite(d_ev) or r_pass_m is None:
        return 0.0
    a = max(abs(float(d_ev)), opts.cpa_floor_m)
    b = max(float(r_pass_m), opts.cpa_floor_m)
    z = np.log(a / b) / opts.cpa_log_sd
    return float(-0.5 * opts.cpa_weight * z * z)


def miss_cost(ev, opts: Options) -> float:
    """Log-likelihood cost of an event no pass explains.

    Default: the all-outlier cost over the event's effective samples. With
    miss_null=True the alternative is a no-Doppler null (constant tone at the same
    sigma), which down-weights noisy/flat events; on the Carlsbad real-world clip
    this made the surface too flat (7.4 km vs 1.0 km error), so it is off.
    """
    w = event_weight(ev, opts)
    full = 8.0 * w * len(ev.times)
    if not opts.miss_null:
        return opts.miss_penalty * full
    o = np.asarray(ev.ratio, float)
    o = o[np.isfinite(o)]
    if len(o) == 0:
        return 0.0
    sigma = event_sigma(ev, opts)
    res = np.clip(o - np.median(o), -4 * sigma, 4 * sigma)
    null = 0.5 * float((res**2).sum()) / sigma**2 * w
    return opts.miss_penalty * min(null, full)


def silence_cost(rm, opts: Options) -> float:
    pd = opts.detect_prob * max(0.0, 1.0 - rm / opts.r_silence_m)
    return float(-np.log1p(-min(pd, 0.999)))


def _obs_span(ev):
    o = np.asarray(ev.ratio)
    o = o[np.isfinite(o) & (o > 0)]
    return float(np.log(o.max() / o.min())) if len(o) > 2 else 0.0


def _pass_span(pl, g, k, half, c):
    lo, hi = max(0, k - int(half)), min(len(pl.tau), k + int(half) + 1)
    if hi - lo < 3:
        return None
    pos = pl.pos[lo:hi]
    d = pos - g
    r = np.linalg.norm(d, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vel = np.gradient(pos, pl.tau[lo:hi], axis=0)
        vr = -np.sum(d * vel, axis=1) / np.maximum(r, 1.0)
        D = c / (c - vr)
    D = D[np.isfinite(D) & (D > 0)]
    if len(D) < 3:
        return None
    return float(np.log(D.max() / D.min()))


def score_points(events, planes, gxyz, c, opts: Options, t0_window=None, clip_dur=None):
    """Joint log-likelihood per grid point. t0_window=(lo, hi) unix bounds on recording start."""
    passes = find_passes(planes, gxyz, opts, c)
    n = len(gxyz)
    logl = np.full(n, -np.inf)
    t0 = np.full(n, np.nan)
    matches = [None] * n
    clip_dur = clip_dur or (max(e.t_end for e in events) + 5 if events else 0)
    spans = {id(e): _obs_span(e) for e in events}
    for i in range(n):
        P = passes[i]
        if not P:
            # no aircraft near this point: every event is a miss (finite, not -inf)
            logl[i] = -sum(miss_cost(e, opts) for e in events)
            if t0_window:
                t0[i] = 0.5 * (t0_window[0] + t0_window[1])
            continue
        # per event, candidate (logl, T0, pass)
        per_ev = []
        for ev in events:
            cand = []
            for pi, k, rm, tr in P:
                if (
                    not opts.include_jets
                    and not is_broadband(ev)
                    and planes[pi].cls not in TONAL_CLASSES
                ):
                    continue  # jets are only loaded for broadband events
                T0c = tr - ev.t_cpa
                if t0_window and not (
                    t0_window[0] - opts.dt_tol_s <= T0c <= t0_window[1] + opts.dt_tol_s
                ):
                    continue
                # cheap scale-free prefilter: the predicted Doppler span over the
                # event's duration must be able to cover the observed span
                half = (ev.times[-1] - ev.times[0]) / 2 + opts.dt_tol_s
                if not is_broadband(ev):
                    sp = _pass_span(planes[pi], gxyz[i], k, half, c)
                    if sp is None or sp + opts.span_tol < spans[id(ev)]:
                        continue
                ll, T0, sc = event_pass_loglik(ev, planes[pi], gxyz[i], k, tr, c, opts)
                if np.isfinite(ll):
                    ll += cpa_consistency(ev, rm, opts)
                    chi2 = -2.0 * ll / (event_weight(ev, opts) * len(ev.times))
                    cand.append((ll, T0, pi, rm, sc, chi2))
            per_ev.append(cand)
        hyps = sorted({round(cnd[1], 1) for cand in per_ev for cnd in cand})
        if not hyps:
            # no event matched any pass: still score "all missed" (+ silence)
            hyps = (
                [round(0.5 * (t0_window[0] + t0_window[1]), 1)]
                if t0_window
                else sorted({round(tr - events[0].t_cpa, 1) for *_x, tr in P})
                if events
                else [0.0]
            )
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
                    if opts.marginalize:
                        # association ambiguity: sum over candidate aircraft and the
                        # "not in ADS-B / not this pass" alternative instead of the max
                        alts = [z[0] for z in ok] + [-miss_cost(events[e_i], opts)]
                        tot += float(np.logaddexp.reduce(alts))
                    else:
                        tot += b[0]
                    used.append((e_i, b[2], b[3], b[4], b[5]))
                else:
                    # as bad as a fit where every sample is an outlier (0.5 * 4^2 per effective sample)
                    tot -= miss_cost(events[e_i], opts)
            # silence: close tonal passes inside the clip that no event explains
            used_pl = {u[1] for u in used}
            for pi, _k, rm, tr in P:
                if pi in used_pl or rm > opts.r_silence_m:
                    continue
                if not opts.include_jets and planes[pi].cls not in TONAL_CLASSES:
                    continue
                if T0h <= tr <= T0h + clip_dur:
                    tot -= opts.silence_penalty / 6.0 * silence_cost(rm, opts)
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
                    "chi2nu": ch,
                    "event_cpa_slant_m": float(events[e_i].cpa_slant_m),
                    "cpa_ok": bool(
                        cpa_consistency(events[e_i], rm, opts) > -2.0
                        or opts.cpa_weight <= 0
                    ),
                }
                for e_i, pi, rm, sc, ch in best[2]
            ]
    return PointResult(logl, t0, matches)


def n_consistent_matches(matches, chi2_max=6.0, opts: Options | None = None):
    """Matched events whose Doppler fit is decent and CPA range is consistent."""
    n = 0
    for m in matches:
        ch = m.get("chi2nu", np.inf)
        if np.isfinite(ch) and ch <= chi2_max and m.get("cpa_ok", True):
            n += 1
    return n


def grid(cx, cy, half, step):
    xs = np.arange(cx - half, cx + half + 1e-6, step)
    ys = np.arange(cy - half, cy + half + 1e-6, step)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()


def birge_temperature(res: PointResult) -> float:
    """Likelihood temperature from the best point's reduced chi-square (Birge ratio).

    If matched events fit worse than the assumed sigma allows (chi2/nu > 1), the
    noise model is too optimistic and the posterior is flattened by that factor.
    """
    fin = np.isfinite(res.logl)
    if not fin.any():
        return 1.0
    j = int(np.argmax(np.where(fin, res.logl, -np.inf)))
    m = res.matches[j] or []
    ch = [x["chi2nu"] for x in m if np.isfinite(x.get("chi2nu", np.nan))]
    return float(max(1.0, np.median(ch))) if ch else 1.0


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
            **{**opts.__dict__, "sigma_model": opts.sigma_model * (step / 50.0) ** 0.5}
        )
        res = score_points(events, planes, g, c, o, t0_window, clip_dur)
        if progress:
            progress(li, len(g))
        temp = birge_temperature(res) * opts.temper_extra
        p, a68, a95 = posterior(res.logl / temp, step * step)
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
                "temperature": temp,
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
            if len(picked) >= (opts.n_seeds if li == 0 else max(3, opts.n_seeds // 2)):
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
