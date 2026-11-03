"""Synthetic flyover audio and ADS-B-like tracks for tests and demos (offline)."""

from __future__ import annotations

import numpy as np

from .physics import C_DEFAULT, R_E, enu


def track(
    t,
    lat0,
    lon0,
    speed=70.0,
    heading_deg=90.0,
    turn_deg_s=0.0,
    alt_m=600.0,
    through=(0.0, 0.0),
    t_through=None,
):
    """Constant-speed track (optional constant turn rate) that is at ENU point `through`
    (metres from lat0/lon0) at time t_through, with heading_deg there. Returns lat, lon, alt."""
    t = np.asarray(t, float)
    t_through = t[len(t) // 2] if t_through is None else t_through
    hd = np.radians(heading_deg + turn_deg_s * (t - t_through))
    dt = np.gradient(t)
    x = np.cumsum(speed * np.sin(hd) * dt)
    y = np.cumsum(speed * np.cos(hd) * dt)
    x += through[0] - np.interp(t_through, t, x)
    y += through[1] - np.interp(t_through, t, y)
    lat = lat0 + np.degrees(y / R_E)
    lon = lon0 + np.degrees(x / (R_E * np.cos(np.radians(lat0))))
    return lat, lon, np.full_like(t, alt_m)


def render(
    rec_start,
    dur,
    sr,
    observer,
    tracks,
    families,
    c=C_DEFAULT,
    noise=1.0,
    seed=0,
    gain=300.0,
):
    """Audio heard at observer=(lat, lon, h) from planes.

    tracks: list of (t, lat, lon, alt); families: per plane list of (freq, amp) lines.
    Emission-time correct: each output sample uses the phase emitted at tau(t).
    """
    rng = np.random.default_rng(seed)
    rt = rec_start + np.arange(0, dur, 1.0 / sr)
    x = rng.normal(0, noise, rt.size)
    for (tt, la, lo, al), lines in zip(tracks, families):
        tau = np.arange(tt[0], tt[-1], 1.0 / sr)
        P = enu(
            np.interp(tau, tt, la),
            np.interp(tau, tt, lo),
            np.interp(tau, tt, al),
            *observer,
        )
        r = np.linalg.norm(P, axis=1)
        trec = tau + r / c
        ok = (rt >= trec[0]) & (rt <= trec[-1])
        te = np.interp(rt[ok], trec, tau)
        amp = gain / np.interp(rt[ok], trec, r)
        for f, a in lines:
            x[ok] += a * amp * np.sin(2 * np.pi * f * (te - tau[0]))
    return x / (np.abs(x).max() * 1.05)


def as_adsb(hex_, t, lat, lon, alt, cls="piston", typ="C172", every=2.0):
    """Package a synthetic track like adsb.fetch_tracks output (sampled every `every` s)."""
    idx = np.arange(0, len(t), max(1, int(every / max(t[1] - t[0], 1e-9))))
    return {
        "hex": hex_,
        "type": typ,
        "reg": hex_.upper(),
        "class": cls,
        "t": list(t[idx]),
        "lat": list(lat[idx]),
        "lon": list(lon[idx]),
        "alt_m": list(alt[idx]),
        "gs_mps": [None] * len(idx),
    }


def render_broadband(
    rec_start,
    dur,
    sr,
    observer,
    tracks,
    mic_h=1.5,
    refl=0.9,
    c=C_DEFAULT,
    noise=0.05,
    seed=0,
    gain=300.0,
    band=(80.0, 1800.0),
):
    """Jet-like broadband flyover heard by a mic mic_h above flat reflecting ground.

    Each plane emits band-limited noise; the mic hears the direct path and the
    ground-reflected path (image receiver at -mic_h), each emission-time correct,
    so the recording carries both Doppler and the Lloyd's-mirror fringes.
    observer = (lat, lon, ground_h).
    """
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(seed)
    rt = rec_start + np.arange(0, dur, 1.0 / sr)
    x = rng.normal(0, noise, rt.size)
    sos = butter(4, [band[0], band[1]], "bp", fs=sr, output="sos")
    lat0, lon0, h0 = observer
    for tt, la, lo, al in tracks:
        tau = np.arange(tt[0], tt[-1], 1.0 / sr)
        src = sosfilt(sos, rng.normal(0, 1, tau.size))
        for hm, a in ((mic_h, 1.0), (-mic_h, refl)):
            P = enu(
                np.interp(tau, tt, la),
                np.interp(tau, tt, lo),
                np.interp(tau, tt, al),
                lat0,
                lon0,
                h0 + hm,
            )
            r = np.linalg.norm(P, axis=1)
            trec = tau + r / c
            ok = (rt >= trec[0]) & (rt <= trec[-1])
            te = np.interp(rt[ok], trec, tau)
            amp = gain / np.interp(rt[ok], trec, r)
            x[ok] += a * amp * np.interp(te, tau, src)
    return x / (np.abs(x).max() * 1.05)
