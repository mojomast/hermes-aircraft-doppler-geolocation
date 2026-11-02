"""One-call pipeline: locate(recording, date_hint=None, region_hint=None)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from . import adsb, detect, env, metadata, score
from .cache import OfflineMiss
from .physics import C_DEFAULT


def _parse_date_hint(h):
    """'YYYY-MM-DD' -> (date, None); ISO datetime -> (date, datetime UTC)."""
    if h is None or h == "":
        return None, None
    if isinstance(h, dt.datetime):
        t = h if h.tzinfo else h.replace(tzinfo=dt.UTC)
        return t.date(), t
    if isinstance(h, dt.date):
        return h, None
    s = str(h).strip()
    if len(s) == 10:
        return dt.date.fromisoformat(s), None
    t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    t = t if t.tzinfo else t.replace(tzinfo=dt.UTC)
    return t.date(), t.astimezone(dt.UTC)


def _parse_region(r):
    """'lat,lon,radius_km' or (lat, lon, radius_km) or {lat, lon, radius_km}."""
    if r is None or r == "":
        return None
    if isinstance(r, dict):
        return float(r["lat"]), float(r["lon"]), float(r.get("radius_km", 15))
    if isinstance(r, str):
        r = [float(v) for v in r.replace(";", ",").split(",")]
    r = list(r)
    return float(r[0]), float(r[1]), float(r[2] if len(r) > 2 else 15)


def locate(
    recording,
    date_hint=None,
    region_hint=None,
    offline=False,
    out_dir=None,
    time_tolerance_s=600.0,
    include_jets=False,
    max_radius_km=60.0,
    tar_parts=None,
    tracks=None,
    elevation_fn=None,
    progress=None,
):
    """Locate a recording from flyover Doppler events and ADS-B.

    recording: audio/video path. date_hint: 'YYYY-MM-DD' (time unknown, the whole
    UTC day is searched) or an ISO datetime of the recording start (searched
    +/- time_tolerance_s). region_hint: 'lat,lon,radius_km' (required unless the
    file carries GPS). offline: use only cached ADS-B/DEM/weather. tracks /
    elevation_fn / tar_parts let tests and power users inject data.
    Returns a JSON-able dict; writes report.json and heatmap.png to out_dir.
    """
    notes = []
    out = Path(out_dir) if out_dir else Path(str(recording) + ".doppler")
    out.mkdir(parents=True, exist_ok=True)
    meta = metadata.hints(recording)
    if meta.get("gps"):
        notes.append(
            "The file carries GPS coordinates in its metadata; that is the location."
        )
        rep = {
            "ok": True,
            "method": "metadata-gps",
            "best": {"lat": meta["gps"][0], "lon": meta["gps"][1]},
            "metadata": _jsonable(meta),
            "notes": notes,
        }
        (out / "report.json").write_text(json.dumps(rep, indent=2))
        return rep
    day, when = _parse_date_hint(date_hint)
    if day is None and meta.get("creation_time"):
        when = meta["creation_time"].astimezone(dt.UTC)
        day = when.date()
        notes.append(
            f"Recording time from file metadata: {when.isoformat()} (+/- {time_tolerance_s:.0f} s searched)."
        )
    if day is None and meta.get("upload_date"):
        day = meta["upload_date"]
        notes.append(
            f"No recording time. Using the upload date {day} (UTC day); the recording may be older. Pass date_hint to search another day."
        )
    region = _parse_region(region_hint)
    x, sr = detect.load_audio(recording)
    clip_dur = len(x) / sr
    events, diag = detect.detect_events(x, sr)
    detect.plot_detection(
        diag,
        events,
        str(out / "detection.png"),
        f"{Path(recording).name}: {len(events)} event(s)",
    )
    aircraft_events = [e for e in events if not e.likely_ground_vehicle]
    if len(aircraft_events) < len(events):
        notes.append(
            f"{len(events) - len(aircraft_events)} slow event(s) (< 35 m/s) set aside as likely road vehicles."
        )
    rep = {
        "ok": False,
        "recording": str(recording),
        "duration_s": clip_dur,
        "events": [e.to_dict() for e in events],
        "rejected_segments": diag["rejected"],
        "metadata": _jsonable(meta),
        "notes": notes,
        "files": {"detection_png": str(out / "detection.png")},
    }
    if not aircraft_events:
        rep["error"] = "no aircraft Doppler events detected"
        return _finish(rep, out)
    if region is None:
        rep["error"] = (
            "region_hint (lat,lon,radius_km) is required: a wider search is not supported"
        )
        return _finish(rep, out)
    if day is None:
        rep["error"] = "date_hint is required (no creation time or upload date found)"
        return _finish(rep, out)
    lat0, lon0, rad_km = region
    rad_km = min(rad_km, max_radius_km)
    opts = score.Options(include_jets=include_jets)
    day_start = dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp()
    if when is not None:
        t0w = (when.timestamp() - time_tolerance_s, when.timestamp() + time_tolerance_s)
    else:
        t0w = (day_start - clip_dur, day_start + 86400)
    margin_deg = (rad_km * 1000 + opts.r_max_m) / 111000.0
    bbox = (
        lat0 - margin_deg,
        lon0 - margin_deg / np.cos(np.radians(lat0)),
        lat0 + margin_deg,
        lon0 + margin_deg / np.cos(np.radians(lat0)),
    )
    if tracks is None:
        tracks = []
        days = sorted(
            {
                dt.datetime.fromtimestamp(t, dt.UTC).date()
                for t in (t0w[0], t0w[1] + clip_dur)
            }
        )
        for d in days:
            try:
                tracks += adsb.fetch_tracks(
                    d,
                    bbox,
                    t0w[0] - 900,
                    t0w[1] + clip_dur + 900,
                    offline=offline,
                    tar_parts=tar_parts,
                    progress=progress,
                )
            except OfflineMiss as e:
                rep["error"] = f"offline: {e}"
                return _finish(rep, out)
    planes = score.prepare_planes(tracks, lat0, lon0, include_jets=include_jets)
    rep["adsb"] = {
        "aircraft_in_crop": len(tracks),
        "tonal_track_segments": len(planes),
        "bbox": bbox,
        "attribution": adsb.ATTRIBUTION,
    }
    if not planes:
        rep["error"] = (
            "no propeller/turboprop/helicopter ADS-B tracks in the region and time window"
        )
        return _finish(rep, out)
    mid_time = dt.datetime.fromtimestamp((t0w[0] + t0w[1]) / 2, dt.UTC)
    c, csrc = env.sound_speed(
        lat0,
        lon0,
        mid_time
        if when is not None
        else dt.datetime.fromtimestamp(day_start + 43200, dt.UTC),
        offline,
    )
    rep["speed_of_sound"] = {"c_mps": c, "source": csrc}
    dem_src = {"src": None}

    def elev(lat, lon):
        if elevation_fn is not None:
            dem_src["src"] = "injected"
            return elevation_fn(lat, lon)
        z, s = env.elevation(lat, lon, offline=offline)
        dem_src["src"] = s
        return z

    res = score.search(
        aircraft_events,
        planes,
        lat0,
        lon0,
        rad_km * 1000,
        elev,
        c,
        opts,
        t0_window=t0w,
        clip_dur=clip_dur,
        progress=progress,
    )
    rep["dem"] = dem_src["src"]
    best = res["best"]
    if best is None or not np.isfinite(best["logl"]):
        rep["error"] = "no location matched any event"
        return _finish(rep, out)
    lv = res["levels"]
    mid = lv[1] if len(lv) > 1 else lv[0]
    best["recording_start_utc"] = dt.datetime.fromtimestamp(
        best["t0_unix"], dt.UTC
    ).isoformat()
    rep.update(
        ok=True,
        method="doppler-adsb",
        best=best,
        top=_top(lv[-1]),
        credible_area_km2={
            "68": _km2(mid["a68_m2"]),
            "95": _km2(mid["a95_m2"]),
            "grid_step_m": mid["step"],
        },
        search={
            "region": {"lat": lat0, "lon": lon0, "radius_km": rad_km},
            "t0_window_utc": [
                dt.datetime.fromtimestamp(v, dt.UTC).isoformat() for v in t0w
            ],
            "levels": [{"step_m": L["step"], "points": len(L["x"])} for L in lv],
        },
    )
    rep["files"]["heatmap_png"] = str(out / "heatmap.png")
    plot_heatmap(res, planes, lat0, lon0, str(out / "heatmap.png"), best)
    return _finish(rep, out)


def _km2(a):
    return None if a is None else round(a / 1e6, 4)


def _top(L, k=5):
    fin = np.isfinite(L["logl"])
    order = np.argsort(-np.where(fin, L["logl"], -np.inf))[:k]
    return [
        {
            "lat": float(L["lat"][j]),
            "lon": float(L["lon"][j]),
            "logl": float(L["logl"][j]),
            "matches": L["matches"][j],
        }
        for j in order
        if fin[j]
    ]


def _jsonable(d):
    return json.loads(json.dumps(d, default=str))


def _finish(rep, out):
    (out / "report.json").write_text(json.dumps(_jsonable(rep), indent=2))
    rep["files"] = {**rep.get("files", {}), "report_json": str(out / "report.json")}
    return _jsonable(rep)


def plot_heatmap(res, planes, lat0, lon0, path, best=None, truth=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lv = res["levels"]
    fig, ax = plt.subplots(figsize=(8, 7))
    L0 = lv[0]
    fin = np.isfinite(L0["logl"])
    ax.scatter(
        L0["lon"][~fin],
        L0["lat"][~fin],
        s=8,
        c="#dddddd",
        marker="s",
        label="no consistent pass",
    )
    for L, size in zip(lv[1:2] or lv[:1], (10,)):
        f2 = np.isfinite(L["logl"])
        p = L["p"]
        sc = ax.scatter(
            L["lon"][f2],
            L["lat"][f2],
            c=np.log10(np.maximum(p[f2], 1e-12)),
            cmap="viridis",
            s=size,
            marker="s",
            vmin=-6,
            vmax=0,
        )
        fig.colorbar(sc, ax=ax, label=f"log10 posterior ({int(L['step'])} m cells)")
        srt = np.argsort(-p)
        cum = np.cumsum(p[srt])
        for lvl, col in ((0.68, "red"), (0.95, "orange")):
            sel = srt[: np.searchsorted(cum, lvl) + 1]
            ax.scatter(
                L["lon"][sel],
                L["lat"][sel],
                s=size * 2.2,
                facecolors="none",
                edgecolors=col,
                lw=0.6,
                label=f"{int(lvl * 100)}% credible",
            )
    used = {m["hex"] for m in (best or {}).get("matches") or []}
    for pl in planes:
        if pl.hex in used:
            la, lo = score.from_enu(pl.pos[:, 0], pl.pos[:, 1], lat0, lon0)
            ax.plot(lo, la, "-", lw=1.5, label=f"ADS-B {pl.label}")
    if best:
        ax.scatter(
            [best["lon"]],
            [best["lat"]],
            marker="*",
            s=260,
            c="gold",
            edgecolors="k",
            zorder=5,
            label="best",
        )
    if truth:
        ax.scatter(
            [truth[1]],
            [truth[0]],
            marker="X",
            s=120,
            c="white",
            edgecolors="k",
            zorder=6,
            label="true",
        )
    span = max(np.ptp(L0["lat"]), 1e-3)
    ax.set_xlim(L0["lon"].min() - 0.05 * span, L0["lon"].max() + 0.05 * span)
    ax.set_ylim(L0["lat"].min() - 0.05 * span, L0["lat"].max() + 0.05 * span)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Doppler/ADS-B location posterior")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def detect_only(recording, out_dir=None):
    x, sr = detect.load_audio(recording)
    events, diag = detect.detect_events(x, sr)
    out = Path(out_dir) if out_dir else Path(str(recording) + ".doppler")
    out.mkdir(parents=True, exist_ok=True)
    png = out / "detection.png"
    detect.plot_detection(
        diag, events, str(png), f"{Path(recording).name}: {len(events)} event(s)"
    )
    rep = {
        "recording": str(recording),
        "duration_s": len(x) / sr,
        "events": [e.to_dict() for e in events],
        "rejected_segments": diag["rejected"],
        "files": {"detection_png": str(png)},
    }
    (out / "detect.json").write_text(json.dumps(rep, indent=2))
    return rep


__all__ = ["C_DEFAULT", "detect_only", "locate"]
