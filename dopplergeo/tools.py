"""Hermes tools: doppler_detect (offline) and doppler_locate.

Handlers return one JSON document. Heavy imports happen inside the handlers.
"""

from __future__ import annotations

import json
import os

TOOL_NAMES = ("doppler_detect", "doppler_locate")


def _params(params) -> dict:
    if params is None:
        return {}
    if isinstance(params, str):
        params = json.loads(params)
    if not isinstance(params, dict):
        raise ValueError("tool arguments must be an object")
    return params


def _dump(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


MEDIA_HINT = (
    "To show these plots to the user, copy each line in `media` into your "
    "reply, on its own line (Hermes turns MEDIA:<path> lines into inline "
    "images or attachments)."
)


def _with_media(rep):
    """Add Hermes media fields for the PNGs a run wrote.

    ``media`` holds one ``MEDIA:<absolute path>`` line per existing PNG in
    ``files`` (detection first, then heatmap), ``generated_image`` the same
    paths, so Hermes clients that index tool artifacts can preview them.
    Nothing is added when no plot exists (``plots: false`` or an early error).
    """
    if not isinstance(rep, dict):
        return rep
    files = rep.get("files") or {}
    order = ["detection_png", "heatmap_png"]
    keys = order + sorted(k for k in files if k.endswith("_png") and k not in order)
    paths = []
    for key in keys:
        value = files.get(key)
        if value and os.path.isfile(value):
            path = os.path.abspath(value)
            if path not in paths:
                paths.append(path)
    if paths:
        rep["generated_image"] = paths
        rep["media"] = [f"MEDIA:{p}" for p in paths]
        rep["media_hint"] = MEDIA_HINT
    return rep


def detect_tool(params, **_kwargs) -> str:
    try:
        args = _params(params)
        from .locate import detect_only

        return _dump(
            _with_media(
                detect_only(
                    str(args["recording"]),
                    args.get("out_dir") or None,
                    broadband=bool(args.get("broadband", False)),
                    plots=bool(args.get("plots", True)),
                    cache=bool(args.get("cache", True)),
                )
            )
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return _dump({"ok": False, "error": str(exc)})


def locate_tool(params, **_kwargs) -> str:
    try:
        args = _params(params)
        from .locate import locate

        rep = locate(
            str(args["recording"]),
            date_hint=args.get("date_hint") or None,
            region_hint=args.get("region_hint") or None,
            offline=bool(args.get("offline", False)),
            out_dir=args.get("out_dir") or None,
            time_tolerance_s=float(args.get("time_tolerance_s", 600)),
            include_jets=bool(args.get("include_jets", False)),
            broadband=bool(args.get("broadband", False)),
            temper_extra=(
                float(args["temper_extra"])
                if args.get("temper_extra") is not None
                else None
            ),
            plots=bool(args.get("plots", True)),
            cache=bool(args.get("cache", True)),
        )
        return _dump(_with_media(rep))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return _dump({"ok": False, "error": str(exc)})


def _schema(name, description, props, required):
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": props, "required": required},
    }


def tool_specs() -> list[dict]:
    rec = {
        "type": "string",
        "description": "Path to an audio or video file on this machine.",
    }
    out = {
        "type": "string",
        "description": "Directory for report.json and PNGs. Default: <recording>.doppler/",
    }
    bb = {
        "type": "boolean",
        "description": "Also detect jet passes from the ground-reflection (Lloyd's mirror) delay. Experimental. Default false.",
    }
    plots = {
        "type": "boolean",
        "description": "Write diagnostic PNGs (detection.png, heatmap.png). Default true.",
    }
    cache = {
        "type": "boolean",
        "description": "Reuse cached events for the same audio content and detector settings. Default true.",
    }
    return [
        {
            "name": "doppler_detect",
            "description": "Find aircraft flyover Doppler tones in a recording (offline).",
            "schema": _schema(
                "doppler_detect",
                "Scan a whole recording for aircraft flyover Doppler tones without hand-picked bands. "
                "Returns events (closest-approach time, speed, slant distance, rest frequency, harmonics, F-test) "
                "and writes detection.png. Local only, no network. Do not write a replacement detector.",
                {
                    "recording": rec,
                    "out_dir": out,
                    "broadband": bb,
                    "plots": plots,
                    "cache": cache,
                },
                ["recording"],
            ),
            "handler": detect_tool,
        },
        {
            "name": "doppler_locate",
            "description": "Estimate where a recording was made from flyover Doppler tones and ADS-B history.",
            "schema": _schema(
                "doppler_locate",
                "Detect flyover events, download (or read cached) adsb.lol ADS-B history for the day cropped to the region, "
                "and score ground locations by predicting each event's Doppler curve from real tracks. "
                "Network: GitHub release downloads (~3 GB streamed per uncached day), AWS Copernicus DEM tiles, Open-Meteo. "
                "offline=true uses only the cache. Returns best point, top-5, credible area, matched aircraft, heatmap.png. "
                "Use only for forensics, research, or the user's own recordings; never to locate a person without consent.",
                {
                    "recording": rec,
                    "date_hint": {
                        "type": "string",
                        "description": "YYYY-MM-DD (UTC day, time unknown) or ISO datetime of the recording start.",
                    },
                    "region_hint": {
                        "type": "string",
                        "description": "lat,lon,radius_km search region (radius <= 60 km).",
                    },
                    "offline": {"type": "boolean"},
                    "time_tolerance_s": {
                        "type": "number",
                        "description": "Slack around an ISO datetime hint. Default 600.",
                    },
                    "include_jets": {"type": "boolean"},
                    "broadband": bb,
                    "temper_extra": {
                        "type": "number",
                        "description": "95% area calibration: extra posterior temper on top of the Birge ratio. Default 4.",
                    },
                    "out_dir": out,
                    "plots": plots,
                    "cache": cache,
                },
                ["recording"],
            ),
            "handler": locate_tool,
        },
    ]


def register_tools(ctx) -> None:
    for spec in tool_specs():
        ctx.register_tool(
            name=spec["name"],
            toolset="aircraft-doppler-geolocation",
            schema=spec["schema"],
            handler=spec["handler"],
            description=spec["description"],
        )
