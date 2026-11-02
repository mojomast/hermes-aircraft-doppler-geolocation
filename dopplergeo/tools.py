"""Hermes tools: doppler_detect (offline) and doppler_locate.

Handlers return one JSON document. Heavy imports happen inside the handlers.
"""

from __future__ import annotations

import json

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


def detect_tool(params, **_kwargs) -> str:
    try:
        args = _params(params)
        from .locate import detect_only

        return _dump(detect_only(str(args["recording"]), args.get("out_dir") or None))
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
        )
        return _dump(rep)
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
    return [
        {
            "name": "doppler_detect",
            "description": "Find aircraft flyover Doppler tones in a recording (offline).",
            "schema": _schema(
                "doppler_detect",
                "Scan a whole recording for aircraft flyover Doppler tones without hand-picked bands. "
                "Returns events (closest-approach time, speed, slant distance, rest frequency, harmonics, F-test) "
                "and writes detection.png. Local only, no network. Do not write a replacement detector.",
                {"recording": rec, "out_dir": out},
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
                    "out_dir": out,
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
