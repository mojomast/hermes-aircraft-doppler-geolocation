"""Time and place hints from the recording: ffprobe tags and yt-dlp sidecar JSON."""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from pathlib import Path

ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")


def ffprobe(path) -> dict:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return json.loads(out)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {}


def _parse_time(s: str) -> dt.datetime | None:
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.UTC)


def hints(path) -> dict:
    """Return {duration_s, creation_time, gps, upload_date, sources[]} (any may be None)."""
    p = Path(path)
    info = ffprobe(p)
    fmt = info.get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    for st in info.get("streams", []):
        for k, v in (st.get("tags") or {}).items():
            tags.setdefault(k.lower(), v)
    res = {
        "duration_s": float(fmt["duration"]) if fmt.get("duration") else None,
        "creation_time": None,
        "gps": None,
        "upload_date": None,
        "sources": [],
    }
    ct = (
        tags.get("creation_time")
        or tags.get("com.apple.quicktime.creationdate")
        or tags.get("date")
    )
    if ct and (t := _parse_time(ct)):
        res["creation_time"] = t
        res["sources"].append("ffprobe creation_time")
    loc = tags.get("location") or tags.get("com.apple.quicktime.location.iso6709")
    if loc and (m := ISO6709.match(loc)):
        res["gps"] = (float(m.group(1)), float(m.group(2)))
        res["sources"].append("ffprobe ISO 6709 location")
    for side in (p.with_suffix(".info.json"), p.parent / (p.name + ".info.json")):
        if side.exists():
            try:
                j = json.loads(side.read_text())
            except ValueError:
                continue
            if j.get("upload_date"):
                res["upload_date"] = dt.datetime.strptime(
                    j["upload_date"], "%Y%m%d"
                ).date()
                res["sources"].append(
                    "yt-dlp upload_date (recording is on or before it)"
                )
            if j.get("timestamp") and not res["creation_time"]:
                res["release_time"] = dt.datetime.fromtimestamp(j["timestamp"], dt.UTC)
            break
    return res
