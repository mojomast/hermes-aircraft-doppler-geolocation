"""Tiny stdlib HTTP helpers (no credentials, explicit User-Agent)."""

from __future__ import annotations

import json
import urllib.request

UA = "hermes-aircraft-doppler-geolocation/0.2 (+https://github.com/mojomast/hermes-aircraft-doppler-geolocation)"


def open_url(url: str, timeout: float = 60.0):
    if not url.startswith("https://"):
        raise ValueError("https URLs only")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url: str, timeout: float = 60.0):
    with open_url(url, timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_bytes(url: str, timeout: float = 120.0) -> bytes:
    with open_url(url, timeout) as r:
        return r.read()
