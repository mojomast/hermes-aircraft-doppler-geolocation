"""Shared constants and the emission-time Doppler model."""

from __future__ import annotations

import numpy as np

R_E = 6371000.0
C_DEFAULT = 343.0


def speed_of_sound(temp_c: float) -> float:
    """Dry-air speed of sound in m/s for a temperature in degrees Celsius."""
    return 331.3 * float(np.sqrt(1.0 + temp_c / 273.15))


def enu(lat, lon, alt, lat0, lon0, h0):
    """Flat-earth local east/north/up in metres (fine within ~50 km)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    alt = np.asarray(alt, float)
    x = np.radians(lon - lon0) * R_E * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R_E
    return np.stack([x, y, alt - h0], axis=-1)


def straight_doppler(t, f0, v, d, tc, c=C_DEFAULT):
    """Received frequency for a constant-velocity straight pass.

    Emission-time corrected: the frequency heard at t comes from the radial
    velocity at emission time tau, with t = tau + r(tau)/c. tc is the time the
    closest-approach sound arrives.
    """
    t = np.asarray(t, float)
    tau = np.linspace(t.min() - 60 - 3 * d / c, t.max() + 60, 4000)
    x = v * (tau - tc)
    r = np.sqrt(x**2 + d**2)
    vr = -v * x / r
    f = f0 * c / (c - vr)
    trecv = tau + r / c - d / c
    return np.interp(t, trecv, f)
