"""Prepared scoring must reproduce the v0.3.0 scorer bit-for-bit on fixed inputs."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref_score_v030 as ref
from test_v02 import LAT0, LON0, SR, scenario

from dopplergeo import broadband, detect, score, synth


def _random_planes(seed=0, n_pl=25, uniform=True):
    rng = np.random.default_rng(seed)
    new, old = [], []
    for _ in range(n_pl):
        n = int(rng.integers(20, 700))
        tau = 1.7e9 + 0.37 + np.arange(n, dtype=float)
        if not uniform:
            tau = tau + np.r_[0, np.cumsum(rng.uniform(0, 0.4, n - 1))]
        p0 = rng.uniform(-15000, 15000, 3)
        p0[2] = 150 + abs(p0[2]) % 2000
        v = rng.normal(0, 60, 3)
        v[2] *= 0.05
        pos = p0 + np.outer(np.arange(n), v) + np.cumsum(rng.normal(0, 3, (n, 3)), 0)
        new.append(score.Plane("h", "piston", tau, pos))
        old.append(ref.Plane("h", "piston", tau, pos))
    return new, old


def _grid(seed=1):
    x, y = score.grid(0, 0, 15000, 600)
    return np.c_[x, y, np.random.default_rng(seed).uniform(0, 50, len(x))]


def test_find_passes_identical():
    for uniform in (True, False):
        new, old = _random_planes(uniform=uniform)
        g = _grid()
        a = ref.find_passes(old, g, ref.Options(), 343.0)
        b = score.find_passes(new, g, score.Options(), 343.0)
        assert a == b and sum(map(len, a)) > 100


def test_slice_velocity_and_predictions_identical():
    for uniform in (True, False):
        new, old = _random_planes(seed=2, n_pl=8, uniform=uniform)
        g = np.array([300.0, -200.0, 12.0])
        for pn, po in zip(new, old):
            for k in (0, 3, len(pn.tau) // 2, len(pn.tau) - 1):
                t = (
                    pn.tau[k]
                    + np.arange(-3, 3.1, 0.5)[:, None]
                    + np.linspace(-25, 25, 30)
                )
                for half in (2, 20, 300):
                    for f in ("predicted_ratio", "predicted_sin_elev"):
                        A = getattr(ref, f)(po, g, t, 343.0, k, half)
                        B = getattr(score, f)(pn, g, t, 343.0, k, half)
                        assert (A is None) == (B is None)
                        if A is not None:
                            np.testing.assert_array_equal(A, B)
                    assert ref._pass_span(po, g, k, half, 343.0) == score._pass_span(
                        pn, g, k, half, 343.0
                    )


def _same_result(a, b):
    np.testing.assert_array_equal(a.logl, b.logl)
    np.testing.assert_array_equal(a.t0, b.t0)
    assert a.matches == b.matches


def test_score_points_and_search_identical_tonal():
    x, tracks, _obs = scenario(crossing=True)
    events, _ = detect.detect_events(x, SR)
    assert events
    pn = score.prepare_planes(tracks, LAT0, LON0)
    po = ref.prepare_planes(tracks, LAT0, LON0)
    xs, ys = score.grid(0, 0, 4000, 250)
    g = np.c_[xs, ys, np.full(len(xs), 151.5)]
    for kw in ({}, {"miss_null": True}, {"marginalize": False, "sigma_model": 0.02}):
        for t0w in (None, (-20.0, 20.0)):
            a = ref.score_points(events, po, g, 343.0, ref.Options(**kw), t0w, 420)
            b = score.score_points(events, pn, g, 343.0, score.Options(**kw), t0w, 420)
            _same_result(a, b)
    ra = ref.search(
        events,
        po,
        LAT0,
        LON0,
        5000,
        lambda la, lo: np.full(len(la), 150.0),
        343.0,
        ref.Options(),
        None,
        420,
    )
    rb = score.search(
        events,
        pn,
        LAT0,
        LON0,
        5000,
        lambda la, lo: np.full(len(la), 150.0),
        343.0,
        score.Options(),
        None,
        420,
    )
    assert ra["best"] == rb["best"]
    for La, Lb in zip(ra["levels"], rb["levels"]):
        np.testing.assert_array_equal(La["logl"], Lb["logl"])
        assert La["temperature"] == Lb["temperature"] and La["a95_m2"] == Lb["a95_m2"]


def test_score_points_identical_broadband():
    t = np.arange(0, 200, 1.0)
    la, lo, al = synth.track(
        t,
        LAT0,
        LON0,
        speed=75.0,
        heading_deg=60.0,
        alt_m=500.0,
        through=(300.0, 0.0),
        t_through=100,
    )
    x = synth.render_broadband(
        0, 200, SR, (LAT0, LON0, 0.0), [(t, la, lo, al)], mic_h=1.5
    )
    ev, _ = broadband.detect_broadband(x, SR)
    assert ev
    tr = [synth.as_adsb("abc", t, la, lo, al, cls="jet", typ="A320")]
    pn = score.prepare_planes(tr, LAT0, LON0, include_jets=True)
    po = ref.prepare_planes(tr, LAT0, LON0, include_jets=True)
    xs, ys = score.grid(0, 0, 3000, 250)
    g = np.c_[xs, ys, np.zeros(len(xs))]
    a = ref.score_points(ev, po, g, 343.0, ref.Options(), (-5, 5), 200)
    b = score.score_points(ev, pn, g, 343.0, score.Options(), (-5, 5), 200)
    _same_result(a, b)
