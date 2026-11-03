import numpy as np

from dopplergeo import broadband, score, synth
from dopplergeo.detect import Event

SR = 4000
LAT0, LON0 = 53.44, -2.10


def _pass(alt=500.0, through=(300.0, 0.0), heading=60.0, speed=75.0):
    t = np.arange(0, 200, 1.0)
    la, lo, al = synth.track(
        t,
        LAT0,
        LON0,
        speed=speed,
        heading_deg=heading,
        alt_m=alt,
        through=through,
        t_through=100,
    )
    return t, la, lo, al


def test_synthetic_pass_recovers_delay_geometry():
    tr = _pass()
    for h in (1.5, 5.0):
        x = synth.render_broadband(0, 200, SR, (LAT0, LON0, 0.0), [tr], mic_h=h)
        ev, _ = broadband.detect_broadband(x, SR)
        assert len(ev) == 1
        e = ev[0]
        # CPA: horizontal 150 m, height 500 m -> slant 522 m; sin(elev) 0.958
        S0 = np.hypot(500.0, 150.0)
        assert abs(e.tau_max_s - 2 * h * (500.0 / S0) / 343.0) < 0.12 * 2 * h / 343
        assert abs(e.T_s - S0 / 75.0) < 0.2 * S0 / 75.0
        assert abs(e.t_cpa - 98.0) < 3.0
        assert e.kind == "broadband"


def test_noise_only_gives_no_broadband_events():
    rng = np.random.default_rng(3)
    ev, _ = broadband.detect_broadband(rng.normal(0, 1, SR * 300), SR)
    assert ev == []


def test_steady_tone_is_not_a_broadband_pass():
    t = np.arange(SR * 120) / SR
    x = sum(np.sin(2 * np.pi * 110 * k * t) / k for k in range(1, 12))
    x = x + np.random.default_rng(0).normal(0, 0.3, t.size)
    ev, _ = broadband.detect_broadband(x, SR)
    assert ev == []


def test_scorer_prefers_the_true_location_for_a_broadband_event():
    tr = _pass()
    x = synth.render_broadband(0, 200, SR, (LAT0, LON0, 0.0), [tr], mic_h=1.5)
    ev, _ = broadband.detect_broadband(x, SR)
    t, la, lo, al = tr
    planes = score.prepare_planes(
        [synth.as_adsb("abc", t, la, lo, al, cls="jet", typ="A320")],
        LAT0,
        LON0,
        include_jets=True,
    )
    opts = score.Options()
    pts = np.array([[0.0, 0.0, 0.0], [2000.0, 0.0, 0.0], [0.0, -2500.0, 0.0]])
    res = score.score_points(
        ev, planes, pts, 343.0, opts, t0_window=(-5, 5), clip_dur=200
    )
    assert np.argmax(res.logl) == 0


def test_tonal_events_ignore_jets_unless_enabled():
    t, la, lo, al = _pass()
    planes = score.prepare_planes(
        [synth.as_adsb("abc", t, la, lo, al, cls="jet", typ="A320")],
        LAT0,
        LON0,
        include_jets=True,
    )
    ev = Event(
        t_start=90,
        t_end=110,
        t_cpa=100,
        f0_hz=100,
        speed_mps=75,
        cpa_slant_m=500,
        rms_resid_hz=0.1,
        snr_z=5,
        harmonics=[100],
        f_test=10,
        p_value=1e-6,
        straight_fit_ok=True,
        times=list(np.arange(90, 110, 0.25)),
        ratio=[1.0] * 80,
        sigma_ratio=0.003,
    )
    res = score.score_points(
        [ev], planes, np.zeros((1, 3)), 343.0, score.Options(), (-5, 5), 200
    )
    assert res.matches[0] in (None, [])
