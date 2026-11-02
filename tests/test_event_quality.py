import numpy as np
from types import SimpleNamespace

from dopplergeo import detect, score


def test_subharmonic_family_relabelled():
    a = np.arange(10.0)
    lines = {2: (a, a, a), 4: (a, a, a), 6: (a, a, a)}
    out = detect._fix_subharmonic(lines)
    assert sorted(out) == [1, 2, 3]
    # genuine missing-fundamental family {2,3,4} is kept
    lines = {2: (a, a, a), 3: (a, a, a), 4: (a, a, a)}
    assert sorted(detect._fix_subharmonic(lines)) == [2, 3, 4]


def test_subharmonic_fit_gives_true_f0():
    # synthetic: lines at 1,2,3 x 80 Hz labelled as 2,4,6 of 40 Hz
    t = np.linspace(0, 40, 160)
    from dopplergeo.physics import straight_doppler
    D = straight_doppler(t, 1.0, 60.0, 300.0, 20.0, 343.0)
    lines = {2 * k: (t, k * 80.0 * D, np.full_like(t, 5.0)) for k in (1, 2, 3)}
    fit, *_ = detect._family_fit(detect._fix_subharmonic(lines), 343.0)
    assert abs(fit[0][0] - 80.0) < 1.0


def test_cpa_consistency():
    o = score.Options()
    ev = SimpleNamespace(cpa_slant_m=500.0)
    assert score.cpa_consistency(ev, 500.0, o) == 0.0
    assert score.cpa_consistency(ev, 5000.0, o) < -2.0
    assert score.cpa_consistency(ev, 900.0, o) > -0.5
    o.cpa_weight = 0
    assert score.cpa_consistency(ev, 5000.0, o) == 0.0


def test_n_consistent_matches():
    m = [{"chi2nu": 2.0, "cpa_ok": True}, {"chi2nu": 9.0, "cpa_ok": True},
         {"chi2nu": 1.0, "cpa_ok": False}]
    assert score.n_consistent_matches(m) == 1
    assert score.n_consistent_matches(m + [{"chi2nu": 3.0}]) == 2
