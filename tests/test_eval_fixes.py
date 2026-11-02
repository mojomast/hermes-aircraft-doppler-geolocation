"""Regression tests for real-clip eval crashes (v0.2.0)."""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dopplergeo import score
from dopplergeo.detect import Event


def one_event(t0=1_700_000_000.0):
    t = np.arange(0, 60, 1.0)
    r = 1 + 0.02 * np.tanh(-(t - 30) / 10)
    return Event(
        0,
        60,
        30,
        130,
        60,
        800,
        0.5,
        8,
        [130],
        10,
        1e-6,
        True,
        times=list(t),
        ratio=list(r),
    )


class EvalFixes(unittest.TestCase):
    def test_single_unmatched_event_is_finite(self):
        ev = [one_event()]
        T0 = 1_700_000_000.0
        tau = np.arange(T0 - 600, T0 + 600, 1.0)
        # an aircraft far in time from the window: no event can match it
        pos = np.c_[
            (tau - T0) * 60.0, np.full_like(tau, 300.0), np.full_like(tau, 600.0)
        ]
        pl = [score.Plane("abc", "piston", tau, pos)]
        g = np.array([[0.0, 0.0, 1.5], [50000.0, 0.0, 1.5]])
        res = score.score_points(
            ev,
            pl,
            g,
            343.0,
            score.Options(),
            t0_window=(T0 + 3000, T0 + 3030),
            clip_dur=60,
        )
        self.assertTrue(np.isfinite(res.logl).all())

    def test_search_never_returns_none_best(self):
        ev = [one_event()]
        res = score.search(
            ev,
            [],
            34.0,
            -118.0,
            2000,
            lambda la, lo: np.zeros_like(la),
            343.0,
            score.Options(),
            t0_window=(0, 30),
            clip_dur=60,
            steps=(1000.0,),
        )
        self.assertIsNotNone(res["best"])

    def test_pass_span_degenerate_track_no_warning(self):
        tau = np.full(10, 5.0)  # duplicate timestamps -> nan velocity
        pos = np.zeros((10, 3))
        pl = score.Plane("x", "piston", tau, pos)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertIsNone(score._pass_span(pl, np.array([0, 0, 0.0]), 5, 4, 343.0))


if __name__ == "__main__":
    unittest.main()
