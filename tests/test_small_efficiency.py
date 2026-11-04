"""Small-efficiency pass: identical numbers to f65a6a2, and skipped work stays skipped."""

import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref_detect_f65a6a2 as ref_detect
import ref_score_f65a6a2 as ref_score
from test_v02 import LAT0, LON0, SR, scenario

from dopplergeo import detect, score

_spec = importlib.util.spec_from_file_location(
    "eval_batch", Path(__file__).resolve().parents[1] / "scripts" / "eval_batch.py"
)
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)


def _eq(a, b):
    """Exact (bit-level) equality for nested events/diagnostics."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _eq(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _eq(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, float) and np.isnan(a):
        assert np.isnan(b)
    else:
        assert a == b


def test_whitened_spectrogram_identical():
    x = np.random.default_rng(3).normal(0, 0.1, SR * 60)
    for A, B in zip(
        ref_detect.whitened_spectrogram(x, SR), detect.whitened_spectrogram(x, SR)
    ):
        assert B.dtype == np.float64
        np.testing.assert_array_equal(A, B)


@pytest.mark.parametrize("bb", [False, True])
def test_detect_events_identical(bb):
    x, _tracks, _obs = scenario(crossing=True)
    ea, da = ref_detect.detect_events(x, SR, broadband=bb)
    eb_, db = detect.detect_events(x, SR, broadband=bb)
    assert ea
    _eq([asdict(e) for e in ea], [asdict(e) for e in eb_])
    _eq(da, db)  # rejected list, spectrogram and harmonic-sum diagnostics


def test_score_points_identical_with_silence():
    x, tracks, _obs = scenario(crossing=True)
    events, _ = detect.detect_events(x, SR)
    pn = score.prepare_planes(tracks, LAT0, LON0, include_jets=True)
    po = ref_score.prepare_planes(tracks, LAT0, LON0, include_jets=True)
    xs, ys = score.grid(0, 0, 4000, 250)
    g = np.c_[xs, ys, np.full(len(xs), 151.5)]
    # drop one event so close passes go unexplained and the silence penalty bites
    for evs in (events, events[:1], []):
        for kw in ({}, {"include_jets": True}, {"silence_penalty": 11.0}):
            for t0w in (None, (-20.0, 20.0)):
                a = ref_score.score_points(
                    evs, po, g, 343.0, ref_score.Options(**kw), t0w, 420
                )
                b = score.score_points(evs, pn, g, 343.0, score.Options(**kw), t0w, 420)
                np.testing.assert_array_equal(a.logl, b.logl)
                np.testing.assert_array_equal(a.t0, b.t0)
                assert a.matches == b.matches


def _entry(tmp_path):
    wav = tmp_path / "n.wav"
    sf.write(wav, np.random.default_rng(1).normal(0, 0.1, 4000 * 20), 4000)
    return {
        "id": "n1",
        "audio": str(wav),
        "t0_unix": 1.7e9,
        "truth_lat": 43.8,
        "truth_lon": -79.3,
        "truth_unc_m": 100,
        "split": "tune",
    }


def test_no_events_skips_adsb(tmp_path, monkeypatch):
    monkeypatch.setattr(eb, "events_for", lambda *a, **k: ([], 20.0))

    def boom(*a, **k):
        raise AssertionError("ADS-B work on a clip without events")

    monkeypatch.setattr(eb, "load_tracks", boom)
    monkeypatch.setattr(eb.score, "prepare_planes", boom)
    row = eb.run_one((_entry(tmp_path), 30.0, {}, {}, str(tmp_path), 15.0, 5.0, False))
    assert row["error"] == "no events"
    assert row["events"] == 0
    assert row["planes"] is None  # not evaluated, not "zero aircraft"


def test_completed_resume_runs_no_detection(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("DOPPLER_GEO_CACHE", str(cache))
    e = _entry(tmp_path)
    man = tmp_path / "m.json"
    man.write_text(json.dumps([e]))
    out = tmp_path / "out"
    args = [str(man), "--out", str(out), "--tol", "30", "--jobs", "1"]
    args += ["--det-jobs", "1"]
    subprocess.run(
        [sys.executable, eb.__file__, *args],
        check=True,
        capture_output=True,
        env={**__import__("os").environ, "DOPPLER_GEO_CACHE": str(cache)},
    )
    before = (out / "rows.jsonl").read_text()

    class NoPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, *a, **k):
            raise AssertionError("detection prepared for a completed resume")

        def submit(self, *a, **k):
            raise AssertionError("scoring job submitted for a completed resume")

    monkeypatch.setattr(eb, "ProcessPoolExecutor", NoPool)
    rows = eb.main([*args, "--resume"])
    assert (out / "rows.jsonl").read_text() == before
    assert len(rows) == 1


def test_resume_detects_each_pending_clip_once(tmp_path, monkeypatch):
    e = _entry(tmp_path)
    man = tmp_path / "m.json"
    man.write_text(json.dumps([e]))
    seen = []

    class FakePool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, fn, it):
            seen.extend(m["id"] for m, _d, _w in it)
            return []

        def submit(self, fn, j):
            from concurrent.futures import Future

            f = Future()
            r = {"id": j[0]["id"], "split": "tune", "tol_s": j[1], "runtime_s": 0.0}
            f.set_result(r)
            return f

    monkeypatch.setattr(eb, "ProcessPoolExecutor", FakePool)
    out = tmp_path / "out"
    eb.main([str(man), "--out", str(out), "--tol", "30", "600", "--jobs", "1"])
    assert seen == ["n1"]  # two tolerances, one detection
