"""Event cache shared by detect / locate: hits, misses, corruption, invalidation."""

import json

import numpy as np
import pytest
import soundfile as sf

from dopplergeo import detect, evcache, locate, synth

SR = 4000


@pytest.fixture
def wav(tmp_path, monkeypatch):
    monkeypatch.setenv("DOPPLER_GEO_CACHE", str(tmp_path / "cache"))
    t = np.arange(0, 200, 1.0)
    la, lo, al = synth.track(
        t,
        43.8,
        -79.3,
        speed=75.0,
        heading_deg=60.0,
        alt_m=500.0,
        through=(300.0, 0.0),
        t_through=100,
    )
    x = synth.render(
        0,
        200,
        SR,
        (43.8, -79.3, 0.0),
        [(t, la, lo, al)],
        [[(110, 0.5), (220, 0.3)]],
        noise=0.3,
    )
    p = tmp_path / "t.wav"
    sf.write(p, x.astype(np.float32), SR)
    return p


@pytest.fixture
def counter(monkeypatch):
    calls = {"n": 0}
    real = detect.detect_events

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(detect, "detect_events", counted)
    return calls


def test_hit_returns_identical_events(wav, counter, tmp_path):
    a = locate.detect_only(str(wav), str(tmp_path / "a"))
    b = locate.detect_only(str(wav), str(tmp_path / "b"))
    assert counter["n"] == 1
    assert not a["detection_cache"]["hit"] and b["detection_cache"]["hit"]
    assert json.dumps(a["events"]) == json.dumps(b["events"]) and a["events"]
    assert a["rejected_segments"] == b["rejected_segments"]
    assert a["duration_s"] == b["duration_s"]
    assert (tmp_path / "b" / "detection.png").stat().st_size > 10000
    ea, *_ = locate.detect_cached(str(wav), cache=False)
    eb, *_ = locate.detect_cached(str(wav))
    assert [e.to_dict(series=True) for e in ea] == [e.to_dict(series=True) for e in eb]


def test_settings_content_and_impl_invalidate(wav, counter, tmp_path, monkeypatch):
    locate.detect_cached(str(wav), plots=False)
    locate.detect_cached(str(wav), broadband=True, plots=False)  # other settings
    assert counter["n"] == 2
    x, sr = sf.read(wav)
    sf.write(wav, (x * 0.9).astype(np.float32), sr)  # other content, same path
    locate.detect_cached(str(wav), plots=False)
    assert counter["n"] == 3
    monkeypatch.setattr(evcache, "_impl", "other-detector")  # detector code changed
    locate.detect_cached(str(wav), plots=False)
    assert counter["n"] == 4
    locate.detect_cached(str(wav), plots=False)
    assert counter["n"] == 4


def test_corrupt_entry_is_a_miss_and_is_replaced(wav, counter):
    locate.detect_cached(str(wav), plots=False)
    (f,) = evcache._dir().glob("*.json")
    f.write_bytes(f.read_bytes()[:50])  # truncated
    ev, *_, info = locate.detect_cached(str(wav), plots=False)
    assert counter["n"] == 2 and not info["hit"] and ev
    assert locate.detect_cached(str(wav), plots=False)[-1]["hit"]
    d = json.loads(f.read_text())
    d["key_material"]["impl"] = "x"  # entry written under another key material
    f.write_text(json.dumps(d))
    assert not locate.detect_cached(str(wav), plots=False)[-1]["hit"]


def test_prune_bounds_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("DOPPLER_GEO_CACHE", str(tmp_path / "c"))
    d = evcache._dir()
    for i in range(5):
        (d / f"{i}.json").write_text("{}")
    evcache.prune(d, max_entries=2)
    assert len(list(d.glob("*.json"))) == 2


def test_new_region_or_date_reuses_detection(wav, counter, tmp_path):
    for i, (date, region) in enumerate(
        [("2026-09-21", "43.8,-79.3,8"), ("2026-09-22", "43.9,-79.2,5")]
    ):
        rep = locate.locate(
            str(wav),
            date,
            region,
            offline=True,
            out_dir=str(tmp_path / f"o{i}"),
            plots=False,
        )
        assert "no cached ADS-B crop" in rep["error"]
        assert rep["detection_cache"]["hit"] == (i == 1)
    assert counter["n"] == 1
    assert not (tmp_path / "o1" / "detection.png").exists()


def test_missing_inputs_fail_before_audio_is_read(wav, monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("audio must not be decoded")

    monkeypatch.setattr(detect, "load_audio", boom)
    rep = locate.locate(str(wav), "2026-09-21", None, out_dir=str(tmp_path / "o"))
    assert "region_hint" in rep["error"]
    rep = locate.locate(str(wav), None, "43.8,-79.3,8", out_dir=str(tmp_path / "o"))
    assert "date_hint" in rep["error"]
