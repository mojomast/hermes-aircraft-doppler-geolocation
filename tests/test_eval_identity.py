"""Eval harness: resume reuses a row only when code, audio and settings match."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_spec = importlib.util.spec_from_file_location(
    "eval_batch", Path(__file__).resolve().parents[1] / "scripts" / "eval_batch.py"
)
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)


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


def test_run_key_covers_every_input(tmp_path, monkeypatch):
    e = _entry(tmp_path)
    k = eb.run_key(e, 30, {}, {}, 15, 5)
    assert k == eb.run_key(e, 30.0, {}, {}, 15.0, 5.0)
    variants = [
        eb.run_key(e, 600, {}, {}, 15, 5),
        eb.run_key(e, 30, {"sigma_model": 0.01}, {}, 15, 5),
        eb.run_key(e, 30, {}, {"broadband": True}, 15, 5),
        eb.run_key(e, 30, {}, {}, 20, 5),
        eb.run_key(e, 30, {}, {}, 15, 3),
        eb.run_key({**e, "t0_unix": 1.7e9 + 1}, 30, {}, {}, 15, 5),
    ]
    monkeypatch.setattr(eb, "_code_hash", lambda: "other-code")
    variants.append(eb.run_key(e, 30, {}, {}, 15, 5))
    monkeypatch.undo()
    eb._audio_hash.clear()
    sf.write(e["audio"], np.zeros(4000 * 20), 4000)  # same path, new audio
    variants.append(eb.run_key(e, 30, {}, {}, 15, 5))
    assert len({k, *variants}) == len(variants) + 1


def test_resume_drops_rows_with_another_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("DOPPLER_GEO_CACHE", str(tmp_path / "cache"))
    e = _entry(tmp_path)
    man = tmp_path / "m.json"
    man.write_text(json.dumps([e]))
    out = tmp_path / "out"
    args = [
        str(man),
        "--out",
        str(out),
        "--tol",
        "30",
        "--jobs",
        "1",
        "--det-jobs",
        "1",
    ]
    script = Path(eb.__file__)
    env = {**os.environ, "DOPPLER_GEO_CACHE": str(tmp_path / "cache")}

    def run(extra=()):
        subprocess.run(
            [sys.executable, str(script), *args, *extra],
            check=True,
            env=env,
            capture_output=True,
        )

    run()
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["run_key"]
    # tamper: pretend the row came from other settings -> must be recomputed
    bad = {**rows[0], "run_key": "stale", "error": "from another run"}
    (out / "rows.jsonl").write_text(json.dumps(bad) + "\n")
    run(["--resume"])
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["run_key"] != "stale"
    # same identity -> kept verbatim
    keep = {**rows[0], "marker": 1}
    (out / "rows.jsonl").write_text(json.dumps(keep) + "\n")
    run(["--resume"])
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    assert rows == [keep]


def test_auto_jobs_respects_memory():
    assert eb.auto_jobs(1e6) == 1
    assert 1 <= eb.auto_jobs(0.5, cpu=3) <= 3
