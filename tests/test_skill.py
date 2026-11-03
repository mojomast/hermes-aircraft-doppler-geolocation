"""Prove the skill file, the plugin entrypoint, and the Doppler fit + scoring offline."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "aircraft-doppler-geolocation"
SCRIPTS = SKILL / "scripts"
sys.path.insert(0, str(SCRIPTS))

import doppler_track
import make_synthetic_test
import score_candidates


def frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "missing frontmatter"
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return {"_raw": m.group(1)}
    return yaml.safe_load(m.group(1))


class SkillFileTests(unittest.TestCase):
    def test_frontmatter_and_sections(self):
        text = (SKILL / "SKILL.md").read_text()
        fm = frontmatter(text)
        if "_raw" not in fm:
            self.assertEqual(fm["name"], "aircraft-doppler-geolocation")
            self.assertIn("Use when", fm["description"])
            self.assertEqual(str(fm["version"]), "0.1.1")
        for h in [
            "## When to Use",
            "## Prerequisites",
            "## How to Run",
            "## Procedure",
            "## Pitfalls",
            "## Verification",
            "## Limitations and Ethics",
        ]:
            self.assertIn(h, text)

    def test_versions_agree(self):
        manifest = (ROOT / "plugin.yaml").read_text()
        pyproject = (ROOT / "pyproject.toml").read_text()
        self.assertIn('version: "0.1.1"', manifest)
        self.assertIn('version = "0.1.1"', pyproject)


class PhysicsTests(unittest.TestCase):
    def test_model_fit_recovers_parameters(self):
        t = np.arange(0, 120, 0.25)
        f = doppler_track.doppler_model(t, 120.0, 70.0, 800.0, 60.0)
        f = f + np.random.default_rng(1).normal(0, 0.2, t.size)
        p, _, rms = doppler_track.fit(t, f)
        self.assertAlmostEqual(p[0], 120.0, delta=0.5)
        self.assertAlmostEqual(p[1], 70.0, delta=3)
        self.assertAlmostEqual(p[2], 800.0, delta=80)
        self.assertLess(rms, 0.5)


class DetectionGateTests(unittest.TestCase):
    def test_pure_noise_is_not_reported_as_a_flyover(self):
        import json
        import subprocess

        import soundfile as sf

        with tempfile.TemporaryDirectory() as d:
            sf.write(
                f"{d}/noise.wav",
                np.random.default_rng(3).normal(0, 0.1, 4000 * 30),
                4000,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "doppler_track.py"),
                    f"{d}/noise.wav",
                    "--out",
                    f"{d}/n",
                ],
                check=True,
                capture_output=True,
            )
            with open(f"{d}/n.json") as fh:
                res = json.load(fh)
        self.assertFalse(res["detected"])
        self.assertTrue(res["rejected_because"])


class TrackSmoothingTests(unittest.TestCase):
    def test_smoothing_removes_doppler_noise_from_jittery_adsb(self):
        # straight 70 m/s pass at 900 m; 1 Hz fixes with ~25 m position jitter
        t = np.arange(0, 200, 1.0)
        lat = 40.0 + np.zeros_like(t)
        lon = -75.0 + np.degrees(
            (70 * (t - 100)) / (score_candidates.R_E * np.cos(np.radians(40)))
        )
        clean = np.c_[t, lat, lon, np.full_like(t, 900.0)]
        noisy = clean.copy()
        noisy[:, 2] += np.degrees(
            np.random.default_rng(0).normal(0, 25, t.size)
            / (score_candidates.R_E * np.cos(np.radians(40)))
        )
        cand = (40.0 + np.degrees(600 / score_candidates.R_E), -75.0, 0.0)
        tc, fc, _, _ = score_candidates.received_curve(clean, cand, 120.0)

        def err(smooth_s):
            tn, fn, _, _ = score_candidates.received_curve(
                noisy, cand, 120.0, smooth_s=smooth_s
            )
            m = (tn > 60) & (tn < 140)
            return np.sqrt(np.mean((fn[m] - np.interp(tn[m], tc, fc)) ** 2))

        self.assertGreater(err(0), 2.0)
        self.assertLess(err(10), err(0) / 3)


class EndToEndTests(unittest.TestCase):
    def test_synthetic_flyover_is_located(self):
        with tempfile.TemporaryDirectory() as d:
            r = make_synthetic_test.run_selftest(
                d, grid_half_m=1200, grid_step_m=200, verbose=False
            )
        self.assertAlmostEqual(r["f0_hz"], 110.0, delta=2)
        self.assertAlmostEqual(r["speed_mps"], 80.0, delta=5)
        self.assertAlmostEqual(r["cpa_slant_m"], r["true_slant_m"], delta=100)
        self.assertLessEqual(r["location_error_m"], 300)


class PluginTests(unittest.TestCase):
    def test_register_discovers_the_skill(self):
        spec = importlib.util.spec_from_file_location(
            "plugin_entry", ROOT / "__init__.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        seen = []

        class Ctx:
            def register_skill(self, name, path):
                seen.append((name, Path(path)))

        module.register(Ctx())
        self.assertEqual([n for n, _ in seen], ["aircraft-doppler-geolocation"])
        self.assertTrue(seen[0][1].is_file())


if __name__ == "__main__":
    unittest.main()
