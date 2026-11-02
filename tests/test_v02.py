"""v0.2.0: automatic detection, ADS-B-geometry scoring, locate(), tools. Offline.

Network tests run only with DOPPLER_GEO_NETWORK_TESTS=1.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dopplergeo import adsb, detect, locate, metadata, score, synth
from dopplergeo.physics import R_E, speed_of_sound

SR = 4000
LAT0, LON0 = 43.80, -79.30
T0 = 1_790_000_000.0  # 2026-09-21T13:46:40Z
NET = os.environ.get("DOPPLER_GEO_NETWORK_TESTS") == "1"
HAS_FFMPEG = shutil.which("ffmpeg") is not None


def observer(dn=700.0, de=-400.0, h=150.0):
    return (
        LAT0 + np.degrees(dn / R_E),
        LON0 + np.degrees(de / (R_E * np.cos(np.radians(LAT0)))),
        h,
    )


def scenario(turn=0.3, dur=420, noise=1.0, seed=0, crossing=False):
    """Two piston flyovers (A turning) and a jet the detector must not need."""
    t = np.arange(T0 - 300, T0 + 800, 0.5)
    A = synth.track(
        t,
        LAT0,
        LON0,
        speed=65,
        heading_deg=80,
        turn_deg_s=turn,
        alt_m=750,
        through=(-400, 0),
        t_through=T0 + 300,
    )
    B = synth.track(
        t,
        LAT0,
        LON0,
        speed=85,
        heading_deg=200,
        alt_m=1050,
        through=(-900, 1500),
        t_through=T0 + 150,
    )
    J = synth.track(
        t,
        LAT0,
        LON0,
        speed=120,
        heading_deg=10,
        alt_m=1500,
        through=(0, 0),
        t_through=T0 + 250,
    )
    fam_a = [(110, 1.0), (220, 0.6), (330, 0.4)]
    if (
        crossing
    ):  # engine family at a non-integer ratio: its lines cross the propeller lines
        fam_a = fam_a + [(77, 0.8), (154, 0.6), (231, 0.5)]
    obs = observer()
    x = synth.render(
        T0,
        dur,
        SR,
        obs,
        [(t, *A), (t, *B)],
        [fam_a, [(92, 1.0), (184, 0.6)]],
        noise=noise,
        seed=seed,
    )
    tracks = [
        synth.as_adsb("aaa001", t, *A),
        synth.as_adsb("bbb002", t, *B),
        synth.as_adsb("ccc003", t, *J, cls="jet", typ="B738"),
    ]
    return x, tracks, obs


def err_m(lat, lon, obs):
    x, y = score.to_enu(lat, lon, obs[0], obs[1])
    return float(np.hypot(x, y))


class DetectionTests(unittest.TestCase):
    def test_finds_both_passes_without_bands(self):
        x, _, _ = scenario()
        ev, _ = detect.detect_events(x, SR)
        self.assertEqual(len(ev), 2, [e.to_dict() for e in ev])
        b, a = ev
        self.assertAlmostEqual(b.t_cpa, 160, delta=6)
        self.assertAlmostEqual(b.speed_mps, 85, delta=8)
        self.assertAlmostEqual(a.t_cpa, 304, delta=6)
        self.assertFalse(
            a.straight_fit_ok
        )  # turning: the straight model is only a proposal

    def test_crossing_harmonic_families_give_one_event_per_pass(self):
        x, _, _ = scenario(crossing=True)
        ev, _ = detect.detect_events(x, SR)
        self.assertEqual(len(ev), 2, [e.to_dict() for e in ev])

    def test_white_noise_and_broadband_roar_are_rejected(self):
        rng = np.random.default_rng(5)
        noise = rng.normal(0, 1, SR * 120)
        ev, _ = detect.detect_events(noise, SR)
        self.assertEqual(ev, [])
        env = np.exp(-(((np.arange(noise.size) / SR) - 60) ** 2) / (2 * 15**2))
        from scipy import signal

        b, a = signal.butter(2, [80, 900], btype="band", fs=SR)
        roar = (
            signal.lfilter(b, a, rng.normal(0, 1, noise.size)) * (0.2 + 3 * env)
            + 0.3 * noise
        )
        ev, _ = detect.detect_events(roar, SR)
        self.assertEqual(ev, [])

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
    def test_survives_aac_and_opus_round_trips(self):
        x, _, _ = scenario()
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "s.wav"
            sf.write(src, (x * 0.9).astype(np.float32), SR)
            src48 = Path(d) / "s48.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(src),
                    "-ar",
                    "48000",
                    str(src48),
                ],
                check=True,
            )
            for codec, ext in (("aac", "m4a"), ("libopus", "opus")):
                out = Path(d) / f"c.{ext}"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(src48),
                        "-c:a",
                        codec,
                        "-b:a",
                        "96k",
                        str(out),
                    ],
                    check=True,
                )
                y, sr = detect.load_audio(str(out))
                ev, _ = detect.detect_events(y, sr)
                self.assertGreaterEqual(
                    len(ev), 2, f"{codec}: {[e.to_dict() for e in ev]}"
                )


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.tracks, cls.obs = scenario()
        cls.events, _ = detect.detect_events(cls.x, SR)
        cls.planes = score.prepare_planes(cls.tracks, LAT0, LON0)

    def test_jets_are_dropped_by_default(self):
        self.assertEqual({p.hex for p in self.planes}, {"aaa001", "bbb002"})

    def _search(self, t0w):
        return score.search(
            self.events,
            self.planes,
            LAT0,
            LON0,
            6000,
            lambda la, lo: np.full(len(la), 150.0),
            343.0,
            score.Options(),
            t0_window=t0w,
            clip_dur=420,
        )

    def test_unknown_time_recovers_place_and_clock(self):
        r = self._search(None)
        b = r["best"]
        self.assertLess(err_m(b["lat"], b["lon"], self.obs), 150)
        self.assertAlmostEqual(b["t0_unix"], T0, delta=3)
        self.assertEqual({m["hex"] for m in b["matches"]}, {"aaa001", "bbb002"})

    def test_known_time_and_credible_area(self):
        r = self._search((T0 - 60, T0 + 60))
        self.assertLess(err_m(r["best"]["lat"], r["best"]["lon"], self.obs), 150)
        self.assertLess(r["levels"][1]["a95_m2"], 5e6)

    def test_silence_penalty_charges_unheard_close_pass(self):
        g = np.array([[*score.to_enu(self.obs[0], self.obs[1], LAT0, LON0), 151.5]])
        on = score.Options(detect_prob=0.9)
        off = score.Options(silence_penalty=0.0)
        with_pen = score.score_points(
            self.events[:1], self.planes, g, 343.0, on, None, 420
        ).logl[0]
        no_pen = score.score_points(
            self.events[:1], self.planes, g, 343.0, off, None, 420
        ).logl[0]
        # only event B kept: plane A's close pass inside the clip is unexplained and costs
        self.assertLess(with_pen, no_pen - 0.3)
        # Bernoulli silence: a lower detection probability costs less
        lo = score.score_points(
            self.events[:1],
            self.planes,
            g,
            343.0,
            score.Options(detect_prob=0.05),
            None,
            420,
        ).logl[0]
        self.assertGreater(lo, with_pen)
        self.assertLess(lo, no_pen)

    def test_likelihood_is_tempered_by_correlation_time(self):
        ev = self.events[0]
        o = score.Options()
        w = score.event_weight(ev, o)
        self.assertLessEqual(w, o.hop_over_win)
        span = ev.times[-1] - ev.times[0]
        self.assertAlmostEqual(
            w * len(ev.times), min(o.n_eff_max, max(3.0, span / o.corr_s)), delta=1e-6
        )
        self.assertGreater(score.event_sigma(ev, o), o.sigma_model - 1e-12)

    def test_rest_frequency_scale_at_bound_rejects_pass(self):
        g = np.array([[*score.to_enu(self.obs[0], self.obs[1], LAT0, LON0), 151.5]])
        tight = score.Options(scale_bounds=(0.99, 1.0))
        import copy

        ev = copy.deepcopy(self.events[0])
        ev.ratio = list(np.asarray(ev.ratio) * 1.2)  # far outside the tight bounds
        r = score.score_points([ev], self.planes, g, 343.0, tight, None, 420)
        self.assertFalse(r.matches[0])


class LocateTests(unittest.TestCase):
    def test_locate_end_to_end_with_injected_tracks(self):
        x, tracks, obs = scenario()
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "clip.wav"
            sf.write(wav, x.astype(np.float32), SR)
            rep = locate.locate(
                str(wav),
                date_hint=dt.datetime.fromtimestamp(T0, dt.UTC).date().isoformat(),
                region_hint=f"{LAT0},{LON0},6",
                offline=True,
                out_dir=d,
                tracks=tracks,
                elevation_fn=lambda la, lo: np.full(len(la), 150.0),
            )
            self.assertTrue(rep["ok"], rep.get("error"))
            self.assertLess(err_m(rep["best"]["lat"], rep["best"]["lon"], obs), 150)
            self.assertTrue(Path(rep["files"]["heatmap_png"]).stat().st_size > 10000)
            self.assertTrue(json.loads((Path(d) / "report.json").read_text())["ok"])

    def test_errors_are_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / "n.wav"
            sf.write(
                wav,
                np.random.default_rng(1).normal(0, 0.1, SR * 30).astype(np.float32),
                SR,
            )
            rep = locate.locate(
                str(wav), "2026-09-21", "43.8,-79.3,5", offline=True, out_dir=d
            )
            self.assertFalse(rep["ok"])
            self.assertIn("no aircraft", rep["error"])


class AdsbTests(unittest.TestCase):
    def _fake_release(self, d):
        """A split tar laid out like adsb.lol globe_history (traces/xx/trace_full_<hex>.json, gzip)."""
        day0 = dt.datetime(2026, 9, 21, tzinfo=dt.UTC).timestamp()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for hx, lat, typ in (("a1b2c3", 43.8, "C172"), ("d4e5f6", 10.0, "B738")):
                trace = [
                    [
                        600.0 + i * 5,
                        lat + i * 1e-4,
                        -79.3,
                        2000 + i,
                        90.0,
                        90.0,
                        0,
                        0,
                        None,
                        "adsb_icao",
                        2100 + i,
                    ]
                    for i in range(40)
                ]
                trace.append(
                    [900.0, lat, -79.3, "ground", 5.0, 0, 0, 0, None, "adsb_icao", None]
                )
                data = gzip.compress(
                    json.dumps(
                        {
                            "icao": hx,
                            "t": typ,
                            "r": "C-TEST",
                            "timestamp": day0,
                            "trace": trace,
                        }
                    ).encode()
                )
                ti = tarfile.TarInfo(f"./traces/{hx[-2:]}/trace_full_{hx}.json")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
        raw = buf.getvalue()
        parts = []
        for i, chunk in enumerate((raw[: len(raw) // 2], raw[len(raw) // 2 :])):
            p = Path(d) / f"x.tar.a{'ab'[i]}"
            p.write_bytes(chunk)
            parts.append(str(p))
        return parts

    def test_stream_crop_and_cache(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["DOPPLER_GEO_CACHE"] = d
            try:
                parts = self._fake_release(d)
                day = dt.date(2026, 9, 21)
                planes = adsb.fetch_tracks(
                    day, (43.5, -79.6, 44.1, -79.0), offline=True, tar_parts=parts
                )
                self.assertEqual([p["hex"] for p in planes], ["a1b2c3"])
                self.assertEqual(len(planes[0]["t"]), 40)  # ground point dropped
                self.assertAlmostEqual(
                    planes[0]["alt_m"][0], 2100 * 0.3048, places=3
                )  # geometric altitude preferred
                again = adsb.fetch_tracks(
                    day, (43.6, -79.5, 44.0, -79.1), offline=True
                )  # served from cache
                self.assertEqual(len(again), 1)
            finally:
                os.environ.pop("DOPPLER_GEO_CACHE", None)

    @unittest.skipUnless(NET, "set DOPPLER_GEO_NETWORK_TESTS=1")
    def test_live_release_assets_exist(self):
        urls = adsb.release_assets(dt.date(2026, 11, 1))
        self.assertTrue(all(".tar.a" in u for u in urls))


class EnvAndMetadataTests(unittest.TestCase):
    def test_speed_of_sound(self):
        self.assertAlmostEqual(speed_of_sound(20.0), 343.2, delta=0.2)

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg not installed")
    def test_creation_time_and_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "a.wav"
            sf.write(src, np.zeros(SR), SR)
            m4a = Path(d) / "clip.m4a"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(src),
                    "-metadata",
                    "creation_time=2026-09-21T13:46:40Z",
                    "-c:a",
                    "aac",
                    str(m4a),
                ],
                check=True,
            )
            (Path(d) / "clip.info.json").write_text(
                json.dumps({"upload_date": "20260922"})
            )
            h = metadata.hints(m4a)
            self.assertEqual(
                h["creation_time"].isoformat(), "2026-09-21T13:46:40+00:00"
            )
            self.assertEqual(str(h["upload_date"]), "2026-09-22")

    @unittest.skipUnless(NET, "set DOPPLER_GEO_NETWORK_TESTS=1")
    def test_live_dem_and_weather(self):
        from dopplergeo import env

        z, src = env.elevation([43.65], [-79.38])
        self.assertEqual(src, "copernicus-glo30")
        self.assertTrue(50 < z[0] < 200)
        self.assertIsNotNone(
            env.temperature_c(
                43.65, -79.38, dt.datetime(2026, 9, 21, 14, tzinfo=dt.UTC)
            )
        )


class ToolTests(unittest.TestCase):
    def test_register_skill_and_tools(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "plugin_entry", ROOT / "__init__.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        seen = {"skills": [], "tools": []}

        class Ctx:
            def register_skill(self, name, path):
                seen["skills"].append(name)

            def register_tool(self, name, toolset, schema, handler, description=""):
                seen["tools"].append(name)
                assert schema["name"] == name and callable(handler)

        mod.register(Ctx())
        self.assertEqual(seen["skills"], ["aircraft-doppler-geolocation"])
        self.assertEqual(seen["tools"], ["doppler_detect", "doppler_locate"])
        manifest = (ROOT / "plugin.yaml").read_text()
        for n in seen["tools"]:
            self.assertIn(f"  - {n}", manifest)

    def test_tool_handlers_return_json_errors(self):
        from dopplergeo.tools import detect_tool, locate_tool

        self.assertFalse(json.loads(detect_tool({}))["ok"])
        self.assertFalse(
            json.loads(locate_tool({"recording": "/nonexistent.wav"}))["ok"]
        )


if __name__ == "__main__":
    unittest.main()


class DaysSpannedTests(unittest.TestCase):
    def test_middle_day_is_included(self):
        a = dt.datetime(2026, 10, 4, 23, 50, tzinfo=dt.UTC).timestamp()
        b = dt.datetime(2026, 10, 6, 0, 10, tzinfo=dt.UTC).timestamp()
        self.assertEqual(
            locate.days_spanned(a, b),
            [dt.date(2026, 10, 4), dt.date(2026, 10, 5), dt.date(2026, 10, 6)],
        )

    def test_single_day(self):
        a = dt.datetime(2026, 10, 5, 0, 0, tzinfo=dt.UTC).timestamp()
        self.assertEqual(locate.days_spanned(a, a + 86399), [dt.date(2026, 10, 5)])
