import json

import numpy as np
import soundfile as sf

from dopplergeo import __main__ as cli
from dopplergeo import locate, synth, tools

SR = 4000
LAT0, LON0 = 53.44, -2.10


def _capture(monkeypatch):
    seen = {}

    def fake_locate(*a, **k):
        seen["locate"] = (a, k)
        return {"ok": True}

    def fake_detect(*a, **k):
        seen["detect"] = (a, k)
        return {"ok": True, "events": []}

    monkeypatch.setattr(locate, "locate", fake_locate)
    monkeypatch.setattr(locate, "detect_only", fake_detect)
    return seen


def test_cli_passes_broadband_and_temper_extra(monkeypatch, capsys):
    seen = _capture(monkeypatch)
    assert cli.main(["detect", "x.wav", "--broadband"]) == 0
    assert seen["detect"][1]["broadband"] is True
    argv = ["locate", "x.wav", "--date", "2026-09-21", "--region", "1,2,3"]
    assert cli.main([*argv, "--broadband", "--temper-extra", "8"]) == 0
    k = seen["locate"][1]
    assert k["broadband"] is True and k["temper_extra"] == 8.0
    assert cli.main(argv) == 0
    k = seen["locate"][1]
    assert k["broadband"] is False and k["temper_extra"] is None
    capsys.readouterr()


def test_tools_expose_and_pass_broadband(monkeypatch):
    seen = _capture(monkeypatch)
    specs = {
        s["name"]: s["schema"]["parameters"]["properties"] for s in tools.tool_specs()
    }
    assert "broadband" in specs["doppler_detect"]
    assert {"broadband", "temper_extra"} <= set(specs["doppler_locate"])
    tools.detect_tool({"recording": "x.wav", "broadband": True})
    assert seen["detect"][1]["broadband"] is True
    tools.locate_tool({"recording": "x.wav", "broadband": True, "temper_extra": 2})
    k = seen["locate"][1]
    assert k["broadband"] is True and k["temper_extra"] == 2.0
    tools.locate_tool({"recording": "x.wav"})
    assert seen["locate"][1]["temper_extra"] is None


def test_detect_only_broadband_finds_synthetic_jet_pass(tmp_path):
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
    x = synth.render_broadband(0, 200, SR, (LAT0, LON0, 0.0), [(t, la, lo, al)])
    wav = tmp_path / "bb.wav"
    sf.write(wav, x.astype(np.float32), SR)
    off = locate.detect_only(str(wav), str(tmp_path / "a"))
    on = locate.detect_only(str(wav), str(tmp_path / "b"), broadband=True)
    assert not any(e["kind"] == "broadband" for e in off["events"])
    assert [e["kind"] for e in on["events"]] == ["broadband"]
    json.dumps(on)


def test_offline_error_has_plain_numbers(tmp_path, monkeypatch):
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
    wav = tmp_path / "t.wav"
    sf.write(wav, x.astype(np.float32), SR)
    rep = locate.locate(
        str(wav),
        "2026-09-21",
        "43.8,-79.3,8",
        offline=True,
        out_dir=str(tmp_path / "o"),
    )
    assert not rep["ok"]
    assert "no cached ADS-B crop" in rep["error"]
    assert "np.float64" not in rep["error"]
