import json

from dopplergeo import tools


def test_with_media_lists_existing_pngs(tmp_path):
    det = tmp_path / "detection.png"
    heat = tmp_path / "heatmap.png"
    det.write_bytes(b"png")
    heat.write_bytes(b"png")
    rep = {
        "files": {
            "heatmap_png": str(heat),
            "detection_png": str(det),
            "report_json": str(tmp_path / "report.json"),
        }
    }
    out = tools._with_media(rep)
    assert out["generated_image"] == [str(det), str(heat)]
    assert out["media"] == [f"MEDIA:{det}", f"MEDIA:{heat}"]
    assert "MEDIA:" in out["media_hint"]


def test_with_media_skips_missing_and_no_plots(tmp_path):
    rep = {"files": {"detection_png": str(tmp_path / "missing.png")}}
    out = tools._with_media(rep)
    assert "media" not in out and "generated_image" not in out
    assert tools._with_media({"ok": False, "error": "x"}) == {"ok": False, "error": "x"}


def test_detect_tool_error_has_no_media():
    out = json.loads(tools.detect_tool({"recording": "/nonexistent.wav"}))
    assert "media" not in out
