from pathlib import Path

from dopplergeo.cache import atomic_write_bytes


def test_atomic_write_replaces_and_leaves_no_temp(tmp_path: Path):
    p = tmp_path / "tile.tif"
    p.write_bytes(b"old")
    atomic_write_bytes(p, b"new-content")
    assert p.read_bytes() == b"new-content"
    assert [x.name for x in tmp_path.iterdir()] == ["tile.tif"]


def test_atomic_write_creates_file(tmp_path: Path):
    p = tmp_path / "x.json"
    atomic_write_bytes(p, b"{}")
    assert p.read_bytes() == b"{}"
