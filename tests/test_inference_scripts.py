"""Test src/detect.py, src/track.py: imgsz lay theo resolve_imgsz va duoc ghi
ra <output>.meta.json. Khong chay YOLO that - thay iter_* bang ham gia."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.detect as detect  # noqa: E402
import src.track as track  # noqa: E402


def _fake_iter(key: str, seen: dict):
    def fake(weights, source, **kwargs):
        seen.update(kwargs)
        yield {"frame_idx": 0, key: []}

    return fake


@pytest.fixture
def train_config(tmp_path: Path) -> Path:
    path = tmp_path / "train_config.yaml"
    path.write_text("common:\n  imgsz: 960\ninference:\n  conf: 0.25\n", encoding="utf-8")
    return path


CASES = [
    (detect, "iter_detections", "detections", []),
    (track, "iter_tracks", "tracks", ["--tracker", str(REPO_ROOT / "configs" / "botsort.yaml")]),
]


@pytest.mark.parametrize("module,func,key,extra", CASES)
def test_default_imgsz_from_train_config(tmp_path, train_config, monkeypatch, module, func, key, extra):
    seen: dict = {}
    monkeypatch.setattr(module, func, _fake_iter(key, seen))
    out = tmp_path / "logs" / "run.jsonl"
    argv = ["--weights", "yolo11n-seg.pt", "--source", "clip.mp4", "--output", str(out),
            "--train-config", str(train_config), *extra]
    assert module.main(argv) == 0

    assert seen["imgsz"] == 960
    meta = json.loads((tmp_path / "logs" / "run.meta.json").read_text(encoding="utf-8"))
    assert meta["imgsz"] == 960
    assert meta["imgsz_source"] == f"train_config:{train_config}"
    assert meta["weights"] == "yolo11n-seg.pt"
    # conf lay tu inference.conf, ghi kem nguon
    assert seen["conf"] == 0.25
    assert (meta["conf"], meta["conf_source"]) == (0.25, f"train_config:{train_config}")


@pytest.mark.parametrize("module,func,key,extra", CASES)
def test_cli_conf_override_recorded(tmp_path, train_config, monkeypatch, module, func, key, extra):
    seen: dict = {}
    monkeypatch.setattr(module, func, _fake_iter(key, seen))
    out = tmp_path / "b.jsonl"
    argv = ["--weights", "w.pt", "--source", "clip.mp4", "--output", str(out),
            "--train-config", str(train_config), "--conf", "0.4", *extra]
    assert module.main(argv) == 0
    meta = json.loads((tmp_path / "b.meta.json").read_text(encoding="utf-8"))
    assert seen["conf"] == 0.4
    assert (meta["conf"], meta["conf_source"], meta["config_conf"]) == (0.4, "cli", 0.25)


@pytest.mark.parametrize("module,func,key,extra", CASES)
def test_cli_and_experiment_dir(tmp_path, train_config, monkeypatch, module, func, key, extra):
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "config.yaml").write_text("detector:\n  imgsz: 800\n", encoding="utf-8")
    seen: dict = {}
    monkeypatch.setattr(module, func, _fake_iter(key, seen))
    base = ["--weights", "w.pt", "--source", "clip.mp4", "--output", str(tmp_path / "a.jsonl"),
            "--train-config", str(train_config), "--experiment-dir", str(exp), *extra]

    assert module.main(base) == 0
    assert seen["imgsz"] == 800

    assert module.main(base + ["--imgsz", "1280"]) == 0
    assert seen["imgsz"] == 1280
    meta = json.loads((tmp_path / "a.meta.json").read_text(encoding="utf-8"))
    assert (meta["imgsz"], meta["imgsz_source"]) == (1280, "cli")


@pytest.mark.parametrize("module", [detect, track])
def test_no_hardcoded_imgsz_default(module):
    """iter_* khong con mac dinh imgsz; CLI mac dinh None."""
    import inspect

    func = module.iter_detections if module is detect else module.iter_tracks
    assert inspect.signature(func).parameters["imgsz"].default is inspect.Parameter.empty
    args = module.parse_args(["--weights", "w", "--source", "s", "--output", "o.jsonl"])
    assert args.imgsz is None
    assert args.conf is None
    assert inspect.signature(func).parameters["conf"].default is inspect.Parameter.empty
