"""Test scripts/run_pipeline.py: imgsz da dung duoc ghi vao run_context.json.
Khong chay YOLO that - thay iter_tracks bang ham gia."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_pipeline as rp  # noqa: E402


def _run(tmp_path: Path, monkeypatch, *extra: str) -> tuple[dict, dict]:
    seen: dict = {}

    def fake_iter_tracks(weights, source, **kwargs):
        seen.update(kwargs)
        yield {"frame_idx": 0, "tracks": [{"track_id": 1, "bbox": [0, 0, 10, 10], "conf": 0.9, "cls": 0}]}
        yield {"frame_idx": 1, "tracks": []}

    monkeypatch.setattr(rp, "iter_tracks", fake_iter_tracks)
    train_config = tmp_path / "train_config.yaml"
    train_config.write_text("common:\n  imgsz: 960\n", encoding="utf-8")
    argv = [
        "--weights", "w.pt", "--source", "scn01.mp4", "--experiment-id", "exp",
        "--target-track-id", "1", "--tracker", str(REPO_ROOT / "configs" / "botsort.yaml"),
        "--output-root", str(tmp_path / "experiments"), "--train-config", str(train_config),
        *extra,
    ]
    assert rp.main(argv) == 0
    context = json.loads((tmp_path / "experiments" / "exp" / "run_context.json").read_text())
    return context, seen


def test_run_context_records_resolved_imgsz(tmp_path: Path, monkeypatch):
    context, seen = _run(tmp_path, monkeypatch)
    assert seen["imgsz"] == 960
    assert context["config"]["imgsz"] == 960
    assert context["inference"]["imgsz"] == 960
    assert context["inference"]["imgsz_source"].startswith("train_config:")


def test_experiment_config_and_cli(tmp_path: Path, monkeypatch):
    exp = tmp_path / "experiments" / "exp"
    exp.mkdir(parents=True)
    (exp / "config.yaml").write_text("detector:\n  imgsz: 800\n", encoding="utf-8")
    context, seen = _run(tmp_path, monkeypatch)
    assert seen["imgsz"] == 800 and context["inference"]["imgsz_source"].startswith("experiment_config:")

    context, seen = _run(tmp_path, monkeypatch, "--imgsz", "1280")
    assert seen["imgsz"] == 1280 and context["inference"]["imgsz_source"] == "cli"


def test_cli_default_imgsz_is_none():
    args = rp.parse_args(["--weights", "w", "--source", "s", "--experiment-id", "e",
                          "--target-track-id", "1"])
    assert args.imgsz is None
