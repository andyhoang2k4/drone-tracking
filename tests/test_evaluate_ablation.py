"""Test kiem tra dieu kien ablation co/khong Motion Memory trong src/evaluate.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import ablation_mismatches, load_pipeline_settings  # noqa: E402

BASE_CONFIG = {
    "weights": "experiments/yolo11_baseline_001/train/weights/best.pt",
    "source": "data/occlusion_test_scenarios/scn01.mp4",
    "tracker": "configs/botsort.yaml",
    "conf": 0.25,
    "imgsz": 960,
    "classes": None,
    "motion_memory": False,
}


def _write_run(tmp_path: Path, name: str, inference_imgsz=960, **overrides) -> Path:
    exp = tmp_path / name
    exp.mkdir()
    context = {"experiment_id": name, "config": {**BASE_CONFIG, **overrides}}
    if inference_imgsz is not None:
        context["inference"] = {"imgsz": inference_imgsz, "imgsz_source": "cli"}
    (exp / "run_context.json").write_text(json.dumps(context), encoding="utf-8")
    return exp


def test_fair_ablation_has_no_mismatch(tmp_path: Path):
    base = load_pipeline_settings(_write_run(tmp_path, "base"))
    mm = load_pipeline_settings(_write_run(tmp_path, "mm", motion_memory=True))
    assert base["imgsz"] == 960
    assert ablation_mismatches(base, mm) == []


def test_different_imgsz_is_reported(tmp_path: Path):
    base = load_pipeline_settings(_write_run(tmp_path, "base"))
    mm = load_pipeline_settings(
        _write_run(tmp_path, "mm", inference_imgsz=640, imgsz=640, motion_memory=True)
    )
    problems = ablation_mismatches(base, mm)
    assert len(problems) == 1 and problems[0].startswith("imgsz khac nhau")


def test_inference_imgsz_preferred_over_config(tmp_path: Path):
    settings = load_pipeline_settings(_write_run(tmp_path, "r", inference_imgsz=1280, imgsz=None))
    assert settings["imgsz"] == 1280


def test_old_run_without_imgsz(tmp_path: Path):
    base = load_pipeline_settings(_write_run(tmp_path, "base", inference_imgsz=None, imgsz=None))
    mm = load_pipeline_settings(
        _write_run(tmp_path, "mm", inference_imgsz=None, imgsz=None, motion_memory=True)
    )
    assert any("thieu imgsz" in p for p in ablation_mismatches(base, mm))


def test_motion_memory_flags_must_differ(tmp_path: Path):
    base = load_pipeline_settings(_write_run(tmp_path, "base"))
    other = load_pipeline_settings(_write_run(tmp_path, "other"))
    assert any("motion_memory" in p for p in ablation_mismatches(base, other))


def test_other_setting_mismatch(tmp_path: Path):
    base = load_pipeline_settings(_write_run(tmp_path, "base"))
    mm = load_pipeline_settings(_write_run(tmp_path, "mm", conf=0.4, motion_memory=True))
    assert ablation_mismatches(base, mm) == ["conf khac nhau: 0.25 vs 0.4"]


def test_with_real_run_pipeline_output(tmp_path: Path, monkeypatch):
    """run_context that do run_pipeline.py ghi -> load + so sanh khong loi."""
    import scripts.run_pipeline as rp
    from src.motion_memory import MotionMemory

    def fake_iter_tracks(weights, source, **kwargs):
        yield {"frame_idx": 0, "tracks": [{"track_id": 1, "bbox": [0, 0, 10, 10], "conf": 0.9, "cls": 0}]}

    monkeypatch.setattr(rp, "iter_tracks", fake_iter_tracks)
    monkeypatch.setattr(MotionMemory, "update", lambda self, obs: _FakePrediction())
    common = ["--weights", "w.pt", "--source", "scn01.mp4", "--target-track-id", "1",
              "--tracker", str(REPO_ROOT / "configs" / "botsort.yaml"),
              "--output-root", str(tmp_path), "--imgsz", "960"]
    assert rp.main(common + ["--experiment-id", "base"]) == 0
    assert rp.main(common + ["--experiment-id", "mm", "--motion-memory"]) == 0

    base = load_pipeline_settings(tmp_path / "base")
    mm = load_pipeline_settings(tmp_path / "mm")
    assert base["imgsz"] == mm["imgsz"] == 960
    assert ablation_mismatches(base, mm) == []


class _FakePrediction:
    from src.motion_memory import TrackState as _TS

    state = _TS.TRACKING
    bbox = None
    confidence = 1.0
    frames_since_confirmed = 0
