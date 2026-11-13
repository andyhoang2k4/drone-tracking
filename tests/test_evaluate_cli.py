"""Test CLI src/evaluate.py: mot sequence VisDrone-MOT (val / test-dev) tu log
cua src/track.py. Cau truc thu muc + annotation + log deu gia, khong chay model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.evaluate as ev  # noqa: E402

SEQ = "uav0000001_00000_v"
# frame 1..3: target 7 (car, cat 4 -> class 3) o (10,10)-(50,50);
# moi frame co mot vung bo qua (category 0) o (200,200)-(300,300).
ANNOTATION = "".join(
    f"{f},7,10,10,40,40,1,4,0,0\n{f},0,200,200,100,100,0,0,0,0\n" for f in (1, 2, 3)
)


def _raw_tree(tmp_path: Path, split_dir: str) -> Path:
    root = tmp_path / "raw"
    base = root / split_dir
    (base / "annotations").mkdir(parents=True)
    (base / "annotations" / f"{SEQ}.txt").write_text(ANNOTATION, encoding="utf-8")
    seq_dir = base / "sequences" / SEQ
    seq_dir.mkdir(parents=True)
    for f in (1, 2, 3):
        (seq_dir / f"{f:07d}.jpg").write_bytes(b"")
    return root


def _eval_config(tmp_path: Path, raw_root: Path, warmup: int = 1) -> dict:
    return {
        "raw_root": str(raw_root),
        "splits": {"val": "VisDrone2019-MOT-val", "test-dev": "VisDrone2019-MOT-test-dev"},
        "dataset_config": str(REPO_ROOT / "configs" / "dataset_visdrone.yaml"),
        "iou_threshold": 0.5,
        "ignore_ioa": 0.5,
        "class_aware": True,
        "fps_warmup_frames": warmup,
    }


def _write_log(tmp_path: Path, records: list[dict], *, source: str, conf: float = 0.25) -> Path:
    log = tmp_path / "logs" / f"{SEQ}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    meta = {"weights": "best.pt", "source": source, "tracker": "configs/botsort.yaml",
            "conf": conf, "conf_source": "train_config:x", "imgsz": 960, "imgsz_source": "cli",
            "device": "0", "classes": None}
    log.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return log


def _record(frame: int, tracks: list[tuple], ms: float) -> dict:
    return {
        "frame_idx": frame - 1,
        "path": f"seq/{SEQ}/{frame:07d}.jpg",
        "frame_time_ms": ms,
        "speed_ms": {"preprocess": 1.0, "inference": 4.0, "postprocess": 1.0},
        "tracks": [{"track_id": tid, "bbox": list(b), "conf": 0.9, "cls": c} for tid, b, c in tracks],
    }


def _standard_records() -> list[dict]:
    car = (10, 10, 50, 50)
    return [
        _record(1, [(1, car, 3), (2, (210, 210, 250, 250), 3)], 500.0),  # pred 2 trong vung bo qua
        _record(2, [(1, car, 3), (3, (400, 400, 420, 420), 0)], 20.0),   # pred 3 la FP
        _record(3, [(5, car, 3)], 30.0),                                  # doi ID 1 -> 5
    ]


@pytest.mark.parametrize("split,split_dir", [("val", "VisDrone2019-MOT-val"),
                                             ("test-dev", "VisDrone2019-MOT-test-dev")])
def test_evaluate_sequence_metrics(tmp_path, split, split_dir):
    raw = _raw_tree(tmp_path, split_dir)
    log = _write_log(tmp_path, _standard_records(), source=str(raw / split_dir / "sequences" / SEQ))
    m = ev.evaluate_sequence(log_path=log, split=split, sequence=SEQ,
                             eval_config=_eval_config(tmp_path, raw), configured_conf=0.25)
    det = m["detection"]
    assert (det["true_positives"], det["false_positives"], det["false_negatives"]) == (3, 1, 0)
    assert det["ignored_predictions"] == 1
    assert det["precision"] == pytest.approx(0.75) and det["recall"] == 1.0
    assert m["tracking"] == {"id_switches": 1, "gt_targets": 1, "gt_targets_matched": 1,
                             "predicted_tracks": 4}
    spd = m["speed"]
    assert spd["warmup_frames_excluded"] == 1 and spd["timed_frames"] == 2
    assert spd["mean_latency_ms"] == pytest.approx(25.0) and spd["mean_fps"] == pytest.approx(40.0)
    assert spd["ultralytics_speed_ms_mean"]["inference"] == pytest.approx(4.0)
    assert m["inference"]["conf"] == 0.25 and m["inference"]["imgsz"] == 960
    assert m["split"] == split and m["frames_evaluated"] == 3 and m["warnings"] == []


def test_main_writes_metrics_json(tmp_path, monkeypatch):
    raw = _raw_tree(tmp_path, "VisDrone2019-MOT-val")
    log = _write_log(tmp_path, _standard_records(), source=f"x/{SEQ}")
    cfg = tmp_path / "eval.yaml"
    import yaml

    cfg.write_text(yaml.safe_dump(_eval_config(tmp_path, raw)), encoding="utf-8")
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text("inference:\n  conf: 0.25\n", encoding="utf-8")
    out = tmp_path / "metrics" / "val" / f"{SEQ}.json"
    assert ev.main(["--log", str(log), "--split", "val", "--sequence", SEQ, "--output", str(out),
                    "--eval-config", str(cfg), "--train-config", str(train_cfg)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["detection"]["recall"] == 1.0 and saved["eval_config"]["path"] == str(cfg)
    assert "commit" in json.dumps(saved["git"])


def _expect_error(tmp_path, records, match, *, source=f"x/{SEQ}", conf=0.25, split="val"):
    raw = _raw_tree(tmp_path, "VisDrone2019-MOT-val")
    log = _write_log(tmp_path, records, source=source, conf=conf)
    with pytest.raises(ev.EvaluationError, match=match):
        ev.evaluate_sequence(log_path=log, split=split, sequence=SEQ,
                             eval_config=_eval_config(tmp_path, raw), configured_conf=0.25)


def test_conf_mismatch_rejected(tmp_path):
    _expect_error(tmp_path, _standard_records(), "conf", conf=0.1)


def test_wrong_sequence_source_rejected(tmp_path):
    _expect_error(tmp_path, _standard_records(), "sequence", source="x/uav0000999_00000_v")


def test_missing_frames_rejected(tmp_path):
    _expect_error(tmp_path, _standard_records()[:2], "thieu 1 frame")


def test_log_without_path_rejected(tmp_path):
    records = _standard_records()
    del records[0]["path"]
    _expect_error(tmp_path, records, "path")


def test_duplicate_frame_rejected(tmp_path):
    records = _standard_records()
    _expect_error(tmp_path, records + records[:1], "hai lan")


def test_unknown_split_rejected(tmp_path):
    _expect_error(tmp_path, _standard_records(), "split", split="train")


def test_speed_summary_not_enough_frames():
    s = ev.speed_summary([{"frame_time_ms": 10.0}], warmup_frames=5)
    assert s["mean_fps"] is None and s["timed_frames"] == 0


def test_repo_eval_config_supports_val_and_test_dev():
    cfg = ev._load_yaml(ev.DEFAULT_EVAL_CONFIG)
    assert set(cfg["splits"]) == {"val", "test-dev"}
    for key in ("iou_threshold", "ignore_ioa", "class_aware", "fps_warmup_frames", "dataset_config"):
        assert key in cfg


def test_end_to_end_with_track_main(tmp_path, monkeypatch):
    """Log do src/track.main that ghi (YOLO gia) -> evaluate doc duoc."""
    import ultralytics

    import src.track as track

    raw = _raw_tree(tmp_path, "VisDrone2019-MOT-val")
    seq_dir = raw / "VisDrone2019-MOT-val" / "sequences" / SEQ

    def result(frame, track_id):
        boxes = [SimpleNamespace(id=np.array([track_id]), xyxy=np.array([[10.0, 10.0, 50.0, 50.0]]),
                                 conf=np.array([0.9]), cls=np.array([3]))]
        return SimpleNamespace(path=str(seq_dir / f"{frame:07d}.jpg"), boxes=boxes,
                               speed={"preprocess": 1.0, "inference": 4.0, "postprocess": 1.0})

    class FakeYOLO:
        def __init__(self, weights):
            pass

        def track(self, **kwargs):
            return iter([result(1, 1), result(2, 1), result(3, 1)])

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text("common:\n  imgsz: 960\ninference:\n  conf: 0.25\n", encoding="utf-8")
    log = tmp_path / "logs" / f"{SEQ}.jsonl"
    assert track.main(["--weights", "w.pt", "--source", str(seq_dir), "--output", str(log),
                       "--tracker", str(REPO_ROOT / "configs" / "botsort.yaml"),
                       "--train-config", str(train_cfg)]) == 0
    m = ev.evaluate_sequence(log_path=log, split="val", sequence=SEQ,
                             eval_config=_eval_config(tmp_path, raw), configured_conf=0.25)
    assert m["detection"]["true_positives"] == 3 and m["tracking"]["id_switches"] == 0
    assert m["speed"]["timed_frames"] == 2
