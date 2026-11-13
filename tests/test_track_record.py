"""Test src/track.py: ban ghi log co path + thoi gian moi frame.
Khong chay YOLO that - dung Results gia."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.track as track  # noqa: E402


def _box(track_id, xyxy, conf, cls):
    return SimpleNamespace(
        id=None if track_id is None else np.array([track_id]),
        xyxy=np.array([xyxy], dtype=float),
        conf=np.array([conf]),
        cls=np.array([cls]),
    )


def _result(path, boxes, speed=None):
    return SimpleNamespace(path=path, boxes=boxes,
                           speed=speed or {"preprocess": 1.0, "inference": 5.0, "postprocess": 2.0})


def test_result_to_record_fields():
    result = _result("seq/0000007.jpg", [_box(3, [1, 2, 11, 22], 0.8, 4), _box(None, [0, 0, 5, 5], 0.3, 0)])
    record = track.result_to_record(6, result, 12.5)
    assert record["frame_idx"] == 6
    assert record["path"] == "seq/0000007.jpg"
    assert record["frame_time_ms"] == 12.5
    assert record["speed_ms"] == {"preprocess": 1.0, "inference": 5.0, "postprocess": 2.0}
    assert record["tracks"] == [
        {"track_id": 3, "bbox": [1.0, 2.0, 11.0, 22.0], "conf": 0.8, "cls": 4},
        {"track_id": None, "bbox": [0.0, 0.0, 5.0, 5.0], "conf": 0.3, "cls": 0},
    ]


def test_result_without_boxes_or_speed():
    record = track.result_to_record(0, SimpleNamespace(path=None, boxes=None, speed=None), None)
    assert record["tracks"] == [] and record["path"] is None and record["speed_ms"] == {}


def test_timed_iter_measures_each_next(monkeypatch):
    clock = iter([0.0, 0.010, 0.010, 0.035, 0.035])  # 2 item + lan StopIteration
    monkeypatch.setattr(track.time, "perf_counter", lambda: next(clock))
    out = list(track.timed_iter(["a", "b"]))
    assert [item for item, _ in out] == ["a", "b"]
    assert [round(ms, 6) for _, ms in out] == [10.0, 25.0]


def test_iter_tracks_uses_timed_records(monkeypatch):
    """iter_tracks ghep timed_iter + result_to_record tren stream gia."""
    results = [_result(f"seq/{i:07d}.jpg", [_box(1, [0, 0, 4, 4], 0.9, 3)]) for i in (1, 2)]

    class FakeYOLO:
        def __init__(self, weights):
            pass

        def track(self, **kwargs):
            assert kwargs["stream"] and kwargs["persist"] and kwargs["conf"] == 0.25
            return iter(results)

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    records = list(track.iter_tracks("w.pt", "seq", conf=0.25, imgsz=960))
    assert [r["path"] for r in records] == ["seq/0000001.jpg", "seq/0000002.jpg"]
    assert all(r["frame_time_ms"] >= 0.0 for r in records)
    assert records[1]["tracks"][0]["track_id"] == 1
