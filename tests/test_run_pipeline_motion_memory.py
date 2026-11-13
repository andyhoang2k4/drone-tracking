"""Test noi Motion Memory vao scripts/run_pipeline.py.

Khong chay YOLO: iter_tracks duoc thay bang ham gia sinh track BoT-SORT gia.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_pipeline as rp  # noqa: E402

# Chuoi track gia dung cho test hoi quy baseline (log mau ghi lai tu code
# TRUOC khi noi Motion Memory, commit fb4cfc4).
BASELINE_RECORDS = [
    {"frame_idx": 0, "tracks": [{"track_id": 2, "bbox": [50, 50, 60, 60], "conf": 0.5, "cls": 1}]},
    {"frame_idx": 1, "tracks": [{"track_id": 1, "bbox": [0, 0, 10, 10], "conf": 0.9, "cls": 0},
                                {"track_id": None, "bbox": [5, 5, 9, 9], "conf": 0.2, "cls": 0}]},
    {"frame_idx": 2, "tracks": []},
    {"frame_idx": 3, "tracks": [{"track_id": 7, "bbox": [3, 0, 13, 10], "conf": 0.8, "cls": 0}]},
    {"frame_idx": 4, "tracks": [{"track_id": 1, "bbox": [4, 0, 14, 10], "conf": 0.7, "cls": 0}]},
]
BASELINE_EXPECTED = [
    {"frame_idx": 0, "detected": False, "tracker_bbox": None, "detector_confidence": None, "state": "LOST"},
    {"frame_idx": 1, "detected": True, "tracker_bbox": [0, 0, 10, 10], "detector_confidence": 0.9, "state": "TRACKING"},
    {"frame_idx": 2, "detected": False, "tracker_bbox": None, "detector_confidence": None, "state": "LOST"},
    {"frame_idx": 3, "detected": False, "tracker_bbox": None, "detector_confidence": None, "state": "LOST"},
    {"frame_idx": 4, "detected": True, "tracker_bbox": [4, 0, 14, 10], "detector_confidence": 0.7, "state": "TRACKING"},
]


def track(tid, cx, cy=100.0, w=40.0, h=20.0, cls=0, conf=0.8):
    return {"track_id": tid, "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            "conf": conf, "cls": cls}


def run(tmp_path: Path, monkeypatch, records, *extra: str, exp: str = "exp"):
    seen: dict = {}

    def fake_iter_tracks(weights, source, **kwargs):
        seen.update({"weights": weights, "source": source, **kwargs})
        yield from records

    monkeypatch.setattr(rp, "iter_tracks", fake_iter_tracks)
    argv = [
        "--weights", "w.pt", "--source", "scn.mp4", "--experiment-id", exp,
        "--target-track-id", "1", "--tracker", str(REPO_ROOT / "configs" / "botsort.yaml"),
        "--output-root", str(tmp_path), "--imgsz", "960", *extra,
    ]
    assert rp.main(argv) == 0
    exp_dir = tmp_path / exp
    rows = [json.loads(line) for line in
            (exp_dir / "logs" / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()]
    context = json.loads((exp_dir / "run_context.json").read_text(encoding="utf-8"))
    return rows, context, seen


# ---------------- baseline giu nguyen ----------------


def test_baseline_log_unchanged(tmp_path, monkeypatch):
    rows, context, _ = run(tmp_path, monkeypatch, BASELINE_RECORDS)
    assert rows == BASELINE_EXPECTED
    assert context["motion_memory"] is None


def test_botsort_called_identically_with_and_without_memory(tmp_path, monkeypatch):
    _, _, seen_base = run(tmp_path, monkeypatch, BASELINE_RECORDS, exp="base")
    _, _, seen_mm = run(tmp_path, monkeypatch, BASELINE_RECORDS, "--motion-memory", exp="mm")
    assert seen_base == seen_mm


def test_cli_timeouts_default_none():
    args = rp.parse_args(["--weights", "w", "--source", "s", "--experiment-id", "e",
                          "--target-track-id", "1"])
    assert args.lost_timeout_frames is None and args.reid_timeout_frames is None


# ---------------- run_context ----------------


def test_run_context_records_resolved_motion_memory(tmp_path, monkeypatch):
    _, context, _ = run(tmp_path, monkeypatch, BASELINE_RECORDS, "--motion-memory",
                        "--lost-timeout-frames", "7")
    mm = context["motion_memory"]
    assert mm["config"]["lost_timeout_frames"] == 7
    assert mm["sources"]["lost_timeout_frames"] == "cli"
    assert mm["config"]["reid_timeout_frames"] == 45
    assert mm["sources"]["reid_timeout_frames"] == "file"
    assert context["config"]["motion_memory"] is True


def test_custom_motion_memory_config_file(tmp_path, monkeypatch):
    cfg = tmp_path / "mm.yaml"
    cfg.write_text("motion_memory:\n  reid_timeout_frames: 9\n", encoding="utf-8")
    _, context, _ = run(tmp_path, monkeypatch, BASELINE_RECORDS, "--motion-memory",
                        "--motion-memory-config", str(cfg))
    assert context["motion_memory"]["config"]["reid_timeout_frames"] == 9
    assert context["motion_memory"]["sources"]["lost_timeout_frames"] == "default"


# ---------------- hanh vi khi bat Motion Memory ----------------


def test_state_null_until_target_seen(tmp_path, monkeypatch):
    rows, _, _ = run(tmp_path, monkeypatch, BASELINE_RECORDS, "--motion-memory")
    assert rows[0]["state"] is None and rows[0]["track_id"] is None
    assert rows[0]["predicted_bbox"] is None and rows[0]["motion_confidence"] is None
    assert rows[1]["state"] == "TRACKING" and rows[1]["track_id"] == 1
    assert rows[2]["state"] == "LOST"


def moving_then_switch():
    """ID 1 di sang phai 3 px/frame o frame 0..9, mat o frame 10, BoT-SORT cap
    ID 7 tu frame 11 tai dung vi tri; co them mot vat xa (ID 9) moi frame."""
    records = []
    for f in range(16):
        tracks = [track(9, 600.0, 400.0, cls=1)]
        x = 100.0 + 3.0 * f
        if f < 10:
            tracks.append(track(1, x))
        elif f >= 11:
            tracks.append(track(7, x))
        records.append({"frame_idx": f, "tracks": tracks})
    return records


def test_follows_new_id_after_gate_match(tmp_path, monkeypatch):
    rows, _, _ = run(tmp_path, monkeypatch, moving_then_switch(), "--motion-memory")
    assert rows[10]["state"] == "LOST" and rows[10]["detected"] is False
    assert rows[10]["predicted_bbox"] is not None
    assert rows[11]["state"] == "TRACKING"
    assert rows[11]["track_id"] == 7
    assert rows[11]["recover_reason"] == "motion_gate"
    assert rows[11]["gate_candidates_passed"] == 1
    for row in rows[12:]:
        assert row["track_id"] == 7 and row["detected"] is True
        assert row["recover_reason"] is None
        assert row["tracker_bbox"] == row["predicted_bbox"]


def test_gate_count_logged_on_lost_frames(tmp_path, monkeypatch):
    records = [{"frame_idx": f, "tracks": [track(1, 100.0 + 3 * f)]} for f in range(10)]
    # mat muc tieu; chi co vat xa ID 9 (cung class) -> 0 ung vien qua gate
    records += [{"frame_idx": f, "tracks": [track(9, 600.0)]} for f in range(10, 13)]
    rows, _, _ = run(tmp_path, monkeypatch, records, "--motion-memory")
    assert rows[10]["state"] == "LOST" and rows[10]["gate_candidates_passed"] is None  # frame chuyen
    assert [r["gate_candidates_passed"] for r in rows[11:13]] == [0, 0]


def test_frame_idx_passed_to_memory(tmp_path, monkeypatch):
    # nguon bo frame 10..12: frame ke tiep (13) phai co frames_since_confirmed = 4
    records = [{"frame_idx": f, "tracks": [track(1, 100.0 + 3 * f)]} for f in range(10)]
    records.append({"frame_idx": 13, "tracks": []})
    rows, _, _ = run(tmp_path, monkeypatch, records, "--motion-memory")
    assert rows[-1]["frames_since_confirmed"] == 4


def test_detector_confidence_not_merged(tmp_path, monkeypatch):
    rows, _, _ = run(tmp_path, monkeypatch, BASELINE_RECORDS, "--motion-memory")
    assert rows[1]["detector_confidence"] == 0.9
    assert rows[1]["motion_confidence"] == 1.0


# ---------------- record_to_candidates ----------------


def test_record_to_candidates_keeps_class_and_drops_unassigned():
    cands = rp.record_to_candidates(BASELINE_RECORDS[1])
    assert len(cands) == 1
    c = cands[0]
    assert c.track_id == 1 and c.cls == 0 and c.frame_idx == 1
    assert c.detector_confidence == 0.9
    assert c.bbox.as_tuple() == (0.0, 0.0, 10.0, 10.0)


def test_record_without_cls_gives_none():
    cands = rp.record_to_candidates(
        {"frame_idx": 0, "tracks": [{"track_id": 3, "bbox": [0, 0, 1, 1], "conf": 0.5}]}
    )
    assert cands[0].cls is None


@pytest.mark.parametrize("flag", [[], ["--motion-memory"]])
def test_runs_end_to_end(tmp_path, monkeypatch, flag):
    rows, _, _ = run(tmp_path, monkeypatch, moving_then_switch(), *flag)
    assert len(rows) == 16
