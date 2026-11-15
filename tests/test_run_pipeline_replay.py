"""Test che do replay cua scripts/run_pipeline.py (--replay-tracks).

Log track gia viet theo dinh dang cua `python -m src.track` (tracks.jsonl +
tracks.meta.json); khong chay YOLO.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_pipeline as rp  # noqa: E402
from src.evaluate import ablation_mismatches, load_pipeline_settings  # noqa: E402

TRACKER = "configs/botsort.yaml"


def trk(tid, cx, cls=0):
    return {"track_id": tid, "bbox": [cx - 20, 90, cx + 20, 110], "conf": 0.8, "cls": cls}


def records():
    out = []
    for k in range(12):
        tracks = [trk(9, 500)]
        if k <= 4:
            tracks.append(trk(1, 100 + 3 * k))
        elif k >= 8:
            tracks.append(trk(2, 100 + 3 * k))
        out.append({"frame_idx": k, "path": f"seq/images/{k + 1:07d}.jpg", "frame_time_ms": 1.0,
                    "speed_ms": {}, "tracks": tracks})
    return out


def write_tracks(tmp_path: Path, recs=None, **meta_overrides) -> Path:
    log = tmp_path / "tr" / "logs" / "tracks.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text("".join(json.dumps(r) + "\n" for r in (recs or records())), encoding="utf-8")
    meta = {"weights": "w.pt", "source": "seq/images", "output": str(log), "tracker": TRACKER,
            "conf": 0.1, "imgsz": 960, "classes": None, "device": "0",
            "imgsz_source": "cli", "imgsz_warnings": [], "conf_source": "train_config",
            "config_conf": 0.1}
    meta.update(meta_overrides)
    (log.parent / "tracks.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return log


@pytest.fixture
def no_yolo(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("replay khong duoc chay YOLO")

    monkeypatch.setattr(rp, "iter_tracks", fail)


@pytest.fixture
def spy(monkeypatch):
    """Ghi lai (ban sao) moi ban ghi tracker ma vong lap cua main da tieu thu."""
    seen: dict[str, list] = {}
    original = rp.tracker_records

    def wrapped(args, replay):
        bucket = seen.setdefault(args.experiment_id, [])
        for record in original(args, replay):
            bucket.append(copy.deepcopy(record))
            yield record

    monkeypatch.setattr(rp, "tracker_records", wrapped)
    return seen


def run(tmp_path, exp, *extra, source="seq/images"):
    argv = ["--weights", "w.pt", "--source", source, "--experiment-id", exp,
            "--target-track-id", "1", "--output-root", str(tmp_path / "out"), *extra]
    assert rp.main(argv) == 0
    d = tmp_path / "out" / exp
    rows = [json.loads(x) for x in (d / "logs" / "pipeline.jsonl").read_text().splitlines()]
    return rows, json.loads((d / "run_context.json").read_text())


def test_both_configs_see_exactly_same_tracker_input(tmp_path, no_yolo, spy):
    log = write_tracks(tmp_path)
    file_records = [json.loads(x) for x in log.read_text().splitlines()]
    _, ctx_b = run(tmp_path, "base", "--replay-tracks", str(log))
    rows_m, ctx_m = run(tmp_path, "mm", "--replay-tracks", str(log), "--motion-memory")
    assert spy["base"] == spy["mm"] == file_records
    assert ctx_b["replay"]["tracks_sha256"] == ctx_m["replay"]["tracks_sha256"]
    assert ctx_b["replay"]["frames"] == 12
    # Motion Memory thuc su chay tren log do (theo ID moi sau khi mat ID 1).
    assert rows_m[-1]["track_id"] == 2


def test_replay_settings_taken_from_meta(tmp_path, no_yolo):
    log = write_tracks(tmp_path, conf=0.25, imgsz=1280, classes=[0, 3], device="1")
    _, ctx = run(tmp_path, "base", "--replay-tracks", str(log))
    cfg = ctx["config"]
    assert (cfg["conf"], cfg["imgsz"], cfg["classes"], cfg["device"]) == (0.25, 1280, [0, 3], "1")
    assert ctx["inference"]["imgsz"] == 1280 and ctx["inference"]["conf"] == 0.25
    settings = load_pipeline_settings(tmp_path / "out" / "base")
    assert settings["imgsz"] == 1280 and settings["tracks_sha256"] == ctx["replay"]["tracks_sha256"]


def test_replay_output_equals_live_run(tmp_path, monkeypatch):
    """Cung chuoi track: replay va chay truc tiep ra log pipeline giong het."""
    log = write_tracks(tmp_path)
    for extra in ((), ("--motion-memory",)):
        monkeypatch.setattr(rp, "iter_tracks", lambda w, s, **k: iter(records()))
        live, _ = run(tmp_path, "live" + str(len(extra)), "--imgsz", "960", "--conf", "0.1", *extra)
        replay, _ = run(tmp_path, "rep" + str(len(extra)), "--replay-tracks", str(log), *extra)
        assert live == replay


@pytest.mark.parametrize("extra,match", [
    (("--conf", "0.3"), "conf"),
    (("--imgsz", "640"), "imgsz"),
])
def test_cli_value_must_match_meta(tmp_path, no_yolo, extra, match):
    log = write_tracks(tmp_path)
    with pytest.raises(rp.ReplayError, match=match):
        run(tmp_path, "x", "--replay-tracks", str(log), *extra)


def test_weights_and_source_must_match_meta(tmp_path, no_yolo):
    log = write_tracks(tmp_path, weights="other.pt")
    with pytest.raises(rp.ReplayError, match="weights"):
        run(tmp_path, "x", "--replay-tracks", str(log))
    log2 = write_tracks(tmp_path / "b")
    with pytest.raises(rp.ReplayError, match="source"):
        run(tmp_path, "y", "--replay-tracks", str(log2), source="other/images")


def test_missing_meta_refused(tmp_path, no_yolo):
    log = write_tracks(tmp_path)
    (log.parent / "tracks.meta.json").unlink()
    with pytest.raises(rp.ReplayError, match="meta"):
        run(tmp_path, "x", "--replay-tracks", str(log))
    assert not (tmp_path / "out" / "x").exists()  # khong ghi gi khi tu choi


def test_appended_or_truncated_log_refused(tmp_path, no_yolo):
    recs = records()
    log = write_tracks(tmp_path, recs + recs[:3])  # ghi noi
    with pytest.raises(rp.ReplayError, match="lien tuc"):
        run(tmp_path, "x", "--replay-tracks", str(log))
    log2 = write_tracks(tmp_path / "b")
    log2.write_text(log2.read_text() + '{"frame_idx": 12, "tra')  # dong dang ghi do
    with pytest.raises(rp.ReplayError, match="JSON"):
        run(tmp_path, "y", "--replay-tracks", str(log2))


def test_compare_refuses_different_tracks_logs():
    common = {"weights": "w", "source": "s", "tracker": "t", "conf": 0.1, "imgsz": 960, "classes": None}
    base = {**common, "motion_memory": False, "tracks_sha256": "aaa"}
    mem = {**common, "motion_memory": True, "tracks_sha256": "bbb"}
    assert any("tracks_sha256" in p for p in ablation_mismatches(base, mem))
    assert ablation_mismatches(base, {**mem, "tracks_sha256": "aaa"}) == []
