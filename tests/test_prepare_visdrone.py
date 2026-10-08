"""Unit test cho scripts/prepare_visdrone.py.

Dataset VisDrone that chua co tren may - test end-to-end dung bo du lieu gia
lap nho theo dung cau truc thu muc/dinh dang annotation cua VisDrone-MOT.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_visdrone import (  # noqa: E402
    VisDroneBox,
    build_frame_labels,
    keep_frame,
    main,
    parse_annotation_line,
    to_yolo_bbox,
)

CATEGORY_MAP = {1: 0, 2: 0, 4: 1, 5: 1, 6: 1, 10: 2}


def make_box(**overrides) -> VisDroneBox:
    values = dict(
        frame_idx=1, target_id=1, left=10, top=20, width=30, height=40,
        score=1, category=4, truncation=0, occlusion=0,
    )
    values.update(overrides)
    return VisDroneBox(**values)


def test_parse_annotation_line():
    box = parse_annotation_line("1,7,684,8,273,116,1,4,0,0\n")
    assert box == VisDroneBox(1, 7, 684, 8, 273, 116, 1, 4, 0, 0)
    assert parse_annotation_line("   \n") is None


def test_parse_annotation_line_rejects_short_line():
    with pytest.raises(ValueError):
        parse_annotation_line("1,7,684,8")


def test_to_yolo_bbox_normalizes():
    cx, cy, w, h = to_yolo_bbox(make_box(left=0, top=0, width=50, height=100), 100, 200)
    assert (cx, cy, w, h) == pytest.approx((0.25, 0.25, 0.5, 0.5))


def test_to_yolo_bbox_clips_to_image():
    # Box tran ra ngoai bien phai -> cat con [90, 100]
    cx, _, w, _ = to_yolo_bbox(make_box(left=90, top=0, width=30, height=10), 100, 100)
    assert w == pytest.approx(0.1)
    assert cx == pytest.approx(0.95)


def test_to_yolo_bbox_drops_small_or_outside():
    assert to_yolo_bbox(make_box(left=200, top=0, width=10, height=10), 100, 100) is None
    assert to_yolo_bbox(make_box(width=1, height=1), 100, 100, min_size=2) is None


def test_build_frame_labels_maps_and_filters():
    boxes = [
        make_box(frame_idx=1, category=1),           # pedestrian -> 0
        make_box(frame_idx=1, category=2),           # people -> 0
        make_box(frame_idx=1, category=5),           # van -> 1
        make_box(frame_idx=2, category=6),           # truck -> 1
        make_box(frame_idx=2, category=10),          # motor -> 2
        make_box(frame_idx=2, category=9),           # bus -> bo
        make_box(frame_idx=2, category=3),           # bicycle -> bo
        make_box(frame_idx=3, category=4, score=0),  # ignored -> bo
    ]
    stats: Counter = Counter()
    labels = build_frame_labels(boxes, CATEGORY_MAP, 100, 100, stats=stats)

    assert [line.split()[0] for line in labels[1]] == ["0", "0", "1"]
    assert [line.split()[0] for line in labels[2]] == ["1", "2"]
    assert 3 not in labels
    assert stats["skipped_unmapped_category"] == 2
    assert stats["skipped_ignored"] == 1


def test_keep_frame_stride():
    assert [i for i in range(1, 12) if keep_frame(i, 5)] == [1, 6, 11]
    assert all(keep_frame(i, 1) for i in range(1, 5))


# ---------------- end-to-end tren dataset gia lap ----------------


def _write_fake_visdrone(root: Path) -> None:
    import cv2
    import numpy as np

    img = np.zeros((60, 80, 3), dtype=np.uint8)
    for folder in ("VisDrone2019-MOT-train", "VisDrone2019-MOT-val", "VisDrone2019-MOT-test-dev"):
        seq = root / folder / "sequences" / "uav0000001_00000_v"
        seq.mkdir(parents=True)
        for i in range(1, 7):
            cv2.imwrite(str(seq / f"{i:07d}.jpg"), img)
        ann = root / folder / "annotations"
        ann.mkdir()
        ann.joinpath("uav0000001_00000_v.txt").write_text(
            "1,1,10,10,20,20,1,1,0,0\n"   # person
            "1,2,30,30,10,10,1,9,0,0\n"   # bus -> bo
            "6,1,12,10,20,20,1,10,0,1\n",  # motorcycle
            encoding="utf-8",
        )


def test_main_end_to_end(tmp_path: Path):
    src = tmp_path / "raw"
    out = tmp_path / "labeled"
    dataset_yaml = tmp_path / "dataset.yaml"
    _write_fake_visdrone(src)

    argv = [
        "--source-root", str(src),
        "--output-dir", str(out),
        "--dataset-yaml", str(dataset_yaml),
        "--link-mode", "copy",
    ]
    assert main(argv) == 0

    # stride train=5 -> frame 1, 6 ; test stride=1 -> 6 frame
    train_labels = sorted((out / "labels" / "train").glob("*.txt"))
    assert [p.stem for p in train_labels] == [
        "uav0000001_00000_v_0000001",
        "uav0000001_00000_v_0000006",
    ]
    assert len(list((out / "images" / "test").glob("*.jpg"))) == 6

    assert train_labels[0].read_text(encoding="utf-8").split()[0] == "0"
    assert train_labels[1].read_text(encoding="utf-8").split()[0] == "2"
    # Frame khong co box -> file label rong
    assert (out / "labels" / "test" / "uav0000001_00000_v_0000002.txt").read_text() == ""

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["frames_kept"] == 2
    assert manifest["splits"]["train"]["box_stats"]["skipped_unmapped_category"] == 1
    assert "names" in dataset_yaml.read_text(encoding="utf-8")

    # Chay lai khong co --overwrite -> tu choi
    with pytest.raises(FileExistsError):
        main(argv)
    assert main(argv + ["--overwrite"]) == 0


def test_main_refuses_to_delete_foreign_dir(tmp_path: Path):
    src = tmp_path / "raw"
    _write_fake_visdrone(src)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "important.txt").write_text("x", encoding="utf-8")

    with pytest.raises(FileExistsError):
        main([
            "--source-root", str(src),
            "--output-dir", str(foreign),
            "--dataset-yaml", str(tmp_path / "d.yaml"),
            "--overwrite",
        ])
    assert (foreign / "important.txt").exists()
