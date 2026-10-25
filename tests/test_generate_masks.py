"""Unit test cho scripts/generate_masks.py (SAM duoc thay bang ham gia)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_masks import (  # noqa: E402
    BOX_LABEL_DIR,
    choose_polygon,
    convert_label_lines,
    parse_box_line,
    process_split,
    summarize_ious,
)

BOX = np.array([0.5, 0.5, 0.2, 0.4])  # xyxy = [0.4, 0.3, 0.6, 0.7]


def test_parse_box_line_rejects_polygon_row():
    assert parse_box_line("1 0.5 0.5 0.2 0.4")[0] == 1
    with pytest.raises(ValueError):
        parse_box_line("1 0.1 0.1 0.2 0.1 0.2 0.2")


def test_choose_polygon_keeps_good_mask():
    poly = np.array([[0.41, 0.31], [0.59, 0.31], [0.59, 0.69], [0.41, 0.69]])
    chosen, fallback, iou = choose_polygon(BOX, poly, 0.5)
    assert not fallback
    assert iou > 0.8
    np.testing.assert_allclose(chosen, poly)


def test_choose_polygon_falls_back_on_bad_mask():
    poly = np.array([[0.40, 0.30], [0.45, 0.30], [0.45, 0.35]])  # chi mot goc nho
    chosen, fallback, iou = choose_polygon(BOX, poly, 0.5)
    assert fallback
    assert iou < 0.5
    np.testing.assert_allclose(chosen, [[0.4, 0.3], [0.6, 0.3], [0.6, 0.7], [0.4, 0.7]])


@pytest.mark.parametrize("poly", [None, np.zeros((0, 2)), np.array([[0.4, 0.3], [0.6, 0.7]])])
def test_choose_polygon_falls_back_on_invalid_mask(poly):
    _, fallback, iou = choose_polygon(BOX, poly, 0.5)
    assert fallback and iou == 0.0


def test_convert_label_lines_keeps_order_and_counts():
    good = np.array([[0.41, 0.31], [0.59, 0.31], [0.59, 0.69], [0.41, 0.69]])
    calls = []

    def fake_segment(xyxy_px):
        calls.append(xyxy_px)
        return [good, None]

    stats: Counter = Counter()
    ious: list[float] = []
    lines = convert_label_lines(
        ["0 0.5 0.5 0.2 0.4", "2 0.5 0.5 0.2 0.4", ""],
        fake_segment, 100, 200, 0.5, stats, ious,
    )
    assert [line.split()[0] for line in lines] == ["0", "2"]
    np.testing.assert_allclose(calls[0][0], [40, 60, 60, 140])  # pixel xyxy
    assert stats["class_0_fallback"] == 0
    assert stats["class_2_fallback"] == 1
    assert len(ious) == 2


def test_convert_label_lines_rejects_count_mismatch():
    with pytest.raises(RuntimeError):
        convert_label_lines(
            ["0 0.5 0.5 0.2 0.4"], lambda xyxy: [], 100, 100, 0.5, Counter(), []
        )


def test_summarize_ious():
    s = summarize_ious([0.2, 0.6, 0.8, 1.0], 0.5)
    assert s["count"] == 4 and s["below_fallback"] == 1
    assert summarize_ious([], 0.5) == {"count": 0}


def test_process_split_moves_box_labels_and_is_rerunnable(tmp_path: Path):
    import cv2

    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images" / "val" / "a.jpg"), np.zeros((20, 40, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "images" / "val" / "b.jpg"), np.zeros((20, 40, 3), np.uint8))
    (tmp_path / "labels" / "val" / "a.txt").write_text("1 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    (tmp_path / "labels" / "val" / "b.txt").write_text("", encoding="utf-8")
    (tmp_path / "labels" / "val.cache").write_text("stale", encoding="utf-8")

    def fake_sam(image, xyxy_px):
        assert image.shape[:2] == (20, 40)
        return [None] * len(xyxy_px)

    for _ in range(2):  # chay lai van doc tu labels_bbox
        report = process_split(tmp_path, "val", fake_sam, 0.5, None)
        assert report["images"] == 2
        assert report["per_class"]["class_1_fallback"] == 1

    assert (tmp_path / BOX_LABEL_DIR / "val" / "a.txt").read_text() == "1 0.5 0.5 0.2 0.4\n"
    seg = (tmp_path / "labels" / "val" / "a.txt").read_text().split()
    assert seg[0] == "1" and len(seg) == 1 + 8  # hinh chu nhat 4 dinh
    assert (tmp_path / "labels" / "val" / "b.txt").read_text() == ""
    assert not (tmp_path / "labels" / "val.cache").exists()
