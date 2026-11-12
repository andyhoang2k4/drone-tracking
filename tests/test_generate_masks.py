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

    def fake_sam(image, xyxy_px, stats):
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


# ---------------- segment_in_batches (SAM gia) ----------------

from scripts.generate_masks import RECOVERY_KEYS, is_oom_error, segment_in_batches  # noqa: E402


class FakeOOM(RuntimeError):
    pass


FakeOOM.__name__ = "OutOfMemoryError"


def _boxes(n: int) -> np.ndarray:
    """Box i co x1 = i -> polygon gia mang chi so box de kiem tra ghep dung."""
    return np.array([[i, 0, i + 1, 1] for i in range(n)], dtype=float)


def _poly(box: np.ndarray) -> np.ndarray:
    return np.full((4, 2), box[0])


def _box_id(poly) -> int:
    return int(poly[0, 0])


def _stats(**expected) -> dict:
    return {k: expected.get(k, 0) for k in RECOVERY_KEYS}


def test_batches_respect_batch_size_and_order():
    calls = []

    def call(chunk):
        calls.append(len(chunk))
        return [_poly(b) for b in chunk], list(range(len(chunk)))

    stats: Counter = Counter()
    out = segment_in_batches(_boxes(10), call, 4, stats)
    assert calls == [4, 4, 2]
    assert [_box_id(p) for p in out] == list(range(10))
    assert {k: stats[k] for k in RECOVERY_KEYS} == _stats(sam_calls=3)


def test_missing_mask_is_retried_per_box_then_rectangle():
    """SAM bo mask cua box 3 (score thap) ca khi goi rieng -> None (hinh chu nhat)."""
    def call(chunk):
        keep = [j for j, b in enumerate(chunk) if int(b[0]) != 3]
        return [_poly(chunk[j]) for j in keep], keep

    stats: Counter = Counter()
    out = segment_in_batches(_boxes(5), call, 5, stats)
    assert [None if p is None else _box_id(p) for p in out] == [0, 1, 2, None, 4]
    assert {k: stats[k] for k in RECOVERY_KEYS} == _stats(
        sam_calls=2, count_mismatch_calls=2, retried_single_boxes=1, rect_missing_mask=1
    )


def test_missing_mask_recovered_on_single_retry():
    """Thieu mask khi goi ca nhom, goi rieng thi co."""
    def call(chunk):
        if len(chunk) > 1:
            return [_poly(b) for b in chunk[1:]], list(range(1, len(chunk)))
        return [_poly(chunk[0])], [0]

    stats: Counter = Counter()
    out = segment_in_batches(_boxes(3), call, 3, stats)
    assert [_box_id(p) for p in out] == [0, 1, 2]
    assert stats["retried_single_boxes"] == 1 and stats["rect_missing_mask"] == 0


def test_out_of_order_masks_are_matched_by_index():
    def call(chunk):
        order = list(reversed(range(len(chunk))))
        return [_poly(chunk[j]) for j in order], order

    stats: Counter = Counter()
    out = segment_in_batches(_boxes(6), call, 6, stats)
    assert [_box_id(p) for p in out] == list(range(6))
    assert stats["count_mismatch_calls"] == 0


def test_oom_on_large_batch_then_succeeds_when_split():
    calls, freed = [], []

    def call(chunk):
        calls.append(len(chunk))
        if len(chunk) > 2:
            raise FakeOOM("CUDA out of memory. Tried to allocate 2.00 GiB")
        return [_poly(b) for b in chunk], list(range(len(chunk)))

    stats: Counter = Counter()
    out = segment_in_batches(_boxes(8), call, 8, stats, on_oom=lambda: freed.append(1))
    assert [_box_id(p) for p in out] == list(range(8))
    assert calls == [8, 4, 2, 2, 4, 2, 2]
    assert len(freed) == 3
    assert {k: stats[k] for k in RECOVERY_KEYS} == _stats(sam_calls=4, oom_retries=3)


def test_oom_even_for_single_box_gives_rectangle():
    def call(chunk):
        if any(int(b[0]) == 1 for b in chunk):
            raise FakeOOM("CUDA out of memory")
        return [_poly(b) for b in chunk], list(range(len(chunk)))

    stats: Counter = Counter()
    out = segment_in_batches(_boxes(4), call, 4, stats)
    assert [None if p is None else _box_id(p) for p in out] == [0, None, 2, 3]
    assert stats["rect_oom"] == 1 and stats["rect_missing_mask"] == 0
    assert stats["oom_retries"] == 2  # 4 -> 2 -> 1


def test_non_oom_error_is_raised():
    def call(chunk):
        raise ValueError("loi khac")

    with pytest.raises(ValueError):
        segment_in_batches(_boxes(2), call, 2, Counter())


def test_is_oom_error_and_invalid_batch_size():
    assert is_oom_error(FakeOOM("x"))
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not is_oom_error(RuntimeError("other"))
    with pytest.raises(ValueError):
        segment_in_batches(_boxes(1), lambda c: ([], []), 0, Counter())
