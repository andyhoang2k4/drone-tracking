import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from src.sim.config import load_config
from src.sim.oracle import GzOracleDetectionSource, boxes_from_msg, detections_from_boxes
from src.sim.target_manager import TargetManager
from src.sim.config import TargetConfig
from src.sim.types import Detection


def _fake_msg(items):
    """Gia lap AnnotatedAxisAligned2DBox_V (chi cac truong oracle doc)."""
    def box(label, x1, y1, x2, y2):
        return SimpleNamespace(label=label, box=SimpleNamespace(min_corner=SimpleNamespace(x=x1, y=y1),
                                                                max_corner=SimpleNamespace(x=x2, y=y2)))
    return SimpleNamespace(annotated_box=[box(*it) for it in items])


def test_detections_use_label_as_track_id_and_class_map():
    dets = detections_from_boxes([(3, 10, 20, 50, 100), (7, 0, 0, 5, 5)], {3: "pedestrian"})
    assert dets[0] == Detection(3, (10.0, 20.0, 50.0, 100.0), "pedestrian", 1.0)
    assert dets[1].label == "label7" and dets[1].track_id == 7


def test_min_box_px_drops_tiny_boxes():
    dets = detections_from_boxes([(1, 0, 0, 3, 40), (2, 0, 0, 30, 40)], min_box_px=4)
    assert [d.track_id for d in dets] == [2]


def test_boxes_from_msg_reads_corners():
    assert boxes_from_msg(_fake_msg([(5, 1.5, 2.0, 30.0, 40.0)])) == [(5, 1.5, 2.0, 30.0, 40.0)]


def test_boxes_from_real_gz_msg():
    pb = pytest.importorskip("gz.msgs10.annotated_axis_aligned_2d_box_v_pb2")
    msg = pb.AnnotatedAxisAligned2DBox_V()
    a = msg.annotated_box.add()
    a.label = 9
    a.box.min_corner.x, a.box.min_corner.y, a.box.max_corner.x, a.box.max_corner.y = 1, 2, 3, 4
    assert boxes_from_msg(msg) == [(9, 1.0, 2.0, 3.0, 4.0)]


def _source_without_gz(**kw):
    """Tao GzOracleDetectionSource ma khong subscribe (khong can Gazebo) de test detect()."""
    src = GzOracleDetectionSource.__new__(GzOracleDetectionSource)
    src._classes, src._min_box_px, src._max_age_s = kw.get("classes", {}), 0.0, kw.get("max_age_s", 1.0)
    import threading
    src._lock, src._boxes, src._stamp = threading.Lock(), [], None
    return src


def test_detect_empty_before_first_message_and_when_stale():
    src = _source_without_gz(max_age_s=0.5)
    frame = np.zeros((10, 10, 3), np.uint8)
    assert src.detect(frame, 0.0) == []
    src._store([(1, 0, 0, 10, 10)], time.monotonic())
    assert [d.track_id for d in src.detect(frame, 0.0)] == [1]
    src._store([(1, 0, 0, 10, 10)], time.monotonic() - 2.0)
    assert src.detect(frame, 0.0) == []


def test_oracle_output_drives_target_manager():
    # cung kieu Detection nhu nguon YOLO + BoT-SORT: khoa duoc va giu TRACKING
    m = TargetManager(TargetConfig())
    m.update(detections_from_boxes([(4, 100, 100, 140, 200)], {4: "pedestrian"}), 0.0)
    assert m.select_at(120, 150) == 4 and m.lock_selected()
    m.update(detections_from_boxes([(4, 105, 100, 145, 200)], {4: "pedestrian"}), 0.1)
    assert m.locked_id == 4


def test_oracle_config_loads_from_yaml(tmp_path):
    p = tmp_path / "sim.yaml"
    p.write_text("oracle:\n  bbox_topic: /probe/bbox\n  label_classes: {1: car, 2: pedestrian}\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.oracle.bbox_topic == "/probe/bbox" and cfg.oracle.label_classes == {1: "car", 2: "pedestrian"}
