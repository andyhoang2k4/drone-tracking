"""run_sim.py + adapters: imgsz lay theo resolve_imgsz, ghi vao log sim.
Khong nap YOLO that, khong can PX4/Gazebo/GUI."""

import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_sim as run_sim  # noqa: E402
from src.sim.adapters import UltralyticsBotSortSource  # noqa: E402


@pytest.fixture
def train_config(tmp_path):
    path = tmp_path / "train_config.yaml"
    path.write_text("common:\n  imgsz: 960\n", encoding="utf-8")
    return path


def test_adapter_requires_imgsz_without_default():
    param = inspect.signature(UltralyticsBotSortSource.__init__).parameters["imgsz"]
    assert param.default is inspect.Parameter.empty
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_resolve_from_train_config_and_cli(train_config):
    args = run_sim.parse_args(["--weights", "w.pt", "--train-config", str(train_config)])
    assert args.imgsz is None
    choice = run_sim.resolve_detector_imgsz(args)
    assert choice.imgsz == 960 and choice.source == f"train_config:{train_config}"

    args = run_sim.parse_args(["--weights", "w.pt", "--imgsz", "1280", "--train-config", str(train_config)])
    choice = run_sim.resolve_detector_imgsz(args)
    assert (choice.imgsz, choice.source) == (1280, "cli")


def test_resolve_from_experiment_dir(tmp_path, train_config):
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "config.yaml").write_text("detector:\n  imgsz: 800\n", encoding="utf-8")
    args = run_sim.parse_args(["--weights", "w.pt", "--experiment-dir", str(exp),
                               "--train-config", str(train_config)])
    assert run_sim.resolve_detector_imgsz(args).imgsz == 800


def test_session_record_contains_imgsz(train_config):
    args = run_sim.parse_args(["--weights", "w.pt", "--video", "clip.mp4", "--train-config", str(train_config)])
    record = run_sim.session_record(args, run_sim.resolve_detector_imgsz(args))
    assert record["event"] == "session_start"
    assert (record["imgsz"], record["imgsz_source"]) == (960, f"train_config:{train_config}")
    assert record["weights"] == "w.pt" and record["video"] == "clip.mp4"


def test_demo_has_no_imgsz():
    args = run_sim.parse_args(["--demo"])
    assert run_sim.resolve_detector_imgsz(args) is None
    assert run_sim.session_record(args, None)["imgsz"] is None


def test_missing_weights_without_demo_errors():
    with pytest.raises(SystemExit):
        run_sim.parse_args([])


def test_main_demo_writes_session_record_first(tmp_path, monkeypatch):
    from src.sim.app import SimApp

    monkeypatch.setattr(SimApp, "run", lambda self: None)  # khong mo GUI
    log = tmp_path / "sim.jsonl"
    run_sim.main(["--demo", "--log", str(log)])
    first = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert first["event"] == "session_start" and first["demo"] is True


def test_oracle_needs_no_weights_and_no_imgsz():
    args = run_sim.parse_args(["--oracle"])
    assert run_sim.resolve_detector_imgsz(args) is None
    record = run_sim.session_record(args, None)
    assert record["detector"] == "gz_oracle" and record["imgsz"] is None
    assert record["imgsz_source"].startswith("oracle")


def test_detector_kind_in_session_record(train_config):
    assert run_sim.session_record(run_sim.parse_args(["--demo"]), None)["detector"] == "demo"
    args = run_sim.parse_args(["--weights", "w.pt", "--train-config", str(train_config)])
    assert run_sim.session_record(args, run_sim.resolve_detector_imgsz(args))["detector"] == "yolo_botsort"


@pytest.mark.parametrize("argv", [["--oracle", "--video", "clip.mp4"], ["--oracle", "--demo"]])
def test_oracle_rejects_incompatible_flags(argv):
    with pytest.raises(SystemExit):
        run_sim.parse_args(argv)


def test_main_oracle_builds_gz_oracle_source(tmp_path, monkeypatch):
    import src.sim.link as link_mod
    import src.sim.oracle as oracle_mod
    import src.sim.sources as sources_mod
    from src.sim.app import SimApp

    built = {}

    class FakeCamera:
        def __init__(self, topic):
            built["camera_topic"] = topic

    class FakeOracle:
        def __init__(self, topic, label_classes, *, min_box_px, max_age_s):
            built.update(oracle_topic=topic, label_classes=label_classes, min_box_px=min_box_px, max_age_s=max_age_s)

        def detect(self, frame, t):
            return []

    class FakeLink:
        def __init__(self, cfg, gimbal_enabled):
            pass

        def connect(self):
            built["connected"] = True

    monkeypatch.setattr(sources_mod, "GzCameraSource", FakeCamera)
    monkeypatch.setattr(oracle_mod, "GzOracleDetectionSource", FakeOracle)
    monkeypatch.setattr(link_mod, "PymavlinkLink", FakeLink)
    monkeypatch.setattr(SimApp, "run", lambda self: None)
    cfg = tmp_path / "sim.yaml"
    cfg.write_text("oracle:\n  bbox_topic: /from/yaml\n  label_classes: {3: car}\n  max_age_s: 2.0\n", encoding="utf-8")
    log = tmp_path / "sim.jsonl"
    run_sim.main(["--oracle", "--config", str(cfg), "--gz-topic", "/cam", "--oracle-topic", "/cam/bbox",
                  "--log", str(log)])
    assert built == {"camera_topic": "/cam", "oracle_topic": "/cam/bbox", "label_classes": {3: "car"},
                     "min_box_px": 0.0, "max_age_s": 2.0, "connected": True}
    first = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert first["detector"] == "gz_oracle" and first["weights"] is None
