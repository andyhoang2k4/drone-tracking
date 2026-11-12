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
