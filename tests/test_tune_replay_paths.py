"""Tune replay voi meta DINH DANG THAT (loi that tren cloud, 2026-11-14).

Ablation nguon chay voi --scenario-root TUONG DOI -> tracks.meta.json (ghi boi
src.track qua write_inference_meta, khong viet tay) chua source tuong doi;
run_pipeline --replay-tracks doi chieu source/weights/tracker voi meta. Tune
phai truyen dung cac gia tri do, voi ca kieu duong dan tuong doi lan tuyet doi.

Replay o day chay qua SUBPROCESS THAT (tune.subprocess_runner, cwd = goc repo)
nen kich ban nam trong thu muc tam DUOI goc repo (data/raw/_pytest_tune_*,
gitignore, xoa sau test) - duong dan tuong doi phai tuong doi voi goc repo,
giong cloud.
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_occlusion_ablation as abl  # noqa: E402
import scripts.run_pipeline as rp  # noqa: E402
import scripts.tune_motion_memory as tune  # noqa: E402
import src.track as tr  # noqa: E402
from tests.test_run_occlusion_ablation import FakeRunner, fake_tracks_v2, make_group_v2  # noqa: E402

DETS = ("yolo11", "yolo26")


@pytest.fixture
def repo_tmp():
    d = REPO_ROOT / "data" / "raw" / f"_pytest_tune_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _arg(path: Path, style: str) -> str:
    return path.relative_to(REPO_ROOT).as_posix() if style == "relative" else str(path)


def build_source(monkeypatch, repo_tmp: Path, tmp_path: Path, style: str) -> dict:
    """Ablation nguon v2_001 (src.track THAT ghi meta, YOLO gia) voi duong dan kieu `style`."""
    monkeypatch.chdir(REPO_ROOT)            # nhu cloud: chay tu goc repo
    monkeypatch.setattr(tr, "iter_tracks", lambda w, s, **k: iter(fake_tracks_v2(s)))

    def no_yolo(*a, **k):
        raise AssertionError("khong duoc chay YOLO khi replay")

    monkeypatch.setattr(rp, "iter_tracks", no_yolo)
    weights = {}
    for name in DETS:
        weights[name] = repo_tmp / f"{name}.pt"
        weights[name].write_bytes(f"weights-{name}".encode())
    scn = repo_tmp / "scn"
    names = make_group_v2(scn, weights)
    exps = tmp_path / "experiments"
    args = ["--group", "dev", "--tag", "v2_001", "--scenario-root", _arg(scn, style), "--output-root", str(exps)]
    for name in DETS:
        args += ["--detector", f"{name}={_arg(weights[name], style)}"]
    assert abl.main(args, FakeRunner()) == 0
    meta_path = exps / "yolo11_botsort_tracks_occl_dev_v2_001" / "scenarios" / names[0] / "logs" / "tracks.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert Path(meta["source"]).is_absolute() == (style == "absolute")
    return {"scn": scn, "exps": exps, "names": names, "weights": weights, "meta": meta}


@pytest.mark.parametrize("source_style,tune_style", [("relative", "absolute"), ("absolute", "relative")])
def test_replay_through_real_subprocess_with_real_meta(monkeypatch, repo_tmp, tmp_path, source_style, tune_style):
    """Loi that: ablation tuong doi + tune tuyet doi (mac dinh) -> moi replay chet. Va chieu nguoc lai."""
    src = build_source(monkeypatch, repo_tmp, tmp_path, source_style)
    argv = ["--round", "1", "--limit-scenarios", "1", "--jobs", "4",
            "--scenario-root", _arg(src["scn"], tune_style), "--output-root", str(src["exps"]),
            "--tuned-output", str(tmp_path / "tuned.yaml")]
    tag = tune.parse_args(argv).tag                                # tag tu danh so, tinh TRUOC khi chay
    assert tune.main(argv) == 0                                    # tune.subprocess_runner: subprocess that
    for det in DETS:
        exp = src["exps"] / f"{det}_botsort_mm_occl_dev_{tag}_default"
        agg = json.loads((exp / "metrics.json").read_text(encoding="utf-8"))
        assert agg["overall"]["scenarios_total"] == 1
        ctx = json.loads((exp / "scenarios" / src["names"][0] / "run_context.json").read_text(encoding="utf-8"))
        meta = json.loads((src["exps"] / f"{det}_botsort_tracks_occl_dev_v2_001" / "scenarios" / src["names"][0]
                           / "logs" / "tracks.meta.json").read_text(encoding="utf-8"))
        for key in ("source", "weights", "tracker"):                  # dung chuoi cua meta (= baseline nguon)
            assert ctx["config"][key] == meta[key]


@pytest.mark.parametrize("source_style", ["relative", "absolute"])
@pytest.mark.parametrize("tune_style", ["relative", "absolute"])
def test_every_meta_checked_param_passed_correctly(monkeypatch, repo_tmp, tmp_path, source_style, tune_style):
    """Moi tham so run_pipeline doi chieu voi meta:
    - REPLAY_PATH_KEYS (weights, source, tracker): bang meta (so theo Path) - tune truyen
      NGUYEN CHUOI cua meta (evaluate compare con so chuoi voi baseline nguon);
    - REPLAY_VALUE_KEYS (conf, imgsz, classes): neu CLI co thi phai bang meta - tune KHONG
      truyen, run_pipeline lay tu meta; meta phai co imgsz + conf.
    Kiem tra bang chinh rp.parse_args + rp.apply_replay_meta (khong nhai lai logic)."""
    assert rp.REPLAY_PATH_KEYS == ("weights", "source", "tracker")
    assert rp.REPLAY_VALUE_KEYS == ("conf", "imgsz", "classes")
    src = build_source(monkeypatch, repo_tmp, tmp_path, source_style)
    argv = ["--round", "1", "--limit-scenarios", "3", "--scenario-root", _arg(src["scn"], tune_style),
            "--output-root", str(src["exps"]), "--dry-run"]
    ctx = tune.build_context(tune.parse_args(argv))
    for det in DETS:
        for scen in ctx.scenarios:
            meta = json.loads(rp.replay_meta_path(ctx.tracks_log(det, scen)).read_text(encoding="utf-8"))
            cmd = tune.unit_pipeline_command(ctx, det, "default", scen, ctx.mm_config_path("default"))
            ns = rp.parse_args(cmd[2:])
            assert ns.replay_tracks is not None and ns.motion_memory
            for key in rp.REPLAY_VALUE_KEYS:
                assert getattr(ns, key) is None                 # khong truyen -> lay tu meta
            rp.apply_replay_meta(ns, meta)                       # ReplayError neu khong khop
            assert cmd[cmd.index("--source") + 1] == meta["source"]
            assert cmd[cmd.index("--weights") + 1] == meta["weights"]
            assert "--tracker" not in cmd and str(rp.DEFAULT_TRACKER_CONFIG) == meta["tracker"]


def test_meta_pointing_elsewhere_is_refused(monkeypatch, repo_tmp, tmp_path):
    src = build_source(monkeypatch, repo_tmp, tmp_path, "relative")
    other = repo_tmp / "copy_of_scn"
    shutil.copytree(src["scn"], other)
    argv = ["--round", "1", "--limit-scenarios", "1", "--scenario-root", str(other),
            "--output-root", str(src["exps"]), "--dry-run"]
    with pytest.raises(SystemExit, match="khong tro toi"):
        tune.build_context(tune.parse_args(argv))


# ---------------- pre-flight ----------------

LONG_ERROR = "\n".join([f"dong traceback {i}" for i in range(12)]
                       + ["scripts.run_pipeline.ReplayError: Log track khong khop tham so:",
                          "  source: CLI '/abs/x/images' khac meta 'x/images'"])


class RecordingRunner:
    def __init__(self, result):
        self.result, self.calls = result, []

    def __call__(self, cmd):
        self.calls.append(cmd)
        return self.result(cmd) if callable(self.result) else self.result


def test_preflight_live_failure_stops_before_round_with_full_output(monkeypatch, repo_tmp, tmp_path):
    src = build_source(monkeypatch, repo_tmp, tmp_path, "relative")
    runner = RecordingRunner((1, LONG_ERROR))
    argv = ["--round", "1", "--scenario-root", _arg(src["scn"], "absolute"), "--output-root", str(src["exps"])]
    with pytest.raises(SystemExit) as exc:
        tune.main(argv, runner)
    msg = str(exc.value)
    assert "Pre-flight THAT BAI (yolo11, default" in msg
    assert "dong traceback 0" in msg and "source: CLI '/abs/x/images'" in msg      # day du, khong chi 5 dong
    assert "--replay-tracks" in msg                                                 # vi du lenh
    assert len(runner.calls) == 1                                                   # dung ngay o detector dau
    assert not list(src["exps"].glob("*_tune_default/scenarios"))                   # chua chay kich ban nao


def test_preflight_static_mismatch_refused_even_in_dry_run(monkeypatch, repo_tmp, tmp_path):
    src = build_source(monkeypatch, repo_tmp, tmp_path, "relative")
    meta_path = (src["exps"] / "yolo26_botsort_tracks_occl_dev_v2_001" / "scenarios" / src["names"][1]
                 / "logs" / "tracks.meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["tracker"] = "configs/botsort_khac.yaml"                   # run_pipeline se tu choi (tracker khac meta)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    runner = RecordingRunner(0)
    for extra in (["--dry-run"], []):
        with pytest.raises(SystemExit, match="1/6 lenh replay KHONG khop meta") as exc:
            tune.main(["--round", "1", "--scenario-root", _arg(src["scn"], "relative"),
                       "--output-root", str(src["exps"])] + extra, runner)
        assert "tracker" in str(exc.value) and "yolo26/" + src["names"][1] in str(exc.value)
    assert runner.calls == []


# ---------------- dung som ----------------

def _replay_error(cmd) -> str:
    scen = cmd[cmd.index("--experiment-id") + 1]
    return LONG_ERROR.replace("/abs/x/images", f"/abs/{scen}/images").replace("'x/images'", f"'{scen}/images'")


def test_error_signature_ignores_per_scenario_details():
    a = tune.error_signature(_replay_error(["--experiment-id", "s1"]), 1)
    b = tune.error_signature(_replay_error(["--experiment-id", "s2"]), 1)
    assert a == b == "scripts.run_pipeline.ReplayError: Log track khong khop tham so:"
    assert tune.error_signature("", 3) == "ma thoat 3, khong co output"
    assert tune.error_signature("FileNotFoundError: '/a/b1.txt' 12 lan", 1) == "FileNotFoundError: '...' N lan"


def _replay_runner(fail):
    """Pre-flight (thu muc tam) thanh cong; lenh replay cua kich ban: fail(n) -> loi."""
    state = {"n": 0}

    def run(cmd):
        if not cmd[1].endswith("run_pipeline.py"):
            return 0
        if "tune_preflight_" in cmd[cmd.index("--output-root") + 1]:
            return 0
        state["n"] += 1
        return (1, _replay_error(cmd)) if fail(state["n"]) else (0, "")
    return run


def test_early_stop_when_most_commands_fail_with_same_message(monkeypatch, repo_tmp, tmp_path, capsys):
    src = build_source(monkeypatch, repo_tmp, tmp_path, "relative")
    runner = RecordingRunner(_replay_runner(lambda n: True))
    argv = ["--round", "1", "--jobs", "1", "--scenario-root", _arg(src["scn"], "absolute"),
            "--output-root", str(src["exps"])]
    assert tune.main(argv, runner) == 1
    out = capsys.readouterr().out
    replays = [c for c in runner.calls if c[1].endswith("run_pipeline.py")]
    assert len(replays) == len(DETS) + tune.EARLY_STOP_MIN_COMMANDS      # pre-flight + 20 roi dung (48 du kien)
    assert not [c for c in runner.calls if "aggregate" in c]               # khong sang dot sau
    assert "DUNG SOM (dot hau ky" in out and "20/20 lenh" in out
    assert "Thong bao: scripts.run_pipeline.ReplayError: Log track khong khop tham so:" in out
    assert "Vi du lenh: " in out and "--replay-tracks" in out
    assert "dong traceback 0" in out                                       # output day du (> 5 dong cuoi)


def test_scattered_failures_do_not_stop_early(monkeypatch, repo_tmp, tmp_path, capsys):
    src = build_source(monkeypatch, repo_tmp, tmp_path, "relative")
    runner = RecordingRunner(_replay_runner(lambda n: n % 3 == 0))        # 1/3 loi (< 50%)
    argv = ["--round", "1", "--jobs", "2", "--scenario-root", _arg(src["scn"], "absolute"),
            "--output-root", str(src["exps"])]
    assert tune.main(argv, runner) == 1                                    # co loi -> ma 1, nhung chay het
    replays = [c for c in runner.calls if c[1].endswith("run_pipeline.py")]
    assert len(replays) == len(DETS) + 8 * len(DETS) * len(src["names"])
    assert "DUNG SOM" not in capsys.readouterr().out
