"""Kiem tra moi truong truoc khi train/test (CLAUDE.md muc 3, 4).

Chay duoc tren ca Windows (local, CPU) va Ubuntu (cloud, GPU). Script CHI BAO
CAO - khong tu sua requirements.txt hay config khi phat hien lech
(xem Dependency & Version Policy o CLAUDE.md muc 4).

Cac muc kiem tra:
    1. Python >= 3.10
    2. Version package da cai so voi pin trong requirements.txt
    3. torch: CUDA co dung duoc khong, GPU co nam trong arch list cua ban
       torch da cai khong (quan trong voi RTX 5060)
    4. configs/botsort.yaml co cung bo key voi botsort.yaml goc cua ultralytics
    5. (tuy chon, --check-weights) tai duoc weight khai bao trong
       configs/train_config.yaml khong (vd yolo26n-seg.pt)

Vi du::

    # Local Windows (CPU)
    python scripts/check_env.py

    # Cloud PC: bat buoc co CUDA, kiem tra ca weight, luu bao cao
    python scripts/check_env.py --require-cuda --check-weights \
        --output env_report.json

Exit code: 0 neu khong co FAIL, 1 neu co it nhat mot FAIL.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_utils import write_json  # noqa: E402
from src.utils.run_context import git_info, hardware_info  # noqa: E402

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
MIN_PYTHON = (3, 10)

_PIN_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s;#]+)")


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


# ---------------- ham thuan (co unit test) ----------------


def parse_pinned_requirements(text: str) -> dict[str, str]:
    """Lay cac dong `name==version` trong requirements.txt.

    Bo qua comment, dong trong, va cac dong khong pin cung (>=, ~=...).
    Ten package duoc chuan hoa ve chu thuong.
    """
    pins: dict[str, str] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        match = _PIN_RE.match(line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def base_version(version: str) -> str:
    """Bo phan local version, vd '2.7.1+cu128' -> '2.7.1'."""
    return version.split("+", 1)[0]


def compare_version(pinned: str, installed: Optional[str]) -> tuple[str, str]:
    """So version da cai voi pin. Tra ve (status, detail)."""
    if installed is None:
        return FAIL, f"chua cai (pin {pinned})"
    if base_version(installed) == base_version(pinned):
        return OK, installed
    return WARN, f"MISMATCH: da cai {installed}, requirements.txt pin {pinned}"


def compare_keys(
    project_keys: set[str], reference_keys: set[str]
) -> tuple[list[str], list[str]]:
    """Tra ve (key thieu so voi goc, key la khong co trong goc)."""
    missing = sorted(reference_keys - project_keys)
    unknown = sorted(project_keys - reference_keys)
    return missing, unknown


# ---------------- cac muc kiem tra ----------------


def check_python() -> CheckResult:
    version = sys.version.split()[0]
    if sys.version_info[:2] >= MIN_PYTHON:
        return CheckResult("python", OK, f"{version} ({sys.executable})")
    return CheckResult(
        "python", FAIL, f"{version} < {'.'.join(map(str, MIN_PYTHON))}"
    )


def check_requirements(requirements_path: Path) -> list[CheckResult]:
    if not requirements_path.exists():
        return [CheckResult("requirements", FAIL, f"khong thay {requirements_path}")]

    pins = parse_pinned_requirements(requirements_path.read_text(encoding="utf-8"))
    results: list[CheckResult] = []
    for name, pinned in pins.items():
        try:
            installed: Optional[str] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        status, detail = compare_version(pinned, installed)
        results.append(CheckResult(f"pkg:{name}", status, detail))
    return results


def check_torch_cuda(require_cuda: bool) -> list[CheckResult]:
    try:
        import torch
    except ImportError:
        return [CheckResult("cuda", FAIL if require_cuda else SKIP, "chua cai torch")]

    info = hardware_info()
    if not info["cuda_available"]:
        status = FAIL if require_cuda else WARN
        detail = (
            f"torch.cuda.is_available() = False (torch {torch.__version__}, "
            f"build CUDA {info['cuda_version']})"
        )
        if not require_cuda:
            detail += " - binh thuong neu dang o may local CPU"
        return [CheckResult("cuda", status, detail)]

    results = [
        CheckResult(
            "cuda",
            OK,
            f"{info['gpu_name']} | torch build CUDA {info['cuda_version']}",
        )
    ]

    # GPU moi (vd RTX 5060, sm_120) can ban torch co build cho arch do.
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    if arch in arch_list:
        results.append(CheckResult("cuda_arch", OK, f"{arch} co trong {arch_list}"))
    else:
        results.append(
            CheckResult(
                "cuda_arch",
                FAIL,
                f"GPU la {arch} nhung ban torch nay chi build cho {arch_list} "
                "- can cai torch build cho CUDA phu hop",
            )
        )
        return results

    # Chay thu mot phep tinh nho de chac chan kernel chay duoc that.
    try:
        x = torch.ones(8, device="cuda")
        _ = float((x * 2).sum().item())
        results.append(CheckResult("cuda_compute", OK, "tensor op tren GPU chay duoc"))
    except Exception as exc:  # noqa: BLE001 - muon bao moi loi CUDA
        results.append(CheckResult("cuda_compute", FAIL, repr(exc)))
    return results


def _load_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_botsort_schema(project_config: Path) -> CheckResult:
    try:
        import ultralytics
    except ImportError:
        return CheckResult("botsort_schema", SKIP, "chua cai ultralytics")
    try:
        import yaml  # noqa: F401
    except ImportError:
        return CheckResult("botsort_schema", SKIP, "chua cai PyYAML")

    reference = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
    if not reference.exists():
        return CheckResult(
            "botsort_schema", FAIL, f"khong thay botsort.yaml goc tai {reference}"
        )
    if not project_config.exists():
        return CheckResult("botsort_schema", FAIL, f"khong thay {project_config}")

    project = _load_yaml(project_config) or {}
    ref = _load_yaml(reference) or {}
    missing, unknown = compare_keys(set(project), set(ref))
    if not missing and not unknown:
        return CheckResult("botsort_schema", OK, f"khop key voi {reference}")
    parts = []
    if missing:
        parts.append(f"thieu so voi goc: {missing}")
    if unknown:
        parts.append(f"key la (goc khong co): {unknown}")
    return CheckResult("botsort_schema", WARN, "; ".join(parts) + f" | goc: {reference}")


def check_weights(train_config: Path, weights_dir: Path) -> list[CheckResult]:
    """Thu nap weight khai bao trong train_config.yaml (co the phai tai ve).

    Tai vao `weights_dir` (bi gitignore) thay vi thu muc goc repo.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return [CheckResult("weights", SKIP, "chua cai ultralytics")]
    try:
        cfg = _load_yaml(train_config)
    except ImportError:
        return [CheckResult("weights", SKIP, "chua cai PyYAML")]

    weights_dir.mkdir(parents=True, exist_ok=True)
    results: list[CheckResult] = []
    previous_cwd = Path.cwd()
    os.chdir(weights_dir)
    try:
        for key, model_cfg in (cfg.get("models") or {}).items():
            name = model_cfg["weights"]
            try:
                model = YOLO(name)
                results.append(
                    CheckResult(f"weights:{key}", OK, f"{name} (task={model.task})")
                )
            except Exception as exc:  # noqa: BLE001 - bao lai nguyen van loi
                # Khong tu doi sang model khac (CLAUDE.md muc 9).
                results.append(CheckResult(f"weights:{key}", FAIL, f"{name}: {exc!r}"))
    finally:
        os.chdir(previous_cwd)
    return results


# ---------------- CLI ----------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--requirements", type=Path, default=REPO_ROOT / "requirements.txt"
    )
    parser.add_argument(
        "--tracker-config", type=Path, default=REPO_ROOT / "configs" / "botsort.yaml"
    )
    parser.add_argument(
        "--train-config", type=Path, default=REPO_ROOT / "configs" / "train_config.yaml"
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Coi viec khong co CUDA la FAIL (dung tren cloud PC)",
    )
    parser.add_argument(
        "--check-weights",
        action="store_true",
        help="Thu nap/tai weight trong train_config.yaml (can mang)",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=REPO_ROOT / "weights",
        help="Thu muc tai weight ve (gitignore)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Ghi bao cao JSON")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    results: list[CheckResult] = [check_python()]
    results += check_requirements(args.requirements)
    results += check_torch_cuda(args.require_cuda)
    results.append(check_botsort_schema(args.tracker_config))
    if args.check_weights:
        results += check_weights(args.train_config, args.weights_dir)
    else:
        results.append(CheckResult("weights", SKIP, "bat bang --check-weights"))

    width = max(len(r.name) for r in results)
    for r in results:
        print(f"[{r.status:<4}] {r.name:<{width}}  {r.detail}")

    counts = {s: sum(r.status == s for r in results) for s in (OK, WARN, FAIL, SKIP)}
    print("\n" + "  ".join(f"{k}={v}" for k, v in counts.items()))
    if counts[WARN] or counts[FAIL]:
        print(
            "Co muc WARN/FAIL: bao lai truoc khi sua requirements.txt/config "
            "(CLAUDE.md muc 4)."
        )

    if args.output:
        write_json(
            args.output,
            {
                "git": git_info(REPO_ROOT),
                "hardware": hardware_info(),
                "results": [asdict(r) for r in results],
                "summary": counts,
            },
        )
        print(f"Da ghi bao cao: {args.output}")

    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
