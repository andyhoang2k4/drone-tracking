"""Thu thap thong tin tai hien experiment (CLAUDE.md muc 14).

Moi lan chay luu: git commit hash, dataset dung, config, version
ultralytics/torch. O vong benchmark cuoi bo sung them Python/CUDA/GPU model.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Union

# Ten file thong ke do scripts/prepare_visdrone.py ghi trong thu muc dataset.
DATASET_MANIFEST_NAME = "manifest.json"


def _git(*args: str, repo_root: Optional[Path] = None) -> Optional[str]:
    """Chay lenh git, tra ve None neu that bai (vd chua init repo)."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def git_info(repo_root: Optional[Path] = None) -> dict[str, Any]:
    """Commit hash + co dirty working tree hay khong."""
    commit = _git("rev-parse", "HEAD", repo_root=repo_root)
    status = _git("status", "--porcelain", repo_root=repo_root)
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", repo_root=repo_root),
        "dirty": bool(status) if status is not None else None,
    }


# Dinh nghia "cay lam viec sach" cho lan chay heldout (khac git_info()["dirty"]:
# `git status --porcelain` mac dinh tinh CA file untracked - vd chinh config.yaml /
# metrics.json ma lan chay vua ghi - nen moi lan chay dev deu bi danh dau dirty).
CLEAN_TREE_DEFINITION = (
    "sach <=> (1) `git status --porcelain --untracked-files=no` rong: khong file DA TRACK nao bi "
    "sua/staged/xoa/doi ten so voi HEAD; (2) moi file ke hoach tham chieu deu da track trong HEAD. "
    "File untracked va file bi .gitignore (output, du lieu) KHONG tinh."
)


def git_tracked_changes(repo_root: Optional[Path] = None) -> Optional[list[str]]:
    """Dong `git status --porcelain --untracked-files=no` (file da track bi doi); None neu khong co git."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo_root) if repo_root else None, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in out.stdout.splitlines() if line.strip()]


def git_untracked_paths(paths: list[Union[str, Path]], repo_root: Optional[Path] = None) -> Optional[list[str]]:
    """Cac duong dan CHUA track trong HEAD (rong = tat ca da track); None neu khong co git."""
    missing = []
    for path in paths:
        try:
            subprocess.run(["git", "cat-file", "-e", f"HEAD:{Path(path).as_posix()}"],
                           cwd=str(repo_root) if repo_root else None, capture_output=True, check=True)
        except OSError:
            return None
        except subprocess.CalledProcessError:
            missing.append(str(path))
    return missing


def package_versions() -> dict[str, Optional[str]]:
    """Version cua cac package chinh. None neu chua cai."""
    versions: dict[str, Optional[str]] = {}

    try:
        import ultralytics

        versions["ultralytics"] = ultralytics.__version__
    except ImportError:
        versions["ultralytics"] = None

    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = None

    try:
        import cv2

        versions["opencv"] = cv2.__version__
    except ImportError:
        versions["opencv"] = None

    return versions


def hardware_info() -> dict[str, Any]:
    """Thong tin GPU/CUDA - can cho vong benchmark cuoi (muc 14)."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_available": None,
        "cuda_version": None,
        "gpu_name": None,
    }
    try:
        import torch
    except ImportError:
        return info

    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def dataset_info(dataset_yaml: Path) -> dict[str, Any]:
    """Ten + version dataset ma mot dataset yaml cua Ultralytics tro toi.

    Doc `path` trong dataset yaml, roi doc `manifest.json` (do
    scripts/prepare_visdrone.py sinh) trong thu muc do. Khong co yaml/manifest
    -> name/version = None (khong raise), de nguoi goi tu canh bao.

    `partial`: True neu dataset la ban chay thu (prepare_visdrone.py
    --max-sequences, hoac generate_masks.py --max-images), False neu chay du,
    None neu khong xac dinh duoc (khong co manifest). Ly do nam trong
    `partial_reasons`.
    """
    info: dict[str, Any] = {
        "yaml": str(dataset_yaml),
        "path": None,
        "manifest": None,
        "name": None,
        "version": None,
        "partial": None,
        "partial_reasons": [],
    }
    try:
        import yaml

        data = yaml.safe_load(Path(dataset_yaml).read_text(encoding="utf-8")) or {}
    except OSError:
        return info
    if not data.get("path"):
        return info

    root = Path(data["path"])
    if not root.is_absolute():
        root = Path(dataset_yaml).parent / root
    info["path"] = str(root)
    manifest_path = root / DATASET_MANIFEST_NAME
    if not manifest_path.is_file():
        return info

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest.get("dataset") or {}
    info.update(
        manifest=str(manifest_path),
        name=dataset.get("name"),
        version=dataset.get("version"),
    )

    reasons: list[str] = []
    max_sequences = manifest.get("max_sequences")
    if max_sequences is not None:
        reasons.append(f"prepare_visdrone.py --max-sequences {max_sequences}")
    # generate_masks.py ghi max_images theo tung split: masks.splits.<split>.max_images
    mask_splits = (manifest.get("masks") or {}).get("splits") or {}
    for split, split_report in sorted(mask_splits.items()):
        mask_max_images = (split_report or {}).get("max_images")
        if mask_max_images is not None:
            reasons.append(f"generate_masks.py --max-images {mask_max_images} ({split})")
    info["partial_reasons"] = reasons
    if reasons:
        info["partial"] = True
    elif "max_sequences" in manifest:
        info["partial"] = False
    # Manifest khong co key max_sequences (ban cu) -> partial = None.
    return info


def partial_dataset_error(info: Mapping[str, Any], allow_partial: bool) -> Optional[str]:
    """Thong bao loi neu dataset la ban chay thu ma khong cho phep; None neu duoc chay."""
    if not info.get("partial") or allow_partial:
        return None
    reasons = ", ".join(info.get("partial_reasons") or []) or "khong ro"
    return (
        f"Dataset {info.get('name')} v{info.get('version')} la ban CHAY THU ({reasons}); "
        f"manifest: {info.get('manifest')}. Chay lai prepare_visdrone.py --overwrite "
        "(va generate_masks.py) khong co gioi han truoc khi train, hoac truyen "
        "--allow-partial neu co y train tren ban nay."
    )


def build_run_context(
    *,
    experiment_id: str,
    config: Mapping[str, Any],
    dataset: Union[str, Mapping[str, Any], None] = None,
    repo_root: Optional[Path] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Gop toan bo metadata cua mot lan chay thanh mot dict de ghi ra file."""
    context: dict[str, Any] = {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "git": git_info(repo_root),
        "versions": package_versions(),
        "hardware": hardware_info(),
        "config": dict(config),
    }
    if extra:
        context.update(extra)
    return context
