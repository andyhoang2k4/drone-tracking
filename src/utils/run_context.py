"""Thu thap thong tin tai hien experiment (CLAUDE.md muc 14).

Moi lan chay luu: git commit hash, dataset dung, config, version
ultralytics/torch. O vong benchmark cuoi bo sung them Python/CUDA/GPU model.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional


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


def build_run_context(
    *,
    experiment_id: str,
    config: Mapping[str, Any],
    dataset: Optional[str] = None,
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
