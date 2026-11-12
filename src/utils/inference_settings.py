"""Chon imgsz cho inference/danh gia - khong hardcode.

Thu tu uu tien (cao -> thap):
    1. `--imgsz` truyen tren CLI.
    2. Experiment:
       a. `detector.imgsz` trong `<experiment_dir>/config.yaml` (neu co gia tri);
       b. `config.imgsz` trong `run_context.json` cua lan train sinh ra weights
          (tim tu thu muc chua weights di len toi da 3 cap, vd
          experiments/<id>/train/weights/best.pt -> experiments/<id>/).
    3. `common.imgsz` trong configs/train_config.yaml.

Khong tim duoc o dau -> raise, KHONG roi ve mot gia tri mac dinh ngam.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train_config.yaml"
RUN_CONTEXT_NAME = "run_context.json"
EXPERIMENT_CONFIG_NAME = "config.yaml"

# experiments/<id>/train/weights/best.pt: weights/ -> train/ -> <id>/
_MAX_LEVELS_UP = 3

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ImgszChoice:
    imgsz: int
    source: str
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"imgsz": self.imgsz, "imgsz_source": self.source, "imgsz_warnings": list(self.warnings)}


def _valid_imgsz(value: Any, where: str) -> Optional[int]:
    """None/"" -> None (chua dien). Gia tri sai kieu/<=0 -> ValueError."""
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"imgsz khong hop le ({value!r}) trong {where}")
    return value


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def find_train_run_context(weights: Optional[PathLike]) -> Optional[Path]:
    """run_context.json gan nhat phia tren file weights (toi da 3 cap)."""
    if not weights:
        return None
    directory = Path(weights).resolve().parent
    for _ in range(_MAX_LEVELS_UP):
        candidate = directory / RUN_CONTEXT_NAME
        if candidate.is_file():
            return candidate
        if directory.parent == directory:
            break
        directory = directory.parent
    return None


def _from_experiment_config(experiment_dir: Optional[PathLike]) -> Optional[tuple[int, str]]:
    if not experiment_dir:
        return None
    path = Path(experiment_dir) / EXPERIMENT_CONFIG_NAME
    if not path.is_file():
        return None
    detector = _load_yaml(path).get("detector") or {}
    value = _valid_imgsz(detector.get("imgsz"), str(path))
    return (value, f"experiment_config:{path}") if value is not None else None


def _from_train_run_context(path: Optional[Path]) -> Optional[tuple[int, str]]:
    if path is None:
        return None
    context = json.loads(path.read_text(encoding="utf-8"))
    value = _valid_imgsz((context.get("config") or {}).get("imgsz"), str(path))
    return (value, f"train_run_context:{path}") if value is not None else None


def _from_train_config(path: PathLike) -> Optional[tuple[int, str]]:
    path = Path(path)
    if not path.is_file():
        return None
    common = _load_yaml(path).get("common") or {}
    value = _valid_imgsz(common.get("imgsz"), str(path))
    return (value, f"train_config:{path}") if value is not None else None


def resolve_imgsz(
    cli_imgsz: Optional[int],
    *,
    experiment_dir: Optional[PathLike] = None,
    weights: Optional[PathLike] = None,
    train_config: PathLike = DEFAULT_TRAIN_CONFIG,
) -> ImgszChoice:
    """Chon imgsz theo thu tu uu tien o docstring cua module.

    `warnings`: canh bao neu imgsz chon duoc khac imgsz luc train weights
    (doc tu run_context.json cua lan train) - van chay, nhung can biet.
    """
    train_context = find_train_run_context(weights)
    trained = _from_train_run_context(train_context)

    if cli_imgsz is not None:
        chosen: Optional[tuple[int, str]] = (_valid_imgsz(cli_imgsz, "--imgsz"), "cli")
    else:
        chosen = (
            _from_experiment_config(experiment_dir)
            or trained
            or _from_train_config(train_config)
        )
    if chosen is None:
        raise ValueError(
            "Khong xac dinh duoc imgsz: khong co --imgsz, khong co detector.imgsz trong "
            f"config.yaml cua experiment, khong co run_context.json canh weights, va "
            f"{train_config} khong co common.imgsz. Truyen --imgsz."
        )

    imgsz, source = chosen
    warnings: list[str] = []
    if trained is not None and trained[0] != imgsz:
        warnings.append(
            f"imgsz={imgsz} ({source}) khac imgsz luc train={trained[0]} ({trained[1]})"
        )
    return ImgszChoice(imgsz=imgsz, source=source, warnings=warnings)


def write_inference_meta(output: PathLike, params: Mapping[str, Any], choice: ImgszChoice) -> Path:
    """Ghi tham so inference da dung (gom imgsz + nguon) canh log: <output>.meta.json.

    Dung de so sanh cong bang giua cac lan chay (vd co/khong Motion Memory).
    """
    from src.utils.logging_utils import write_json

    meta_path = Path(output).with_suffix(".meta.json")
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()}
    meta.update(choice.as_dict())
    write_json(meta_path, meta)
    return meta_path
