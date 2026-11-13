"""Resolve tham so Motion Memory - khong hardcode.

Thu tu uu tien (cao -> thap):
    1. Gia tri CLI (chi cac tham so run_pipeline.py mo ra, vd
       --lost-timeout-frames, --reid-timeout-frames); None = khong truyen.
    2. Khoa `motion_memory` trong configs/motion_memory.yaml (hoac file chi dinh).
    3. Mac dinh cua MotionMemoryConfig.

File config bat buoc ton tai; khoa la -> ValueError (tranh go sai ten tham
so ma khong biet). Gia tri null trong file = dung mac dinh dataclass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from src.motion_memory import MotionMemoryConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOTION_MEMORY_CONFIG = REPO_ROOT / "configs" / "motion_memory.yaml"
CONFIG_KEY = "motion_memory"

PathLike = Union[str, Path]


@dataclass(frozen=True)
class MotionMemoryChoice:
    config: MotionMemoryConfig
    config_path: str
    sources: dict[str, str] = field(default_factory=dict)  # ten tham so -> cli|file|default

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "config_path": self.config_path,
            "sources": dict(self.sources),
        }


def _field_types() -> dict[str, str]:
    # `from __future__ import annotations` -> f.type la chuoi ("int"/"float")
    return {f.name: str(f.type) for f in fields(MotionMemoryConfig)}


def _check_type(name: str, value: Any, type_name: str, where: str) -> Any:
    if isinstance(value, bool):
        raise ValueError(f"{name}={value!r} trong {where}: khong nhan bool")
    if type_name == "int":
        if not isinstance(value, int):
            raise ValueError(f"{name}={value!r} trong {where}: can so nguyen")
        return value
    if type_name == "float":
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name}={value!r} trong {where}: can so thuc")
        return float(value)
    raise ValueError(f"Kieu chua ho tro cho {name}: {type_name}")


def load_motion_memory_file(config_path: PathLike) -> dict[str, Any]:
    """Doc khoa `motion_memory` (bo cac gia tri null)."""
    import yaml

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Khong tim thay config Motion Memory: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = data.get(CONFIG_KEY)
    if not isinstance(section, Mapping):
        raise ValueError(f"{path}: thieu khoa '{CONFIG_KEY}' dang mapping")
    types = _field_types()
    unknown = sorted(set(section) - set(types))
    if unknown:
        raise ValueError(f"{path}: tham so Motion Memory khong ton tai: {unknown}")
    return {
        name: _check_type(name, value, types[name], str(path))
        for name, value in section.items()
        if value is not None
    }


def resolve_motion_memory_config(
    cli_overrides: Optional[Mapping[str, Any]] = None,
    config_path: PathLike = DEFAULT_MOTION_MEMORY_CONFIG,
) -> MotionMemoryChoice:
    types = _field_types()
    from_file = load_motion_memory_file(config_path)
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name, type_name in types.items():
        cli_value = (cli_overrides or {}).get(name)
        if cli_value is not None:
            values[name] = _check_type(name, cli_value, type_name, "CLI")
            sources[name] = "cli"
        elif name in from_file:
            values[name] = from_file[name]
            sources[name] = "file"
        else:
            sources[name] = "default"
    unknown_cli = sorted(set(cli_overrides or {}) - set(types))
    if unknown_cli:
        raise ValueError(f"Tham so CLI Motion Memory khong ton tai: {unknown_cli}")
    config = MotionMemoryConfig(**values)  # validate khoang gia tri o __post_init__
    return MotionMemoryChoice(config=config, config_path=str(config_path), sources=sources)
