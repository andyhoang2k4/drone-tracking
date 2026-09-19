"""Log ket qua ra file co cau truc (CLAUDE.md muc 8).

Dung JSONL: moi dong la mot ban ghi JSON doc lap -> de append trong khi chay,
de doc lai bang pandas/json de tinh metric sau.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, TextIO


class JsonlLogger:
    """Ghi tung ban ghi ra file .jsonl, dong thoi co the in ra console.

    Dung nhu context manager::

        with JsonlLogger(Path("logs/run.jsonl")) as log:
            log.write({"frame": 0, "state": "TRACKING"})
    """

    def __init__(self, path: Path, *, echo: bool = False) -> None:
        self.path = Path(path)
        self.echo = echo
        self._handle: Optional[TextIO] = None

    def __enter__(self) -> "JsonlLogger":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, record: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlLogger chua duoc open()")
        line = json.dumps(record, ensure_ascii=False, default=str)
        self._handle.write(line + "\n")
        self._handle.flush()
        if self.echo:
            print(line)


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Ghi mot dict ra file JSON (tao thu muc cha neu chua co)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path
