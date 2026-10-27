"""Bang thong bao hien tren man hinh camera (OSD). Chu khong dau vi font OpenCV khong ve duoc dau."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    text: str
    level: str  # info | warn | alert
    expires_at: float | None  # None = sticky


class MessageBoard:
    def __init__(self) -> None:
        self._timed: list[Message] = []
        self._sticky: dict[str, Message] = {}

    def post(self, text: str, now: float, level: str = "info", ttl_s: float = 4.0) -> None:
        self._timed.append(Message(text, level, now + ttl_s))

    def set_sticky(self, key: str, text: str | None, level: str = "warn") -> None:
        if text is None:
            self._sticky.pop(key, None)
        else:
            self._sticky[key] = Message(text, level, None)

    def active(self, now: float) -> list[Message]:
        self._timed = [m for m in self._timed if m.expires_at is not None and m.expires_at > now]
        return list(self._sticky.values()) + list(self._timed)
