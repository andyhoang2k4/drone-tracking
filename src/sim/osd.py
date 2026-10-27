"""Ve OSD len anh camera. Chi ve len BAN SAO de hien thi; luong anh dua cho AI phai la anh goc."""
from __future__ import annotations

import cv2
import numpy as np

from .messages import Message
from .types import Detection, FlightMode, RCInput, TargetEstimate, TargetState, Telemetry

GREEN, YELLOW, RED, ORANGE, WHITE, GRAY = (80, 220, 80), (0, 230, 255), (60, 60, 255), (0, 165, 255), (240, 240, 240), (170, 170, 170)
LEVEL_COLOR = {"info": GREEN, "warn": YELLOW, "alert": RED}
BAR_H = 36
BUTTONS = {"lock": (10, "LOCK [L]"), "unlock": (130, "UNLOCK [U]"), "next": (270, "NEXT [N]")}
BTN_W = {"lock": 110, "unlock": 130, "next": 110}


def button_rects(frame_h: int) -> dict[str, tuple[int, int, int, int]]:
    y1, y2 = frame_h - BAR_H + 4, frame_h - 4
    return {k: (BUTTONS[k][0], y1, BUTTONS[k][0] + BTN_W[k], y2) for k in BUTTONS}


def _put(img: np.ndarray, s: str, org: tuple[int, int], scale: float = 0.5, color=WHITE, thick: int = 1) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _wrap(text: str, max_w: int, scale: float) -> list[str]:
    """Ngat dong theo chieu rong anh (OpenCV khong tu xuong dong)."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _dashed_rect(img: np.ndarray, b: tuple[int, int, int, int], color, step: int = 8) -> None:
    x1, y1, x2, y2 = b
    for x in range(x1, x2, step * 2):
        cv2.line(img, (x, y1), (min(x + step, x2), y1), color, 2)
        cv2.line(img, (x, y2), (min(x + step, x2), y2), color, 2)
    for y in range(y1, y2, step * 2):
        cv2.line(img, (x1, y), (x1, min(y + step, y2)), color, 2)
        cv2.line(img, (x2, y), (x2, min(y + step, y2)), color, 2)


def draw_osd(
    frame: np.ndarray,
    detections: list[Detection],
    selected_id: int | None,
    estimate: TargetEstimate | None,
    telem: Telemetry,
    rc: RCInput,
    desired_mode: FlightMode,
    status: str,
    messages: list[Message],
    gimbal: tuple[float, float] | None,
) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    locked_id = estimate.track_id if estimate else None

    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        if d.track_id == locked_id:
            continue
        sel = d.track_id == selected_id
        cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW if sel else GRAY, 3 if sel else 1)
        _put(img, f"{d.label}#{d.track_id} {d.conf:.2f}", (x1, max(14, y1 - 6)), 0.45, YELLOW if sel else GRAY)

    if estimate is not None:
        if estimate.state == TargetState.TRACKING and estimate.bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in estimate.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 3)
            _put(img, f"LOCK #{estimate.track_id}", (x1, max(14, y1 - 6)), 0.5, GREEN)
        elif estimate.state == TargetState.LOST and estimate.bbox is not None:
            _dashed_rect(img, tuple(int(v) for v in estimate.bbox), ORANGE)
            _put(img, f"LOST - du doan {estimate.prediction_confidence:.0%}",
                 (int(estimate.bbox[0]), max(14, int(estimate.bbox[1]) - 6)), 0.5, ORANGE)

    cv2.drawMarker(img, (w // 2, h // 2), WHITE, cv2.MARKER_CROSS, 18, 1)

    armed = "ARMED" if telem.armed else "DISARM"
    rows = [
        f"MODE {desired_mode.value}  [{telem.mode_name}]",
        f"{armed}  {status}",
        f"ALT {telem.rel_alt_m:5.1f} m   SPD {telem.ground_speed_ms:4.1f} m/s",
        f"BAT {telem.battery_pct:3.0f}%  GPS fix{telem.gps_fix} sat{telem.gps_sats}"
        if telem.battery_pct >= 0 else f"GPS fix{telem.gps_fix} sat{telem.gps_sats}",
        f"THR {rc.throttle:4.2f}",
    ]
    for i, r in enumerate(rows):
        _put(img, r, (10, 20 + i * 18), 0.5, RED if (i == 1 and not telem.armed) else WHITE)
    if gimbal is not None:
        _put(img, f"GIMBAL pitch {gimbal[0]:+.0f} yaw {gimbal[1]:+.0f}", (10, h - BAR_H - 10), 0.45, WHITE)
    if not telem.connected:
        _put(img, "NO LINK", (w - 110, 22), 0.6, RED, 2)

    y = 120
    for m in messages[-4:]:
        for line in _wrap(m.text, w - 24, 0.6):
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            _put(img, line, (max(8, (w - tw) // 2), y), 0.6, LEVEL_COLOR.get(m.level, WHITE), 2)
            y += 24
        y += 6

    cv2.rectangle(img, (0, h - BAR_H), (w, h), (30, 30, 30), -1)
    for k, r in button_rects(h).items():
        cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (90, 90, 90), -1)
        _put(img, BUTTONS[k][1], (r[0] + 8, r[3] - 9), 0.45, WHITE)
    _put(img, "Click vat the de chon, LOCK de khoa", (w - 270, h - 12), 0.42, GRAY)
    return img
