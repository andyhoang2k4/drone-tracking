"""Logic chuyen che do bay + sinh lenh. Thuan logic, khong goi PX4 truc tiep (de test duoc)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import SimConfig
from .follow_controller import ZERO_VELOCITY, BodyVelocity, bearing_error_rad, follow_command
from .gimbal import GimbalTracker
from .messages import MessageBoard
from .search import BearingMemory, search_command
from .types import Detection, FlightMode, FrameSize, RCInput, TargetEstimate, TargetState, Telemetry


@dataclass
class FlightOutput:
    desired_mode: FlightMode  # che do link can dang o
    manual: tuple[float, float, float, float]  # pitch, roll, throttle (0..1), yaw
    arm: bool | None  # canh len/xuong cua cong tac arm; None = khong doi
    velocity: BodyVelocity  # setpoint offboard (0 khi khong bam)
    gimbal: tuple[float, float] | None  # (pitch_deg, yaw_deg)
    status: str  # dong trang thai cho OSD
    auto_engaged: bool


class FlightSupervisor:
    def __init__(self, cfg: SimConfig, board: MessageBoard) -> None:
        self._cfg = cfg
        self._board = board
        self._gimbal = GimbalTracker(cfg.gimbal)
        self._bearings = BearingMemory(cfg.search.fit_window_s)
        self._prev_arm_sw = False
        self._prev_req = FlightMode.STABILIZED
        self._last_manual = FlightMode.STABILIZED
        self._auto_armed = False  # luc gat sang Auto da co muc tieu
        self._prev_state: TargetState | None = None
        self._last_fwd = 0.0
        self._lost_since: float | None = None
        self._search_ref: tuple[float, float] | None = None  # (huong cuoi, toc do doi huong)
        self._had_lock = False

    def step(
        self,
        now: float,
        rc: RCInput,
        estimate: TargetEstimate | None,
        candidates: list[Detection],
        frame: FrameSize,
        telem: Telemetry,
    ) -> FlightOutput:
        cfg, board = self._cfg, self._board

        arm_edge: bool | None = None
        if rc.arm_switch != self._prev_arm_sw:
            arm_edge = rc.arm_switch
            self._prev_arm_sw = rc.arm_switch

        self._notify_state_change(now, estimate)
        self._update_bearing(now, estimate, frame, telem)

        req = rc.mode_switch
        if req != FlightMode.AUTO_FOLLOW:
            self._last_manual = req
            self._auto_armed = False
            board.set_sticky("need_target", None)
            board.set_sticky("relatch", None)
        elif self._prev_req != FlightMode.AUTO_FOLLOW:
            self._auto_armed = estimate is not None
        self._prev_req = req

        engaged = False
        if req == FlightMode.AUTO_FOLLOW:
            if estimate is None:
                names = ", ".join(f"{d.label}#{d.track_id}" for d in candidates) or "(chua thay vat the nao)"
                board.set_sticky(
                    "need_target",
                    f"CHUA CO MUC TIEU - chon vat the roi bam Lock. Vat the: {names}",
                    "alert",
                )
                board.set_sticky("relatch", None)
            else:
                board.set_sticky("need_target", None)
                if self._auto_armed or cfg.auto_engage_on_lock:
                    engaged = True
                    board.set_sticky("relatch", None)
                else:
                    board.set_sticky("relatch", "DA KHOA MUC TIEU - gat cong tac Auto xuong roi len lai de vao Auto", "warn")

        velocity = ZERO_VELOCITY
        status = "MANUAL"
        if engaged and estimate is not None:
            velocity, status = self._auto_velocity(now, estimate, frame, telem)

        gimbal_cmd = self._gimbal_step(estimate, frame) if cfg.gimbal.enabled else None

        desired = FlightMode.AUTO_FOLLOW if engaged else self._last_manual
        if not engaged and req == FlightMode.AUTO_FOLLOW:
            status = "MANUAL (cho muc tieu)"
        return FlightOutput(
            desired_mode=desired,
            manual=(rc.pitch, rc.roll, rc.throttle, rc.yaw),
            arm=arm_edge,
            velocity=velocity,
            gimbal=gimbal_cmd,
            status=status,
            auto_engaged=engaged,
        )

    # ---- noi bo ----
    def _notify_state_change(self, now: float, est: TargetEstimate | None) -> None:
        state = est.state if est else None
        if state != self._prev_state:
            b = self._board
            if state == TargetState.LOST:
                b.post("MUC TIEU BIEN MAT", now, "alert", 5.0)
            elif state == TargetState.REIDENTIFYING:
                b.post("DANG TIM MUC TIEU THEO HUONG CU", now, "warn", 6.0)
            elif state == TargetState.FAILED:
                b.post("KHONG TIM THAY MUC TIEU - GIU VI TRI", now, "alert", 8.0)
            elif state == TargetState.TRACKING and self._prev_state in (
                TargetState.LOST,
                TargetState.REIDENTIFYING,
                TargetState.FAILED,
            ):
                b.post("DA BAT LAI MUC TIEU", now, "info", 4.0)
            elif state == TargetState.TRACKING and self._prev_state is None:
                b.post("DA KHOA MUC TIEU", now, "info", 3.0)
            elif state is None and self._prev_state is not None:
                b.post("DA BO KHOA MUC TIEU", now, "info", 3.0)
            self._prev_state = state

    def _update_bearing(self, now: float, est: TargetEstimate | None, frame: FrameSize, telem: Telemetry) -> None:
        if est is None:
            self._bearings.clear()
            self._lost_since = None
            self._search_ref = None
            return
        if est.state == TargetState.TRACKING and est.bbox is not None:
            err = bearing_error_rad(est.bbox, frame, self._cfg.camera.hfov_deg, self._gimbal.yaw_deg)
            self._bearings.add(now, telem.yaw_rad + err)
            self._lost_since = None
            self._search_ref = None
        elif est.state != TargetState.TRACKING and self._lost_since is None:
            self._lost_since = now - est.seconds_since_confirmed
            if not self._bearings.empty:
                self._search_ref = self._bearings.last_and_rate()

    def _auto_velocity(
        self, now: float, est: TargetEstimate, frame: FrameSize, telem: Telemetry
    ) -> tuple[BodyVelocity, str]:
        cfg = self._cfg
        if est.state in (TargetState.TRACKING, TargetState.LOST) and est.bbox is not None:
            v = follow_command(est.bbox, frame, cfg.camera.hfov_deg, cfg.follow, self._gimbal.yaw_deg)
            self._last_fwd = v.forward
            return v, f"AUTO:{est.state.value}"
        if est.state == TargetState.REIDENTIFYING:
            if self._search_ref is None:
                return ZERO_VELOCITY, "AUTO:REIDENTIFYING"
            last, rate = self._search_ref
            since = now - (self._lost_since if self._lost_since is not None else now)
            v = search_command(last, rate, since, telem.yaw_rad, self._last_fwd, cfg.search)
            return v, "AUTO:REIDENTIFYING"
        return ZERO_VELOCITY, "AUTO:FAILED"

    def _gimbal_step(self, est: TargetEstimate | None, frame: FrameSize) -> tuple[float, float]:
        if est is None:
            if self._had_lock:
                self._gimbal.reset()
            self._had_lock = False
        else:
            self._had_lock = True
            if est.bbox is not None:  # TRACKING hoac LOST (du doan); REID/FAILED thi giu nguyen goc
                self._gimbal.update(est.bbox, frame, self._cfg.camera.hfov_deg)
        return self._gimbal.pitch_deg, self._gimbal.yaw_deg
