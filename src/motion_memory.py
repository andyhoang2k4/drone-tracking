"""Motion memory / temporal propagation.

Module TU VIET, hoat dong theo kieu POST-PROCESSING: nhan output/state tu
BoT-SORT lam input, khong sua ma nguon noi bo cua ultralytics.
Xem CLAUDE.md muc 5 va muc 6.

Pham vi: chi lam viec tren BOUNDING BOX, khong phai segmentation mask.

Trang thai file (roadmap buoc 5, dang lam):
    - State machine theo bang transition o CLAUDE.md muc 6: DA implement.
    - Ngoai suy vi tri (`_extrapolate`): DA implement - van toc khong doi co
      suy giam, uoc luong bang hoi quy tuyen tinh tam bbox theo frame_idx.
    - Prediction confidence: DA implement - giam tuyen tinh theo motion
      horizon, nhan he so chat luong mo hinh (du lieu + residual).
    - Chua co: khop lai bang gating, chinh sach REIDENTIFYING (commit sau).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional, Sequence


class TrackState(str, Enum):
    """Trang thai cua muc tieu dang duoc theo doi."""

    TRACKING = "TRACKING"
    LOST = "LOST"
    REIDENTIFYING = "REIDENTIFYING"
    FAILED = "FAILED"


@dataclass(frozen=True)
class BBox:
    """Bounding box dinh dang xyxy, don vi pixel."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True)
class Observation:
    """Mot lan quan sat confirmed tu detector + tracker.

    detector_confidence duoc giu RIENG, khong gop vao prediction confidence
    (xem CLAUDE.md muc 6).
    """

    bbox: BBox
    frame_idx: int
    timestamp: Optional[float] = None
    track_id: Optional[int] = None
    detector_confidence: Optional[float] = None


@dataclass(frozen=True)
class MotionPrediction:
    """Output cua motion memory - dung interface o CLAUDE.md muc 6."""

    bbox: Optional[BBox]
    confidence: float
    frames_since_confirmed: int
    state: TrackState


@dataclass
class MotionMemoryConfig:
    """Nguong dieu khien state machine.

    lost_timeout_frames:
        So frame toi da o state LOST truoc khi chuyen sang REIDENTIFYING.
        Day la "motion horizon" - qua nguong nay ngoai suy chuyen dong khong
        con dang tin.
    reid_timeout_frames:
        So frame toi da o state REIDENTIFYING truoc khi chuyen sang FAILED.
    history_size:
        So observation gan nhat duoc giu lai de uoc luong chuyen dong.
    velocity_window_frames:
        Cua so uoc luong van toc, tinh theo FRAME (khong theo so observation):
        chi dung observation co frame_idx > frame_idx_cuoi - window. Sau mot
        khoang mat dai, observation cu roi khoi cua so nen khong tron chuyen
        dong cu vao van toc moi.
    min_history:
        So observation toi thieu trong cua so de uoc luong van toc. It hon
        -> gia dinh dung yen (van toc 0).
    velocity_decay:
        He so suy giam van toc moi frame du doan, trong (0, 1]. Van toc o
        frame du doan thu i la v * decay**(i-1) -> do doi bi chan boi
        v / (1 - decay) khi decay < 1. 1.0 = van toc khong doi.
    min_speed_ratio:
        Nguong toc do toi thieu, TI LE theo kich thuoc bbox cuoi
        ((w + h) / 2) moi frame. Toc do nho hon -> coi la dung yen (loc jitter
        cua bbox), khong ngoai suy.

    Cac gia tri mac dinh la gia tri khoi dau CHUA tinh chinh.
    """

    lost_timeout_frames: int = 15
    reid_timeout_frames: int = 45
    history_size: int = 30
    velocity_window_frames: int = 8
    min_history: int = 3
    velocity_decay: float = 0.95
    min_speed_ratio: float = 0.02

    def __post_init__(self) -> None:
        if self.lost_timeout_frames < 1 or self.reid_timeout_frames < 1:
            raise ValueError("lost_timeout_frames va reid_timeout_frames phai >= 1")
        if self.history_size < 2:
            raise ValueError("history_size phai >= 2")
        if self.velocity_window_frames < 2:
            raise ValueError("velocity_window_frames phai >= 2")
        if not 2 <= self.min_history <= self.history_size:
            raise ValueError("min_history phai trong [2, history_size]")
        if not 0.0 < self.velocity_decay <= 1.0:
            raise ValueError("velocity_decay phai trong (0, 1]")
        if self.min_speed_ratio < 0.0:
            raise ValueError("min_speed_ratio phai >= 0")


@dataclass(frozen=True)
class VelocityFit:
    """Ket qua uoc luong van toc tam bbox (pixel/frame)."""

    vx: float
    vy: float
    n_samples: int  # so observation trong cua so da dung
    # RMS khoang cach (pixel) giua tam quan sat va duong hoi quy - do lech
    # cua chuyen dong so voi mo hinh thang deu. 0 khi chua du observation.
    residual_rms: float = 0.0


class MotionMemory:
    """Theo doi mot track, du doan vi tri khi detection bi mat.

    Vong doi moi frame::

        pred = memory.update(observation)   # observation=None neu mat detection

    Bang transition (CLAUDE.md muc 6)::

        TRACKING      --mat detection--------------> LOST
        LOST          --detect lai, khop track-----> TRACKING
        LOST          --qua lost_timeout_frames----> REIDENTIFYING
        REIDENTIFYING --Re-ID match thanh cong-----> TRACKING
        REIDENTIFYING --qua reid_timeout_frames----> FAILED
    """

    def __init__(self, config: Optional[MotionMemoryConfig] = None) -> None:
        self.config = config or MotionMemoryConfig()
        self._history: Deque[Observation] = deque(maxlen=self.config.history_size)
        self._state: TrackState = TrackState.TRACKING
        self._frames_since_confirmed: int = 0
        self._frames_in_state: int = 0

    # ---------------- thuoc tinh doc ----------------

    @property
    def state(self) -> TrackState:
        return self._state

    @property
    def frames_since_confirmed(self) -> int:
        return self._frames_since_confirmed

    @property
    def history(self) -> Sequence[Observation]:
        return tuple(self._history)

    @property
    def last_confirmed(self) -> Optional[Observation]:
        return self._history[-1] if self._history else None

    # ---------------- API chinh ----------------

    def update(
        self,
        observation: Optional[Observation] = None,
        *,
        reid_matched: bool = False,
    ) -> MotionPrediction:
        """Cap nhat memory bang quan sat cua frame hien tai.

        Args:
            observation: quan sat confirmed cua frame nay, hoac None neu
                detector/tracker khong tra ve muc tieu.
            reid_matched: chi co y nghia khi dang o state REIDENTIFYING - bao
                rang `observation` den tu mot lan Re-ID match thanh cong.

        Returns:
            MotionPrediction cua frame hien tai.
        """
        if self._state is TrackState.FAILED:
            return self._emit(bbox=None, confidence=0.0)

        if observation is not None:
            accepted = self._state is not TrackState.REIDENTIFYING or reid_matched
            if accepted:
                self._accept(observation)
                return self._emit(bbox=observation.bbox, confidence=1.0)

        self._on_missing_detection()

        if self._state is TrackState.LOST:
            bbox = self._extrapolate(self._frames_since_confirmed)
            if bbox is None:  # chua co observation nao -> khong co gi de du doan
                return self._emit(bbox=None, confidence=0.0)
            return self._emit(bbox=bbox, confidence=self._prediction_confidence())

        # REIDENTIFYING / FAILED: ngoai suy chuyen dong khong con dang tin,
        # cho co che appearance-based Re-ID (xem CLAUDE.md muc 6).
        return self._emit(bbox=None, confidence=0.0)

    def reset(self) -> None:
        """Xoa toan bo lich su, quay ve TRACKING."""
        self._history.clear()
        self._state = TrackState.TRACKING
        self._frames_since_confirmed = 0
        self._frames_in_state = 0

    # ---------------- noi bo ----------------

    def _accept(self, observation: Observation) -> None:
        self._history.append(observation)
        self._transition(TrackState.TRACKING)
        self._frames_since_confirmed = 0

    def _on_missing_detection(self) -> None:
        self._frames_since_confirmed += 1
        self._frames_in_state += 1

        if self._state is TrackState.TRACKING:
            self._transition(TrackState.LOST)
        elif (
            self._state is TrackState.LOST
            # Dem tu lan confirmed gan nhat: motion horizon dai dung
            # lost_timeout_frames frame du doan.
            and self._frames_since_confirmed > self.config.lost_timeout_frames
        ):
            self._transition(TrackState.REIDENTIFYING)
        elif (
            self._state is TrackState.REIDENTIFYING
            and self._frames_in_state > self.config.reid_timeout_frames
        ):
            self._transition(TrackState.FAILED)

    def _transition(self, new_state: TrackState) -> None:
        if new_state is not self._state:
            self._state = new_state
            self._frames_in_state = 0

    def _emit(self, bbox: Optional[BBox], confidence: float) -> MotionPrediction:
        return MotionPrediction(
            bbox=bbox,
            confidence=confidence,
            frames_since_confirmed=self._frames_since_confirmed,
            state=self._state,
        )

    def _prediction_confidence(self) -> float:
        """Do tin cay cua DU DOAN trong LOST (khong phai cua detection).

        KHONG doc va KHONG gop voi detector_confidence (CLAUDE.md muc 6).

            conf = q * (1 - k / (lost_timeout_frames + 1))
            q    = h / (1 + residual_rms / scale)
            h    = min(1, n_samples / min_history)

        k = so frame tu lan confirmed cuoi; scale = (w + h) / 2 cua bbox cuoi.
        Thanh phan thoi gian gan voi motion horizon: giam tuyen tinh, van > 0
        o frame LOST cuoi cung. q phat khi it du lieu (h) hoac chuyen dong lech
        nhieu khoi duong thang (residual lon so voi kich thuoc bbox).
        Ket qua trong [0, 1] va khong tang theo k.
        """
        if not self._history:
            return 0.0
        k = self._frames_since_confirmed
        time_factor = max(0.0, 1.0 - k / (self.config.lost_timeout_frames + 1))

        last = self._history[-1].bbox
        scale = (last.width + last.height) / 2.0
        if scale <= 0.0:
            return 0.0
        fit = self._fit_velocity()
        history_factor = min(1.0, fit.n_samples / self.config.min_history)
        quality = history_factor / (1.0 + fit.residual_rms / scale)
        return min(1.0, max(0.0, quality * time_factor))

    def _window(self) -> list[Observation]:
        """Observation nam trong cua so van toc (theo frame_idx)."""
        if not self._history:
            return []
        last_frame = self._history[-1].frame_idx
        start = last_frame - self.config.velocity_window_frames
        return [obs for obs in self._history if obs.frame_idx > start]

    def _fit_velocity(self) -> VelocityFit:
        """Uoc luong van toc tam bbox bang hoi quy tuyen tinh theo frame_idx.

        Truc x la frame_idx that nen khoang trong (frame bi mat giua cac
        observation) duoc tinh dung. Tra ve van toc 0 khi:
            - it hon `min_history` observation trong cua so,
            - cac observation trung frame_idx (khong xac dinh duoc slope),
            - toc do < min_speed_ratio * kich thuoc bbox cuoi (dung yen/jitter).
        """
        window = self._window()
        n = len(window)
        if n < self.config.min_history:
            return VelocityFit(0.0, 0.0, n)

        ts = [float(obs.frame_idx) for obs in window]
        xs = [obs.bbox.center[0] for obs in window]
        ys = [obs.bbox.center[1] for obs in window]
        mean_t = sum(ts) / n
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        denom = sum((t - mean_t) ** 2 for t in ts)
        if denom <= 0.0:
            vx = vy = 0.0
        else:
            vx = sum((t - mean_t) * (x - mean_x) for t, x in zip(ts, xs)) / denom
            vy = sum((t - mean_t) * (y - mean_y) for t, y in zip(ts, ys)) / denom

        # Residual tinh tren duong hoi quy (truoc khi snap ve dung yen).
        sq = 0.0
        for t, x, y in zip(ts, xs, ys):
            fx = mean_x + vx * (t - mean_t)
            fy = mean_y + vy * (t - mean_t)
            sq += (x - fx) ** 2 + (y - fy) ** 2
        residual = (sq / n) ** 0.5

        last = window[-1].bbox
        scale = (last.width + last.height) / 2.0
        speed = (vx * vx + vy * vy) ** 0.5
        if scale <= 0.0 or speed < self.config.min_speed_ratio * scale:
            return VelocityFit(0.0, 0.0, n, residual)
        return VelocityFit(vx, vy, n, residual)

    def _extrapolate(self, frames_ahead: int) -> Optional[BBox]:
        """Ngoai suy bbox `frames_ahead` frame sau observation confirmed cuoi.

        Mo hinh: van toc khong doi co suy giam tren TAM bbox, giu nguyen kich
        thuoc bbox cuoi. Do doi sau k frame::

            d = v * (1 - decay**k) / (1 - decay)   (decay < 1)
            d = v * k                              (decay == 1)

        Tra ve None neu history rong. Khong bu chuyen dong camera (homography
        CMC nam trong BoT-SORT) va khong clamp vao khung anh.
        """
        if not self._history:
            return None
        last = self._history[-1].bbox
        k = max(0, frames_ahead)
        fit = self._fit_velocity()

        decay = self.config.velocity_decay
        if decay >= 1.0:
            gain = float(k)
        else:
            gain = (1.0 - decay**k) / (1.0 - decay)

        dx = fit.vx * gain
        dy = fit.vy * gain
        return BBox(last.x1 + dx, last.y1 + dy, last.x2 + dx, last.y2 + dy)
