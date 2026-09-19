"""Motion memory / temporal propagation.

Module TU VIET, hoat dong theo kieu POST-PROCESSING: nhan output/state tu
BoT-SORT lam input, khong sua ma nguon noi bo cua ultralytics.
Xem CLAUDE.md muc 5 va muc 6.

Pham vi: chi lam viec tren BOUNDING BOX, khong phai segmentation mask.

Trang thai file: KHUNG (scaffold).
    - State machine theo bang transition o CLAUDE.md muc 6: DA implement.
    - Thuat toan ngoai suy vi tri (`_extrapolate`) va prediction confidence:
      CHUA implement (roadmap buoc 5). Goi khi dang LOST se raise
      NotImplementedError.
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
    """

    lost_timeout_frames: int = 15
    reid_timeout_frames: int = 45
    history_size: int = 30


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
        """Do tin cay cua DU DOAN (khong phai cua detection).

        Khong duoc cong/gop voi detector confidence (CLAUDE.md muc 6).
        """
        raise NotImplementedError(
            "Prediction confidence chua implement - roadmap buoc 5."
        )

    def _extrapolate(self, frames_ahead: int) -> Optional[BBox]:
        """Ngoai suy bbox tu lich su chuyen dong gan nhat.

        CHUA IMPLEMENT - noi dung chinh cua roadmap buoc 5 (CLAUDE.md muc 16).
        Input co san: `self._history` (cac Observation confirmed gan nhat, kem
        frame_idx/timestamp) va `frames_ahead`.
        """
        raise NotImplementedError(
            "Temporal propagation chua implement - roadmap buoc 5."
        )
