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
    - Khop lai: gating chuyen dong trong LOST; REIDENTIFYING chi bat lai khi
      dung ID cu quay lai (hoac phia goi bao reid_matched).
    - Chua co: appearance-based Re-ID.
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

    Cung kieu nay dung cho UNG VIEN khop lai (`candidates` cua update()).
    `cls`: class id cua detector (tuy chon) - dung cho gating cung class.
    """

    bbox: BBox
    frame_idx: int
    timestamp: Optional[float] = None
    track_id: Optional[int] = None
    detector_confidence: Optional[float] = None
    cls: Optional[int] = None


@dataclass(frozen=True)
class MotionPrediction:
    """Output cua motion memory - dung interface o CLAUDE.md muc 6.

    track_id: ID tracker cua muc tieu dang theo (doi khi khop lai sang ID moi
    bang gating); None neu chua biet.
    recover_reason: CHI co gia tri o frame muc tieu duoc bat lai (tu LOST
    hoac REIDENTIFYING ve TRACKING), None o moi frame khac:
        "same_id"      - dung ID tracker cu quay lai,
        "motion_gate"  - ung vien ID khac qua gate chuyen dong (chi trong LOST),
        "reid_matched" - phia goi bao Re-ID match (update(reid_matched=True)),
        "anchor_id"    - (chi khi config.anchor_recovery) ID goc (anchor, ID
                         truoc lan khop motion_gate dau tien) xuat hien lai ->
                         quay ve ID goc, o MOI state tru FAILED, ke ca khi
                         dang TRACKING tren ID khac.
    gate_candidates_passed: so ung vien (ID khac) qua gate chuyen dong o lan
    goi nay; None neu gate KHONG duoc xet (khong o LOST, khong co ung vien
    ID khac, hoac da khop bang dung ID). 1 khi khop bang motion_gate; 0 hoac
    >= 2 (mo ho) khi van LOST.
    """

    bbox: Optional[BBox]
    confidence: float
    frames_since_confirmed: int
    state: TrackState
    track_id: Optional[int] = None
    recover_reason: Optional[str] = None
    gate_candidates_passed: Optional[int] = None


RECOVER_SAME_ID = "same_id"
RECOVER_MOTION_GATE = "motion_gate"
RECOVER_REID_MATCHED = "reid_matched"
RECOVER_ANCHOR_ID = "anchor_id"


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

    Gating khop lai trong LOST (ung vien co track ID khac). `scale` =
    (w + h) / 2 cua bbox confirmed cuoi; `gap` = frame_idx ung vien -
    frame_idx confirmed cuoi (chenh lech frame THAT, khong theo so lan goi).
    match_iou:
        IoU toi thieu voi predicted bbox (tai `gap`) de qua gate.
    match_center_ratio, match_gate_growth:
        Hoac khoang cach tam <= scale * match_center_ratio
        * (1 + match_gate_growth * gap) - gate noi dan theo thoi gian mat.
    max_scale_change:
        Ti le dien tich ung vien / bbox cuoi phai trong
        [1 / max_scale_change, max_scale_change].
    Ung vien phai cung class (neu muc tieu co class) va CHI chap nhan khi
    dung mot ung vien qua gate.

    anchor_recovery:
        Ghi nho ID goc (anchor) khi khop motion_gate sang ID khac; anchor xuat
        hien lai -> quay ve anchor (xem MotionMemory, "Anchor"). Mac dinh
        False = hanh vi cu (tai lap moi ket qua truoc 2026-11-15).

    Cac gia tri mac dinh la gia tri khoi dau CHUA tinh chinh.
    """

    lost_timeout_frames: int = 15
    reid_timeout_frames: int = 45
    history_size: int = 30
    velocity_window_frames: int = 8
    min_history: int = 3
    velocity_decay: float = 0.95
    min_speed_ratio: float = 0.02
    match_iou: float = 0.3
    match_center_ratio: float = 0.5
    match_gate_growth: float = 0.1
    max_scale_change: float = 2.0
    anchor_recovery: bool = False

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
        if not 0.0 < self.match_iou <= 1.0:
            raise ValueError("match_iou phai trong (0, 1]")
        if self.match_center_ratio <= 0.0:
            raise ValueError("match_center_ratio phai > 0")
        if self.match_gate_growth < 0.0:
            raise ValueError("match_gate_growth phai >= 0")
        if self.max_scale_change < 1.0:
            raise ValueError("max_scale_change phai >= 1")
        if not isinstance(self.anchor_recovery, bool):
            raise ValueError("anchor_recovery phai la bool")


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

    FAILED la trang thai cuoi (chi reset() moi thoat).

    Anchor (chi khi config.anchor_recovery = True; mac dinh tat):
        - Dat anchor_id = ID dang theo dung luc chap nhan motion_gate sang ID
          KHAC, chi khi anchor dang trong (B -> C van giu anchor A).
        - Moi lan goi (tru FAILED; sau nhanh reid_matched), TRUOC nhanh
          same_id va TRUOC gate: neu track co ID = anchor co trong observation
          hoac candidates -> chap nhan no, quay ve anchor (recover_reason
          "anchor_id"), xoa anchor. Khong them dieu kien khoang cach.
        - Xoa anchor khi quay ve anchor, khi reset(), khi vao FAILED.
        - Anchor KHONG co han. Gia dinh: tracker (BoT-SORT cua ultralytics)
          KHONG tai su dung track ID trong mot lan chay, va bo track da mat
          sau track_buffer (configs/botsort.yaml: 30 frame) nen ID goc khong
          the "quay lai" sai thoi diem. Doi tracker / track_buffer / cach gan
          ID -> xem lai gia dinh nay.
        - Gioi han: lich su KHONG bi xoa khi quay ve anchor - quan sat cua
          track da "cuop" (B) van nam trong _history, nen van toc uoc luong bi
          tron quan sat cua B trong khoang velocity_window_frames frame sau
          khi quay ve.
        - Rui ro: tracker doi ID nham (ID anchor bi gan cho vat khac) ->
          quay ve sai vat; hoac B moi dung muc tieu -> roi muc tieu dung.
    """

    def __init__(self, config: Optional[MotionMemoryConfig] = None) -> None:
        self.config = config or MotionMemoryConfig()
        self._history: Deque[Observation] = deque(maxlen=self.config.history_size)
        self._state: TrackState = TrackState.TRACKING
        self._frames_since_confirmed: int = 0
        self._track_id: Optional[int] = None
        self._anchor_id: Optional[int] = None  # chi dung khi config.anchor_recovery
        self._last_frame_idx: Optional[int] = None  # frame_idx cua lan goi truoc (neu co)

    # ---------------- thuoc tinh doc ----------------

    @property
    def track_id(self) -> Optional[int]:
        """ID tracker hien tai cua muc tieu (co the doi sau khi khop lai)."""
        return self._track_id

    @property
    def anchor_id(self) -> Optional[int]:
        """ID goc dang ghi nho (None neu khong co / co anchor_recovery tat)."""
        return self._anchor_id

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
        candidates: Optional[Sequence[Observation]] = None,
        *,
        reid_matched: bool = False,
        frame_idx: Optional[int] = None,
    ) -> MotionPrediction:
        """Cap nhat memory bang quan sat cua frame hien tai.

        Args:
            observation: quan sat confirmed cua muc tieu trong frame nay, hoac
                None neu detector/tracker khong tra ve muc tieu. Neu
                observation co track_id KHAC ID hien tai cua memory, no duoc
                xu ly nhu mot ung vien (phai qua gate).
            candidates: cac detection khac trong frame (tuy chon), dung de
                khop lai khi dang LOST va BoT-SORT da cap ID moi. Ung vien
                trung ID hien tai duoc coi la observation cua muc tieu.
            reid_matched: chi co y nghia khi dang o state REIDENTIFYING - bao
                rang `observation` den tu mot lan Re-ID match thanh cong (khong
                can trung ID). O state khac flag nay bi bo qua.
            frame_idx: chi so frame hien tai (tuy chon). Neu co,
                frames_since_confirmed = frame_idx - frame_idx confirmed cuoi
                (dung khi nguon bo frame hoac phia goi bo qua loi goi), va cac
                timeout so sanh tren so do. Neu khong co, dem +1 moi lan goi
                thieu detection (hanh vi cu). Phai tang dan qua cac lan goi
                va khop observation.frame_idx (neu co observation).

        Hai moc timeout deu tinh tren frames_since_confirmed (k)::

            k <= L                 LOST            (L = lost_timeout_frames)
            L < k <= L + R         REIDENTIFYING   (R = reid_timeout_frames)
            k >  L + R             FAILED

        Tuc la LOST dai dung L frame va REIDENTIFYING dai dung R frame.

        Neu k nhay qua nhieu moc trong mot lan goi (bo frame), state di qua
        cac moc lien tiep ngay trong lan goi do.

        Chinh sach REIDENTIFYING (quyet dinh B): dung ID tracker cu quay lai
        duoc coi la match hop le. KHONG dung gate chuyen dong voi ID khac -
        de danh cho appearance-based Re-ID sau nay. Khi config.anchor_recovery:
        nhan lai ID hien tai HOAC anchor (ID goc), van KHONG dung gate.

        Returns:
            MotionPrediction cua frame hien tai.
        """
        self._check_frame_idx(frame_idx, observation)
        if self._state is TrackState.FAILED:
            return self._emit(bbox=None, confidence=0.0)

        prev_state = self._state
        if prev_state is TrackState.REIDENTIFYING and reid_matched and observation is not None:
            self._accept(observation)
            return self._emit(observation.bbox, 1.0, RECOVER_REID_MATCHED)

        anchor = self._find_anchor(observation, candidates)
        if anchor is not None:
            self._anchor_id = None
            self._accept(anchor)
            return self._emit(anchor.bbox, 1.0, RECOVER_ANCHOR_ID)

        observation, others = self._split_target(observation, candidates)

        if observation is not None and (
            prev_state is not TrackState.REIDENTIFYING or self._is_current_id(observation)
        ):
            self._accept(observation)
            reason = RECOVER_SAME_ID if prev_state is not TrackState.TRACKING else None
            return self._emit(observation.bbox, 1.0, reason)

        # Gating chuyen dong CHI trong LOST (con trong motion horizon).
        gate_passed: Optional[int] = None
        if prev_state is TrackState.LOST and others:
            match, gate_passed = self._gate(others)
            if match is not None:
                if (
                    self.config.anchor_recovery
                    and self._anchor_id is None
                    and self._track_id is not None
                    and match.track_id != self._track_id
                ):
                    self._anchor_id = self._track_id
                self._accept(match)
                return self._emit(match.bbox, 1.0, RECOVER_MOTION_GATE, gate_passed)

        self._on_missing_detection(frame_idx)

        if self._state is TrackState.LOST:
            bbox = self._extrapolate(self._frames_since_confirmed)
            if bbox is None:  # chua co observation nao -> khong co gi de du doan
                return self._emit(None, 0.0, None, gate_passed)
            return self._emit(bbox, self._prediction_confidence(), None, gate_passed)

        # REIDENTIFYING / FAILED: ngoai suy chuyen dong khong con dang tin,
        # cho co che appearance-based Re-ID (xem CLAUDE.md muc 6).
        return self._emit(None, 0.0, None, gate_passed)

    def reset(self) -> None:
        """Xoa toan bo lich su, quay ve TRACKING."""
        self._history.clear()
        self._state = TrackState.TRACKING
        self._frames_since_confirmed = 0
        self._track_id = None
        self._anchor_id = None
        self._last_frame_idx = None

    # ---------------- noi bo ----------------

    def _find_anchor(
        self,
        observation: Optional[Observation],
        candidates: Optional[Sequence[Observation]],
    ) -> Optional[Observation]:
        """Track co ID = anchor trong frame nay (None neu tat / khong co anchor)."""
        if not self.config.anchor_recovery or self._anchor_id is None:
            return None
        for obs in ([observation] if observation is not None else []) + list(candidates or ()):
            if obs.track_id is not None and obs.track_id == self._anchor_id:
                return obs
        return None

    def _check_frame_idx(
        self, frame_idx: Optional[int], observation: Optional[Observation]
    ) -> None:
        if frame_idx is None:
            return
        if self._last_frame_idx is not None and frame_idx <= self._last_frame_idx:
            raise ValueError(
                f"frame_idx phai tang dan: {frame_idx} <= {self._last_frame_idx}"
            )
        if observation is not None and observation.frame_idx != frame_idx:
            raise ValueError(
                f"frame_idx={frame_idx} khac observation.frame_idx={observation.frame_idx}"
            )
        self._last_frame_idx = frame_idx

    def _accept(self, observation: Observation) -> None:
        self._history.append(observation)
        self._transition(TrackState.TRACKING)
        self._frames_since_confirmed = 0
        if observation.track_id is not None:
            self._track_id = observation.track_id

    def _split_target(
        self,
        observation: Optional[Observation],
        candidates: Optional[Sequence[Observation]],
    ) -> tuple[Optional[Observation], list[Observation]]:
        """Tach observation cua muc tieu (trung ID hien tai) khoi ung vien khac."""
        others = list(candidates or ())
        if self._track_id is None:
            return observation, others
        if (
            observation is not None
            and observation.track_id is not None
            and observation.track_id != self._track_id
        ):
            others.append(observation)
            observation = None
        if observation is None:
            for i, cand in enumerate(others):
                if cand.track_id == self._track_id:
                    observation = others.pop(i)
                    break
        return observation, others

    def _is_current_id(self, observation: Observation) -> bool:
        return observation.track_id is not None and observation.track_id == self._track_id

    def _gate(
        self, candidates: Sequence[Observation]
    ) -> tuple[Optional[Observation], int]:
        """Khop lai bang chuyen dong.

        Tra ve (ung vien DUY NHAT qua gate hoac None, so ung vien qua gate).
        """
        if not self._history:
            return None, 0
        last_obs = self._history[-1]
        last = last_obs.bbox
        scale = (last.width + last.height) / 2.0
        last_area = last.width * last.height
        if scale <= 0.0 or last_area <= 0.0:
            return None, 0
        cfg = self.config

        passed: list[Observation] = []
        for cand in candidates:
            if last_obs.cls is not None and cand.cls != last_obs.cls:
                continue
            area_ratio = (cand.bbox.width * cand.bbox.height) / last_area
            if not 1.0 / cfg.max_scale_change <= area_ratio <= cfg.max_scale_change:
                continue
            gap = max(1, cand.frame_idx - last_obs.frame_idx)
            if gap > cfg.lost_timeout_frames:  # ngoai motion horizon
                continue
            predicted = self._extrapolate(gap)
            if predicted is None:
                continue
            (px, py), (cx, cy) = predicted.center, cand.bbox.center
            radius = scale * cfg.match_center_ratio * (1.0 + cfg.match_gate_growth * gap)
            close = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 <= radius
            if close or _iou(predicted, cand.bbox) >= cfg.match_iou:
                passed.append(cand)
        return (passed[0] if len(passed) == 1 else None), len(passed)

    def _on_missing_detection(self, frame_idx: Optional[int] = None) -> None:
        if frame_idx is not None and self._history:
            self._frames_since_confirmed = frame_idx - self._history[-1].frame_idx
        else:
            self._frames_since_confirmed += 1
        k = self._frames_since_confirmed
        cfg = self.config

        if self._state is TrackState.TRACKING:
            self._transition(TrackState.LOST)
        # Moc tinh tu lan confirmed cuoi: motion horizon dai dung L frame du
        # doan. Khi dem +1 moi lan goi, REIDENTIFYING luon bat dau o k = L+1
        # nen dieu kien nay tuong duong dem so frame trong state.
        if self._state is TrackState.LOST and k > cfg.lost_timeout_frames:
            self._transition(TrackState.REIDENTIFYING)
        if (
            self._state is TrackState.REIDENTIFYING
            and k > cfg.lost_timeout_frames + cfg.reid_timeout_frames
        ):
            self._transition(TrackState.FAILED)
            self._anchor_id = None  # FAILED la trang thai cuoi -> bo anchor

    def _transition(self, new_state: TrackState) -> None:
        self._state = new_state

    def _emit(
        self,
        bbox: Optional[BBox],
        confidence: float,
        recover_reason: Optional[str] = None,
        gate_candidates_passed: Optional[int] = None,
    ) -> MotionPrediction:
        return MotionPrediction(
            bbox=bbox,
            confidence=confidence,
            frames_since_confirmed=self._frames_since_confirmed,
            state=self._state,
            track_id=self._track_id,
            recover_reason=recover_reason,
            gate_candidates_passed=gate_candidates_passed,
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


def _iou(a: BBox, b: BBox) -> float:
    """IoU hai bbox xyxy. Viet rieng (khong import src.evaluate) de tranh
    import vong: evaluate.py da import BBox tu module nay."""
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0.0 else 0.0
