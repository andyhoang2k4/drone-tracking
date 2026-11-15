"""Adapter: dung `MotionMemory` (src/motion_memory.py) lam bo uoc luong muc tieu cho lop sim.

Thay heuristic ngoai suy tam anh cua `TargetManager` (CLAUDE.md muc 2, moc 2 cua lop mo phong) ma
giu nguyen interface ma `SimApp`/`CameraView`/`FlightSupervisor` dang dung:
`select_at`, `lock_selected`, `unlock`, `update(detections, t) -> TargetEstimate | None`,
cac thuoc tinh `candidates`, `selected_id`, `locked_id`, `locked_label`, `estimate`.

Khong sua motion_memory.py. Quy uoc:
- `MotionMemory` tinh theo frame; sim tinh theo giay. `frame_idx` = round((t - t_lock) * frame_rate_hz),
  luon tang ngat (neu hai lan goi roi vao cung frame thi day len +1). Nguong timeout cua MotionMemory
  (lost_timeout_frames, reid_timeout_frames) vi vay mang nghia giay = frame / frame_rate_hz.
- Confidence du doan cua MotionMemory -> `TargetEstimate.prediction_confidence`. Khong gop voi
  confidence cua detector (CLAUDE.md muc 6).
- `Observation.cls` = chi so on dinh cua nhan lop (theo thu tu gap lan dau) de MotionMemory gate cung lop.
- FAILED la trang thai cuoi cua MotionMemory: muon bam tiep phai khoa lai (giong TargetManager).

Lop sim khong dung de danh gia Motion Memory (CLAUDE.md muc 2, 12).
"""
from __future__ import annotations

from ..motion_memory import BBox as MMBox
from ..motion_memory import MotionMemory, MotionMemoryConfig, Observation
from .config import TargetConfig
from .target_manager import TargetManager
from .types import Detection, TargetEstimate, TargetState


def build_target_manager(cfg: TargetConfig) -> TargetManager:
    """Chon bo uoc luong theo `cfg.estimator`: "heuristic" (TargetManager) hoac "motion_memory"."""
    if cfg.estimator == "heuristic":
        return TargetManager(cfg)
    if cfg.estimator == "motion_memory":
        from ..utils.motion_memory_settings import resolve_motion_memory_config

        choice = resolve_motion_memory_config(config_path=cfg.motion_memory_config)
        return MotionMemoryTargetManager(cfg, choice.config)
    raise ValueError(f"target.estimator khong hop le: {cfg.estimator!r} (heuristic | motion_memory)")


class MotionMemoryTargetManager(TargetManager):
    def __init__(self, cfg: TargetConfig, mm_config: MotionMemoryConfig | None = None) -> None:
        super().__init__(cfg)
        self._mm_config = mm_config or MotionMemoryConfig()
        self._mm = MotionMemory(self._mm_config)
        self._rate = float(cfg.mm_frame_rate_hz)
        if self._rate <= 0:
            raise ValueError("mm_frame_rate_hz phai > 0")
        self._t0: float | None = None
        self._last_frame: int | None = None
        self._classes: dict[str, int] = {}
        self.last_prediction = None  # MotionPrediction gan nhat (de log/debug)

    # ---- thao tac nguoi dung ----
    def lock_selected(self) -> bool:
        det = next((d for d in self.candidates if d.track_id == self.selected_id), None)
        if det is None:
            return False
        t = self._last_t if self._last_t is not None else 0.0
        self._mm.reset()
        self._t0, self._last_frame = t, None
        self.locked_id, self.locked_label = det.track_id, det.label
        fi = self._frame(t)
        pred = self._mm.update(self._obs(det, fi, t), frame_idx=fi)
        self._apply(pred)
        return True

    def unlock(self) -> None:
        super().unlock()
        self._mm.reset()
        self._t0, self._last_frame = None, None
        self.last_prediction = None

    # ---- cap nhat moi frame ----
    def update(self, detections: list[Detection], t: float) -> TargetEstimate | None:
        self.candidates = list(detections)
        if self.selected_id is not None and all(d.track_id != self.selected_id for d in detections):
            if self.selected_id != self.locked_id:
                self.selected_id = None
        self._last_t = t
        if self.locked_id is None:
            self.estimate = None
            return None
        fi = self._frame(t)
        current = self._mm.track_id if self._mm.track_id is not None else self.locked_id
        obs, cands = None, []
        for d in detections:
            o = self._obs(d, fi, t)
            if obs is None and d.track_id == current:
                obs = o
            else:
                cands.append(o)
        pred = self._mm.update(obs, cands, frame_idx=fi)
        self._apply(pred)
        return self.estimate

    # ---- noi bo ----
    def _frame(self, t: float) -> int:
        assert self._t0 is not None
        fi = int(round((t - self._t0) * self._rate))
        if self._last_frame is not None and fi <= self._last_frame:
            fi = self._last_frame + 1
        self._last_frame = fi
        return fi

    def _cls(self, label: str) -> int:
        return self._classes.setdefault(label, len(self._classes))

    def _obs(self, d: Detection, fi: int, t: float) -> Observation:
        return Observation(bbox=MMBox(*d.bbox), frame_idx=fi, timestamp=t, track_id=d.track_id,
                           detector_confidence=d.conf, cls=self._cls(d.label))

    def _apply(self, pred) -> None:
        self.last_prediction = pred
        if pred.track_id is not None:
            self.locked_id = pred.track_id
        bbox = pred.bbox.as_tuple() if pred.bbox is not None else None
        if pred.state is TargetState.TRACKING and bbox is not None:
            self._last_bbox = bbox
        self.estimate = TargetEstimate(self.locked_id, pred.state, bbox,
                                       pred.frames_since_confirmed / self._rate, pred.confidence)
