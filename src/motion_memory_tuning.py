"""Quy tac tune Motion Memory DA DANG KY TRUOC (configs/motion_memory_tuning_dev.yaml).

Module nay chi chua HAM THUAN (khong doc/ghi o dia): sinh danh sach cau hinh
tu luoi, tinh diem, kiem tra hop le, chon gia tri thang vong 1, sinh cau hinh
ket hop vong 2 va chon cau hinh cuoi. Du lieu vao la dict metrics gop
(khoa `overall` trong experiments/<id>/metrics.json, metrics v2), de test bang
du lieu gia. Doc config/metrics tu dia: scripts/tune_motion_memory.py.

Quy tac (CLAUDE.md muc 13 - viet TRUOC khi co ket qua that):
    - Luoi quet TUNG tham so mot quanh mac dinh (cac tham so khac giu mac
      dinh). Mac dinh lay tu configs/motion_memory.yaml va phai trung
      `expected_defaults` cua config tuning (doi file mac dinh -> tu choi).
    - Vong 1: cau hinh mac dinh (`default`, chay lai cung code) + moi gia tri
      khac mac dinh cua moi tham so (dung MOT tham so khac mac dinh).
    - Diem: correct_prediction_ratio GOP tren frame bi che, cong tu so va
      mau so cua CA HAI detector (Fraction - so sanh chinh xac).
    - Hop le: tren MOI detector, reidentification_rate >= cua mac dinh VA
      correct_prediction_ratio > cua mac dinh.
    - Gia tri thang cua tham so: cau hinh hop le diem cao nhat; hoa -> sai so
      vi tri gop thap hon; van hoa -> giu mac dinh (khong co gia tri thang).
    - Vong 2: DUNG MOT cau hinh ket hop cac gia tri thang (tham so khong co
      gia tri thang giu mac dinh).
    - Cau hinh cuoi: trong {cau hinh thang vong 1, cau hinh vong 2} chi xet
      cau hinh hop le VA diem - diem mac dinh >= min_improvement; lay diem
      cao nhat (hoa nhu tren). Khong co -> KET LUAN giu mac dinh.
    - Khong co vong 3, khong bo sung luoi sau khi co ket qua.

Ket qua chay mot phan (--limit-scenarios, aggregate incomplete) bi tu choi o
moi ham chon (`complete` phai True).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import Any, Mapping, Optional, Sequence

from src.motion_memory import MotionMemoryConfig

DEFAULT_ID = "default"
ROUND2_ID = "r2_combined"
SUPPORTED_VERSIONS = (1,)
SUPPORTED_MAX_ROUNDS = 2

CONCLUSION_TUNED = "use_tuned"
CONCLUSION_KEEP_DEFAULT = "keep_default"

# Truong can trong metrics gop (overall cua metrics v2) cho moi detector.
REQUIRED_METRIC_KEYS = (
    "scenarios_evaluated",
    "occlusion_frames",
    "correct_occluded_frames",
    "reidentified",
    "occluded_frames_with_output",
    "mean_position_error_px",
)


class TuningError(ValueError):
    """Config tuning hoac ket qua khong dung quy tac -> khong chon cau hinh."""


def _field_types() -> dict[str, str]:
    return {f.name: str(f.type) for f in fields(MotionMemoryConfig)}


def _check_value(param: str, value: Any, type_name: str) -> Any:
    if isinstance(value, bool):
        raise TuningError(f"{param}: khong nhan bool ({value!r})")
    if type_name == "int":
        if not isinstance(value, int):
            raise TuningError(f"{param}: can so nguyen, nhan {value!r}")
        return value
    if not isinstance(value, (int, float)):
        raise TuningError(f"{param}: can so thuc, nhan {value!r}")
    return float(value)


@dataclass(frozen=True)
class TuningRules:
    version: int
    name: str
    group: str
    detectors: tuple[str, ...]
    metrics_config_version: int
    source_tag: str
    grid: tuple[tuple[str, tuple[Any, ...]], ...]   # thu tu tham so + gia tri co dinh
    expected_defaults: tuple[tuple[str, Any], ...]
    min_improvement: Fraction
    max_rounds: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TuningRules":
        try:
            version = int(data["version"])
            grid_raw = data["grid"]
            defaults_raw = data["expected_defaults"]
            selection = data["selection"]
            detectors = tuple(str(d) for d in data["detectors"])
            min_improvement = Fraction(str(selection["min_improvement"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise TuningError(f"Config tuning thieu/sai khoa: {exc}") from exc
        if version not in SUPPORTED_VERSIONS:
            raise TuningError(f"version={version} chua ho tro (co: {SUPPORTED_VERSIONS})")
        if not detectors or len(set(detectors)) != len(detectors):
            raise TuningError("detectors phai khac rong va khong trung")
        if min_improvement < 0:
            raise TuningError("selection.min_improvement phai >= 0")
        max_rounds = int(data.get("max_rounds", SUPPORTED_MAX_ROUNDS))
        if max_rounds != SUPPORTED_MAX_ROUNDS:
            raise TuningError(f"max_rounds chi ho tro {SUPPORTED_MAX_ROUNDS} (khong co vong 3)")
        if not isinstance(grid_raw, Mapping) or not grid_raw:
            raise TuningError("grid phai la mapping khac rong")
        if not isinstance(defaults_raw, Mapping) or set(defaults_raw) != set(grid_raw):
            raise TuningError("expected_defaults phai co dung cac tham so cua grid")
        types = _field_types()
        grid = []
        defaults = []
        for param, values in grid_raw.items():
            if param not in types:
                raise TuningError(f"Tham so {param!r} khong co trong MotionMemoryConfig")
            if not isinstance(values, Sequence) or isinstance(values, str):
                raise TuningError(f"grid.{param} phai la danh sach")
            vals = tuple(_check_value(param, v, types[param]) for v in values)
            if len(vals) < 2 or len(set(vals)) != len(vals):
                raise TuningError(f"grid.{param}: can >= 2 gia tri khong trung")
            default = _check_value(param, defaults_raw[param], types[param])
            if default not in vals:
                raise TuningError(f"grid.{param} phai chua gia tri mac dinh {default!r}")
            grid.append((str(param), vals))
            defaults.append((str(param), default))
        return cls(
            version=version,
            name=str(data.get("name", "")),
            group=str(data["group"]),
            detectors=detectors,
            metrics_config_version=int(data["metrics_config_version"]),
            source_tag=str(data["source_tag"]),
            grid=tuple(grid),
            expected_defaults=tuple(defaults),
            min_improvement=min_improvement,
            max_rounds=max_rounds,
        )

    @property
    def defaults(self) -> dict[str, Any]:
        return dict(self.expected_defaults)


def check_defaults(rules: TuningRules, base_values: Mapping[str, Any]) -> None:
    """Gia tri mac dinh (configs/motion_memory.yaml da resolve) phai trung quy tac."""
    diff = {
        p: (base_values.get(p), expected)
        for p, expected in rules.expected_defaults
        if base_values.get(p) != expected
    }
    if diff:
        raise TuningError(
            "Config Motion Memory mac dinh khac expected_defaults cua quy tac tuning (file mac dinh "
            f"da doi sau khi dang ky?): {diff}"
        )


def value_token(value: Any) -> str:
    """0.9 -> '0p9', 25 -> '25' (dung trong ID cau hinh / ten thu muc)."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def round1_config_id(param: str, value: Any) -> str:
    return f"r1_{param}_{value_token(value)}"


def config_values(base_values: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Tham so day du cua mot cau hinh; kiem tra hop le qua MotionMemoryConfig."""
    unknown = sorted(set(overrides) - set(base_values))
    if unknown:
        raise TuningError(f"Tham so khong ton tai: {unknown}")
    values = {**dict(base_values), **dict(overrides)}
    try:
        MotionMemoryConfig(**values)
    except (TypeError, ValueError) as exc:
        raise TuningError(f"Cau hinh khong hop le cho MotionMemoryConfig: {exc}") from exc
    return values


def round1_configs(rules: TuningRules) -> list[dict[str, Any]]:
    """`default` + moi gia tri khac mac dinh cua tung tham so (thu tu co dinh)."""
    out: list[dict[str, Any]] = [
        {"id": DEFAULT_ID, "round": 1, "param": None, "value": None, "overrides": {}}
    ]
    defaults = rules.defaults
    for param, values in rules.grid:
        for value in values:
            if value == defaults[param]:
                continue
            out.append({
                "id": round1_config_id(param, value), "round": 1,
                "param": param, "value": value, "overrides": {param: value},
            })
    return out


def round2_config(rules: TuningRules, winners: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Cau hinh ket hop cac gia tri thang vong 1.

    `same_as`: ID cau hinh da chay co cung tham so (DEFAULT_ID khi khong tham so
    nao thang; ID vong 1 khi chi mot tham so thang) - khi do KHONG chay lai,
    dung ket qua cua cau hinh do. None = cau hinh moi, phai chay.
    """
    missing = [p for p, _ in rules.grid if p not in winners]
    if missing:
        raise TuningError(f"Thieu ket qua vong 1 cho tham so {missing}")
    overrides = {
        p: winners[p]["value"] for p, _ in rules.grid if winners[p].get("value") is not None
    }
    if not overrides:
        same_as: Optional[str] = DEFAULT_ID
    elif len(overrides) == 1:
        (param, value), = overrides.items()
        same_as = round1_config_id(param, value)
    else:
        same_as = None
    return {"id": ROUND2_ID, "round": 2, "param": None, "value": None,
            "overrides": overrides, "same_as": same_as}


# ---------------- diem + hop le ----------------

def _det_counts(config_id: str, det: str, overall: Mapping[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED_METRIC_KEYS if k not in overall]
    if missing:
        raise TuningError(f"{config_id}/{det}: metrics gop thieu {missing} (can metrics v2)")
    if not overall["occlusion_frames"] or not overall["scenarios_evaluated"]:
        raise TuningError(f"{config_id}/{det}: khong co kich ban da khoa / frame bi che nao")
    return {k: overall[k] for k in REQUIRED_METRIC_KEYS}


def check_entry(rules: TuningRules, config_id: str, entry: Optional[Mapping[str, Any]]) -> None:
    """Ket qua mot cau hinh: du, khong partial, dung tap detector."""
    if entry is None:
        raise TuningError(f"Chua co ket qua cua {config_id}")
    if not entry.get("complete"):
        raise TuningError(
            f"Ket qua cua {config_id} chua day du/partial (vd --limit-scenarios) - khong dung de chon"
        )
    dets = set(entry.get("detectors") or {})
    if dets != set(rules.detectors):
        raise TuningError(f"{config_id}: detector {sorted(dets)} khac quy tac {sorted(rules.detectors)}")
    for det in rules.detectors:
        _det_counts(config_id, det, entry["detectors"][det])


def check_same_scenarios(rules: TuningRules, config_id: str, entry: Mapping[str, Any],
                         default: Mapping[str, Any]) -> None:
    """Cung tap kich ban da khoa (cung so kich ban + so frame bi che) voi mac dinh."""
    for det in rules.detectors:
        a, b = entry["detectors"][det], default["detectors"][det]
        for key in ("scenarios_evaluated", "occlusion_frames"):
            if a[key] != b[key]:
                raise TuningError(
                    f"{config_id}/{det}: {key}={a[key]} khac mac dinh {b[key]} - khong cung tap kich ban "
                    "da khoa, khong so sanh duoc"
                )


def _cpr(overall: Mapping[str, Any]) -> Fraction:
    return Fraction(overall["correct_occluded_frames"], overall["occlusion_frames"])


def _reid_rate(overall: Mapping[str, Any]) -> Fraction:
    return Fraction(overall["reidentified"], overall["scenarios_evaluated"])


def score(rules: TuningRules, entry: Mapping[str, Any]) -> Fraction:
    """correct_prediction_ratio gop ca hai detector (cong tu so va mau so)."""
    dets = [entry["detectors"][d] for d in rules.detectors]
    return Fraction(sum(o["correct_occluded_frames"] for o in dets), sum(o["occlusion_frames"] for o in dets))


def pooled_position_error(rules: TuningRules, entry: Mapping[str, Any]) -> Optional[float]:
    """Sai so vi tri trung binh gop (trong so = so frame bi che co output)."""
    total = n = 0.0
    for d in rules.detectors:
        o = entry["detectors"][d]
        if o["occluded_frames_with_output"] and o["mean_position_error_px"] is not None:
            total += o["mean_position_error_px"] * o["occluded_frames_with_output"]
            n += o["occluded_frames_with_output"]
    return total / n if n else None


def validity(rules: TuningRules, entry: Mapping[str, Any], default: Mapping[str, Any]) -> dict[str, Any]:
    """Hop le khi tren MOI detector: reid_rate >= mac dinh VA cpr > mac dinh."""
    per_det = {}
    for d in rules.detectors:
        o, od = entry["detectors"][d], default["detectors"][d]
        reid_ok = _reid_rate(o) >= _reid_rate(od)
        cpr_ok = _cpr(o) > _cpr(od)
        per_det[d] = {
            "reidentification_rate": float(_reid_rate(o)),
            "default_reidentification_rate": float(_reid_rate(od)),
            "correct_prediction_ratio": float(_cpr(o)),
            "default_correct_prediction_ratio": float(_cpr(od)),
            "reid_not_worse": reid_ok,
            "cpr_better": cpr_ok,
        }
    return {"valid": all(v["reid_not_worse"] and v["cpr_better"] for v in per_det.values()),
            "per_detector": per_det}


def assess(rules: TuningRules, config_id: str, entry: Mapping[str, Any],
           default: Mapping[str, Any]) -> dict[str, Any]:
    """Diem, chenh lech so voi mac dinh, sai so gop, hop le cua mot cau hinh."""
    check_entry(rules, config_id, entry)
    check_entry(rules, DEFAULT_ID, default)
    check_same_scenarios(rules, config_id, entry, default)
    s, s0 = score(rules, entry), score(rules, default)
    v = validity(rules, entry, default)
    return {
        "config_id": config_id,
        "score": float(s),
        "score_fraction": f"{s.numerator}/{s.denominator}",
        "delta_vs_default": float(s - s0),
        "meets_min_improvement": (s - s0) >= rules.min_improvement,
        "pooled_mean_position_error_px": pooled_position_error(rules, entry),
        "valid": v["valid"] if config_id != DEFAULT_ID else False,
        "validity": v,
        "_score": s,
    }


def _errors_tie(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def pick_best(assessed: Sequence[Mapping[str, Any]]) -> tuple[Optional[str], str]:
    """Diem cao nhat; hoa -> sai so gop thap hon (None = te nhat); van hoa -> None."""
    if not assessed:
        return None, "khong co cau hinh hop le"
    top = max(a["_score"] for a in assessed)
    best = [a for a in assessed if a["_score"] == top]
    if len(best) == 1:
        return best[0]["config_id"], "diem cao nhat"
    errs = [a["pooled_mean_position_error_px"] for a in best]
    known = [e for e in errs if e is not None]
    if not known:
        return None, f"hoa diem giua {[a['config_id'] for a in best]}, khong co sai so -> giu mac dinh"
    low = min(known)
    lowest = [a for a in best if a["pooled_mean_position_error_px"] is not None
              and _errors_tie(a["pooled_mean_position_error_px"], low)]
    if len(lowest) == 1:
        return lowest[0]["config_id"], "hoa diem, sai so vi tri gop thap hon"
    return None, f"hoa ca diem va sai so giua {[a['config_id'] for a in lowest]} -> giu mac dinh"


def _public(a: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in a.items() if not k.startswith("_")}


def round1_winners(rules: TuningRules, results: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Gia tri thang cua tung tham so (None = giu mac dinh)."""
    configs = round1_configs(rules)
    default = results.get(DEFAULT_ID)
    for cfg in configs:
        check_entry(rules, cfg["id"], results.get(cfg["id"]))
    winners: dict[str, dict[str, Any]] = {}
    for param, _ in rules.grid:
        own = [c for c in configs if c["param"] == param]
        assessed = [assess(rules, c["id"], results[c["id"]], default) for c in own]
        best_id, reason = pick_best([a for a in assessed if a["valid"]])
        best = next((c for c in own if c["id"] == best_id), None)
        winners[param] = {
            "value": best["value"] if best else None,
            "config_id": best_id,
            "default_value": rules.defaults[param],
            "reason": reason,
            "candidates": [{**_public(a), "value": c["value"]} for a, c in zip(assessed, own)],
        }
    return winners


def final_selection(rules: TuningRules, results: Mapping[str, Mapping[str, Any]],
                    winners: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Cau hinh cuoi tu {thang vong 1, ket hop vong 2}; khong co -> giu mac dinh."""
    default = results.get(DEFAULT_ID)
    check_entry(rules, DEFAULT_ID, default)
    r2 = round2_config(rules, winners)
    ids: list[str] = [w["config_id"] for p, _ in rules.grid if (w := winners[p])["config_id"]]
    if r2["same_as"] is None:
        if results.get(ROUND2_ID) is None:
            raise TuningError("Chua co ket qua vong 2 (r2_combined) - chay --round 2 truoc khi chon")
        ids.append(ROUND2_ID)
    assessed = [assess(rules, i, results[i], default) for i in ids]
    eligible = [a for a in assessed if a["valid"] and a["meets_min_improvement"]]
    best_id, reason = pick_best(eligible)
    s0 = score(rules, default)
    out = {
        "round2": r2,
        "default_score": float(s0),
        "default_score_fraction": f"{s0.numerator}/{s0.denominator}",
        "min_improvement": float(rules.min_improvement),
        "candidates": [_public(a) for a in assessed],
        "eligible": [a["config_id"] for a in eligible],
    }
    if best_id is None:
        out.update({
            "selected_config_id": None,
            "overrides": {},
            "conclusion": CONCLUSION_KEEP_DEFAULT,
            "reason": (
                "Khong cau hinh nao hop le tren ca hai detector VA hon mac dinh it nhat "
                f"{float(rules.min_improvement)} diem -> giu cau hinh mac dinh"
                if not eligible else reason
            ),
        })
        return out
    if best_id == ROUND2_ID:
        overrides = dict(r2["overrides"])
    else:
        param = next(p for p, _ in rules.grid if winners[p]["config_id"] == best_id)
        overrides = {param: winners[param]["value"]}
    out.update({
        "selected_config_id": best_id,
        "overrides": overrides,
        "conclusion": CONCLUSION_TUNED,
        "reason": reason,
    })
    return out
