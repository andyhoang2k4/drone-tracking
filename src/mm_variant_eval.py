"""Danh gia bien the Motion Memory tren dev theo file khai bao truoc
(vd configs/motion_memory_anchor_dev.yaml).

HAM THUAN (khong doc/ghi o dia): nap + kiem tra file khai bao. Doc/ghi va chay
replay: scripts/tune_motion_memory.py --variant.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
from typing import Any, Mapping, Optional

from src.motion_memory import MotionMemoryConfig

SUPPORTED_VERSIONS = (1,)
ROLES = ("regression", "decision", "descriptive")


class VariantPlanError(ValueError):
    """File khai bao hoac ket qua khong dung quy tac."""


def _field_types() -> dict[str, str]:
    return {f.name: str(f.type) for f in fields(MotionMemoryConfig)}


def _check_params(where: str, values: Mapping[str, Any]) -> dict[str, Any]:
    types = _field_types()
    out = {}
    for name, value in (values or {}).items():
        if name not in types:
            raise VariantPlanError(f"{where}: tham so {name!r} khong co trong MotionMemoryConfig")
        kind = types[name]
        ok = (isinstance(value, bool) if kind == "bool"
              else isinstance(value, int) and not isinstance(value, bool) if kind == "int"
              else isinstance(value, (int, float)) and not isinstance(value, bool))
        if not ok:
            raise VariantPlanError(f"{where}: {name}={value!r} sai kieu (can {kind})")
        out[name] = float(value) if kind == "float" else value
    return out


@dataclass(frozen=True)
class Variant:
    id: str
    role: str
    overrides: dict
    check_file: Optional[str] = None
    check_extra: Optional[dict] = None


@dataclass(frozen=True)
class VariantPlan:
    version: int
    name: str
    group: str
    detectors: tuple[str, ...]
    metrics_config: str
    metrics_config_version: int
    base_motion_memory_config: str
    source_tag: str
    tag: str
    expected_defaults: dict
    variants: tuple[Variant, ...]
    references: dict
    regression: dict
    decision: dict

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "VariantPlan":
        try:
            version = int(data["version"])
            detectors = tuple(str(d) for d in data["detectors"])
            variants_raw = list(data["variants"])
            references = {str(k): str(v) for k, v in dict(data["references"]).items()}
            regression = dict(data["regression"])
            decision = dict(data["decision"])
            plan_fields = {k: data[k] for k in ("name", "group", "metrics_config", "base_motion_memory_config",
                                                "source_tag", "tag")}
            metrics_version = int(data["metrics_config_version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VariantPlanError(f"File khai bao thieu/sai khoa: {exc}") from exc
        if version not in SUPPORTED_VERSIONS:
            raise VariantPlanError(f"version={version} chua ho tro")
        if not detectors or len(set(detectors)) != len(detectors):
            raise VariantPlanError("detectors phai khac rong, khong trung")
        variants = []
        for raw in variants_raw:
            vid, role = str(raw.get("id", "")), str(raw.get("role", ""))
            if not vid or role not in ROLES:
                raise VariantPlanError(f"Bien the {raw!r}: can id va role trong {ROLES}")
            variants.append(Variant(
                id=vid, role=role, overrides=_check_params(f"variants.{vid}", raw.get("overrides") or {}),
                check_file=raw.get("check_file"),
                check_extra=_check_params(f"variants.{vid}.check_extra", raw.get("check_extra") or {})
                if raw.get("check_file") else None,
            ))
        ids = [v.id for v in variants]
        if len(set(ids)) != len(ids):
            raise VariantPlanError("id bien the bi trung")
        for role in ("regression", "decision"):
            if [v.role for v in variants].count(role) != 1:
                raise VariantPlanError(f"Can dung MOT bien the role={role}")
        by_role = {v.role: v.id for v in variants if v.role != "descriptive"}
        if regression.get("variant") != by_role["regression"] or decision.get("variant") != by_role["decision"]:
            raise VariantPlanError("regression.variant / decision.variant phai trung bien the co role tuong ung")
        if regression.get("reference") not in references:
            raise VariantPlanError("regression.reference khong co trong references")
        if not set(regression.get("sections") or []) <= {"overall", "by_duration_frames", "by_occluder"} \
                or not regression.get("sections"):
            raise VariantPlanError("regression.sections phai la tap con khac rong cua overall/by_duration_frames/"
                                   "by_occluder")
        reid_rules = []
        for rule in decision.get("reid_not_below") or []:
            if rule.get("reference") not in references or not rule.get("durations"):
                raise VariantPlanError(f"decision.reid_not_below sai: {rule!r}")
            reid_rules.append({"reference": rule["reference"], "durations": [int(d) for d in rule["durations"]]})
        cpr = decision.get("cpr_max_drop") or {}
        if cpr.get("reference") not in references or cpr.get("value") is None:
            raise VariantPlanError("decision.cpr_max_drop can reference + value")
        if Fraction(str(cpr["value"])) < 0:
            raise VariantPlanError("decision.cpr_max_drop.value phai >= 0")
        for name, template in references.items():
            if "{det}" not in template:
                raise VariantPlanError(f"references.{name} phai chua {{det}}")
        return cls(
            version=version, detectors=detectors, metrics_config_version=metrics_version,
            expected_defaults=_check_params("expected_defaults", data.get("expected_defaults") or {}),
            variants=tuple(variants), references=references, regression=regression,
            decision={"variant": decision["variant"], "reid_not_below": reid_rules,
                      "cpr_max_drop": {"reference": cpr["reference"], "value": str(cpr["value"])}},
            **{k: str(v) for k, v in plan_fields.items()},
        )

    def variant(self, vid: str) -> Variant:
        return next(v for v in self.variants if v.id == vid)

    def reference_experiment(self, name: str, det: str) -> str:
        return self.references[name].format(det=det)


def check_defaults(plan: VariantPlan, base_values: Mapping[str, Any]) -> None:
    """Config MM mac dinh (da resolve) phai trung expected_defaults luc dang ky."""
    diff = {k: (base_values.get(k), v) for k, v in plan.expected_defaults.items() if base_values.get(k) != v}
    if diff:
        raise VariantPlanError(f"Config Motion Memory mac dinh khac expected_defaults cua file khai bao: {diff}")


def check_file_values(variant: Variant, variant_values: Mapping[str, Any],
                      file_values: Mapping[str, Any]) -> None:
    """Bien the co check_file: tham so file (da resolve) + check_extra phai trung tham so bien the."""
    expected = {**dict(file_values), **(variant.check_extra or {})}
    diff = {k: (variant_values.get(k), expected.get(k)) for k in set(expected) | set(variant_values)
            if variant_values.get(k) != expected.get(k)}
    if diff:
        raise VariantPlanError(f"Bien the {variant.id}: tham so khac {variant.check_file} (+ check_extra): {diff}")


# ---------------- ket qua: hoi quy, quyet dinh, mo ta ----------------
# Dau vao: metrics.json GOP (src.evaluate aggregate, metrics v2) cua tung
# experiment, dang {det: agg}. Moi ti le so bang Fraction tren count.

def check_complete(label: str, aggs: Mapping[str, Mapping[str, Any]], detectors) -> None:
    for det in detectors:
        agg = aggs.get(det)
        if agg is None:
            raise VariantPlanError(f"{label}/{det}: chua co metrics.json gop")
        if not (agg.get("completeness") or {}).get("complete"):
            raise VariantPlanError(f"{label}/{det}: ket qua chua day du/partial - khong dung de quyet dinh")


def _diff_paths(a: Any, b: Any, path: str) -> list[str]:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                out.append(f"{path}.{key}: chi co o mot ben")
            else:
                out.extend(_diff_paths(a[key], b[key], f"{path}.{key}"))
        return out
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def regression_check(plan: VariantPlan, variant_aggs: Mapping[str, Mapping[str, Any]],
                     reference_aggs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Bien the hoi quy phai TRUNG tham chieu o moi section (so sanh chinh xac)."""
    diffs: list[str] = []
    for det in plan.detectors:
        for sec in plan.regression["sections"]:
            a = (variant_aggs.get(det) or {}).get(sec)
            b = (reference_aggs.get(det) or {}).get(sec)
            if a is None or b is None:
                diffs.append(f"{det}.{sec}: thieu")
            else:
                diffs.extend(_diff_paths(a, b, f"{det}.{sec}"))
    return {"ok": not diffs, "differences": diffs[:50], "n_differences": len(diffs)}


def _pool_of(agg: Mapping[str, Any], duration: Optional[int]) -> Mapping[str, Any]:
    if duration is None:
        return agg["overall"]
    pool = (agg.get("by_duration_frames") or {}).get(str(duration))
    if pool is None:
        raise VariantPlanError(f"Khong co by_duration_frames[{duration}]")
    return pool


def _reid(pool: Mapping[str, Any]) -> Fraction:
    if not pool.get("scenarios_evaluated"):
        raise VariantPlanError("Khong co kich ban da khoa trong nhom - khong tinh duoc ti le re-ID")
    return Fraction(pool["reidentified"], pool["scenarios_evaluated"])


def _cpr(pool: Mapping[str, Any]) -> Fraction:
    if not pool.get("occlusion_frames"):
        raise VariantPlanError("Khong co frame bi che - khong tinh duoc cpr")
    return Fraction(pool["correct_occluded_frames"], pool["occlusion_frames"])


def _same_set(a: Mapping[str, Any], b: Mapping[str, Any], where: str) -> None:
    for key in ("scenarios_evaluated", "occlusion_frames"):
        if a.get(key) != b.get(key):
            raise VariantPlanError(f"{where}: {key} {a.get(key)} != {b.get(key)} - khong cung tap kich ban da khoa")


def _frac(x: Fraction) -> dict[str, Any]:
    return {"value": float(x), "fraction": f"{x.numerator}/{x.denominator}"}


def decide(plan: VariantPlan, variant_aggs: Mapping[str, Mapping[str, Any]],
           reference_aggs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Quy tac quyet dinh cua file khai bao, TUNG detector; dat <=> moi detector dat moi dieu kien.

    reference_aggs: {ten tham chieu: {det: agg}}."""
    vid = plan.decision["variant"]
    check_complete(vid, variant_aggs, plan.detectors)
    for ref in {r["reference"] for r in plan.decision["reid_not_below"]} | {plan.decision["cpr_max_drop"]["reference"]}:
        check_complete(ref, reference_aggs.get(ref) or {}, plan.detectors)
    drop = Fraction(plan.decision["cpr_max_drop"]["value"])
    per_det = {}
    for det in plan.detectors:
        conds = []
        mine = variant_aggs[det]
        for rule in plan.decision["reid_not_below"]:
            ref = reference_aggs[rule["reference"]][det]
            for d in rule["durations"]:
                a, b = _pool_of(mine, d), _pool_of(ref, d)
                _same_set(a, b, f"{det} d{d} vs {rule['reference']}")
                ra, rb = _reid(a), _reid(b)
                conds.append({"condition": f"reid(d{d}) >= {rule['reference']}", "variant": _frac(ra),
                              "reference": _frac(rb), "ok": ra >= rb})
        ref_name = plan.decision["cpr_max_drop"]["reference"]
        a, b = _pool_of(mine, None), _pool_of(reference_aggs[ref_name][det], None)
        _same_set(a, b, f"{det} overall vs {ref_name}")
        ca, cb = _cpr(a), _cpr(b)
        conds.append({"condition": f"cpr(overall) >= {ref_name} - {plan.decision['cpr_max_drop']['value']}",
                      "variant": _frac(ca), "reference": _frac(cb), "delta": float(ca - cb), "ok": ca >= cb - drop})
        per_det[det] = {"passed": all(c["ok"] for c in conds), "conditions": conds}
    passed = all(v["passed"] for v in per_det.values())
    return {"variant": vid, "passed": passed, "per_detector": per_det,
            "conclusion": (f"DAT: bien the {vid} dat moi dieu kien tren ca hai detector" if passed else
                           f"KHONG DAT: bien the {vid} truot it nhat mot dieu kien (ket qua am, ghi nhan)")}


def box_precision(pool: Mapping[str, Any]) -> Optional[float]:
    """correct_occluded_frames / occluded_frames_with_output (frame dung luon co box); None neu khong co box."""
    n = pool.get("occluded_frames_with_output") or 0
    return pool["correct_occluded_frames"] / n if n else None


def describe_pool(pool: Mapping[str, Any]) -> dict[str, Any]:
    occ, ev = pool.get("occlusion_frames") or 0, pool.get("scenarios_evaluated") or 0
    return {
        "scenarios_evaluated": ev,
        "occlusion_frames": occ,
        "correct_occluded_frames": pool.get("correct_occluded_frames"),
        "correct_prediction_ratio": pool["correct_occluded_frames"] / occ if occ else None,
        "occluded_frames_with_output": pool.get("occluded_frames_with_output"),
        "box_precision": box_precision(pool),
        "reidentified": pool.get("reidentified"),
        "reidentification_rate": pool["reidentified"] / ev if ev else None,
        "reidentified_with_new_track_id": pool.get("reidentified_with_new_track_id"),
        "mean_time_to_reidentify_frames": pool.get("mean_time_to_reidentify_frames"),
        "longest_tracking_gap_max": pool.get("longest_tracking_gap_max"),
        "longest_tracking_gap_mean": pool.get("longest_tracking_gap_mean"),
        "mean_position_error_px": pool.get("mean_position_error_px"),
    }


def describe(aggs: Mapping[str, Mapping[str, Any]], detectors) -> dict[str, Any]:
    """Bang mo ta (khong quyet dinh) cua mot cot: tung detector, overall + moi duration."""
    out = {}
    for det in detectors:
        agg = aggs.get(det)
        if agg is None:
            out[det] = None
            continue
        out[det] = {"overall": describe_pool(agg["overall"]),
                    "by_duration_frames": {d: describe_pool(p) for d, p in
                                           sorted((agg.get("by_duration_frames") or {}).items(),
                                                  key=lambda kv: int(kv[0]))},
                    "complete": bool((agg.get("completeness") or {}).get("complete"))}
    return out
