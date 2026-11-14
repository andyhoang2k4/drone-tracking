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
