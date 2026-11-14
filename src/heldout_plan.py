"""Nap + kiem tra ke hoach bao cao heldout (configs/heldout_report_plan.yaml). HAM THUAN
(khong doc o dia); doc file / kiem tra sha256 / git: scripts/run_occlusion_ablation.py va
scripts/report_heldout.py."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

SUPPORTED_VERSIONS = (1,)
KINDS = ("baseline", "motion_memory", "extra")
ROLES = ("reference", "decision", "descriptive")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class HeldoutPlanError(ValueError):
    pass


@dataclass(frozen=True)
class Column:
    name: str
    kind: str
    role: str
    config: Optional[str] = None
    sha256: Optional[str] = None
    extra_name: Optional[str] = None


def _file_ref(where: str, raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping) or not raw.get("path") or not SHA_RE.match(str(raw.get("sha256", ""))):
        raise HeldoutPlanError(f"{where}: can path + sha256 (64 hex)")
    return {"path": str(raw["path"]), "sha256": str(raw["sha256"])}


@dataclass(frozen=True)
class HeldoutPlan:
    version: int
    name: str
    group: str
    tag: str
    detectors: tuple[str, ...]
    scenario_config: dict
    metrics_config: dict
    weights_must_match: tuple[str, ...]
    columns: tuple[Column, ...]
    decision: dict
    dev_evidence: tuple[dict, ...]
    report: dict
    bootstrap: dict

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "HeldoutPlan":
        try:
            version = int(data["version"])
            detectors = tuple(str(d) for d in data["detectors"])
            columns_raw = list(data["columns"])
            decision = dict(data["decision"])
            boot = dict(data["bootstrap"])
            report = dict(data["report"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HeldoutPlanError(f"Ke hoach thieu/sai khoa: {exc}") from exc
        if version not in SUPPORTED_VERSIONS:
            raise HeldoutPlanError(f"version={version} chua ho tro")
        if str(data.get("group")) != "heldout":
            raise HeldoutPlanError("group cua ke hoach heldout phai la 'heldout'")
        if not detectors or len(set(detectors)) != len(detectors):
            raise HeldoutPlanError("detectors phai khac rong, khong trung")
        columns = []
        for raw in columns_raw:
            name, kind, role = str(raw.get("name", "")), str(raw.get("kind", "")), str(raw.get("role", ""))
            if not NAME_RE.match(name) or kind not in KINDS or role not in ROLES:
                raise HeldoutPlanError(f"Cot {raw!r}: can name (a-z0-9_), kind {KINDS}, role {ROLES}")
            config = raw.get("config")
            sha = raw.get("sha256")
            extra = raw.get("extra_name")
            if kind == "baseline":
                if config or sha or extra:
                    raise HeldoutPlanError(f"Cot {name}: baseline khong co config")
            else:
                if not config or not SHA_RE.match(str(sha or "")):
                    raise HeldoutPlanError(f"Cot {name}: can config + sha256 (64 hex)")
            if kind == "extra":
                if not extra or not NAME_RE.match(str(extra)):
                    raise HeldoutPlanError(f"Cot {name}: cot extra can extra_name (a-z0-9_)")
            elif extra:
                raise HeldoutPlanError(f"Cot {name}: chi cot extra co extra_name")
            columns.append(Column(name, kind, role, config and str(config), sha and str(sha),
                                  extra and str(extra)))
        names = [c.name for c in columns]
        extras = [c.extra_name for c in columns if c.extra_name]
        if len(set(names)) != len(names) or len(set(extras)) != len(extras):
            raise HeldoutPlanError("Ten cot / extra_name bi trung")
        for kind in ("baseline", "motion_memory"):
            if [c.kind for c in columns].count(kind) != 1:
                raise HeldoutPlanError(f"Can dung MOT cot kind={kind}")
        decision_cols = [c.name for c in columns if c.role == "decision"]
        if decision_cols != [decision.get("column")]:
            raise HeldoutPlanError("Can dung MOT cot role=decision, trung decision.column")
        refs = dict(decision.get("reference_columns") or {})
        if not refs or any(v not in names for v in refs.values()):
            raise HeldoutPlanError("decision.reference_columns phai tro toi cot da khai bao")
        boot_cols = list(boot.get("paired_vs") or [])
        if any(c not in names for c in boot_cols):
            raise HeldoutPlanError("bootstrap.paired_vs phai la cot da khai bao")
        if int(boot.get("resamples", 0)) < 1 or not 0 < float(boot.get("ci", 0)) < 1 or "seed" not in boot:
            raise HeldoutPlanError("bootstrap can resamples >= 1, 0 < ci < 1, seed")
        diag = list(report.get("diagnosis_columns") or [])
        if any(c not in names or next(x for x in columns if x.name == c).kind == "baseline" for c in diag):
            raise HeldoutPlanError("report.diagnosis_columns phai la cot Motion Memory da khai bao")
        return cls(
            version=version, name=str(data.get("name", "")), group="heldout", tag=str(data["tag"]),
            detectors=detectors,
            scenario_config=_file_ref("scenario_config", data.get("scenario_config")),
            metrics_config=_file_ref("metrics_config", data.get("metrics_config")),
            weights_must_match=tuple(str(g) for g in (data.get("weights") or {}).get("must_match_screening_of") or ()),
            columns=tuple(columns),
            decision={"column": decision["column"], "rules_from": _file_ref("decision.rules_from",
                                                                            decision.get("rules_from")),
                      "reference_columns": {str(k): str(v) for k, v in refs.items()}},
            dev_evidence=tuple(_file_ref("dev_evidence", r) for r in data.get("dev_evidence") or []),
            report=report,
            bootstrap={"cluster": str(boot.get("cluster", "sequence")), "resamples": int(boot["resamples"]),
                       "seed": int(boot["seed"]), "ci": float(boot["ci"]), "paired_vs": boot_cols,
                       "metrics": list(boot.get("metrics") or [])},
        )

    def column(self, name: str) -> Column:
        return next(c for c in self.columns if c.name == name)

    @property
    def extras(self) -> dict[str, Column]:
        return {c.extra_name: c for c in self.columns if c.kind == "extra"}

    @property
    def mm_default(self) -> Column:
        return next(c for c in self.columns if c.kind == "motion_memory")

    def experiment_id(self, column: str, det: str, tag: Optional[str] = None) -> str:
        """Ten experiment cua cot (cung quy uoc scripts/run_occlusion_ablation.py)."""
        c = self.column(column)
        t = tag or self.tag
        if c.kind == "baseline":
            return f"{det}_botsort_occl_{self.group}_{t}"
        if c.kind == "motion_memory":
            return f"{det}_botsort_mm_occl_{self.group}_{t}"
        return f"{det}_botsort_mm_{c.extra_name}_occl_{self.group}_{t}"

    def referenced_files(self) -> dict[str, str]:
        """{duong dan: sha256} moi file ke hoach tham chieu (phai track trong HEAD + dung sha)."""
        out = {self.scenario_config["path"]: self.scenario_config["sha256"],
               self.metrics_config["path"]: self.metrics_config["sha256"],
               self.decision["rules_from"]["path"]: self.decision["rules_from"]["sha256"]}
        for c in self.columns:
            if c.config:
                out[c.config] = c.sha256
        for ev in self.dev_evidence:
            out[ev["path"]] = ev["sha256"]
        return out
