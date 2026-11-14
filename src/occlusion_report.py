"""Tang bao cao occlusion (B1): HAM THUAN tren metrics da co, KHONG doi src/evaluate.py hay
version metrics.

- box_precision = correct_occluded_frames / occluded_frames_with_output (frame dung luon co
  box) - CHI MO TA; khong dung de chon cau hinh.
- Gop hai detector: CONG count roi chia (khong trung binh ti le).
- Bootstrap theo cluster (sequence): moi lan lay lai CO HOAN LAI cac sequence, CUNG mot mau
  cho moi cot va moi detector; uoc luong lai ti le tu tong count; hieu THEO CAP (cot - tham
  chieu) tren cung mau. Khoang percentile. MO TA do bat dinh, KHONG phai kiem dinh y nghia.

Dau vao: metrics.json gop (`overall`, `by_duration_frames`) va metrics.json tung kich ban
(metrics v2, co `target_locked_at_occlusion`).
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Optional, Sequence

COUNT_KEYS = ("scenarios_evaluated", "reidentified", "occlusion_frames", "correct_occluded_frames",
              "occluded_frames_with_output")
RATIOS = {
    "correct_prediction_ratio": ("correct_occluded_frames", "occlusion_frames"),
    "box_precision": ("correct_occluded_frames", "occluded_frames_with_output"),
    "reidentification_rate": ("reidentified", "scenarios_evaluated"),
}
BOOTSTRAP_NOTE = ("Bootstrap theo sequence (cluster), cung mau cho moi cot: MO TA do bat dinh, KHONG phai "
                  "kiem dinh y nghia thong ke; so sequence it -> khoang rong/khong on dinh.")


class ReportError(ValueError):
    pass


def ratio(counts: Mapping[str, Any], name: str) -> Optional[float]:
    num, den = RATIOS[name]
    d = counts.get(den) or 0
    return counts.get(num, 0) / d if d else None


def box_precision(pool: Mapping[str, Any]) -> Optional[float]:
    return ratio(pool, "box_precision")


def _counts(pool: Mapping[str, Any]) -> dict[str, int]:
    return {k: int(pool.get(k) or 0) for k in COUNT_KEYS}


def _with_ratios(counts: Mapping[str, int]) -> dict[str, Any]:
    return {**counts, **{name: ratio(counts, name) for name in RATIOS}}


def pooled_across_detectors(aggs: Mapping[str, Mapping[str, Any]], detectors: Sequence[str]) -> dict[str, Any]:
    """Cong count cua cac detector (overall + moi duration) roi tinh ti le."""
    def add(pools):
        total = {k: 0 for k in COUNT_KEYS}
        for p in pools:
            for k, v in _counts(p).items():
                total[k] += v
        return _with_ratios(total)

    durations = sorted({d for det in detectors for d in (aggs[det].get("by_duration_frames") or {})}, key=int)
    return {
        "overall": add(aggs[det]["overall"] for det in detectors),
        "by_duration_frames": {d: add((aggs[det].get("by_duration_frames") or {}).get(d, {}) for det in detectors)
                               for d in durations},
    }


# ---------------- bootstrap theo sequence ----------------

def scenario_counts(item: Mapping[str, Any]) -> Optional[dict[str, int]]:
    """Count cua MOT kich ban (None neu chua khoa -> khong tinh, nhu metrics v2)."""
    m = item["metrics"]
    if "target_locked_at_occlusion" not in m:
        raise ReportError(f"{item.get('scenario_id')}: metrics khong phai v2")
    if not m["target_locked_at_occlusion"]:
        return None
    return {"scenarios_evaluated": 1, "reidentified": int(bool(m["reidentified"])),
            "occlusion_frames": int(m["occlusion_frames"]), "correct_occluded_frames": int(m["correct_occluded_frames"]),
            "occluded_frames_with_output": int(m["occluded_frames_with_output"])}


def _scopes(durations: Sequence[int]) -> list[tuple[str, Optional[int]]]:
    return [("overall", None)] + [(f"d{d}", int(d)) for d in durations]


def sequence_tables(items: Sequence[Mapping[str, Any]], durations: Sequence[int]) -> dict[str, dict[str, dict]]:
    """{scope: {sequence: counts}} tu metrics tung kich ban (chi kich ban da khoa)."""
    out: dict[str, dict[str, dict]] = {name: {} for name, _ in _scopes(durations)}
    for it in items:
        c = scenario_counts(it)
        if c is None:
            continue
        seq = str(it["scenario"]["sequence"])
        dur = int(it["scenario"]["duration_frames"])
        for name, d in _scopes(durations):
            if d is not None and d != dur:
                continue
            row = out[name].setdefault(seq, {k: 0 for k in COUNT_KEYS})
            for k, v in c.items():
                row[k] += v
    return out


def _locked_set(items: Sequence[Mapping[str, Any]]) -> set:
    return {(it["scenario_id"], bool(it["metrics"]["target_locked_at_occlusion"])) for it in items}


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    pos = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _summary(point: Optional[float], values: list[float], ci: float, resamples: int) -> dict[str, Any]:
    vals = sorted(values)
    alpha = (1.0 - ci) / 2.0
    return {"point": point, "ci_low": _quantile(vals, alpha) if vals else None,
            "ci_high": _quantile(vals, 1.0 - alpha) if vals else None,
            "n_valid": len(vals), "n_resamples": resamples}


def bootstrap(columns: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], detectors: Sequence[str], *,
              resamples: int, seed: int, ci: float, paired_vs: Sequence[str], metrics: Sequence[str],
              durations: Sequence[int]) -> dict[str, Any]:
    """columns: {cot: {det: [metrics tung kich ban]}}. Tra ve uoc luong + khoang cho tung detector va
    'pooled' (cong hai detector), moi scope (overall, d<duration>), moi cot va hieu theo cap."""
    if resamples < 1 or not 0 < ci < 1:
        raise ReportError("resamples >= 1 va 0 < ci < 1")
    for m in metrics:
        if m not in RATIOS:
            raise ReportError(f"Chi so bootstrap khong ho tro: {m}")
    cols = list(columns)
    for ref in paired_vs:
        if ref not in columns:
            raise ReportError(f"Cot tham chieu {ref} khong co")
    for det in detectors:                          # cung tap kich ban da khoa o moi cot
        sets = {c: _locked_set(columns[c][det]) for c in cols}
        first = sets[cols[0]]
        bad = [c for c in cols if sets[c] != first]
        if bad:
            raise ReportError(f"{det}: cot {bad} khong cung tap kich ban/da khoa voi {cols[0]}")
    tables = {c: {det: sequence_tables(columns[c][det], durations) for det in detectors} for c in cols}
    seqs = sorted({s for c in cols for det in detectors for s in tables[c][det]["overall"]})
    if not seqs:
        raise ReportError("Khong co kich ban da khoa nao")
    rng = random.Random(seed)
    samples = [[rng.randrange(len(seqs)) for _ in seqs] for _ in range(resamples)]
    multiplicities = []
    for idx in samples:
        mult: dict[int, int] = {}
        for i in idx:
            mult[i] = mult.get(i, 0) + 1
        multiplicities.append(mult)

    def vectors(col: str, det_key: str, scope: str) -> list[dict[str, int]]:
        dets = detectors if det_key == "pooled" else [det_key]
        out = []
        for s in seqs:
            row = {k: 0 for k in COUNT_KEYS}
            for d in dets:
                for k, v in tables[col][d][scope].get(s, {}).items():
                    row[k] += v
            out.append(row)
        return out

    def total(vec: list[dict[str, int]], mult: Optional[dict[int, int]]) -> dict[str, int]:
        t = {k: 0 for k in COUNT_KEYS}
        items = mult.items() if mult is not None else ((i, 1) for i in range(len(vec)))
        for i, w in items:
            for k in COUNT_KEYS:
                t[k] += vec[i][k] * w
        return t

    estimates: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for det_key in list(detectors) + ["pooled"]:
        estimates[det_key], paired[det_key] = {}, {}
        for scope, _ in _scopes(durations):
            vecs = {c: vectors(c, det_key, scope) for c in cols}
            reps = {c: [total(vecs[c], m) for m in multiplicities] for c in cols}
            points = {c: total(vecs[c], None) for c in cols}
            estimates[det_key][scope] = {
                c: {m: _summary(ratio(points[c], m),
                                [v for v in (ratio(r, m) for r in reps[c]) if v is not None], ci, resamples)
                    for m in metrics} for c in cols}
            paired[det_key][scope] = {}
            for ref in paired_vs:
                for c in cols:
                    if c == ref:
                        continue
                    row = {}
                    for m in metrics:
                        pa, pb = ratio(points[c], m), ratio(points[ref], m)
                        diffs = [a - b for a, b in ((ratio(rc, m), ratio(rr, m)) for rc, rr in zip(reps[c], reps[ref]))
                                 if a is not None and b is not None]
                        row[m] = _summary(pa - pb if pa is not None and pb is not None else None, diffs, ci, resamples)
                    paired[det_key][scope][f"{c} - {ref}"] = row
    return {
        "settings": {"cluster": "sequence", "resamples": resamples, "seed": seed, "ci": ci, "method": "percentile",
                     "same_sample_all_columns": True, "paired_vs": list(paired_vs), "metrics": list(metrics),
                     "rng": "python random.Random(seed).randrange"},
        "sequences": seqs,
        "estimates": estimates,
        "paired_differences": paired,
        "note": BOOTSTRAP_NOTE,
    }
