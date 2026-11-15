"""Gom summary.json cua cac lan chay run_closed_loop.py thanh bang (Markdown).

Chi la so sanh hanh vi vong kin trong sim (CLAUDE.md muc 2): mot world, moi cau hinh mot lan chay. KHONG phai
benchmark hay danh gia Motion Memory (CLAUDE.md muc 12, 13).

    python scripts/sim_probe/compare_runs.py ~/sim_probe/follow_v2/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def row(name: str, s: dict) -> str:
    occ = [o for o in s["occlusions"] if o.get("reappeared")]
    ok = sum(1 for o in occ if o.get("reacquired_on_target"))
    st = s["state_ticks"]
    ratio = s["on_target_ratio_when_visible"]
    return (f"| {name} | {s['detector']} | {s['estimator']}{' (mm: ' + Path(s['mm_config']).stem + ')' if s.get('mm_config') else ''} "
            f"| {ratio if ratio is not None else '-'} | {s['locked_on_other_object_ticks']} | {s['locked_id_changes']} "
            f"| {s['first_failed_t'] if s['first_failed_t'] is not None else '-'} | {ok}/{len(occ)} "
            f"| {st.get('TRACKING', 0)}/{st.get('LOST', 0)}/{st.get('REIDENTIFYING', 0)}/{st.get('FAILED', 0)} "
            f"| {s['loop_hz']} | {s['tilt_deg']['max'] if s.get('tilt_deg') else '-'} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir", type=Path)
    a = ap.parse_args()
    print("| lan chay | detector | estimator | ti le bam dung (khi muc tieu thay duoc) | tick bam nham vat khac "
          "| doi ID | FAILED luc (s) | che khuat bat lai dung / so lan | tick TRACK/LOST/REID/FAILED | Hz | nghieng max (do) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(a.runs_dir.glob("*/summary.json")):
        print(row(p.parent.name, json.loads(p.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
