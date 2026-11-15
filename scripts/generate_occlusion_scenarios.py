"""Sinh kich ban occlusion nhan tao co ground truth (src/occlusion_scenarios.py).

Mac dinh dung configs/occlusion_scenarios_v2.yaml (tu 2026-11-14). v2 CHI sinh
kich ban cho muc tieu da qua screening khoa muc tieu
(scripts/screen_occlusion_targets.py) va bat buoc --detector cho MOI detector
trong target_lock.required_detectors de doi chieu weights sha256 voi
screening.json. Tu choi khi screening partial / co loi / khac config_sha256 /
khac danh sach ung vien / khac weights sha256.

Vi du (tren cloud, sau khi giai nen VisDrone2019-MOT val + test-dev vao data/raw/)::

    # v2: sau screening day du
    python scripts/generate_occlusion_scenarios.py --group dev \\
        --detector yolo11=weights/yolo11_best.pt --detector yolo26=weights/yolo26_best.pt --dry-run
    python scripts/generate_occlusion_scenarios.py --group dev \\
        --detector yolo11=weights/yolo11_best.pt --detector yolo26=weights/yolo26_best.pt

    # v1 (chi de tai hien lan chay thu): khong can screening
    python scripts/generate_occlusion_scenarios.py --config configs/occlusion_scenarios.yaml --group dev

Ket qua ghi vao <output_root>/<group>/ (v2: data/occlusion_test_scenarios/v2/,
v1: data/occlusion_test_scenarios/; bi gitignore tru index.json/screening.json).
Kich ban da sinh voi cung config duoc bo qua.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_occlusion_ablation import parse_detector  # noqa: E402
from src.occlusion_scenarios import file_sha256, generate_group, load_scenario_config  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "occlusion_scenarios_v2.yaml"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--group", required=True, help="Ten nhom trong config (dev | heldout)")
    parser.add_argument("--max-sequences", type=int, default=None, help="Chi N sequence dau (smoke test)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Sinh lai kich ban da ton tai voi config khac")
    parser.add_argument("--dry-run", action="store_true", help="Chi liet ke, khong ghi file")
    parser.add_argument("--screening", type=Path, default=None,
                        help="v2: screening.json (mac dinh <output_root>/<group>/screening.json)")
    parser.add_argument("--detector", type=parse_detector, action="append", default=[],
                        help="v2: ten=duong_dan_weights cho MOI detector cua screening (doi chieu sha256)")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_scenario_config(args.config)
    weights_sha256 = None
    if cfg.uses_screening:
        names = [n for n, _ in args.detector]
        if len(set(names)) != len(names):
            raise SystemExit("--detector bi trung ten")
        weights_sha256 = {}
        for name, weights in args.detector:
            if not Path(weights).is_file():
                raise SystemExit(f"Khong thay weights {weights} cua {name}")
            weights_sha256[name] = file_sha256(weights)
    elif args.detector or args.screening:
        raise SystemExit("--detector/--screening chi dung cho config v2 (co target_lock)")
    index = generate_group(
        cfg, args.group,
        overwrite=args.overwrite, max_sequences=args.max_sequences, dry_run=args.dry_run,
        screening_file=args.screening, weights_sha256=weights_sha256,
    )
    print(f"Nhom {args.group} (config v{cfg.version}): {len(index['sequences'])} sequence, "
          f"{len(index['scenarios'])} kich ban "
          f"({len(index['skipped_existing'])} da co, bo qua)")
    if cfg.uses_screening:
        for seq in index["sequences"]:
            print(f"  {seq}: {index['candidates_per_sequence'][seq]} ung vien, "
                  f"{index['eligible_per_sequence'][seq]} dat, chon {index['selected_targets'][seq]}")
    if index["sequences_without_target"]:
        print(f"Sequence khong co muc tieu hop le: {index['sequences_without_target']}")
    if index["partial"]:
        print("CANH BAO: --max-sequences -> index danh dau partial, khong dung de benchmark")
    if args.dry_run:
        for name in index["scenarios"]:
            print(f"  {name}")
    else:
        print(f"Index: {cfg.output_root / args.group / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
