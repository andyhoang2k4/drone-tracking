"""Sinh kich ban occlusion nhan tao co ground truth (src/occlusion_scenarios.py).

Vi du (tren cloud, sau khi giai nen VisDrone2019-MOT val + test-dev vao data/raw/)::

    # Xem truoc danh sach kich ban, khong ghi gi
    python scripts/generate_occlusion_scenarios.py --group dev --dry-run
    # Smoke test 1 sequence (index danh dau partial)
    python scripts/generate_occlusion_scenarios.py --group dev --max-sequences 1
    # Day du
    python scripts/generate_occlusion_scenarios.py --group dev
    python scripts/generate_occlusion_scenarios.py --group heldout

Ket qua ghi vao <output_root>/<group>/ (mac dinh data/occlusion_test_scenarios/,
bi gitignore). Kich ban da sinh voi cung config duoc bo qua.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.occlusion_scenarios import generate_group, load_scenario_config  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "occlusion_scenarios.yaml"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--group", required=True, help="Ten nhom trong config (dev | heldout)")
    parser.add_argument("--max-sequences", type=int, default=None, help="Chi N sequence dau (smoke test)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Sinh lai kich ban da ton tai voi config khac")
    parser.add_argument("--dry-run", action="store_true", help="Chi liet ke, khong ghi file")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_scenario_config(args.config)
    index = generate_group(
        cfg, args.group,
        overwrite=args.overwrite, max_sequences=args.max_sequences, dry_run=args.dry_run,
    )
    print(f"Nhom {args.group}: {len(index['sequences'])} sequence, "
          f"{len(index['scenarios'])} kich ban "
          f"({len(index['skipped_existing'])} da co, bo qua)")
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
