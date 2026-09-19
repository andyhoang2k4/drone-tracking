#!/usr/bin/env bash
# Train yolo26-seg. CHI CHAY TREN UBUNTU/CLOUD PC (CLAUDE.md muc 3).
#
# Doc tham so tu configs/train_config.yaml - khong hardcode o day.
# Usage: bash scripts/train_yolo26.sh <experiment_id> [extra yolo args...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENT_ID="${1:?Thieu experiment_id, vd: yolo26_baseline_001}"
shift || true

MODEL_KEY="yolo26_seg"
CONFIG="configs/train_config.yaml"

echo "[train] experiment_id = $EXPERIMENT_ID"
echo "[train] model_key     = $MODEL_KEY"
echo "[train] config        = $CONFIG"

python - "$CONFIG" "$MODEL_KEY" "$EXPERIMENT_ID" "$@" <<'PY'
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path.cwd()))
from src.utils.logging_utils import write_json
from src.utils.run_context import build_run_context

config_path, model_key, experiment_id, *extra = sys.argv[1:]
cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

model_cfg = cfg["models"][model_key]
train_kwargs = dict(cfg["common"])
train_kwargs["data"] = cfg["data"]

# Ghi de tu CLI: dang key=value
for item in extra:
    key, _, value = item.partition("=")
    train_kwargs[key] = yaml.safe_load(value)

exp_dir = Path(cfg["output_root"]) / experiment_id
exp_dir.mkdir(parents=True, exist_ok=True)

write_json(
    exp_dir / "run_context.json",
    build_run_context(
        experiment_id=experiment_id,
        config={"model": model_cfg, **train_kwargs},
        dataset=cfg["data"],
        repo_root=Path.cwd(),
    ),
)

from ultralytics import YOLO

model = YOLO(model_cfg["weights"])
results = model.train(project=str(exp_dir), name="train", **train_kwargs)
print(f"[train] xong. Ket qua trong {exp_dir}")
PY
