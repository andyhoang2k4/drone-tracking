#!/usr/bin/env bash
# Train yolo11-seg. CHI CHAY TREN UBUNTU/CLOUD PC (CLAUDE.md muc 3).
#
# Doc tham so tu configs/train_config.yaml - khong hardcode o day.
# Usage: bash scripts/train_yolo11.sh <experiment_id> [--allow-partial] [key=value...]
#   --allow-partial: cho phep train tren dataset chay thu (mac dinh: dung lai).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXPERIMENT_ID="${1:?Thieu experiment_id, vd: yolo11_baseline_001}"
shift || true

MODEL_KEY="yolo11_seg"
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
from src.utils.run_context import build_run_context, dataset_info, partial_dataset_error

config_path, model_key, experiment_id, *extra = sys.argv[1:]
allow_partial = "--allow-partial" in extra
extra = [item for item in extra if item != "--allow-partial"]
cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

model_cfg = cfg["models"][model_key]
train_kwargs = dict(cfg["common"])
train_kwargs["data"] = cfg["data"]

# Ghi de tu CLI: dang key=value
for item in extra:
    key, _, value = item.partition("=")
    train_kwargs[key] = yaml.safe_load(value)

local_weights = Path(cfg.get("weights_dir", "weights")) / model_cfg["weights"]
if local_weights.is_file():
    weights = str(local_weights)
else:
    weights = model_cfg["weights"]
    print(f"[train] CANH BAO: khong thay {local_weights}, Ultralytics se tai {weights}")
print(f"[train] weights       = {weights}")

dataset = dataset_info(Path(cfg["data"]))
if dataset["name"] is None:
    print(f"[train] CANH BAO: khong doc duoc ten/version dataset tu {cfg['data']} "
          "(thieu manifest.json cua prepare_visdrone.py?)")
else:
    print(f"[train] dataset       = {dataset['name']} v{dataset['version']}")

error = partial_dataset_error(dataset, allow_partial)
if error:
    sys.exit(f"[train] DUNG: {error}")
if dataset["partial"]:
    print(f"[train] CANH BAO: train tren dataset CHAY THU ({', '.join(dataset['partial_reasons'])}) "
          "vi co --allow-partial")

exp_dir = Path(cfg["output_root"]) / experiment_id
exp_dir.mkdir(parents=True, exist_ok=True)

write_json(
    exp_dir / "run_context.json",
    build_run_context(
        experiment_id=experiment_id,
        config={"model": {**model_cfg, "weights_path": weights}, **train_kwargs},
        dataset=dataset,
        repo_root=Path.cwd(),
        extra={"allow_partial": allow_partial},
    ),
)

from ultralytics import YOLO

model = YOLO(weights)
results = model.train(project=str(exp_dir), name="train", **train_kwargs)
print(f"[train] xong. Ket qua trong {exp_dir}")
PY
