# Drone AI Tracking

Pipeline AI theo doi muc tieu cho drone quadcopter trinh sat: **image
segmentation** de phat hien muc tieu, **BoT-SORT** de tracking, va **motion
memory / temporal propagation** de du doan vi tri khi muc tieu bi che khuat
hoac mat detection do nhieu.

Ngu canh day du, quy tac kien truc va coding rules: xem [CLAUDE.md](CLAUDE.md).

## Giai doan hien tai

**Mo phong / train tren cloud GPU. CHUA deploy len phan cung that.**

Khong toi uu edge (Jetson/Hailo), khong export ONNX/TensorRT, khong tich hop
PX4 o giai doan nay - xem CLAUDE.md muc 2.

## Pipeline

```
[Video/Camera] -> [YOLO11-seg / YOLO26-seg] -> [BoT-SORT] -> [Motion Memory]
                                                                    |
                                          [track ID + bbox + trang thai]
```

Motion Memory la **post-processing**: nhan output cua BoT-SORT lam input,
khong sua ma nguon noi bo cua `ultralytics`.

## Cau truc

| Duong dan | Noi dung |
|---|---|
| `configs/botsort.yaml` | Config BoT-SORT rieng cua project (Re-ID mac dinh TAT) |
| `configs/train_config.yaml` | Config train dung chung cho YOLO11-seg va YOLO26-seg |
| `src/detect.py` | Chay detection, log JSONL |
| `src/track.py` | Chay detection + BoT-SORT, log JSONL |
| `src/motion_memory.py` | Motion memory / temporal propagation |
| `src/evaluate.py` | Metric danh gia (detection, tracking, occlusion) |
| `src/utils/` | Logger JSONL, thu thap metadata tai hien experiment |
| `scripts/train_yolo11.sh`, `train_yolo26.sh` | Train - **chi chay tren Ubuntu/cloud** |
| `scripts/run_pipeline.py` | Pipeline day du, co co `--motion-memory` cho ablation |
| `scripts/check_env.py` | Kiem tra Python, version package, CUDA, schema BoT-SORT, weight |
| `data/` | Dataset (khong commit) - xem [data/README.md](data/README.md) |
| `experiments/` | Ket qua tung lan chay - xem [experiments/README.md](experiments/README.md) |

## Trang thai implement

| Phan | Trang thai |
|---|---|
| Khung thu muc, config, logging, metadata experiment | Xong |
| `detect.py`, `track.py`, `run_pipeline.py` | Smoke test CPU qua voi yolo11n-seg/yolo26n-seg pretrained (ultralytics 8.4.174), **chua chay tren GPU / dataset that** |
| `motion_memory.py` state machine | Xong, co unit test |
| `motion_memory.py` ngoai suy vi tri + prediction confidence | **Chua implement** (roadmap buoc 5) |
| `evaluate.py` cac ham tinh metric | **Chua implement** - cho chot dinh dang ground truth (roadmap buoc 4) |

## Cai dat

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# torch TRUOC, dung build cho may (xem dau requirements.txt)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu  # local CPU
pip install -r requirements.txt
```

Cai torch/torchvision tu index cua PyTorch TRUOC (xem chu thich dau
`requirements.txt`: CPU cho local, CUDA cho cloud), roi moi cai requirements.
Version da xac minh tren Windows CPU, **chua xac minh tren cloud GPU**.
Kiem tra truoc khi train:

```bash
pip show ultralytics torch opencv-python
```

Neu lech so voi `requirements.txt`: bao mismatch, khong tu sua file
(CLAUDE.md muc 4).

Hoac dung script kiem tra tong hop:

```bash
# Local (CPU)
python scripts/check_env.py --check-weights
# Cloud PC (bat buoc co CUDA)
python scripts/check_env.py --require-cuda --check-weights --output env_report.json
```

## Quy trinh lam viec

```
Windows (viet code) -> git push -> Cloud PC Ubuntu (git pull) -> train/test GPU
```

## Chay

```bash
# Detection
python -m src.detect --weights <best.pt> --source data/raw/clip01.mp4 \
    --output experiments/<exp_id>/logs/detect.jsonl

# Detection + BoT-SORT
python -m src.track --weights <best.pt> --source data/raw/clip01.mp4 \
    --tracker configs/botsort.yaml \
    --output experiments/<exp_id>/logs/track.jsonl

# Pipeline day du (baseline, chua co Motion Memory)
python scripts/run_pipeline.py --weights <best.pt> \
    --source data/occlusion_test_scenarios/scn01.mp4 \
    --experiment-id <exp_id> --target-track-id 1

# Train (chi tren Ubuntu/cloud)
bash scripts/train_yolo11.sh yolo11_baseline_001
```

## Test

```bash
python -m pytest tests/ -q
```

## Danh gia

Precision va recall bao cao **song song**. Moi kich ban occlusion bat buoc co
ground truth cho toan bo sequence. Danh gia Motion Memory bat buoc co ablation
co/khong Motion Memory tren cung detector + tracker. Chi tiet: CLAUDE.md muc 12.
