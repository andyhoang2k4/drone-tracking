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
| `configs/train_config.yaml` | Config train dung chung cho YOLO11-seg va YOLO26-seg; `inference.conf` dung chung khi inference/danh gia |
| `configs/motion_memory.yaml` | Tham so Motion Memory (gia tri khoi dau chua tinh chinh; CLI > file > mac dinh dataclass) |
| `configs/eval_visdrone.yaml` | Danh gia VisDrone2019-MOT (val/test-dev): nguong IoU/IoA, class-aware, warmup FPS; chon muc tieu theo GT |
| `configs/occlusion_scenarios.yaml` | Kich ban occlusion nhan tao: seed, quy tac chon muc tieu, thoi luong, kieu che, nhom dev/heldout (co dinh truoc khi chay) |
| `configs/occlusion_metrics.yaml` | Dinh nghia + nguong do do occlusion (co dinh truoc khi chay) |
| `src/occlusion_scenarios.py` | Sinh kich ban occlusion co GT moi frame |
| `src/detect.py` | Chay detection, log JSONL |
| `src/track.py` | Chay detection + BoT-SORT, log JSONL |
| `src/motion_memory.py` | Motion memory / temporal propagation |
| `src/evaluate.py` | Metric danh gia (detection, tracking, occlusion) |
| `src/utils/` | Logger JSONL, thu thap metadata tai hien experiment, resolve tham so inference/Motion Memory |
| `scripts/train_yolo11.sh`, `train_yolo26.sh` | Train - **chi chay tren Ubuntu/cloud** |
| `scripts/run_pipeline.py` | Pipeline day du, co co `--motion-memory` cho ablation |
| `scripts/generate_occlusion_scenarios.py` | Sinh kich ban occlusion cho mot nhom (dev/heldout) |
| `scripts/run_occlusion_ablation.py` | Chay ablation co/khong Motion Memory tren mot nhom kich ban, roi danh gia/gop/so sanh |
| `scripts/prepare_visdrone.py` | Chuyen VisDrone2019-MOT sang dinh dang Ultralytics (giai doan A, 10 class) |
| `scripts/generate_masks.py` | Sinh mask polygon tu box bang SAM, fallback hinh chu nhat khi mask lech |
| `scripts/check_env.py` | Kiem tra Python, version package, CUDA, schema BoT-SORT, weight |
| `data/` | Dataset (khong commit) - xem [data/README.md](data/README.md) |
| `experiments/` | Ket qua tung lan chay - xem [experiments/README.md](experiments/README.md) |

## Trang thai implement

| Phan | Trang thai |
|---|---|
| Khung thu muc, config, logging, metadata experiment | Xong |
| `detect.py`, `track.py`, `run_pipeline.py` | Smoke test CPU qua voi yolo11n-seg/yolo26n-seg pretrained (ultralytics 8.4.174), **chua chay tren GPU / dataset that** |
| `motion_memory.py` state machine (LOST dung L frame, REIDENTIFYING dung R frame, dem theo `frame_idx`) | Xong, unit test du lieu gia |
| `motion_memory.py` ngoai suy vi tri + prediction confidence | Xong, unit test du lieu gia; **chua chay tren log that** |
| `motion_memory.py` khop lai: gate chuyen dong trong LOST, dung ID cu trong REIDENTIFYING | Xong, unit test du lieu gia; **chua chay tren log that** |
| `run_pipeline.py --motion-memory` (config, ung vien, theo ID moi) | Xong, test voi track gia (khong YOLO); **chua chay tren GPU / dataset that** |
| `evaluate.py` precision/recall + ID switch tren ground truth VisDrone-MOT | Xong (unit test du lieu gia, chua chay tren log that) |
| Kich ban occlusion nhan tao (`src/occlusion_scenarios.py`) | Xong, test voi anh/annotation gia; **chua sinh tren VisDrone that** |
| `evaluate.py` metric occlusion + gop + so sanh ablation | Xong, test voi log/manifest gia; **chua chay tren log that** |
| `run_pipeline.py --replay-tracks` + ablation chay BoT-SORT mot lan / kich ban | Xong, test voi log track gia; **chua chay tren GPU / dataset that** |

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

# Pipeline day du: baseline (YOLO + BoT-SORT) va ablation (+ Motion Memory)
python scripts/run_pipeline.py --weights <best.pt> \
    --source data/occlusion_test_scenarios/scn01.mp4 \
    --experiment-id <exp_id> --target-track-id 1
python scripts/run_pipeline.py ... --motion-memory \
    [--motion-memory-config configs/motion_memory.yaml] \
    [--lost-timeout-frames N] [--reid-timeout-frames N]
# Tham so Motion Memory: CLI > configs/motion_memory.yaml > mac dinh dataclass;
# gia tri da dung ghi vao run_context.json (khoa "motion_memory").
# Log moi frame (khi bat): state (null truoc khi thay muc tieu), predicted_bbox,
# motion_confidence (RIENG, khong gop voi detector_confidence), track_id (doi
# khi khop lai bang ID moi), recover_reason, gate_candidates_passed.
# Chon muc tieu theo ground truth thay vi --target-track-id:
#   --gt-target-id <id> --gt-annotation <annotation VisDrone .txt> [--gt-first-frame 1]
# (ghep GT voi track BoT-SORT bang IoU o frame dau muc tieu xuat hien; ket qua
# ghi experiments/<id>/target_selection.json).
# Replay: --replay-tracks <tracks.jsonl cua src.track> -> khong chay YOLO, doc
# track tu log (can tracks.meta.json canh log; weights/source/tracker phai khop,
# conf/imgsz/classes lay tu meta; SHA-256 log ghi vao run_context.json).
# Log pipeline co truong `path` (anh nguon) de kiem tra thu tu anh.

# Ablation tren kich ban occlusion (roadmap buoc 4-6), chay tren cloud:
python scripts/generate_occlusion_scenarios.py --group dev
python scripts/run_occlusion_ablation.py --group dev --tag 001 \
    --detector yolo11=<yolo11 best.pt> --detector yolo26=<yolo26 best.pt> --device 0
# Moi detector x kich ban: src.track chay YOLO + BoT-SORT MOT lan
# (experiments/<detector>_botsort_tracks_occl_dev_001/), roi baseline va Motion
# Memory hau ky tren CUNG log do (run_pipeline --replay-tracks).
# --dry-run: in lenh + so kich ban + so lan goi du kien. --limit-scenarios N:
# chi N kich ban dau de do thoi gian (ban gop danh dau incomplete, khong phai
# benchmark; dung --tag rieng).
# -> experiments/<detector>_botsort[_mm]_occl_dev_001/{config.yaml,metrics.json}
#    + experiments/<detector>_botsort_mm_occl_dev_001/ablation_vs_<baseline>.json
# Commit: index.json cua nhom kich ban, config.yaml, metrics.json gop,
# ablation_vs_*.json. Gitignore: anh/manifest kich ban, experiments/**/scenarios/.
# Nhom heldout CHI chay sau khi da chot tham so Motion Memory tren dev.

# imgsz: --imgsz > detector.imgsz trong experiments/<id>/config.yaml >
# run_context.json canh weights > configs/train_config.yaml. Gia tri da dung
# ghi vao <output>.meta.json (detect/track) va run_context.json (pipeline).
# conf: co dinh o inference.conf trong configs/train_config.yaml, dung chung
# cho YOLO11/YOLO26; --conf chi de thu nghiem (nguon ghi vao meta, va
# src/evaluate.py tu choi log co conf khac config).

# Danh gia tracking tren VisDrone2019-MOT (tung sequence rieng, val hoac test-dev).
# --source cua track phai la thu muc anh cua DUNG sequence do.
SEQ=uav0000086_00000_v
python -m src.track --weights <best.pt> \
    --source data/raw/VisDrone2019-MOT/VisDrone2019-MOT-val/sequences/$SEQ \
    --output experiments/<exp_id>/logs/val/$SEQ.jsonl
python -m src.evaluate --split val --sequence $SEQ \
    --log experiments/<exp_id>/logs/val/$SEQ.jsonl \
    --output experiments/<exp_id>/metrics/val/$SEQ.json
# metrics JSON: precision + recall (song song), TP/FP/FN, so box bi loai do vung
# bo qua, ID switch, FPS/latency thuc do (bo fps_warmup_frames frame dau).
# Tu choi: log thieu frame, log khong co `path`, sai sequence, conf khac config.

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

### Do do kich ban occlusion (`python -m src.evaluate occlusion|aggregate|compare`)

Dinh nghia + nguong co dinh trong `configs/occlusion_metrics.yaml` (commit
truoc moi lan chay that). Moi kich ban co GT bbox o MOI frame (manifest), khoang
che la mot doan lien tuc [a, b]:

- **Output** o frame k: co Motion Memory -> `predicted_bbox` (TRACKING = bbox
  tracker, LOST = bbox ngoai suy, REIDENTIFYING/FAILED = khong co); baseline ->
  `tracker_bbox` cua track ID muc tieu. **Output xac nhan** = den tu detection
  (Motion Memory: state TRACKING; baseline: luon).
- **Du doan dung**: co output va IoU(output, GT) >= `correct_iou` (0.5).
- **% frame du doan dung** = so frame dung trong [a, b] / (b - a + 1).
- **Tracking gap dai nhat**: doan dai nhat cac frame lien tiep khong du doan
  dung, tu frame dung dau tien den het (chua tung dung = ca kich ban).
- **Re-ID thanh cong**: sau b co `reid_min_consecutive_frames` (3) frame lien
  tiep du doan dung VA la output xac nhan - ngoai suy trung vi tri KHONG tinh.
  **Time-to-reidentify** = frame dau cua doan do - b (>= 1).
- **Sai so vi tri trong luc che**: trung binh khoang cach tam tren cac frame
  trong [a, b] co output (kem so frame co output).
- Ghi them: bat lai bang track ID moi hay cu, `recover_reason`, muc tieu co duoc
  bat truoc khi che khong (kich ban khong bat duoc van tinh vao tong).
- **Gop nhieu kich ban**: cong gop tu so/mau so (khong trung binh cac ti le),
  tach theo thoi luong che va kieu che; tu choi gop khac nhom (dev/heldout),
  khac tham so chay, hoac thieu kich ban so voi `index.json` (tru
  `--allow-incomplete`, danh dau incomplete).
- **So sanh ablation** (`compare`): tu choi neu bat ky cap kich ban nao khac
  weights/source/tracker/conf/imgsz/classes hoac khac log track da replay
  (`tracks_sha256`) (`ablation_mismatches`).
- **Kiem tra dau vao**: `occlusion` tu choi log co so frame khac manifest, hoac
  `path` theo thu tu frame_idx khong khop ten anh da sap xep cua kich ban (lech
  thu tu, thieu, thua anh).

```bash
python -m src.evaluate occlusion --scenario-dir data/occlusion_test_scenarios/dev/<ten> \
    --run-dir experiments/<exp_id>/scenarios/<ten>
python -m src.evaluate aggregate --exp-dir experiments/<exp_id> \
    --index data/occlusion_test_scenarios/dev/index.json
python -m src.evaluate compare --baseline experiments/<base_id> \
    --memory experiments/<mm_id> --output experiments/<mm_id>/ablation_vs_<base_id>.json
```

Han che: vat che nhan tao khong giong che khuat that - xem `data/README.md`.

### Quy tac ground truth VisDrone2019-MOT (`src/evaluate.py`)

1. GT hop le: `score=1` va category 1..10 (map sang class 0..9 theo
   `configs/dataset_visdrone.yaml`). Con lai la **vung bi bo qua**: `score=0`,
   category 0 (ignored region), category ngoai map (vd 11 "others").
2. Moi frame, truoc khi ghep cap: prediction co IoA (dien tich giao / dien tich
   prediction) >= `ignore_ioa` (mac dinh 0.5) voi mot vung bi bo qua bi **loai**
   (khong TP, khong FP). GT hop le nam trong vung bi bo qua (IoA theo dien tich
   GT >= `ignore_ioa`) cung bi loai (khong FN).
3. Ghep cap tham lam theo IoU giam dan, IoU >= `iou_threshold` (0.5), cung
   class neu `class_aware`.
4. ID switch: cung mot `target_id` GT duoc ghep voi track ID khac lan ghep truoc
   (tinh ca khi co khoang mat o giua).
