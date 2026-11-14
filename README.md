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
| `configs/occlusion_scenarios.yaml` | Kich ban occlusion **v1** (lan chay thu "timing", giu de truy vet): chon muc tieu chi theo GT |
| `configs/occlusion_scenarios_v2.yaml` | Kich ban occlusion **v2** (mac dinh): 12 ung vien/sequence, screening khoa muc tieu (`target_lock`), giu toi da 4 muc tieu dat/sequence; output `data/occlusion_test_scenarios/v2/` |
| `configs/occlusion_metrics.yaml` | Dinh nghia + nguong do do occlusion **v1** (kich ban chua khoa van tinh vao tong) |
| `configs/occlusion_metrics_v2.yaml` | Do do occlusion **v2** (mac dinh): chi tinh ti le tren kich ban da khoa truoc khi che, dem rieng kich ban chua khoa |
| `configs/motion_memory_tuning_dev.yaml` | Quy tac tune Motion Memory tren dev **dang ky truoc** (luoi, diem, hop le, bien 0.01, toi da 2 vong) |
| `configs/motion_memory_tuned.yaml` | (chi co khi tuning chon duoc cau hinh) tham so Motion Memory da tune - dung cho heldout |
| `configs/motion_memory_anchor_dev.yaml` | Khai bao truoc danh gia anchor tren dev (bien the, hoi quy, quy tac quyet dinh, cam ket sau ket qua) |
| `src/occlusion_scenarios.py` | Sinh kich ban occlusion co GT moi frame (v2: loc theo screening.json) |
| `src/motion_memory_tuning.py` | Ham thuan chon cau hinh Motion Memory theo quy tac tuning (sinh luoi, diem, hop le, chon) |
| `src/occlusion_screening.py` | Screening khoa muc tieu tren clip sach truoc khi che (luat chon muc tieu that cua pipeline) |
| `src/detect.py` | Chay detection, log JSONL |
| `src/track.py` | Chay detection + BoT-SORT, log JSONL |
| `src/motion_memory.py` | Motion memory / temporal propagation |
| `src/evaluate.py` | Metric danh gia (detection, tracking, occlusion) |
| `src/utils/` | Logger JSONL, thu thap metadata tai hien experiment, resolve tham so inference/Motion Memory |
| `scripts/train_yolo11.sh`, `train_yolo26.sh` | Train - **chi chay tren Ubuntu/cloud** |
| `scripts/run_pipeline.py` | Pipeline day du, co co `--motion-memory` cho ablation |
| `scripts/generate_occlusion_scenarios.py` | Sinh kich ban occlusion cho mot nhom (dev/heldout) |
| `scripts/run_occlusion_ablation.py` | Chay ablation co/khong Motion Memory tren mot nhom kich ban, roi danh gia/gop/so sanh |
| `scripts/tune_motion_memory.py` | Tune Motion Memory tren dev: replay song song log tracker cua ablation v2_001 (CPU), chon cau hinh theo quy tac; `--variant <file khai bao>` danh gia bien the co dinh (vd anchor) |
| `scripts/diagnose_mm_hijack.py` | Chan doan mo ta (dev): frame bi che co box cua MM theo LOST / TRACKING ID goc / TRACKING ID khac |
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
| `motion_memory.py` khop lai: gate chuyen dong trong LOST, dung ID cu trong REIDENTIFYING | Xong, unit test du lieu gia; da chay replay dev v2_001 |
| `motion_memory.py` anchor (`anchor_recovery`, mac dinh TAT): ghi nho ID goc khi motion_gate sang ID khac, quay ve khi ID goc xuat hien lai (moi state tru FAILED); quyet dinh B thanh "nhan lai ID hien tai hoac anchor, van khong dung gate" | Xong, unit test du lieu gia; **chua chay replay dev** |
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
# khi khop lai bang ID moi), recover_reason (same_id / motion_gate / reid_matched /
# anchor_id - anchor_id chi khi anchor_recovery: true), gate_candidates_passed.
# anchor_recovery (bool, mac dinh false; KHONG co trong configs/motion_memory.yaml
# de giu sha256 file mac dinh/tuned): bat bang file config rieng.
# Chon muc tieu theo ground truth thay vi --target-track-id:
#   --gt-target-id <id> --gt-annotation <annotation VisDrone .txt> [--gt-first-frame 1]
# (ghep GT voi track BoT-SORT bang IoU o frame dau muc tieu xuat hien; ket qua
# ghi experiments/<id>/target_selection.json).
# Replay: --replay-tracks <tracks.jsonl cua src.track> -> khong chay YOLO, doc
# track tu log (can tracks.meta.json canh log; weights/source/tracker phai khop,
# conf/imgsz/classes lay tu meta; SHA-256 log ghi vao run_context.json).
# Log pipeline co truong `path` (anh nguon) de kiem tra thu tu anh.

# Ablation tren kich ban occlusion v2 (roadmap buoc 4-6), chay tren cloud:
DET="--detector yolo11=<yolo11 best.pt> --detector yolo26=<yolo26 best.pt>"
# 1. Screening khoa muc tieu (GPU): YOLO + BoT-SORT tren 30 frame SACH truoc
#    khi che cua moi ung vien, luat chon muc tieu that + 3 frame cuoi dung vi
#    tri; dat <=> CA HAI detector khoa duoc. --dry-run / --limit N (partial).
python scripts/screen_occlusion_targets.py --group dev $DET --device 0
# 2. Sinh kich ban CHI cho muc tieu dat (tu choi screening partial / khac
#    config_sha256 / khac danh sach ung vien / khac weights sha256)
python scripts/generate_occlusion_scenarios.py --group dev $DET
# 3. Ablation (metrics v2 mac dinh; detector + weights phai khop screening)
python scripts/run_occlusion_ablation.py --group dev --tag v2_001 \
    --scenario-root data/occlusion_test_scenarios/v2 $DET --device 0
# Kich ban v1 (lan chay thu): --config configs/occlusion_scenarios.yaml khi sinh,
# --metrics-config configs/occlusion_metrics.yaml khi chay ablation.
# Moi detector x kich ban: src.track chay YOLO + BoT-SORT MOT lan
# (experiments/<detector>_botsort_tracks_occl_dev_001/), roi baseline va Motion
# Memory hau ky tren CUNG log do (run_pipeline --replay-tracks).
# --dry-run: in lenh + so kich ban + so lan goi du kien. --limit-scenarios N:
# chi N kich ban dau de do thoi gian (ban gop danh dau incomplete, khong phai
# benchmark; dung --tag rieng).
# -> experiments/<detector>_botsort[_mm]_occl_dev_001/{config.yaml,metrics.json}
#    + experiments/<detector>_botsort_mm_occl_dev_001/ablation_vs_<baseline>.json
# Commit: index.json (+ screening.json voi v2) cua nhom kich ban, config.yaml,
# metrics.json gop, ablation_vs_*.json. Gitignore: anh/manifest kich ban,
# data/occlusion_screening/ (clip + log track screening), experiments/**/scenarios/.
# Nhom heldout CHI chay sau khi da chot tham so Motion Memory tren dev.

# 4. Tune Motion Memory tren dev (CPU, chi replay log tracker cua v2_001 - khong
#    YOLO). Quy tac dang ky truoc: configs/motion_memory_tuning_dev.yaml.
python scripts/tune_motion_memory.py --round 1 --dry-run
python scripts/tune_motion_memory.py --round 1 --jobs 16
python scripts/tune_motion_memory.py --round 2 --jobs 16   # tu choi neu vong 1 chua du
python scripts/tune_motion_memory.py --select              # chi ghi configs/motion_memory_tuned.yaml khi chon duoc
# Tuning cu (tag tune) CHI DOC: code tu 2026-11-15 co them anchor_recovery nen chay lai
# --round/--select tren tag cu bi tu choi (mm_configs/tuned khac noi dung).
# 4b. Danh gia anchor tren dev (khai bao truoc: configs/motion_memory_anchor_dev.yaml):
python scripts/tune_motion_memory.py --variant configs/motion_memory_anchor_dev.yaml --dry-run
python scripts/tune_motion_memory.py --variant configs/motion_memory_anchor_dev.yaml --jobs 16
# -> experiments/motion_memory_anchor_dev/summary.json (hoi quy, quyet dinh, mo ta)
python scripts/diagnose_mm_hijack.py --experiment <exp MM> [...] --output experiments/motion_memory_anchor_dev/diagnose_hijack.json
# -> experiments/<detector>_botsort_mm_occl_dev_tune_<id cau hinh>/ (moi cau hinh, ke ca cau hinh xau)
#    + experiments/motion_memory_tuning_dev/{config.yaml,summary.json,mm_configs/}
# Ket luan co the la "giu cau hinh mac dinh" - ghi ro trong summary.json.
# 5. Heldout (MOT lan, sau khi chot): run_occlusion_ablation.py --group heldout ...
#    --motion-memory-config configs/motion_memory_tuned.yaml (neu tuning chon duoc;
#    neu khong, de mac dinh configs/motion_memory.yaml).

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

Dinh nghia + nguong co dinh trong `configs/occlusion_metrics.yaml` (v1) va
`configs/occlusion_metrics_v2.yaml` (v2, mac dinh - xem muc Metrics v2 duoi day). Moi kich ban co GT bbox o MOI frame (manifest), khoang
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
  bat truoc khi che khong (v1: kich ban khong bat duoc van tinh vao tong;
  v2: loai khoi ti le, dem rieng).
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

#### Metrics v2 (`configs/occlusion_metrics_v2.yaml`, mac dinh tu 2026-11-14)

Lan chay thu cho thay quy tac v1 chon ca muc tieu ma detector khong thay; cac
kich ban "chua khoa" do bi tinh la Re-ID that bai, % dung = 0 va gap = ca kich
ban, lam loang ti le cua CA baseline lan Motion Memory. v2:

- **Khoa duoc truoc khi che** (`src/utils/gt_target.py` `evaluate_target_lock`,
  tham so trong `target_lock` cua `configs/occlusion_scenarios_v2.yaml`):
  `GtTargetSelector` chon duoc track trong `max_search_frames` (10) frame dau
  (IoU >= 0.5, KHONG xet class) VA track da chon co IoU >= 0.5 voi GT o 3 frame
  lien truoc khi che. Tinh tren log track replay dung chung -> baseline va
  Motion Memory co cung tap kich ban da khoa. Dung CUNG dinh nghia cho
  screening, run_pipeline (tham so lay tu manifest) va danh gia.
- Moi ti le / tong / gap / time-to-reidentify chi tren kich ban da khoa. Bao
  cao rieng `scenarios_total`, `scenarios_not_locked`, `not_locked_scenario_ids`
  (ca theo thoi luong va kieu che), them `independent_targets`,
  `independent_sequences`, `per_sequence` (so kich ban / muc tieu moi
  sequence - de sau nay lam khoang tin cay theo sequence; 16 kich ban cua mot
  muc tieu tuong quan manh).
- `compare` tu choi neu hai cau hinh co tap kich ban da khoa khac nhau;
  `aggregate`/`compare` tu choi gop v1 voi v2. `metrics_config` ghi
  `config_sha256`; `provenance` ghi sha256 config kich ban va metrics.
- `evaluate occlusion` khong truyen `--metrics-config` -> chon theo manifest
  (v2 neu co `target_lock`); runner luon truyen ro rang.

**Thien lech chon mau (ghi trong moi bao cao dung v2):** ket qua chi co dieu
kien tren muc tieu ma CA HAI detector khoa duoc truoc khi che (nghieng ve vat
lon, tuong phan cao, lop pho bien); khong suy rong cho muc tieu kho. So sanh
YOLO11 vs YOLO26 tren occlusion chi tren tap giao do. Ti le dat va phan bo
lop/kich thuoc ung vien vs dat nam trong `screening.json`.

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
