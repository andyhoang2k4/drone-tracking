# heldout_report_v2_001

Bao cao heldout v2_001 cua Motion Memory (`scripts/report_heldout.py`) theo ke hoach khai bao
truoc `configs/heldout_report_plan.yaml` (version 1). Heldout chay DUNG MOT LAN; khong tune lai,
khong chon lai cot sau khi xem ket qua (CLAUDE.md muc 13).

## Muc tieu

Kiem tra tren tap heldout (sequence khong dung khi tinh chinh) xem bien the `anchor` cua Motion
Memory (cot `mm_anchor`) co dat quy tac quyet dinh da khai bao truoc trong
`configs/motion_memory_anchor_dev.yaml` tren CA HAI detector hay khong, so voi cung detector +
BoT-SORT khong co Motion Memory (`no_mm`) va Motion Memory mac dinh (`mm_default`).

## Cau hinh

| | |
|---|---|
| Ke hoach | `configs/heldout_report_plan.yaml`, sha256 `ede00d887d952b382e6be6b9869fc66dac2ca495795da02b4ea78254f0c0a3c8` |
| Quy tac quyet dinh | `configs/motion_memory_anchor_dev.yaml`, sha256 `8baf5b9d19eb31dae9cbdbc62fc7904b1c39d2021b46c2cb2e56e731c0a6291c` |
| Git | commit `aa1299f794d6ce4a62b827b61b8b5ddbac874434` (nhanh feat/heldout-report-plan), cay SACH theo `CLEAN_TREE_DEFINITION` |
| Weights | yolo11 `82f0be26b04e3e4fdf09dea0b038a587aa01a262c380e6abab58afc751339fed`, yolo26 `6b7223db8f0e8f16a18bb53d0c82ff48bd33902eb2100e9337142b2f63f11c3a` (trung screening dev + heldout) |
| Kich ban | `data/occlusion_test_scenarios/v2/heldout` (`configs/occlusion_scenarios_v2.yaml`), metrics v2 (`configs/occlusion_metrics_v2.yaml`) |
| Moi truong | RTX 5060 Ti, CUDA 13.0, Python 3.12.3, ultralytics 8.4.174, torch 2.14.1+cu130, opencv 4.10.0 |

5 cot tren CUNG log tracker (replay):

| Cot | Vai tro | Cau hinh MM | Experiment (moi detector `<det>`) |
|---|---|---|---|
| no_mm | reference | - | `<det>_botsort_occl_heldout_v2_001` |
| mm_default | reference | `configs/motion_memory.yaml` | `<det>_botsort_mm_occl_heldout_v2_001` |
| mm_tuned | descriptive | `configs/motion_memory_tuned.yaml` | `<det>_botsort_mm_tuned_occl_heldout_v2_001` |
| mm_anchor | **decision** | `experiments/motion_memory_anchor_dev/mm_configs/anchor.yaml` | `<det>_botsort_mm_anchor_occl_heldout_v2_001` |
| mm_anchor_lost25 | descriptive | `experiments/motion_memory_anchor_dev/mm_configs/anchor_lost25.yaml` | `<det>_botsort_mm_anchor_lost25_occl_heldout_v2_001` |

Log tracker dung chung: `<det>_botsort_tracks_occl_heldout_v2_001`.

## Lenh da chay

```bash
DET="--detector yolo11=weights/eval/yolo11_best.pt --detector yolo26=weights/eval/yolo26_best.pt"
ARGS="--group heldout --tag v2_001 --scenario-root data/occlusion_test_scenarios/v2 $DET --device 0 --report-plan configs/heldout_report_plan.yaml --motion-memory-config configs/motion_memory.yaml --extra-motion-memory-config tuned=configs/motion_memory_tuned.yaml --extra-motion-memory-config anchor=experiments/motion_memory_anchor_dev/mm_configs/anchor.yaml --extra-motion-memory-config anchor_lost25=experiments/motion_memory_anchor_dev/mm_configs/anchor_lost25.yaml"
python scripts/run_occlusion_ablation.py $ARGS --dry-run      # 2026-11-15 02:03, guard OK
python scripts/run_occlusion_ablation.py $ARGS                # 2026-11-15 02:04 -> ~05:41, mot lan
python scripts/report_heldout.py --plan configs/heldout_report_plan.yaml   # 05:41, status complete
```

Lan chay: `Tracker: chay 1440, dung lai log 0`; `Hau ky: chay moi 7200, bo qua (da co): 0,
danh gia lai: 0, loi: 0`; khong dung `--resume-incomplete`. Thoi gian: src.track 1440 lan tong
6048.0 s; run_pipeline 7200 lan tong 6515.4 s; evaluate occlusion 7200 lan tong 431.3 s.

## Ket qua

Muc kiem chung: chay that tren GPU (tracker) + replay CPU (Motion Memory), danh gia tren ground
truth cua kich ban. Moi so duoi day lay tu `report.json` (lam tron 4 chu so); khong tinh lai.

So kich ban: moi cot x detector 720 kich ban (180 moi do dai che 5/15/30/60 frame), 0 chua khoa;
45 muc tieu, `independent_sequences` = 15. `report.json`: `status: complete`, `problems: []`.

### Quy tac quyet dinh (cot mm_anchor)

| Detector | Dieu kien | mm_anchor | Tham chieu | Ket qua |
|---|---|---|---|---|
| yolo11 | `reid(d15) >= no_mm` | 37/45 (0.8222) | 29/36 (0.8056) | dat |
| yolo11 | `reid(d30) >= no_mm` | 109/180 (0.6056) | 101/180 (0.5611) | dat |
| yolo11 | `reid(d5) >= mm_default` | 167/180 (0.9278) | 37/45 (0.8222) | dat |
| yolo11 | `cpr(overall) >= mm_default - 0.01` | 5197/19800 (0.2625) | 43/165 (0.2606) | dat |
| yolo26 | `reid(d15) >= no_mm` | 29/36 (0.8056) | 143/180 (0.7944) | dat |
| yolo26 | `reid(d30) >= no_mm` | 37/60 (0.6167) | 107/180 (0.5944) | dat |
| yolo26 | `reid(d5) >= mm_default` | 89/90 (0.9889) | 17/20 (0.8500) | dat |
| yolo26 | `cpr(overall) >= mm_default - 0.01` | 5201/19800 (0.2627) | 1019/3960 (0.2573) | dat |

**Ket luan cua tool: `COT QUYET DINH mm_anchor: DAT`** (yolo11: DAT, yolo26: DAT).

### Tong quan theo cot (overall)

| Cot | Vai tro | Detector | correct_prediction_ratio | box_precision | reidentification_rate | reid d60 | reidentified_with_new_track_id | mean_position_error_px |
|---|---|---|---|---|---|---|---|---|
| no_mm | reference | yolo11 | 0.0393 | 0.5317 | 0.5833 | 0.0722 | 0 | 14.3068 |
| no_mm | reference | yolo26 | 0.0366 | 0.4292 | 0.6042 | 0.0500 | 0 | 13.4333 |
| mm_default | reference | yolo11 | 0.2606 | 0.3503 | 0.4264 | 0.0722 | 27 | 48.4770 |
| mm_default | reference | yolo26 | 0.2573 | 0.3397 | 0.4181 | 0.0444 | 17 | 45.4278 |
| mm_tuned | descriptive | yolo11 | 0.2796 | 0.3090 | 0.4292 | 0.0778 | 43 | 51.9331 |
| mm_tuned | descriptive | yolo26 | 0.2782 | 0.3040 | 0.4125 | 0.0500 | 33 | 48.8522 |
| mm_anchor | decision | yolo11 | 0.2625 | 0.3537 | 0.6097 | 0.0833 | 21 | 48.2474 |
| mm_anchor | decision | yolo26 | 0.2627 | 0.3463 | 0.6194 | 0.0667 | 11 | 42.5158 |
| mm_anchor_lost25 | descriptive | yolo11 | 0.2824 | 0.3121 | 0.6319 | 0.0944 | 37 | 51.8219 |
| mm_anchor_lost25 | descriptive | yolo26 | 0.2835 | 0.3095 | 0.6417 | 0.0722 | 27 | 46.3488 |

### Chieu xau

- `box_precision` cua moi cot Motion Memory thap hon `no_mm` tren ca hai detector (bang tren).
- `reidentification_rate` cua `mm_default` va `mm_tuned` thap hon `no_mm`.
- Box cua track khac (`tracking_other_id`, tu `diagnosis` trong `report.json`, overall):

| Cot | Detector | frames_with_box | correct | precision |
|---|---|---|---|---|
| mm_default | yolo11 | 6138 | 54 | 0.0088 |
| mm_default | yolo26 | 6997 | 108 | 0.0154 |
| mm_tuned | yolo11 | 7884 | 54 | 0.0068 |
| mm_tuned | yolo26 | 8789 | 108 | 0.0123 |
| mm_anchor | yolo11 | 5918 | 54 | 0.0091 |
| mm_anchor | yolo26 | 6619 | 69 | 0.0104 |
| mm_anchor_lost25 | yolo11 | 7648 | 55 | 0.0072 |
| mm_anchor_lost25 | yolo26 | 8407 | 69 | 0.0082 |

Bootstrap theo sequence (2000 lan, seed 20261115) cho `correct_prediction_ratio`,
`box_precision`, `reidentification_rate`: xem `bootstrap` trong `report.json`.

## Nhan xet

- Motion Memory la "motion-only + ID anchor", KHONG co appearance Re-ID; re-ID o d60 bi gioi han
  boi `track_buffer=30` cua BoT-SORT va thiet ke motion-only.
- Selection bias: chi gom muc tieu ma CA HAI detector khoa duoc truoc khi che; khong suy rong cho
  muc tieu kho.
- Cot mo ta (`mm_tuned`, `mm_anchor_lost25`), `box_precision`, phan loai frame, bootstrap KHONG
  dung de chon cau hinh sau khi thay ket qua.
- `mean_position_error_px` do tren tap frame CO BOX cua tung cot (khong cung tap);
  `reidentified_with_new_track_id` cua `no_mm` = 0 do dinh nghia.
- Bootstrap la MO TA do bat dinh, khong phai kiem dinh y nghia; so sequence it (15).
- Screening/sinh kich ban heldout chay luc nhanh tren may cloud bi doi giua chung (01:30, sang
  feat/sim-gazebo-probe do phien khac). Diff cac file screening/sinh kich ban/track/cau hinh v2
  giua hai nhanh rong (da kiem lai). Lan chay heldout (02:04-05:41) tren commit aa1299f, cay sach.
- `git.dirty: true` trong `report.json` la truong cu tinh CA file untracked; khong mau thuan voi
  cay sach (xem `experiments/README.md`).

## Luu tru

| Noi dung | Vi tri |
|---|---|
| config.yaml, metrics.json gop, ablation_vs_*.json, report.json, index/screening heldout | git (`main`, commit f0011ae) |
| scenarios/ (log tracker + pipeline, metrics tung kich ban) + annotations/index/screening | may cloud (ban goc, gitignore) + ban sao ngoai may: `heldout_logs_gt_2026-11-15.tgz`, sha256 `2a4e73b4c4cadd49529aaf1d41d0554f9a0c54c9b790d71f870dc12aedad5a28` |
| Anh kich ban (jpg) | chi tren may cloud; sinh lai duoc tu VisDrone theo seed/config trong index (chua kiem tra trung byte) |
