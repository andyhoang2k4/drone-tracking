# experiments/

Moi lan train/benchmark co mot thu muc rieng, ten la experiment ID ro rang
(vi du `yolo11_baseline_001`).

## Commit gi, bo qua gi

| Duong dan | Git |
|---|---|
| `config.yaml` | commit |
| `metrics.json` | commit |
| `README.md` | commit |
| `run_context.json` | commit (tru trong `scenarios/`, xem duoi) |
| `weights/` | gitignore |
| `videos/` | gitignore |
| `logs/` | gitignore |
| `*.png`, `*.jpg`, `results.csv` (output Ultralytics, vd `train/`) | gitignore - so lieu chinh chep sang `metrics.json` |
| `scenarios/` (ablation occlusion: ket qua + log tung kich ban) | gitignore - hang nghin file |
| `ablation_vs_*.json` | commit |

## Quy tac (CLAUDE.md muc 13, 14)

- **Khong xoa lich su cac lan chay that bai.** Ghi lai ke ca khi ket qua xau,
  de sau nay biet da thu gi.
- Khong doi dataset/test split sau khi da xem ket qua ma khong tao experiment
  moi.
- Khong cherry-pick ket qua tot nhat tu nhieu lan chay roi bao cao nhu mot
  experiment duy nhat.
- Chi goi la "benchmark" khi da chay tren dataset/split co dinh tu truoc.

## Bat dau mot experiment moi

Copy `_template/` sang `<experiment_id>/` roi dien. `run_context.json` duoc
`scripts/run_pipeline.py` va `scripts/train_*.sh` tu sinh (git commit hash,
dataset, version ultralytics/torch, hardware).

## Ablation bat buoc khi danh gia Motion Memory

So sanh tren CUNG detector + tracker (CLAUDE.md muc 12). Tren kich ban
occlusion, `scripts/run_occlusion_ablation.py` dat ten theo quy uoc:

| Cau hinh | Experiment ID |
|---|---|
| YOLO11 + BoT-SORT | `yolo11_botsort_occl_<nhom>_<tag>` |
| YOLO11 + BoT-SORT + Motion Memory | `yolo11_botsort_mm_occl_<nhom>_<tag>` |
| YOLO26 + BoT-SORT | `yolo26_botsort_occl_<nhom>_<tag>` |
| YOLO26 + BoT-SORT + Motion Memory | `yolo26_botsort_mm_occl_<nhom>_<tag>` |
| Log YOLO + BoT-SORT dung chung (moi detector) | `<detector>_botsort_tracks_occl_<nhom>_<tag>` |

BoT-SORT chay MOT lan cho moi detector x kich ban (log track trong
`<detector>_botsort_tracks_...`); baseline va Motion Memory hau ky tren cung log
do (`run_pipeline --replay-tracks`), nen hai cau hinh thay chinh xac cung dau
vao tracker. `compare` tu choi neu `tracks_sha256` khac nhau.

Tag dung voi `--limit-scenarios` (do thoi gian) phai khac tag cua lan chay du;
ban gop cua no co `completeness.complete = false`.

Kich ban v2 (`--scenario-root data/occlusion_test_scenarios/v2`) danh gia bang
metrics v2 (mac dinh `--metrics-config configs/occlusion_metrics_v2.yaml`):
kich ban chua khoa truoc khi che duoc dem rieng (`scenarios_not_locked`), khong
lam loang ti le. Runner tu choi detector/weights khong khop `screening.json`.
Dung tag moi (vd `v2_001`), khong dung lai tag cua lan chay v1. Cac experiment
`*_timing` (v1, lan chay thu) giu nguyen, khong xoa, khong gop voi v2. Moi bao
cao dung v2 phai ghi thien lech chon mau (chi muc tieu ca hai detector khoa
duoc) - xem README goc, muc Metrics v2.

Commit: `config.yaml`, `metrics.json` (gop; khoa `provenance` giu commit git,
so lan chay co working tree dirty, version package cua cac lan chay) va
`<..._mm_...>/ablation_vs_<baseline>.json`. Gitignore: toan bo `scenarios/`
(`run_context.json`, `target_selection.json`, `metrics.json`, `logs/` cua tung
kich ban) - nho backup ca thu muc truoc khi tat cloud neu can xem lai tung
kich ban.

Nhom `dev` dung de tinh chinh tham so Motion Memory; nhom `heldout` chi chay
MOT lan sau khi da chot tham so (doi tham so sau khi xem heldout -> tag moi va
ghi ro trong README cua experiment).

## Tune Motion Memory tren dev (`scripts/tune_motion_memory.py`)

Quy tac dang ky truoc: `configs/motion_memory_tuning_dev.yaml` (version +
sha256 ghi vao moi config.yaml/summary.json). Chi replay log tracker cua
ablation nguon (`source_tag`, vd `v2_001`), khong chay YOLO.

| Duong dan | Git |
|---|---|
| `motion_memory_tuning_dev/config.yaml` | commit |
| `motion_memory_tuning_dev/summary.json` (moi cau hinh, metrics gop, per_sequence, vong 1/2, cau hinh chon, ket luan, sha256 quy tac + log tracker) | commit |
| `motion_memory_tuning_dev/mm_configs/<id>.yaml` (tham so day du tung cau hinh) | commit |
| `<det>_botsort_mm_occl_dev_tune_<id>/{config.yaml,metrics.json,ablation_vs_*.json}` | commit |
| `<det>_botsort_mm_occl_dev_tune_<id>/scenarios/` | gitignore - sao luu |
| `motion_memory_tuning_dev_tune_limit<N>_run<k>/`, `*_tune_limit<N>_run<k>_*` (`--limit-scenarios`, partial; tag tu danh so, k nho nhat chua dung) | thu nghiem, khong dung de chon; giu lai (khong xoa lich su) |
| `motion_memory_tuning_dev_tune_limit6/`, `*_tune_limit6_*` (tag dinh dang cu) | lan chay thu LOI tren cloud 2026-11-14 (moi replay bi tu choi vi source tuyet doi khac meta tuong doi) - giu nguyen, khong xoa; tag cu bi tu choi nen khong bi ghi tiep |

- Tham so replay (`--source`, `--weights`) lay nguyen chuoi tu `tracks.meta.json` cua log tracker
  nguon (run_pipeline va `evaluate compare` doi chieu voi chuoi do), khong tu dung tu `--scenario-root`.
- Truoc moi vong: pre-flight (kiem tinh moi lenh replay bang `apply_replay_meta` + chay that 1 replay
  moi detector vao thu muc tam); > 50% lenh cua mot dot loi cung thong bao -> dung som, in day du.
- Luu MOI cau hinh, ke ca cau hinh xau; khong co vong 3, khong them luoi sau
  khi co ket qua.
- Cau hinh `default` duoc chay lai cung code; summary ghi
  `default_reproduces_source_motion_memory` (so voi `<det>_botsort_mm_occl_dev_v2_001`).
- Ket luan "giu cau hinh mac dinh" la ket qua hop le, ghi ro trong summary.
- `configs/motion_memory_tuned.yaml` chi duoc ghi khi chon duoc cau hinh;
  `configs/motion_memory.yaml` khong bao gio bi ghi de.
- Heldout: chay `run_occlusion_ablation.py --group heldout --tag <tag moi>
  --motion-memory-config configs/motion_memory_tuned.yaml` MOT lan.
- Tuning cu CHI DOC: code tu 2026-11-15 them `anchor_recovery` vao MotionMemoryConfig nen
  chay lai `--round`/`--select` tren tag `tune` bi tu choi (noi dung mm_configs/tuned khac).

## Danh gia anchor tren dev (`tune_motion_memory.py --variant`)

Khai bao truoc: `configs/motion_memory_anchor_dev.yaml` (version 1, sha256 ghi vao moi ket qua).
Bien the: `default_regression` (hoi quy, phai trung `tune_default`), `anchor` (QUYET DINH),
`anchor_lost25` (chi mo ta). Chay DUNG MOT LAN, giu ket qua du xau.

| Duong dan | Git |
|---|---|
| `motion_memory_anchor_dev/{config.yaml,summary.json,mm_configs/}` | commit |
| `motion_memory_anchor_dev/diagnose_hijack.json` (scripts/diagnose_mm_hijack.py) | commit |
| `<det>_botsort_mm_occl_dev_anchor_v1_<bien the>/{config.yaml,metrics.json,ablation_vs_*.json}` | commit |
| `<det>_botsort_mm_occl_dev_anchor_v1_<bien the>/scenarios/` | gitignore - sao luu |
| `motion_memory_anchor_dev_anchor_v1_limit<N>_run<k>/`, `*_anchor_v1_limit<N>_run<k>_*` | thu nghiem (partial), khong quyet dinh; giu lai |

`summary.json`: `status` = complete | blocked_by_regression (default_regression lech
tune_default -> DUNG, khong dien giai anchor) | not_available. Cot mo ta va `anchor_lost25`
khong dung de chon cau hinh; chi vao heldout neu ke hoach heldout khai bao truoc.
