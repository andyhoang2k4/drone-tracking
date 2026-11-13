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

Commit: `config.yaml`, `metrics.json` (gop; khoa `provenance` giu commit git,
so lan chay co working tree dirty, version package cua cac lan chay) va
`<..._mm_...>/ablation_vs_<baseline>.json`. Gitignore: toan bo `scenarios/`
(`run_context.json`, `target_selection.json`, `metrics.json`, `logs/` cua tung
kich ban) - nho backup ca thu muc truoc khi tat cloud neu can xem lai tung
kich ban.

Nhom `dev` dung de tinh chinh tham so Motion Memory; nhom `heldout` chi chay
MOT lan sau khi da chot tham so (doi tham so sau khi xem heldout -> tag moi va
ghi ro trong README cua experiment).
