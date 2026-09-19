# experiments/

Moi lan train/benchmark co mot thu muc rieng, ten la experiment ID ro rang
(vi du `yolo11_baseline_001`).

## Commit gi, bo qua gi

| Duong dan | Git |
|---|---|
| `config.yaml` | commit |
| `metrics.json` | commit |
| `README.md` | commit |
| `run_context.json` | commit |
| `weights/` | gitignore |
| `videos/` | gitignore |
| `logs/` | gitignore |

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

So sanh tren CUNG detector + tracker (CLAUDE.md muc 12):

| Cau hinh | Experiment ID |
|---|---|
| YOLO11 + BoT-SORT | |
| YOLO11 + BoT-SORT + Motion Memory | |
| YOLO26 + BoT-SORT | |
| YOLO26 + BoT-SORT + Motion Memory | |
