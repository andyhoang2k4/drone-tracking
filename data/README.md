# data/

Thu muc dataset. **Khong commit dataset that** (xem `.gitignore`).

| Thu muc | Noi dung |
|---|---|
| `raw/` | Video/anh goc chua gan nhan |
| `labeled/` | Dataset da gan nhan, dung dinh dang Ultralytics segmentation |
| `occlusion_test_scenarios/` | Kich ban occlusion nhan tao + ground truth |

## Split theo video/sequence

Chia train/val/test theo **video**, khong chia ngau nhien theo frame
(CLAUDE.md muc 7). Mot video chi thuoc dung mot split.

## Ground truth cho kich ban occlusion

Moi kich ban trong `occlusion_test_scenarios/` **bat buoc** co ground-truth
bounding box cho **toan bo sequence**, ke ca cac frame bi che khuat hoac
nhieu (CLAUDE.md muc 12). Khong co ground truth o cac frame do thi khong
tinh duoc "% frame du doan dung" va "sai so vi tri trong luc occlusion".

## Dataset yaml

Copy `dataset.example.yaml` thanh `dataset.yaml` (bi gitignore) va sua duong
dan cho khop may dang chay. `configs/train_config.yaml` tro toi
`data/dataset.yaml`.
