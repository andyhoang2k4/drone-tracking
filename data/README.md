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

## Giai doan A: VisDrone2019-MOT (3 class)

Class: `person` (pedestrian + people), `car` (car + van + truck),
`motorcycle` (motor). Anh xa va split: `configs/dataset_visdrone.yaml`.
Giai doan B (them tau thuyen) se la dataset version rieng.

1. Tai VisDrone2019-MOT (train, val, test-dev) tu trang chinh thuc cua
   VisDrone, giai nen vao `data/raw/VisDrone2019-MOT/` sao cho co
   `VisDrone2019-MOT-train/sequences/` va `.../annotations/`.
   Kiem tra license tren trang VisDrone truoc khi dung.
2. Chuyen sang dinh dang Ultralytics (bbox):

   ```bash
   python scripts/prepare_visdrone.py --max-sequences 1   # smoke test
   python scripts/prepare_visdrone.py --overwrite         # day du
   ```

   Sinh `data/labeled/visdrone_mot_a/` (kem `manifest.json` thong ke) va
   `data/dataset.yaml`. Anh duoc hardlink, khong ton them dung luong.
3. Sinh mask cho YOLO-seg bang SAM (`yolo_bbox2segment`) - buoc sau, chay
   tren cloud.
