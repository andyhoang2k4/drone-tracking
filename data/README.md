# data/

Thu muc dataset. **Khong commit dataset that** (xem `.gitignore`).

| Thu muc | Noi dung |
|---|---|
| `raw/` | Video/anh goc chua gan nhan |
| `labeled/` | Dataset da gan nhan, dung dinh dang Ultralytics segmentation |
| `occlusion_test_scenarios/` | Kich ban occlusion nhan tao + ground truth (v1 o `<nhom>/`, v2 o `v2/<nhom>/`) |
| `occlusion_screening/` | Clip sach + log track cua screening khoa muc tieu v2 (gitignore, sao luu) |

## Split theo video/sequence

Chia train/val/test theo **video**, khong chia ngau nhien theo frame
(CLAUDE.md muc 7). Mot video chi thuoc dung mot split.

## Ground truth cho kich ban occlusion

Moi kich ban trong `occlusion_test_scenarios/` **bat buoc** co ground-truth
bounding box cho **toan bo sequence**, ke ca cac frame bi che khuat hoac
nhieu (CLAUDE.md muc 12). Khong co ground truth o cac frame do thi khong
tinh duoc "% frame du doan dung" va "sai so vi tri trong luc occlusion".

### Sinh kich ban occlusion nhan tao (roadmap buoc 4)

Can VisDrone2019-MOT val + test-dev da giai nen o `data/raw/VisDrone2019-MOT/`.

**v2 (mac dinh, `configs/occlusion_scenarios_v2.yaml`)** - muc tieu phai
"khoa duoc truoc khi che" voi CA HAI detector:

1. Moi sequence lay 12 ung vien (thu tu tron co seed) trong cac muc tieu thoa
   quy tac GT (giong v1).
2. `scripts/screen_occlusion_targets.py` (GPU): voi moi ung vien, clip 30
   frame SACH truoc khi che (hardlink, giong het frame 0..29 cua moi kich ban
   cua muc tieu do) -> `src.track` moi detector -> luat chon muc tieu that
   (`GtTargetSelector`, khong xet class) + track da chon dung vi tri o 3 frame
   cuoi. Ket qua: `occlusion_test_scenarios/v2/<nhom>/screening.json`.
3. `scripts/generate_occlusion_scenarios.py`: moi sequence giu toi da 4 ung
   vien DAT dau tien theo thu tu co seed, chi sinh kich ban cho chung.

```bash
DET="--detector yolo11=<yolo11 best.pt> --detector yolo26=<yolo26 best.pt>"
python scripts/screen_occlusion_targets.py --group dev $DET --device 0 --dry-run
python scripts/screen_occlusion_targets.py --group dev $DET --device 0
python scripts/generate_occlusion_scenarios.py --group dev $DET --dry-run
python scripts/generate_occlusion_scenarios.py --group dev $DET
```

Cung quy tac cho heldout: screening chi chay detector + tracker tren clip sach
(khong Motion Memory, khong kich ban che), khong tinh chinh gi tren heldout.
Du lieu train KHONG dung de chon muc tieu.

**v1 (`configs/occlusion_scenarios.yaml`, lan chay thu "timing")** - chon muc
tieu chi theo GT, giu de truy vet:

```bash
python scripts/generate_occlusion_scenarios.py --config configs/occlusion_scenarios.yaml --group dev
```

Moi kich ban `occlusion_test_scenarios/<nhom>/<sequence>__t<id>__d<thoi luong>__<kieu che>/`:
`images/` (frame bi che ghi lai, frame khac hardlink tu anh goc),
`annotations.txt` (annotation VisDrone cua cua so, danh so lai tu 1),
`manifest.json` (GT bbox cua muc tieu o MOI frame, khoang frame bi che, kieu
che, seed, sequence nguon). Cung config + seed -> cung kich ban.

Kich ban v2 co them trong `manifest.json`: `target_lock` (tham so khoa +
`occlusion_start_frame_idx`) va `screening` (sha256 screening.json, weights
sha256 tung detector, thu hang ung vien).

Commit gi: **chi `occlusion_test_scenarios/<nhom>/index.json`** (v2: them
`occlusion_test_scenarios/v2/<nhom>/screening.json`) (danh sach kich
ban, sequence, seed, `config_sha256`; v2: so ung vien / so dat / muc tieu chon
moi sequence). Anh, `manifest.json`, `annotations.txt`
cua tung kich ban bi gitignore - sinh lai duoc tu config da commit + dataset
VisDrone (sinh xac dinh). `config_sha256` tinh tren noi dung da bo ky tu CR nen
khong phu thuoc line ending (Windows/Linux). index.json khong ghi danh sach
kich ban "bo qua vi da co" nen chay lai khong tao diff.

- Nhom **dev** (VisDrone val): duoc dung de tinh chinh tham so Motion Memory.
- Nhom **heldout** (VisDrone test-dev): KHONG dung de tinh chinh; chi chay sau
  khi da chot tham so tren dev (CLAUDE.md muc 13).

**Han che:** vat che nhan tao (khoi mau dac hoac nhieu, bam theo bbox GT cua
muc tieu tung frame) KHONG giong che khuat that nhu cay hay cong trinh: khong
co bien canh tu nhien, khong dung yen trong the gioi, va luon che dung muc
tieu. Ket qua tren kich ban nay khong thay the danh gia tren che khuat that.

## Dataset yaml

Copy `dataset.example.yaml` thanh `dataset.yaml` (bi gitignore) va sua duong
dan cho khop may dang chay. `configs/train_config.yaml` tro toi
`data/dataset.yaml`.

## Giai doan A: VisDrone2019-MOT (10 class, dataset version 2)

Class: giu nguyen 10 class cua VisDrone (pedestrian, people, bicycle, car,
van, truck, tricycle, awning-tricycle, bus, motor); bo category 0 (ignored
region) va 11 (others). Vung ignored chua duoc xu ly o version nay.
Anh xa va split: `configs/dataset_visdrone.yaml`.
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

   Sinh `data/labeled/visdrone_mot_a_v2/` (kem `manifest.json` thong ke) va
   `data/dataset.yaml`. Anh duoc hardlink, khong ton them dung luong.
   Ban smoke test (`--max-sequences`, hay `generate_masks.py --max-images`)
   duoc danh dau `partial` trong manifest; `scripts/train_*.sh` tu dung neu
   gap dataset partial, tru khi truyen `--allow-partial`.
3. Sinh mask cho YOLO-seg bang SAM - chay tren cloud (CPU ~10 s/anh):

   ```bash
   python scripts/generate_masks.py --splits val --max-images 5 --device cpu  # smoke
   python scripts/generate_masks.py --splits train val --device 0                # day du
   ```

   SAM ma hoa moi anh mot lan roi giai ma theo nhom `masks.sam_box_batch` box;
   `masks.sam_conf: 0.0` de Ultralytics khong bo mask diem thap. Box thieu mask
   duoc goi lai rieng, OOM thi tu chia nho nhom; van that bai thi dung hinh chu
   nhat. So lan xay ra ghi o `manifest.json` -> `masks.splits.<split>.sam_recovery`
   va `flagged_images`.

   Baseline (dataset version 2) chi sinh mask cho **train va val**. Test giu
   nhan bbox goc trong `labels/test` (khong co `labels_bbox/test`), nen tren
   test chi danh gia box, khong danh gia mask.

   Box goc duoc chuyen sang `labels_bbox/`, polygon ghi vao `labels/`. Mask
   lech (IoU bbox-cua-mask vs box goc < `masks.fallback_iou`) thay bang hinh
   chu nhat cua box goc, vi YOLO-seg tinh box train tu mask.
   **Danh gia detection/occlusion luon dung box goc trong `labels_bbox/`.**
