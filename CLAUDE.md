# CLAUDE.md — Drone AI Tracking Project

Hướng dẫn ngữ cảnh cho Claude Code khi làm việc trên dự án này. Đọc file này đầu tiên trước khi bắt đầu bất kỳ task nào, và đọc lại nếu có nghi ngờ về kiến trúc/quy tắc trước khi sửa code.

**Bảo vệ file này:** Không tự ý sửa CLAUDE.md để hợp thức hóa cho implementation hiện tại pass (ví dụ: gặp rule "không dùng dependency X" mà code cần X, thì báo lại cho người dùng — không âm thầm sửa rule thành "X được phép"). Chỉ cập nhật file này khi có quyết định kiến trúc/workflow/requirement mới thực sự, và phải giữ nguyên các nguyên tắc hiện có trừ khi người dùng yêu cầu đổi.

## 1. Mục tiêu dự án

Dự án drone **quadcopter** trinh sát/giám sát, tích hợp bộ xử lý AI để **tự động khóa mục tiêu và điều khiển drone bay bám theo mục tiêu** qua camera, dùng:
- **Image segmentation** để phát hiện mục tiêu
- **Motion memory + temporal propagation** để dự đoán vị trí mục tiêu giữa các lần detect, và **tự động bắt lại mục tiêu** khi mục tiêu biến mất rồi xuất hiện lại do **bị che khuất** (khuất sau cây, vật cản) hoặc **nhiễu**

Dự án gồm 2 phần lớn, phát triển song song nhưng độc lập:
1. **Flight controller firmware** (PX4 port cho board FK743M2-IIT6, STM32H743, drone quadcopter) — không nằm trong repo này trừ khi có chỉ định riêng.
2. **AI tracking pipeline** (Python) — trọng tâm của repo này. Mục tiêu cuối là điều khiển drone bay bám theo mục tiêu, nhưng ở giai đoạn hiện tại repo này chỉ dừng ở detect + track + motion memory (xem mục 2) — phần ghép output đó thành lệnh điều khiển bay thực tế là giai đoạn sau, chưa nằm trong scope bây giờ.

## 2. Giai đoạn hiện tại (ĐỌC KỸ — tránh over-engineering)

Đang ở **giai đoạn mô phỏng/train trên cloud GPU**, CHƯA deploy lên phần cứng thật.

**Việc CẦN làm ở giai đoạn này:**
- Fine-tune và đánh giá YOLO11-seg và YOLO26-seg
- Tích hợp BoT-SORT
- Viết + test module motion memory/temporal propagation
- Dựng kịch bản occlusion nhân tạo để test re-ID

**Việc KHÔNG cần làm ở giai đoạn này (để dành cho giai đoạn deploy sau):**
- Không tối ưu tốc độ cho edge (Jetson Orin Nano Super / Pi5+Hailo)
- Không export ONNX/TensorRT/HEF
- Không thêm correlation tracker (CSRT/KCF) để giảm tần suất gọi YOLO
- Không tích hợp với PX4/flight controller
- Không lo về pin, nhiệt độ, ràng buộc vật lý

Nếu một task có vẻ dẫn tới các việc ở nhóm "không cần" phía trên, dừng lại và hỏi lại người dùng trước khi làm.

## 3. Môi trường phát triển

| | Local Dev | Cloud PC |
|---|---|---|
| OS | Windows | Ubuntu 24.04 |
| Vai trò | Viết code, chỉnh sửa, commit git | Train, inference, benchmark |
| GPU | Không có / không dùng để train | RTX 5060 (CUDA) |
| Kết nối | Local | Remote Desktop |

**Quy tắc bắt buộc:**
- Không dùng path cứng kiểu Windows (`C:\Users\...`) hay Unix cứng — luôn dùng `pathlib.Path`, path tương đối, hoặc biến môi trường.
- Không giả định OS trong code — tránh gọi lệnh shell đặc thù qua `os.system`; dùng `platform.system()` nếu cần rẽ nhánh.
- Repo có `.gitattributes` chuẩn hoá LF cho file code.
- `.gitignore` đầy đủ: venv, `__pycache__`, cấu hình IDE cá nhân, dataset thật, checkpoint model.
- Code trên Windows chủ yếu để viết/chỉnh sửa, không cần tối ưu CUDA — nhưng nên chạy được ở mức "không lỗi cú pháp/import" (CPU-only) trước khi đẩy lên cloud.
- File `.sh` trong `scripts/` chỉ chạy trên Ubuntu/cloud — không cố chạy trên Windows; validation trên Windows dùng lệnh Python/PowerShell tương đương.

**Quy trình:** code trên Windows → `git push` → cloud PC `git pull` → chạy training/test có GPU.

## 4. Stack kỹ thuật đã quyết định

- **Detector:** Ultralytics YOLO — train song song **YOLO11-seg** và **YOLO26-seg**.
- **Tracker:** **BoT-SORT** (`model.track(tracker="botsort.yaml")`) — Kalman filter + camera motion compensation (CMC) bật mặc định. **Re-ID mặc định TẮT** (`with_reid: false` trong config gốc của Ultralytics) — phải bật rõ ràng (`with_reid: true`) trong config riêng của project nếu experiment cần Re-ID, không giả định nó luôn bật.
- **Framework:** Python + PyTorch + `ultralytics`.
- **Xử lý ảnh phụ trợ:** OpenCV — optical flow/debug visualization ở giai đoạn này; correlation tracker (CSRT/KCF) để dành cho giai đoạn deploy edge.
- **Phần cứng deploy tương lai (KHÔNG tối ưu bây giờ):** Jetson Orin Nano Super (ưu tiên) hoặc Raspberry Pi 5 + Hailo-8L AI HAT.

### Dependency & Version Policy

- Không giả định API mới nhất của `ultralytics`. Trước khi code phần liên quan model, kiểm tra version đã cài (`pip show ultralytics`) và xác nhận API tương ứng.
- Khóa version `ultralytics`, `torch`, `opencv-python` trong `requirements.txt` — không tự ý upgrade major version trừ khi được yêu cầu.
- Nếu version thực tế trong môi trường khác với version đã khóa trong `requirements.txt`, không tự sửa `requirements.txt` hay code để thích nghi — báo mismatch trước.
- Nếu tên model hoặc API không khớp mô tả ở đây, dừng lại và báo cho người dùng thay vì tự suy đoán.

## 5. Kiến trúc pipeline

```
[Video/Camera input]
        │
        ▼
[YOLO11-seg / YOLO26-seg detect]  ← chạy MỖI FRAME ở giai đoạn sim
        │
        ▼
[BoT-SORT tracker]  ← Kalman filter, Re-ID (nếu bật), CMC
        │
        ▼
[Motion memory / temporal propagation]  ← module TỰ VIẾT (xem mục 6)
        │
        ▼
[Output: track ID + bounding box + trạng thái]
```

Quyết định kiến trúc: **không tự viết lại tracker** — tận dụng BoT-SORT cho Kalman filter/Re-ID/CMC cơ bản; phần tự viết chỉ giới hạn ở motion memory. Motion Memory hoạt động theo kiểu **post-processing**: nhận state/output từ BoT-SORT làm input, không sửa mã nguồn nội bộ của BoT-SORT (không đụng vào code bên trong package `ultralytics`). Nếu sau này muốn Motion Memory phản hồi ngược để ảnh hưởng association bên trong BoT-SORT (kiến trúc feedback thay vì post-processing), đó là thay đổi kiến trúc cần bàn trước, không phải mặc định.

"Chạy mỗi frame" ở trên nghĩa là Motion Memory hiện chỉ được test trong các trường hợp YOLO **tự nó bỏ sót** detection (occlusion, nhiễu...), không phải trường hợp chủ động bỏ qua frame để tiết kiệm compute — việc chủ động skip frame để dành cho giai đoạn deploy edge (mục 2).

## 6. Motion Memory — interface bắt buộc

`motion_memory.py` phải có interface rõ ràng, độc lập với nội bộ Ultralytics/BoT-SORT.

**Input:** last confirmed bounding box · previous bounding boxes · timestamps/frame index · current tracker state

**Output:** predicted bounding box · confidence của dự đoán · số frame kể từ lần detect confirmed gần nhất · state (`TRACKING`/`LOST`/`REIDENTIFYING`/`FAILED`)

**Quy tắc quan trọng:** confidence của motion memory (độ tin cậy của *dự đoán*) là khái niệm khác với confidence của detector (độ tin cậy của *detection*). Không tự ý cộng/gộp hai giá trị này thành một con số chung (ví dụ `final_confidence = yolo_conf + motion_conf`) trừ khi có quy tắc gộp được định nghĩa rõ ràng.

**Transition cơ bản của state** (chưa cần implement ngay, nhưng logic phải theo bảng này):

| Từ state | Sự kiện | Sang state |
|---|---|---|
| TRACKING | mất detection | LOST |
| LOST | detect lại được, khớp track | TRACKING |
| LOST | quá ngưỡng thời gian (timeout) | REIDENTIFYING |
| REIDENTIFYING | Re-ID match thành công | TRACKING |
| REIDENTIFYING | hết ngưỡng, không match | FAILED |

Ý nghĩa: `LOST` là khoảng thời gian ngắn ngay sau khi mất detection, motion memory dự đoán vị trí dựa trên chuyển động gần nhất (còn trong "motion horizon"). Khi vượt ngưỡng thời gian đó (dự đoán motion không còn đáng tin), chuyển sang `REIDENTIFYING` — lúc này cần cơ chế khác (ví dụ appearance-based Re-ID) để bắt lại mục tiêu, không chỉ dựa vào ngoại suy chuyển động nữa.

**Phạm vi:** Motion Memory hoạt động trên bounding box, không phải segmentation mask. Dù detector là YOLO-seg (có mask sẵn), input/output của `motion_memory.py` chỉ là bounding box như mô tả trên. Khi được yêu cầu "viết temporal propagation", không tự mở rộng thành hệ thống propagate mask (ví dụ optical-flow mask propagation) — nếu sau này cần motion memory ở cấp độ mask, đó là thay đổi interface cần bàn trước, không phải mặc định.

## 7. Cấu trúc thư mục

```
drone-tracking/
├── CLAUDE.md
├── README.md
├── .gitignore
├── .gitattributes
├── requirements.txt
├── configs/
│   ├── botsort.yaml
│   └── train_config.yaml
├── data/                          (không commit dataset thật)
│   ├── raw/
│   ├── labeled/
│   └── occlusion_test_scenarios/
├── experiments/
│   └── yolo11_baseline_001/
│       ├── config.yaml            ← commit
│       ├── metrics.json           ← commit
│       ├── README.md              ← commit
│       ├── weights/               ← gitignore
│       ├── videos/                ← gitignore
│       └── logs/                  ← gitignore
├── src/
│   ├── detect.py
│   ├── track.py
│   ├── motion_memory.py
│   ├── evaluate.py
│   └── utils/
├── scripts/
│   ├── train_yolo11.sh
│   ├── train_yolo26.sh
│   └── run_pipeline.py
└── notebooks/
```

**Dataset split:** train/val/test phải chia theo **video/sequence**, không chia ngẫu nhiên theo frame — các frame liền kề trong cùng video rất giống nhau, chia ngẫu nhiên theo frame sẽ làm train và test lẫn cả frame gần như giống hệt nhau (data leakage).

## 8. Coding Rules

- Python 3.10+ (khớp Ubuntu 24.04 và yêu cầu Ultralytics).
- Tên biến/hàm tiếng Anh; comment/docstring tiếng Việt được, không bắt buộc.
- Mọi script train/test nhận tham số qua CLI args hoặc config file — không hardcode path, learning rate, tên model.
- Log kết quả ra file có cấu trúc, không chỉ console.

## 9. Claude Code Agent Rules

- Đọc file này trước khi sửa bất kỳ code nào.
- Kiểm tra cấu trúc repo hiện có trước khi tạo file mới — không tạo module trùng lặp nếu module hiện có mở rộng được.
- Không viết lại code đang chạy tốt mà không có lý do rõ ràng.
- Trước khi đổi API/interface của một module, tìm tất cả nơi đang gọi nó và cập nhật đồng bộ.
- Giữ mỗi thay đổi nhỏ và tập trung đúng task được giao.
- Không thêm dependency mới trừ khi thực sự cần thiết.
- Nếu yêu cầu chưa rõ, kiểm tra code/config hiện có trước khi hỏi lại hoặc đoán.
- Không tự ý thay đổi quyết định kiến trúc đã ghi trong file này.
- Không tự ý thay thế model, tracker, hay thuật toán đã chỉ định trong file này bằng lựa chọn khác để code chạy được (ví dụ BoT-SORT lỗi → tự đổi sang DeepSORT cho xong việc, hoặc YOLO26-seg không nhận → tự đổi sang YOLO11-seg). Khi gặp trường hợp này, dừng lại, báo rõ lỗi gặp phải, và chờ xác nhận trước khi đổi — không tự quyết rồi giải thích sau.
- Sau khi sửa code: (1) chạy kiểm tra cú pháp/import, (2) chạy test liên quan nếu có, (3) báo rõ đã thay đổi gì và phần nào chưa được test.
- Không bịa kết quả benchmark/thực nghiệm. Khi báo cáo kết quả code hay experiment, phân biệt rõ mức đã kiểm chứng: syntax/import check, unit test, hay benchmark GPU thực tế — không gộp chung thành "đã pass"/"thành công" nếu chưa chạy đủ bước. Nếu chưa có experiment thực tế cho một so sánh (ví dụ YOLO11 vs YOLO26), nói rõ là chưa có số liệu thay vì tự ước tính rồi trình bày như dữ liệu đã đo.

**Xử lý code tham khảo từ repo bên ngoài:** khi được yêu cầu xem một repo/dự án khác để tham khảo cách làm (ví dụ cách xử lý occlusion, cách setup PX4 SITL), đọc để hiểu ý tưởng/thiết kế — không copy-paste nguyên file hoặc khối code lớn vào project. Viết lại logic theo interface và convention đã định trong file này (ví dụ Motion Memory contract ở mục 6), không giữ nguyên cấu trúc code của tác giả gốc. Thư viện chính thức có version rõ ràng (`ultralytics`, `mavsdk`, PX4-Autopilot) là dependency, cài qua package manager như bình thường — khác với các dự án mẫu cá nhân trên GitHub (thường không có license rõ ràng), chỉ dùng để đọc và học ý tưởng.

## 10. Git Rules

- Không commit file sinh ra, dataset, checkpoint, log, hay file môi trường cá nhân.
- Không rewrite git history trừ khi được yêu cầu rõ ràng.
- Không dùng `git reset --hard`, `git clean -fd`, hay force push trừ khi được yêu cầu rõ ràng.
- Mỗi commit tập trung vào một thay đổi logic.
- Kiểm tra `git diff`/`git status` trước khi commit.

## 11. Testing Rules

Với mọi thay đổi code Python:
1. `python -m py_compile` cho file đã sửa
2. Import test cho module bị ảnh hưởng
3. Unit test nếu có
4. Smoke test nhẹ trên CPU nếu áp dụng được

**Không chạy full training (`yolo train ...`) như một bước validation trừ khi được yêu cầu rõ ràng** — training tốn thời gian/chi phí cloud GPU thật, không dùng để "kiểm tra syntax".

## 12. Evaluation Metrics

**Ưu tiên các metric đơn giản, tự đo được, bám sát mục tiêu occlusion:**
- Detection recall và precision được báo cáo **song song**. Recall đặc biệt quan trọng vì miss detection tạo tracking gap, nhưng không dùng recall đơn độc để đánh giá chất lượng tracking — recall cao kèm precision thấp có thể tạo nhiều false positive, khiến tracker association sai và tăng ID switch.
- ID switch count
- FPS/latency thực đo

**Occlusion-specific metrics (bắt buộc cho mỗi kịch bản test occlusion):**
- Thời lượng occlusion · % frame dự đoán đúng vị trí · tracking gap dài nhất · re-identification thành công/thất bại · time-to-reidentify · sai số vị trí trong lúc occlusion
- **Bắt buộc có ground truth:** mọi kịch bản occlusion dùng để benchmark phải có ground-truth vị trí/bounding box của mục tiêu trong toàn bộ sequence, kể cả các frame bị occlude/nhiễu — nếu không có ground truth, không thể tính được "% frame dự đoán đúng" hay "sai số vị trí" ở trên.

**Baseline/ablation bắt buộc khi đánh giá Motion Memory:** để biết Motion Memory thực sự có cải thiện hay không (chứ không phải do đổi detector), luôn so sánh có/không Motion Memory trên cùng detector+tracker: `YOLO11+BoT-SORT` vs `YOLO11+BoT-SORT+Motion Memory`, và tương tự cho YOLO26 — không chỉ chạy một cấu hình đầy đủ rồi kết luận.

**Metric học thuật chuẩn MOT (IDF1, MOTA, HOTA) — KHÔNG bắt buộc ở giai đoạn này**, chỉ cần nếu sau này viết báo cáo/đồ án cần so sánh chuẩn học thuật (cần ground-truth MOT-format + TrackEval/py-motmetrics).

## 13. Research Integrity

Áp dụng vì kết quả có thể dùng cho báo cáo/đồ án sau này:
- Không gọi một kết quả là "benchmark" nếu chưa chạy trên dataset/split đã cố định từ trước.
- Không đổi dataset/test split sau khi đã xem kết quả mà không tạo experiment mới (tránh vô tình p-hack).
- Không cherry-pick kết quả tốt nhất từ nhiều lần chạy để báo cáo như một experiment duy nhất.
- Không xóa lịch sử các lần chạy thất bại — ghi lại kể cả khi kết quả xấu, để sau này biết đã thử gì.

## 14. Experiment Reproducibility (mức tối giản cho giai đoạn đầu)

- Mỗi lần train có thư mục riêng trong `experiments/` với ID rõ ràng.
- Mỗi thư mục lưu: git commit hash tại thời điểm chạy, tên/phiên bản dataset dùng, config, version `ultralytics`/`torch`, kết quả (xem mục 7 cho phần commit vs gitignore) — để sau này biết chính xác một thay đổi kết quả (ví dụ recall tăng từ 91% lên 94%) đến từ thay đổi code, dataset, hay config nào.
- Ở vòng **benchmark cuối cùng** (bước 6 roadmap) mới cần siết: cùng dataset split, resolution, epoch, augmentation, ghi lại seed, và bổ sung thêm Python/CUDA/GPU model vào thông tin lưu trữ (không cần cho mỗi lần chạy thử ở giai đoạn đầu).

## 15. Definition of Done

- Code đã implement và chạy được.
- Không phá vỡ chức năng hiện có mà không có lý do.
- Test liên quan (nếu có) đã pass; kiểm tra cú pháp/import đã pass.
- Không có path hardcode đặc thù máy cá nhân.
- Không commit checkpoint/dataset/artifact sinh ra.
- Nếu task liên quan benchmark, kết quả đã lưu vào `experiments/`.
- README/doc liên quan cập nhật nếu hành vi public thay đổi.

## 16. Roadmap (giai đoạn mô phỏng)

1. Setup môi trường + dataset ban đầu (chia split theo video/sequence — xem mục 7)
2. Fine-tune YOLO11-seg và YOLO26-seg trên cùng dataset/split/config
3. Tích hợp BoT-SORT, test cơ bản (chưa có occlusion)
4. Thiết kế kịch bản occlusion nhân tạo (có ground truth) + test khả năng re-ID
5. Viết module motion memory/temporal propagation, tích hợp bổ trợ cho BoT-SORT
6. Benchmark tổng thể theo baseline/ablation matrix (mục 12): so sánh detector (YOLO11 vs YOLO26), rồi tác động của Motion Memory trên từng detector, rồi hiệu năng khi occlusion — không chỉ so sánh detection metric đơn thuần (siết reproducibility ở bước này)
7. Sửa lỗi phát sinh + tinh chỉnh

## 17. Bối cảnh phần cứng tương lai (KHÔNG triển khai bây giờ — xem mục 2)

- Flight controller: STM32H743 (board FK743M2-IIT6), tự port PX4 — tách biệt khỏi AI pipeline trong repo này.
- Drone dạng **quadcopter** (đã cân nhắc VTOL/quadplane trước đó, nhưng chốt dùng quadcopter cho hướng hiện tại) — không ảnh hưởng code AI.
- Phần cứng AI deploy dự kiến: Jetson Orin Nano Super hoặc Pi5+Hailo-8L — export TensorRT/HEF, correlation tracker CSRT/KCF, để dành cho giai đoạn sau khi thuật toán tracking đã ổn định ở sim.

## 18. Về người phát triển

Solo developer, sinh viên kỹ thuật phần mềm, đã có kinh nghiệm PX4/embedded, đang học dần AI/CV cho dự án này. Làm việc theo nhịp ~3 tiếng/ngày — ưu tiên task nhỏ, có thể hoàn thành trọn vẹn trong 1 buổi, hơn là task lớn dở dang qua nhiều ngày.
