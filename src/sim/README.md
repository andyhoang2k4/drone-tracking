# src/sim — mô phỏng bay có tay điều khiển ảo và màn hình camera OSD

Lớp này **không thuộc pipeline detect/track**. Nó nhận `track_id + bbox + label` từ pipeline (YOLO-seg + BoT-SORT),
rồi biến thành lệnh bay gửi tới PX4. Khi lên drone thật: PX4 chạy trên FC (H743), phần AI + lớp này chạy trên Jetson.

> Phạm vi và ràng buộc của thư mục này: xem ngoại lệ "lớp mô phỏng bay" (quyết định 2026-10-26) ở mục 2 của `CLAUDE.md`.

## Liên hệ với code có sẵn

- `TargetState` chính là `TrackState` của `src/motion_memory.py` (không định nghĩa trùng).
- `motion_memory.py` đã có ngoại suy vị trí, prediction confidence và khớp lại (roadmap bước 5; mới kiểm chứng bằng unit test dữ liệu giả).
  `target_manager.py` VẪN dùng ngoại suy tâm ảnh đơn giản riêng để lái drone (tính theo giây). Đó **không phải** motion memory và không
  dùng cho đánh giá (CLAUDE.md mục 12). Chưa thay bằng `MotionMemory`: interface của `MotionMemory` tính theo frame
  (`update(observation, candidates, frame_idx=...)`), cần một adapter trong `src/sim/` nếu muốn dùng chung.
- `src/track.py::iter_tracks` chạy theo cả nguồn video. Mô phỏng cần suy luận từng khung hình do Gazebo đẩy sang, nên `adapters.py`
  gọi `model.track(frame, persist=True, ...)` mỗi frame. Nếu sau này `track.py` có hàm theo từng frame, hãy chuyển adapter sang dùng nó.

## Chạy thử không cần PX4/Gazebo

```bash
pip install -r requirements-sim.txt
python scripts/run_sim.py --demo
```

Hai cửa sổ hiện ra: **Camera + OSD** và **Virtual radio**. Thế giới giả lập có 2 người đi bộ, 1 xe, 2 vật cản che khuất.
Người số 1 đi sau vật cản rộng; sau khi ra lại, bộ phát hiện giả cấp ID mới (giống BoT-SORT đổi ID sau che khuất).

## Điều khiển

| Việc | Cách làm |
|---|---|
| Arm | Công tắc ARMED (click) hoặc `Space`. Chỉ arm được khi ga ≈ 0 |
| Chọn mode | Công tắc 3 nấc (click) hoặc `1` Angle (PX4 Stabilized), `2` Acro, `3` Auto bám mục tiêu |
| Stick trái (ga/yaw) | Kéo chuột, ga giữ nguyên khi thả, yaw tự về giữa. Hoặc `W/S` ga, `A/D` yaw, `X` ga = 0 |
| Stick phải (pitch/roll) | Kéo chuột, tự về giữa. Hoặc `I/K` pitch, `J/L` roll |
| Chọn vật thể | Click vào khung vật thể trên màn hình camera (hoặc `N` để chuyển qua vật thể kế) |
| Khóa / bỏ khóa | Nút LOCK / UNLOCK hoặc `L` / `U` |
| Thoát | `Esc` hoặc `Q` |

Chuột chỉ kéo được một stick một lúc nên có phím song song cho stick còn lại.

### Hành vi

- Bay thủ công + đã khóa mục tiêu: gimbal ảo xoay giữ mục tiêu ở giữa hình.
- Gạt sang Auto khi **chưa khóa**: vẫn bay thủ công, hiện thông báo "CHUA CO MUC TIEU" kèm danh sách vật thể đang thấy. Khóa xong
  (khi công tắc đang ở Auto) thì vào Auto (`auto_engage_on_lock`, tắt được trong `configs/sim.yaml`).
- Auto + mục tiêu: drone quay mũi về mục tiêu và giữ khoảng cách theo kích thước bbox (`follow.desired_height_frac`).
- Mất mục tiêu theo bảng `TRACKING → LOST → REIDENTIFYING → FAILED` (CLAUDE.md mục 6):
  - `LOST`: thông báo "MUC TIEU BIEN MAT", vẫn bám theo vị trí dự đoán (khung cam đứt nét).
  - `REIDENTIFYING`: thông báo đang tìm; quay về hướng cuối cùng của mục tiêu (có cộng tốc độ đổi hướng) và tiến lên.
  - `FAILED`: thông báo, đứng yên. Muốn bám tiếp phải khóa lại thủ công hoặc nhận lại đúng ID cũ.
- Chuyển công tắc về Angle/Acro bất cứ lúc nào để lấy lại điều khiển.

## Nối với PX4 SITL + Gazebo

```bash
python scripts/run_sim.py --weights weights/<ten>.pt --tracker-cfg configs/botsort.yaml \
    --gz-topic /camera --mavlink-url udpin:0.0.0.0:14540 --log experiments/sim_runs/run01.jsonl
```

Việc cần làm **trước** (chưa kiểm chứng ở đây, kiểm tra theo tài liệu đúng phiên bản PX4/Gazebo của bạn):

1. Chạy PX4 SITL với model có camera và gimbal (họ `x500`, ví dụ `x500_gimbal`). Xem `gz topic -l` để lấy tên topic camera.
2. Cài binding Python của Gazebo: `python3-gz-transport13`, `python3-gz-msgs10` (Harmonic).
3. Tham số PX4 cần xem lại:
   - Nhận `MANUAL_CONTROL` qua MAVLink làm nguồn điều khiển tay: `COM_RC_IN_MODE`.
   - Không failsafe "mất RC" khi chạy sim không có RC thật: `NAV_RCL_ACT`, `COM_RCL_EXCEPT`.
   - Gimbal qua MAVLink Gimbal Protocol v2: `MNT_MODE_IN`, `MNT_MODE_OUT` theo model gimbal.
   - Offboard: PX4 chỉ vào được khi đã có setpoint đều đặn (>2 Hz). Code stream setpoint mỗi vòng lặp.
4. QGroundControl chạy song song được (cổng 14550), còn lớp này dùng cổng 14540.

Mode PX4: `STABILIZED` (= "Angle"), `ACRO`, `OFFBOARD` (cho Auto). PX4 không có mode tên "Angle".

### Trạng thái trên cloud (cập nhật 2026-11-15)

Đã chạy thật trên cloud (Ubuntu 24.04, RTX 5060 Ti, driver 595.99.02):
- Gazebo Harmonic 8.15.0 (`gz sim --version`), cài cùng PX4-Autopilot bằng `Tools/setup/ubuntu.sh --no-nuttx` (PX4 v1.18.0-beta1-1572-g53afcc82b2 ở ~/PX4-Autopilot; chưa build/chạy PX4).
- Render headless bằng GPU: `DISPLAY= gz sim -v 4 -s -r --headless-rendering <world>.sdf`. World dùng plugin Sensors với `<render_engine>ogre2</render_engine>`. `~/.gz/rendering/ogre2.log` có "Starting EGL Subsystem" và GL_RENDERER là RTX 5060 Ti. Cảnh báo `libEGL ... /dev/dri/...: Permission denied` vô hại (EGL vẫn dùng NVIDIA qua /dev/nvidia*).
- Camera chỉ render khi có subscriber trên topic của nó (ví dụ chạy kèm `gz topic -e -t <topic> > /dev/null &`). Thẻ `<save enabled="true"><path>…</path></save>` trong `<camera>` ghi ảnh PNG; tên file chứa `::`, cần đổi tên trước khi tải về Windows.
- Hình học đo trên world thử (camera cao 12 m, pitch 0.7854, hfov 1.5708, 1344x756): vật cao 1.7 m cách ~18 m cao ~64 px trong ảnh, khớp ước tính tay ~63 px (f ≈ 672 px).

Cập nhật thêm (2026-11-15, phiên probe):
- PX4 SITL build được: `env -u VIRTUAL_ENV nice -n 19 make -j6 px4_sitl` → `build/px4_sitl_default/bin/px4`. Phải bỏ `VIRTUAL_ENV`
  nếu shell đang trỏ tới một venv khác: CMake FindPython3 theo biến này và báo thiếu `kconfiglib`.
- Venv riêng cho sim tạo bằng `python3 -m venv --system-site-packages` thì import được `gz.transport13`/`gz.msgs10` (apt). Nếu
  `~/.local` có gói lệch version (ví dụ pandas cần numpy 2), chạy với `PYTHONNOUSERSITE=1`.
- Asset Fuel: `gz fuel download -u https://fuel.gazebosim.org/1.0/OpenRobotics/models/<Tên>` (Hatchback, SUV, Pickup, Walking person,
  Casual female, Male visitor, Standing person) dùng được trong `<include><uri>` bằng URL Fuel. Hatchback tham chiếu
  `model://hatchback/...`: cần `GZ_SIM_RESOURCE_PATH` trỏ tới thư mục có `hatchback/` (symlink tới cache phiên bản); thân xe vẫn thiếu texture.
- Render bằng CPU (Mesa llvmpipe qua EGL) không chạy được: ogre2 báo "OpenGL 3.3 is not supported" rồi segfault. Cần GPU.
- `boundingbox_camera` với `<box_type>2d</box_type>` + `<save>` ghi `boxes/boxes_*.csv` (`label,x_center,y_center,width,height`) theo plugin Label.

Chưa kiểm: `make px4_sitl gz_x500_gimbal` (mới build, chưa chạy kèm Gazebo); `GzCameraSource` với Gazebo thật; YOLO11-seg/YOLO26-seg trên ảnh
Gazebo (probe domain-gap, CLAUDE.md §2: đã render khung + GT bbox ngoài repo, chưa chạy detector). Probe không phải benchmark.

File probe: scripts/sim_probe/ (smoke_world.sdf, check_pixels.py). Chạy từ thư mục ngoài repo: ảnh ghi vào frames_smoke/ theo thư mục đang chạy gz sim.

## Cấu trúc

| File | Vai trò |
|---|---|
| `virtual_radio.py` | Tay điều khiển ảo (OpenCV) |
| `camera_view.py`, `osd.py` | Cửa sổ camera, chọn/khóa bằng chuột, OSD |
| `target_manager.py` | Chọn/khóa mục tiêu, trạng thái LOST/REID/FAILED, bắt lại khi đổi ID (heuristic) |
| `follow_controller.py`, `search.py`, `gimbal.py` | Lệnh bám, tìm theo hướng cũ, gimbal |
| `supervisor.py` | Chuyển mode, nhịp arm, thông báo, ghép lệnh |
| `link.py` | `PymavlinkLink` (PX4 thật/SITL) |
| `sources.py`, `adapters.py` | Camera Gazebo/video, YOLO + BoT-SORT theo từng frame (cùng quy ước track dict với `src/track.py`, dùng `configs/botsort.yaml`) |
| `demo.py` | Thế giới giả lập để chạy thử |
| `app.py` | Vòng lặp chính |

## Mức đã kiểm chứng (và chưa)

Đã kiểm: `py_compile`, import, unit test cho logic (chọn/khóa, trạng thái, bám, tìm kiếm, gimbal, chuyển mode, tay điều khiển ảo, OSD),
test `PymavlinkLink` với "autopilot giả" qua UDP loopback (đúng khuôn tin nhắn MAVLink), và chạy cả vòng lặp trong thế giới demo.

**Chưa kiểm** với phần cứng/phần mềm thật: PX4 SITL, Gazebo (`GzCameraSource`), YOLO + BoT-SORT (`UltralyticsBotSortSource`), gimbal
thật trong Gazebo, tên tham số PX4 ở trên, và cửa sổ OpenCV trên máy cloud qua remote desktop. Số tuning (`config.py`) là giá trị khởi đầu, chưa đo. Riêng render Gazebo headless đã kiểm một phần, xem "Trạng thái trên cloud" ở trên.

## Giới hạn đã biết

- Bắt lại mục tiêu sau khi đổi ID chỉ là gating theo vị trí dự đoán + nhãn lớp. Re-ID thật thuộc pipeline/Motion Memory.
- Góc gimbal dùng góc lệnh (open-loop). Có thể đọc `GIMBAL_DEVICE_ATTITUDE_STATUS` để chính xác hơn.
- Thông báo trên OSD không dấu vì font OpenCV không vẽ được dấu tiếng Việt.
- Drone thật hiện chưa có gimbal; gimbal ảo chỉ có ở mô phỏng.
