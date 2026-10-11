# Xử lý video và theo dõi đa đối tượng

Module này thực hiện đầy đủ pipeline cổ điển (không cần mô hình học sâu):

1. Phân đoạn foreground bằng MOG2 và morphology.
2. Phát hiện contour, bounding box và centroid.
3. Dự đoán trạng thái `[x, y, vx, vy]` bằng Kalman Filter.
4. Ghép detection giữa các frame bằng Nearest-Neighbor hoặc Hungarian.
5. Vẽ ID/quỹ đạo và lưu kết quả CSV, JSON, video chú thích, video mask.

Cost ghép cặp kết hợp khoảng cách centroid, IoU của bounding box, tỉ lệ diện
tích và histogram màu HSV. Histogram được cập nhật dần theo track để giảm đổi
ID khi hai vật thể giao nhau.
Track mới ở trạng thái tạm thời, chỉ được xác nhận sau `min_hits` lần phát hiện;
nhiễu xuất hiện một frame sẽ không bị lưu thành quỹ đạo thật.

## Chạy từ thư mục gốc repository

Các lệnh khuyến nghị dưới đây chạy tại
`D:\image-processing-object-tracking`:

```powershell
python -m pip install -r requirements.txt
python main.py dataset\video.mp4 --output python\outputs --algorithm both --device auto
```

Ép dùng GPU OpenCL:

```powershell
python main.py dataset\video.mp4 --output python\outputs --algorithm both --device gpu
```

Chạy thử nhanh 300 frame:

```powershell
python main.py dataset\video.mp4 --output python\outputs --algorithm both --device gpu --max-frames 300
```

File `main.py` ở gốc là launcher; toàn bộ phần cài đặt vẫn nằm trong thư mục
`python`.

## Cài đặt và chạy trong thư mục `python`

Từ thư mục `python`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Chạy

Một video:

```powershell
python main.py ..\dataset\video.mp4 --output outputs --algorithm both
```

Toàn bộ video trong `dataset`:

```powershell
python main.py ..\dataset --output outputs --algorithm both
```

Mặc định `--device auto` dùng OpenCL GPU nếu OpenCV nhận được GPU, nếu không sẽ
tự dùng CPU. Ép chạy GPU hoặc CPU bằng:

```powershell
python main.py ..\dataset\video.mp4 --device gpu --algorithm both
python main.py ..\dataset\video.mp4 --device cpu --algorithm both
```

GPU tăng tốc MOG2, threshold và morphology. Nearest-Neighbor, Hungarian,
Kalman và ghi video vẫn chạy trên CPU. Trường `processing_device_name` trong
`summary.json` cho biết backend thực tế, tránh trường hợp tưởng rằng GPU đã bật.

Chương trình in tiến độ sau mỗi 100 frame. Để chạy thử nhanh 300 frame đầu:

```powershell
python main.py ..\dataset\video.mp4 --algorithm both --max-frames 300
```

`--algorithm both` xử lý cùng detection bằng cả hai thuật toán để so sánh công bằng. Có thể chọn riêng `nearest` hoặc `hungarian`.

Các tham số thường cần chỉnh theo video:

```powershell
python main.py ..\dataset\video.mp4 `
  --min-area 300 `
  --max-area 50000 `
  --max-distance 80 `
  --iou-weight 0.25 `
  --area-weight 0.1 `
  --max-area-ratio 4 `
  --appearance-weight 0.5 `
  --max-missed 10 `
  --warmup-frames 15
```

- `--min-area`, `--max-area`: lọc nhiễu hoặc blob quá lớn.
- `--max-distance`: cổng khoảng cách ghép centroid, tính bằng pixel.
- `--max-missed`: số frame tối đa giữ track khi bị che khuất (mặc định 15).
- `--iou-weight`: mức ảnh hưởng của độ chồng lấp bounding box khi ghép.
- `--area-weight`: mức phạt khi diện tích detection thay đổi.
- `--max-area-ratio`: không ghép hai đối tượng chênh diện tích quá tỉ lệ này.
- `--appearance-weight`: mức ảnh hưởng của histogram màu khi ghép ID.
- `--appearance-momentum`: độ ổn định của mẫu màu theo thời gian.
- `--initial-velocity-variance`: độ bất định vận tốc ban đầu (đơn vị pixel/giây);
  tăng khi vật thể xuất hiện với vận tốc lớn.
- `--max-foreground-ratio`: bỏ detection ở frame đổi cảnh/rung quá mạnh.
- `--warmup-frames`: thời gian để MOG2 học nền trước khi tạo track.
- `--learning-rate`: tốc độ cập nhật nền (`-1` là tự động, `0` là đóng băng nền).
- `--display`: xem trực tiếp; nhấn `q` để dừng.

Mỗi video tạo một thư mục gồm:

- `annotated_nearest.mp4`, `annotated_hungarian.mp4`;
- `foreground_mask.mp4`;
- `trajectories_<algorithm>.csv` và `.json`;
- `summary.json` ở thư mục output, có tốc độ xử lý, thời gian ghép cặp,
  độ dài track và tỉ lệ điểm đo/dự đoán.

Trong CSV/JSON, `matched=false` nghĩa là vị trí ở frame đó do Kalman dự đoán khi detection tạm thời bị mất.
Các cột `vx`, `vy`, `speed` là vận tốc Kalman theo pixel/giây.

## Kiểm thử

Các test thuật toán không cần video thật:

```powershell
python -m pytest -q
```

Để kiểm thử nhiều video, đặt video vào `dataset`, chạy cả hai thuật toán, rồi so sánh `track_count`, `association_seconds` trong `summary.json` và kiểm tra trực quan ID/quỹ đạo ở video đầu ra. Với đánh giá định lượng, có thể đối chiếu CSV quỹ đạo với ground truth bằng IDF1/MOTA trong thư mục `evaluation` của nhóm.

## Gợi ý tối ưu

- Video camera rung: ổn định ảnh trước MOG2 hoặc tăng `variance-threshold`.
- Nhiều nhiễu nhỏ: tăng `min-area`, `kernel-size` hoặc số lần morphology.
- Vật di chuyển nhanh: tăng `max-distance` và process variance.
- Hay đổi ID lúc che khuất: tăng `max-missed`; ưu tiên Hungarian.
- Nền thay đổi nhanh: tăng learning rate; nền tĩnh thì giảm hoặc đặt `0` sau warm-up.
