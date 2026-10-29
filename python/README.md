# Xử lý video và theo dõi đa đối tượng

Đặt các file video hoặc TIFF trực tiếp vào thư mục `dataset`. Từ thư mục gốc,
chạy:

```powershell
python main.py
```

Trong lần thiết lập đầu tiên, cài thư viện bằng
`python -m pip install -r requirements.txt`.

Chương trình xử lý lần lượt mọi file hỗ trợ, sử dụng cùng các detection cho hai
thuật toán Nearest-Neighbor và Hungarian. Mỗi track có Kalman filter để dự đoán
vị trí, giữ ID qua các frame và xây dựng quỹ đạo. Viền track và quỹ đạo dày
1 pixel để dễ quan sát vật thể nhỏ.

Video thông thường dùng MOG2 để tách foreground. TIFF dùng white top-hat để tách
các đốm sáng nhỏ; FPS được đọc từ metadata ImageJ khi có. Trước phân đoạn,
chương trình lọc nhiễu nhẹ và giữ biên; sau đó tự điều chỉnh ngưỡng diện tích
theo các vùng phát hiện trong từng cảnh. Ngưỡng đốm sáng TIFF được điều chỉnh
theo mức nhiễu nền. Với cảnh có nhiều vùng lớn, chương
trình nối các mảnh foreground gần nhau và loại bớt vùng nhỏ nằm trong khung
lớn. Các vật thể thật nằm sát nhau có thể bị gộp; trong cảnh có cả vật thể lớn
và rất nhỏ, các vật thể nhỏ nhất vẫn có thể bị lọc nhầm.

Track được xác nhận sau ba lần phát hiện liên tiếp; trong cảnh có nhiều vật thể
lớn, các vùng nhỏ cần năm lần để giảm ID nhiễu. Với video thông thường, track cần dịch chuyển
ít nhất 2 pixel trước khi được hiển thị; track đã di chuyển vẫn được giữ khi
tạm dừng. Kalman tự tăng độ linh hoạt theo vận tốc để theo kịp vật tăng tốc.
Video đầu ra chỉ vẽ track đang có detection, rút ngắn đường vẽ và giới hạn
nhãn ID khi cảnh quá đông;
điểm dự đoán trong lúc mất detection vẫn có trong CSV/JSON với `matched=false`.
Số `visible tracks` là số track được xác nhận và đang có detection, không phải
số vật thể thực tế. Chất lượng giữ ID còn phụ thuộc độ tương phản và che khuất;
vật thể quá mờ hoặc dịch chuyển dưới ngưỡng có thể chưa được hiển thị.

Kết quả nằm trong `python/outputs/<tên-file>/`:

- `annotated_nearest.mp4` và `annotated_hungarian.mp4`: video có ID và quỹ đạo.
- `trajectories_nearest.csv`, `trajectories_nearest.json`,
  `trajectories_hungarian.csv`, `trajectories_hungarian.json`: vị trí, vận tốc
  và thông tin điểm được đo hoặc dự đoán.
- `python/outputs/summary.json`: thống kê cho tất cả file đã xử lý.

Các file cùng tên gốc nhưng khác phần mở rộng được lưu vào thư mục riêng có
thêm phần mở rộng vào tên. Chương trình hỗ trợ MP4, AVI, MOV, MKV, WebM, WMV,
MPEG/MPG, FLV, 3GP, TS/MTS/M2TS, VOB, MJPEG và TIFF; khả năng đọc codec phụ
thuộc bản OpenCV được cài.
