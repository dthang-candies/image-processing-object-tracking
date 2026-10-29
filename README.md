# Object Tracking Demo

## Demo trong một lệnh

1. Cài dependency:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Đặt video vào `dataset` rồi chạy ở thư mục này:

   ```powershell
   python main.py
   ```

3. Chương trình tự xử lý tất cả media được hỗ trợ trong `dataset`; kết quả nằm
   ở `python\outputs\<tên-file>`.

Chỉ cần lệnh `python main.py`: chương trình chạy lần lượt tất cả media, đưa mỗi
video qua cả Nearest và Hungarian trên cùng detection, rồi lưu video đã đánh
dấu và quỹ đạo riêng cho từng thuật toán. Chương trình tự chọn cách đọc và phân
đoạn phù hợp với định dạng, đồng thời tự chọn GPU nếu được hỗ trợ.

Hỗ trợ video MP4, AVI, MOV, MKV, WebM, WMV, MPEG/MPG, FLV, 3GP, TS/MTS/M2TS,
VOB, MJPEG; TIFF `.tif`/`.tiff` một ảnh hoặc nhiều frame. Decoder codec tùy theo
bản OpenCV được cài.

Xem chi tiết kết quả trong [python/README.md](python/README.md).
