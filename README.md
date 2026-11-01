# Object Detection and Tracking using Python and Fiji/ImageJ

## 📌 Giới thiệu

Đây là dự án môn **Xử lý ảnh** với đề tài:

> **Nghiên cứu, cài đặt, thực nghiệm và đánh giá thuật toán phát hiện và theo dõi đối tượng trong video, sử dụng Fiji/ImageJ làm nền tảng tham chiếu và kiểm chứng.**

Dự án tập trung vào bài toán **Multi-Object Tracking (MOT)**, trong đó các đối tượng chuyển động trong video được phát hiện và gán một **ID cố định** xuyên suốt các frame.

Nhóm thực hiện hai hướng tiếp cận:

1. **Fiji/ImageJ + TrackMate**
   Sử dụng TrackMate như một công cụ tham chiếu để thực hiện object detection và tracking.

2. **Python**
   Tự cài đặt pipeline xử lý ảnh và tracking, sau đó so sánh kết quả với TrackMate và Ground Truth.

Mục tiêu cuối cùng là đánh giá định lượng hiệu quả của các phương pháp tracking thay vì chỉ đánh giá bằng quan sát trực quan.

---

## 🎯 Mục tiêu

Dự án hướng tới các mục tiêu:

* Tìm hiểu lý thuyết về phát hiện và theo dõi đối tượng trong video.
* Tìm hiểu quy trình tracking của Fiji/ImageJ và plugin TrackMate.
* Cài đặt pipeline xử lý video bằng Python.
* Thực hiện phát hiện đối tượng từ từng frame.
* Tính toán centroid/bounding box của đối tượng.
* Cài đặt thuật toán **Nearest Neighbor** để liên kết đối tượng giữa các frame.
* Cài đặt **Hungarian Algorithm** để giải bài toán data association.
* Nghiên cứu khả năng sử dụng **Kalman Filter** để dự đoán vị trí đối tượng.
* Sử dụng Ground Truth đi kèm dataset (CAVIAR XML, CSV/TrackMate) để đánh giá kết quả.
* So sánh định lượng giữa các phương pháp.
* Phân tích các trường hợp mất track, nhầm ID và ID Switch.

---

## ⚡ Demo trong một lệnh

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

---

# 🧠 Tổng quan hệ thống

Pipeline chính của dự án:

```text
                    VIDEO
                      │
                      ▼
              Frame Extraction
                      │
                      ▼
               Preprocessing
                      │
                      ▼
             Object Detection
                      │
                      ▼
          Bounding Box / Centroid
                      │
                      ▼
              Data Association
                 /           \
                /             \
               ▼               ▼
       Nearest Neighbor    Hungarian
               │               │
               └───────┬───────┘
                       ▼
                 Track ID
                       │
                       ▼
                Trajectory
                       │
                       ▼
                  Evaluation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      Precision      Recall         F1
          │            │            │
          └────────────┼────────────┘
                       ▼
                  ID Switch
                       │
                       ▼
                Final Analysis
```

---

# 🔬 Các phương pháp được nghiên cứu

## 1. Object Detection

Từ mỗi frame của video, hệ thống xác định các đối tượng cần tracking.

Đầu ra:

```text
Frame
  ↓
Object 1 → Bounding Box
Object 2 → Bounding Box
Object 3 → Bounding Box
...
```

Bounding box được biểu diễn bởi:

```text
(x, y, width, height)
```

Trong đó có thể tính centroid:

$$
c_x = x + \frac{w}{2}
$$

$$
c_y = y + \frac{h}{2}
$$

---

## 2. Nearest Neighbor Tracking

Nearest Neighbor là phương pháp baseline đơn giản.

Với centroid của object ở frame hiện tại:

$$
p_i = (x_i,y_i)
$$

và centroid của object ở frame tiếp theo:

$$
q_j = (x_j,y_j)
$$

Khoảng cách Euclidean:

$$
d(p_i,q_j)
=
\sqrt{(x_i-x_j)^2+(y_i-y_j)^2}
$$

Object mới được gán cho track gần nhất nếu:

$$
d(p_i,q_j) < d_{max}
$$

### Ưu điểm

* Dễ cài đặt.
* Tốc độ nhanh.
* Phù hợp làm baseline.

### Nhược điểm

* Dễ nhầm ID khi hai object tiến gần nhau.
* Dễ xảy ra ID Switch khi các object crossing paths.
* Không giải quyết tốt occlusion.

---

# 3. Hungarian Algorithm

Hungarian Algorithm được sử dụng để giải bài toán **global data association**.

Từ các track hiện tại và detections mới, xây dựng cost matrix:

$$
C_{ij}=d(p_i,q_j)
$$

Sau đó tìm phép gán tối ưu:

$$
\min \sum C_{ij}
$$

so với việc lựa chọn từng object độc lập bằng Nearest Neighbor.

Mục đích của thí nghiệm là kiểm tra liệu Hungarian Algorithm có giúp giảm:

* ID Switch
* Tracking error
* Track fragmentation

hay không.

---

# 4. Kalman Filter

Kalman Filter được dùng để dự đoán vị trí của từng track giữa các frame (`python/object_tracking/kalman.py`).

Mô hình trạng thái:

$$
x_k = F x_{k-1}+w_k
$$

Trong đó:

* \(x_k\): trạng thái của object.
* \(F\): state transition matrix.
* \(w_k\): process noise.

Measurement:

$$
z_k = Hx_k+v_k
$$

Kalman Filter cho phép dự đoán vị trí object khi detection bị mất hoặc bị che khuất trong một số frame.

---

# 🖥️ Fiji/ImageJ và TrackMate

**Fiji/ImageJ** được sử dụng làm nền tảng tham chiếu và kiểm chứng.

Plugin **TrackMate** được sử dụng để:

1. Import video.
2. Phát hiện object.
3. Tạo tracks.
4. Xuất tracking results.
5. Quan sát trajectory.
6. So sánh với kết quả Python.

Workflow:

```text
Video
  ↓
Fiji/ImageJ
  ↓
TrackMate
  ↓
Detector
  ↓
Tracker
  ↓
Tracks
  ↓
Export Results
  ↓
Evaluation
```

TrackMate không được xem là Ground Truth.

Ground Truth độc lập được lấy từ annotation đi kèm dataset (ví dụ XML của CAVIAR cho `data1`).

---

# 📊 Dataset

Bốn bộ dữ liệu nằm trong `dataset/` và được commit cùng repository:

| Thư mục | File | Nội dung | Ground Truth |
| ------- | ---- | -------- | ------------ |
| `data1/` | `video1.mpg` | Video CAVIAR `EnterExitCrossingPaths1cor`, 384 × 288, 383 frame có annotation | `ground_truth_video1.xml` (định dạng CAVIAR) |
| `data2/` | `video2.mp4` | Video 1280 × 720, dài khoảng 199 giây | Chưa có |
| `data3/` | `track_2D_A_0000.tif` | Chuỗi ảnh TIFF nhiều frame (đốm sáng nhỏ) | `track_2D_A_0000.csv`, `track_2D_A_0000.tif.xml` (TrackMate) |
| `data4/` | `track_2D_C_0002.tif` | Chuỗi ảnh TIFF nhiều frame (đốm sáng nhỏ) | `track_2D_C_0002.csv`, `track_2D_C_0002.tif.xml` (TrackMate) |

## CAVIAR (`data1`)

`video1.mpg` là sequence `EnterExitCrossingPaths1cor` của dataset **CAVIAR** (University of Edinburgh): hai người đi qua nhau tại khu vực cửa hàng, có hiện tượng **crossing paths** nên phù hợp để kiểm tra khả năng giữ ID. Ground Truth là XML, mỗi frame chứa thông tin object:

```xml
<frame number="0">
    <objectlist>
        <object id="0">
            <orientation>90</orientation>
            <box h="79" w="30" xc="145" yc="111"/>
            ...
        </object>
    </objectlist>
</frame>
```

| Thuộc tính     | Ý nghĩa                 |
| -------------- | ----------------------- |
| `frame number` | Số frame (bắt đầu từ 0) |
| `object id`    | ID Ground Truth         |
| `xc`           | Tọa độ tâm X            |
| `yc`           | Tọa độ tâm Y            |
| `w`            | Chiều rộng bounding box |
| `h`            | Chiều cao bounding box  |

## TIFF + TrackMate (`data3`, `data4`)

Mỗi bộ gồm một file TIFF và hai file đi kèm:

* `*.csv`: danh sách điểm theo track, các cột `TrackID,x,y,frame`.
* `*.tif.xml`: file TrackMate (XML) cho cùng chuỗi ảnh.

```csv
TrackID,x,y,frame
0,82.000000000,116.000000000,0
0,79.157736602,116.959968111,1
```

> **Lưu ý:** `python main.py` quét toàn bộ `dataset/` kể cả thư mục con (bỏ qua file/thư mục bắt đầu bằng dấu `.`), nên chạy một lần sẽ xử lý cả bốn bộ dữ liệu trên. Các file `.csv` và `.xml` đi kèm không phải media nên bị bỏ qua.

---

# 📁 Cấu trúc repository

```text
image-processing-object-tracking/
│
├── README.md
├── main.py                      # Launcher ở thư mục gốc, gọi python/main.py
├── requirements.txt             # Trỏ tới python/requirements.txt
│
├── dataset/
│   ├── data1/                   # CAVIAR: video1.mpg + ground_truth_video1.xml
│   ├── data2/                   # video2.mp4
│   ├── data3/                   # track_2D_A_0000.tif + .csv + .tif.xml
│   └── data4/                   # track_2D_C_0002.tif + .csv + .tif.xml
│
├── python/
│   ├── main.py                  # Xử lý toàn bộ media trong dataset/
│   ├── README.md                # Chi tiết thuật toán và tham số
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── object_tracking/
│   │   ├── frame_source.py      # Đọc video / TIFF nhiều frame
│   │   ├── detection.py         # Tách foreground, lọc nhiễu, tạo detection
│   │   ├── tracker.py           # Quản lý track, xác nhận track
│   │   ├── assignment.py        # Nearest Neighbor và Hungarian
│   │   ├── kalman.py            # Kalman Filter dự đoán vị trí
│   │   ├── pipeline.py          # Ghép các bước, xử lý một video
│   │   ├── visualization.py     # Vẽ bounding box, ID, quỹ đạo
│   │   ├── export.py            # Xuất CSV / JSON
│   │   └── models.py
│   ├── tests/                   # Unit test (pytest)
│   └── outputs/                 # Kết quả chạy (bị .gitignore)
│
├── fiji/                        # Fiji/ImageJ + TrackMate (chưa có nội dung)
├── evaluation/                  # Đánh giá định lượng (chưa có nội dung)
├── experiments/                 # Cấu hình / log thí nghiệm (chưa có nội dung)
└── report/                      # Báo cáo (chưa có nội dung)
```

---

# ⚙️ Requirements

## Python

Khuyến nghị:

```text
Python >= 3.10
```

Các thư viện sử dụng (khai báo trong `python/requirements.txt`):

```text
numpy>=1.24
opencv-python>=4.8
Pillow>=10.0
```

Kiểm tra:

```bash
python --version
```

---

# 🚀 Installation

Clone repository:

```bash
git clone git@github.com:dthang-candies/image-processing-object-tracking.git
```

Đi vào thư mục:

```bash
cd image-processing-object-tracking
```

Tạo virtual environment:

```bash
python -m venv .venv
```

Kích hoạt trên macOS/Linux:

```bash
source .venv/bin/activate
```

Kích hoạt trên Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Cài dependencies (file `requirements.txt` ở thư mục gốc trỏ tới `python/requirements.txt`):

```bash
python -m pip install -r requirements.txt
```

---

# ▶️ Running the project

Mã nguồn nằm trong `python/`; file `main.py` ở thư mục gốc chỉ là launcher gọi `python/main.py`. Chương trình không nhận tham số dòng lệnh: nó tự xử lý toàn bộ media trong `dataset/` và chạy cả hai thuật toán (Nearest Neighbor và Hungarian) trên cùng một kết quả detection.

## 1. Chuẩn bị dữ liệu

Đặt file video hoặc TIFF vào thư mục `dataset/` hoặc bất kỳ thư mục con nào của nó (chương trình quét đệ quy):

```text
dataset/
├── data1/video1.mpg
├── data2/video2.mp4
└── data3/track_2D_A_0000.tif
```

Kết quả của mỗi file nằm trong `python/outputs/<tên-file>/`; nếu hai file trùng tên gốc, chương trình tự thêm phần mở rộng hoặc số thứ tự vào tên thư mục.

Định dạng hỗ trợ: MP4, AVI, MOV, MKV, WebM, WMV, MPEG/MPG, FLV, 3GP, TS/MTS/M2TS, VOB, MJPEG và TIFF (`.tif`/`.tiff`, một ảnh hoặc nhiều frame). Khả năng đọc codec phụ thuộc bản OpenCV được cài.

---

## 2. Chạy tracking

Từ thư mục gốc của repository:

```bash
python main.py
```

Chương trình xử lý lần lượt từng file và in tiến độ (số frame, FPS). Video thông thường dùng MOG2 để tách foreground; TIFF dùng white top-hat để tách các đốm sáng nhỏ. Mỗi track có Kalman Filter để dự đoán vị trí khi mất detection.

---

## 3. Kết quả

Kết quả nằm trong `python/outputs/` (thư mục này bị `.gitignore`):

```text
python/outputs/
├── summary.json
└── <tên-file>/
    ├── annotated_nearest.mp4
    ├── annotated_hungarian.mp4
    ├── trajectories_nearest.csv
    ├── trajectories_nearest.json
    ├── trajectories_hungarian.csv
    └── trajectories_hungarian.json
```

* `annotated_*.mp4`: video có bounding box, Track ID và quỹ đạo.
* `trajectories_*.csv/json`: vị trí, vận tốc và thông tin điểm được đo hoặc dự đoán (`matched=false` khi chỉ có dự đoán Kalman).
* `summary.json`: thống kê cho tất cả file đã xử lý.

Các file cùng tên gốc nhưng khác phần mở rộng được lưu vào thư mục riêng có thêm phần mở rộng vào tên.

Chi tiết thuật toán và tham số: xem [python/README.md](python/README.md).

---

## 4. Chạy unit test

```bash
python -m pip install pytest
cd python
python -m pytest
```

---

# 📐 Ground Truth Processing

Ground Truth hiện có ở hai dạng:

| Bộ dữ liệu | Định dạng | Thông tin |
| ---------- | --------- | --------- |
| `data1` | XML kiểu CAVIAR | `frame`, `object id`, bounding box (`xc`, `yc`, `w`, `h`) |
| `data3`, `data4` | CSV (`TrackID,x,y,frame`) và TrackMate XML | Tọa độ tâm theo track |

Để so sánh với kết quả của chương trình, Ground Truth cần được đưa về cùng một định dạng bảng, ví dụ:

```csv
frame,id,xc,yc,w,h
0,0,145,111,30,79
```

> **Trạng thái:** chưa có script chuyển đổi (`parse_xml.py`) và chưa có đoạn mã đánh giá nào trong repository; thư mục `evaluation/` đang trống.

Kết quả tracking của chương trình (`python/outputs/<tên-file>/trajectories_*.csv`) có các cột:

```text
track_id, frame, time_seconds, x, y, vx, vy, speed,
bbox_x, bbox_y, bbox_width, bbox_height, matched
```

---

# 📊 Evaluation

Kết quả của các phương pháp sẽ được đánh giá định lượng.

> **Trạng thái:** phần dưới đây là kế hoạch đánh giá; chưa được cài đặt trong code. Hiện chương trình chỉ xuất video đã đánh dấu, quỹ đạo (CSV/JSON) và tốc độ xử lý (FPS in ra khi chạy).

## Metrics

### Precision

$$
Precision=\frac{TP}{TP+FP}
$$

Đo tỷ lệ detection/tracking prediction đúng trong tổng số prediction.

---

### Recall

$$
Recall=\frac{TP}{TP+FN}
$$

Đo tỷ lệ Ground Truth được hệ thống phát hiện/tracking đúng.

---

### F1-score

$$
F1=2\frac{Precision \times Recall}
{Precision+Recall}
$$

F1 được sử dụng làm metric tổng hợp giữa Precision và Recall.

---

### Centroid Error

Khoảng cách giữa centroid dự đoán và centroid Ground Truth:

$$
E=
\sqrt{(x_p-x_{gt})^2+(y_p-y_{gt})^2}
$$

Có thể báo cáo:

```text
Mean Centroid Error
Median Centroid Error
Maximum Centroid Error
```

---

### ID Switch

ID Switch xảy ra khi cùng một Ground Truth object bị hệ thống gán sang một tracking ID khác.

Ví dụ:

```text
Ground Truth:

Object A → ID 1
Object B → ID 2

Prediction:

Frame 1:
A → ID 1
B → ID 2

Frame 20:
A → ID 2
B → ID 1
```

Trường hợp này tạo ra ID Switch.

---

### Track Fragmentation

Đo mức độ một trajectory bị chia thành nhiều đoạn tracking riêng biệt.

---

### FPS

Đo tốc độ xử lý:

$$
FPS=\frac{\text{Number of processed frames}}
{\text{Processing time}}
$$

Dùng để so sánh hiệu năng giữa các phương pháp.

---

# 🧪 Experimental Design

Các thí nghiệm chính:

### Experiment 1 — Nearest Neighbor (baseline)

Liên kết detection với track bằng khoảng cách gần nhất.

Mục đích:

* Kiểm tra pipeline cơ bản.
* Làm baseline cho phương pháp còn lại.

---

### Experiment 2 — Hungarian Algorithm

Gán toàn cục giữa track và detection bằng Hungarian, chạy trên **cùng detection** với Nearest Neighbor.

Các metric dự kiến:

```text
Precision
Recall
F1
ID Switch
Centroid Error
FPS
```

---

### Experiment 3 — Kalman Filter

Trong code hiện tại, mỗi track luôn có Kalman Filter để dự đoán vị trí, nên cả hai thuật toán gán đều dùng Kalman. Việc bật/tắt Kalman để đối chứng riêng chưa được cài đặt.

Mục đích:

* Kiểm tra khả năng duy trì track.
* Giảm ảnh hưởng của detection noise.
* Xử lý trường hợp object bị mất detection trong một số frame (điểm dự đoán được ghi với `matched=false`).

---

### Experiment 4 — Fiji/ImageJ

Chạy cùng dữ liệu bằng:

```text
Fiji/ImageJ
      ↓
TrackMate
```

Sau đó so sánh với Ground Truth (`data1` có XML CAVIAR; `data3`, `data4` có CSV/TrackMate):

```text
Ground Truth
      │
      ├── Python + Nearest Neighbor
      │
      ├── Python + Hungarian
      │
      └── Fiji/ImageJ + TrackMate
```

---

# 📈 Expected Results

Bảng kết quả dự kiến:

| Method             | Precision | Recall | F1 | Centroid Error | ID Switch | FPS |
| ------------------ | --------: | -----: | -: | -------------: | --------: | --: |
| Nearest Neighbor   |         - |      - |  - |              - |         - |   - |
| Hungarian          |         - |      - |  - |              - |         - |   - |
| TrackMate          |         - |      - |  - |              - |         - |   - |

> Các giá trị sẽ được điền sau khi cài đặt phần đánh giá và thực hiện thực nghiệm.

---

# 📷 Visualization

Với mỗi file đầu vào, chương trình lưu hai video đã đánh dấu (`annotated_nearest.mp4`, `annotated_hungarian.mp4`) có bounding box, Track ID và quỹ đạo (nét dày 1 pixel để dễ quan sát vật thể nhỏ). Video chỉ vẽ track đang có detection; khi cảnh quá đông, đường vẽ được rút ngắn và nhãn ID bị giới hạn.

Quỹ đạo của từng object cũng được lưu ở `trajectories_*.csv` / `.json` để phân tích.

---

# 🔬 So sánh Fiji/ImageJ và Python

Dự án không chỉ thực hiện tracking bằng Python mà còn sử dụng Fiji/ImageJ để tham chiếu.

| Tiêu chí                | Fiji/ImageJ + TrackMate | Python             |
| ----------------------- | ----------------------- | ------------------ |
| Object detection        | TrackMate               | Tự cài đặt         |
| Tracking                | TrackMate               | NN / Hungarian     |
| Kalman Filter           | Có thể cấu hình         | Tự cài đặt         |
| Customization           | Trung bình              | Cao                |
| Quantitative evaluation | Có                      | Có                 |
| Source code             | Plugin                  | Nhóm tự phát triển |
| Experimental control    | Trung bình              | Cao                |

Mục tiêu không phải chứng minh Python luôn tốt hơn TrackMate mà là:

> **Hiểu, cài đặt và đánh giá một pipeline tracking bằng code, đồng thời sử dụng Fiji/ImageJ như một nền tảng tham chiếu và kiểm chứng.**

---

# 👥 Phân công thành viên

| Thành viên                    | Nhiệm vụ                                |
| ----------------------------- | --------------------------------------- |
| Nguyễn Văn Hiếu – B23DCCE033  | Lý thuyết, thuật toán, mô hình toán học |
| Nguyễn Vĩnh Hưng – B23DCCE042 | Python implementation                   |
| Nguyễn Vũ Quang – B23DCVT363  | Fiji/ImageJ + TrackMate                 |
| Đặng Văn Thắng – B23DCCE086   | Dataset, Ground Truth, Evaluation       |

---

# 📝 Report

Báo cáo dự kiến gồm:

```text
1. Introduction
2. Theoretical Background
3. Proposed Method
4. Fiji/ImageJ and TrackMate
5. Python Implementation
6. Experimental Design
7. Results and Discussion
8. Conclusion
9. References
10. Appendix
```

Nội dung trọng tâm:

* Cơ sở lý thuyết.
* Mathematical model.
* Object detection.
* Centroid extraction.
* Data association.
* Nearest Neighbor.
* Hungarian Algorithm.
* Kalman Filter.
* TrackMate.
* Experimental setup.
* Ground Truth.
* Quantitative evaluation.
* Error analysis.

---

# 📚 References

### CAVIAR Dataset

University of Edinburgh — School of Informatics:

https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/

### Fiji/ImageJ

https://fiji.sc/

### ImageJ

https://imagej.net/

### TrackMate

https://imagej.net/plugins/trackmate

### OpenCV

https://opencv.org/

---

# ⚠️ Dataset and GitHub

Các file trong `dataset/` (tổng cộng khoảng 100 MB, trong đó `video2.mp4` khoảng 27 MB) hiện được commit trực tiếp vào repository. Khi thêm dữ liệu lớn hơn, nên cân nhắc Git LFS hoặc chỉ ghi nguồn tải thay vì commit.

Kết quả chạy trong `python/outputs/` không được commit (đã có trong `.gitignore`).

Nguồn dataset CAVIAR:

https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/

---

# 📌 Project Status

* [x] Chọn bài toán Object Tracking
* [x] Chọn dữ liệu (CAVIAR, video, TIFF) và đưa vào `dataset/`
* [x] Có Ground Truth cho `data1` (XML) và `data3`, `data4` (CSV, TrackMate XML)
* [x] Implement đọc video / TIFF (`frame_source.py`)
* [x] Implement preprocessing và object detection (`detection.py`)
* [x] Implement tracker, quản lý track (`tracker.py`)
* [x] Implement Nearest Neighbor và Hungarian Algorithm (`assignment.py`)
* [x] Implement Kalman Filter (`kalman.py`)
* [x] Xuất video đã đánh dấu và quỹ đạo CSV/JSON
* [x] Unit test cho các module chính
* [ ] Parse Ground Truth về định dạng chung
* [ ] Run TrackMate
* [ ] Export TrackMate results
* [ ] Implement evaluation
* [ ] Generate charts
* [ ] Compare results
* [ ] Complete report
* [ ] Complete presentation

---

# 📜 License / Dataset Notice

Dataset được sử dụng cho mục đích học tập và nghiên cứu. Người sử dụng cần tuân thủ điều khoản sử dụng và nguồn gốc của từng bộ dữ liệu (ví dụ CAVIAR cho `data1`).

Source:

https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/

---

# 👨‍💻 Authors

**Digital Image Processing Course Project**

PTIT — Posts and Telecommunications Institute of Technology

Academic Year: **2026**

---
