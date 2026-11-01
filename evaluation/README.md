# Đánh giá định lượng: Python và Fiji/TrackMate so với Ground Truth

Một bộ đánh giá dùng chung chấm cả kết quả Python (Nearest, Hungarian) và Fiji/TrackMate
theo cùng một quy tắc ghép cặp. Nhờ vậy số liệu hai bên so sánh công bằng được.

## Chạy

```bash
pip install -r requirements.txt            # cần numpy, scipy, matplotlib
python main.py                              # tạo python/outputs/<tên>/trajectories_*.csv
python evaluation/evaluate.py --all         # chấm toàn bộ dataset/
```

`--all` tự tìm:

| Thành phần | Nơi tìm |
|---|---|
| Ground truth | `dataset/<dataX>/<stem>.csv` (hạt) hoặc `ground_truth_*.xml` (CAVIAR) |
| Python | `python/outputs/<stem>/trajectories_{nearest,hungarian}.csv` |
| Fiji/TrackMate | `<stem>_spots.csv` hoặc `<stem>_TrackMate.xml` trong `fiji/` hoặc `Báo cáo XLA/`; thêm `dataset/.../<file>.tif.xml` |

Chấm riêng từng cặp:

```bash
python evaluation/evaluate.py \
  --gt dataset/data3/track_2D_A_0000.csv \
  --pred Hungarian=python/outputs/track_2D_A_0000/trajectories_hungarian.csv \
  --pred TrackMate="Báo cáo XLA/TrackMate_GUI_A/results/track_2D_A_0000_spots.csv" \
  --out evaluation/results/A
```

Tuỳ chọn chính: `--gate` (px, mặc định 5), `--iou` (mặc định 0.3), `--match auto|distance|iou|inbox`,
`--pixel-size` (µm/pixel cho CSV của TrackMate, tự đọc từ XML nếu có), `--include-predicted`
(tính cả điểm Kalman dự đoán `matched=0`), `--keep-untracked`, `--gt-offset DX,DY`, `--no-plots`.

Test: `cd evaluation && python -m pytest tests`.

## Định dạng đọc được

| Định dạng | File |
|---|---|
| `gt-csv` | `TrackID,x,y,frame` (data3, data4) |
| `caviar` | XML CAVIAR có box (data1) |
| `python` | `trajectories_*.csv` do `export.py` ghi |
| `trackmate-csv` | Fiji TrackMate → Export to CSV (`*_spots.csv`, đổi µm → pixel) |
| `trackmate-xml` | file phiên TrackMate (ghép spot với track qua danh sách Edge, bỏ track bị filter) |

Tất cả được đưa về pixel, frame tính từ 0.

## Quy tắc ghép cặp (mỗi frame)

* Cả GT và kết quả có box (CAVIAR + Python): **IoU ≥ 0.3**
* GT có box, kết quả chỉ có điểm (TrackMate trên video): điểm **nằm trong box GT**
* GT là điểm (data3/data4): **khoảng cách centroid ≤ 5 px**

Ghép bằng Hungarian (`scipy.optimize.linear_sum_assignment`) trên ma trận chi phí.

## Các chỉ số

| Nhóm | Chỉ số | Ý nghĩa |
|---|---|---|
| Phát hiện | Precision, Recall, F1, TP/FP/FN | Chỉ xét từng frame, không quan tâm ID |
| Định vị | RMSE, mean error, bias x/y (px) | Sai số điểm đã ghép được |
| CLEAR-MOT | MOTA = 1 − (FN+FP+IDSW)/#GT, MOTP, IDSW | Giữ cặp GT↔ID khi còn hợp lệ nên IDSW chỉ đếm lần đổi ID thật |
| Định danh | IDTP, IDP, IDR, **IDF1** | Một phép gán toàn cục GT-track ↔ pred-track |
| Mức track | MT (≥80%), PT, ML (≤20%), FM | FM = số lần track GT bị đứt rồi nối lại |
| Mô tả | số track, độ dài trung bình, tốc độ | Dùng cho video không có GT (`data2`) |

MOTA có thể âm khi FP rất nhiều (như `video1`). MOTP là IoU trung bình (càng cao càng tốt) với box
và là px (càng thấp càng tốt) với điểm.

Ngoài ra có thêm `agreement_with_Fiji-TrackMate` trong `metrics.json`: chấm Python với Fiji làm
tham chiếu, dùng khi chưa có GT.

## Kết quả

Mỗi sequence được lưu ở `evaluation/results/<stem>/`:
`metrics.json`, `metrics.csv`, `metrics.md`, `frames_<method>.csv` (TP/FP/FN/IDSW theo frame),
`scores.png`, `per_frame.png`, `trajectories.png`. Bảng tổng hợp: `evaluation/results/summary.md|csv`.

## Lưu ý quan trọng

* **GT CSV của data3/data4 dùng toạ độ pixel bắt đầu từ 1** (kiểu MATLAB), còn OpenCV/TrackMate bắt đầu
  từ 0. Bằng chứng là cả bốn phương pháp độc lập đều lệch đúng khoảng −1 px ở x và y
  (bias ≈ −1.0 với TrackMate, −0.7 đến −0.9 với Python). Vì vậy `--gt-offset auto` dịch GT
  `(−1, −1)` cho file `TrackID,x,y,frame`. Sau khi dịch, bias còn ≈ 0 và RMSE của TrackMate giảm từ
  1.5 xuống 0.6 px. Dùng `--gt-offset 0,0` để tắt.
* Mặc định chỉ chấm điểm Python có `matched=1` (thật sự được phát hiện). Điểm Kalman dự đoán trong lúc
  mất detection chỉ được tính khi có `--include-predicted`.
* Chỉ chấm trong khoảng frame có GT (video1: frame 0–382, dù video dài hơn).
* `TrackMate-dataset-XML` là file `*.tif.xml` có sẵn trong `dataset/` (DoG + Sparse LAP, `THRESHOLD=2.5`).
  Nó không phải GT và được chấm như một kết quả Fiji khác.
