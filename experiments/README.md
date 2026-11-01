# Thí nghiệm: cải thiện Python trên ảnh hạt (data4 = C, kiểm chéo data3 = A)

Script: `tune_detector_tiff.py`. Kết quả thô: `results/*.csv`. Mỗi cấu hình chạy ~1 giây.

```bash
python experiments/tune_detector_tiff.py --data dataset/data4/track_2D_C_0002                      # từng tham số một
python experiments/tune_detector_tiff.py --data dataset/data4/track_2D_C_0002 --mode grid --grid tracker
python experiments/tune_detector_tiff.py --data dataset/data3/track_2D_A_0000 --config "measurement_variance=1.0,process_variance=1000,mahalanobis_gate=100"
```

## Thiết kế

| Mục | Quyết định |
|---|---|
| Câu hỏi | Vì sao C chỉ đạt F1 0,79 / IDF1 0,78 (311 FP, 491 FN, 73 lần đứt track) trong khi TrackMate đạt 0,96 / 0,95? |
| Baseline | `main.build_config` trước thí nghiệm, Hungarian |
| Chấm điểm | `evaluation/` (GT dịch −1,−1, ghép khoảng cách ≤ 5 px) |
| Chống quá khớp | Chia **theo thời gian**: nửa đầu (tune) để chọn, nửa sau (test) để kiểm. Cấu hình thắng phải kiểm chéo trên A |
| Tiêu chí chọn | **IDF1 nửa tune**, tie-break F1. Không chọn theo F1 (xem bài học dưới) |
| Ngẫu nhiên | Không có (pipeline xác định), mỗi cấu hình chạy một lần |
| Quy trình | (1) quét từng tham số một, (2) lưới nhỏ trên tham số có tác dụng, (3) kiểm trên nửa test, (4) kiểm chéo A |

Giả thuyết:

| | Giả thuyết | Kết quả |
|---|---|---|
| H1 | Hệ số σ của ngưỡng thích nghi sai | Bác bỏ: 4,0–6,0 gần như không đổi; σ ≤ 3,5 làm hỏng (precision 0,07) |
| H2 | Cửa sổ top-hat lệch kích thước hạt | Ảnh hưởng nhỏ (size 7: F1 +0,02); không phải nguyên nhân chính |
| H3 | `min_area` quá nhỏ nên nhiễu thành FP | Bác bỏ: không đổi kết quả |
| H4 | `min_hits`, `max_missed` sửa được đứt track | Bác bỏ: không đổi kết quả |
| H5 | **Vị trí xuất ra là trạng thái Kalman lọc quá mạnh nên tụt sau hạt** | **Xác nhận** |

## Phát hiện

1. **Không phải lỗi phát hiện mà là lỗi định vị tăng theo thời gian.** Số điểm Python mỗi frame vẫn ~180 như TrackMate, nhưng độ lệch tới GT gần nhất tăng đều: trung vị 1,3 → 2,3 → 3,2 → 4,1 → 4,6 px qua 5 đoạn 20 frame (TrackMate ổn định ~0,45 px). Vượt ngưỡng 5 px là bị tính vừa FP vừa FN.
2. **Nguyên nhân:** `measurement_variance=25` (σ = 5 px) quá lớn so với độ chính xác đo thực tế (~1 px), nên bộ lọc bám chậm.
3. **Bài học về tiêu chí:** giảm `max_distance` còn 5 hoặc `measurement_variance` còn 1 (riêng lẻ) nâng F1 lên 0,95 nhưng IDF1 sụp còn 0,34–0,49, vì cổng quá chặt làm track bị cắt vụn. F1 chỉ đo vị trí, không đo danh tính, nên chọn theo F1 sẽ chọn nhầm.
4. **Lời giải cần chỉnh đồng thời:** tin phép đo (`measurement_variance` nhỏ) và cho phép hạt nhảy (`process_variance` lớn, cổng Mahalanobis rộng).

## Cấu hình được chọn (áp dụng cho TIFF trong `python/main.py`)

`measurement_variance=1.0`, `process_variance=1000`, `mahalanobis_gate=100` (video giữ 25 / 5 / 25).
Mốc "trước" tái tạo bằng `--legacy-tracker`. Mọi số dưới đây loại GT nằm ngoài khung ảnh (xem `evaluation/README.md`).

| Bộ | | F1 | IDF1 | RMSE (px) | IDSW | FM |
|---|---|---|---|---|---|---|
| C toàn bộ | trước | 0,807 | 0,797 | 2,70 | 3 | 73 |
| C toàn bộ | sau | **0,902** | **0,889** | 2,57 | 3 | 63 |
| C nửa test (không dùng để chọn) | trước → sau | 0,689 → **0,871** | 0,682 → **0,865** | | | 50 → 42 |
| A (kiểm chéo) | trước | 0,976 | 0,971 | 1,75 | 3 | 9 |
| A (kiểm chéo) | sau | 0,979 | 0,978 | **0,65** | 1 | 8 |

Nửa test cải thiện nhiều hơn nửa tune, và A không xấu đi, nên không có dấu hiệu quá khớp.

## Còn lại và hạn chế

- C vẫn thua TrackMate (F1 0,902 so với 0,978; tham chiếu của bộ dữ liệu). Sai số định vị vẫn tăng theo thời gian dù đã giảm: trung vị khoảng cách tới GT theo 5 đoạn 20 frame là 1,19 / 2,18 / 3,05 / 3,50 / 2,72 px (trước khi sửa: 1,30 / 2,25 / 3,22 / 4,06 / 4,57; TrackMate ~0,45). Nguyên nhân còn lại chưa được kiểm chứng; giả thuyết: tâm vùng sau ngưỡng lệch khi hạt mờ dần.
- Lưu ý khi đọc nhận xét cũ: recall ~0,48 của ảnh A **không** do nhiễu mà do 51% điểm GT nằm ngoài khung ảnh (A: 977/2000 điểm trong khung). Sau khi loại các điểm đó, recall của Python là 0,963 và của TrackMate 0,982.
- Vì bộ lọc Kalman từng làm RMSE của A là 1,75 px, phần nhận xét "TrackMate định vị tốt hơn do DoG" chỉ đúng một phần: sau khi sửa, Python trên A đạt RMSE 0,65 px, ngang TrackMate (0,61).
- Tuning dùng chính GT của C (có chia tune/test nhưng cùng một video); A là bộ kiểm độc lập duy nhất.
- Video1 không bị ảnh hưởng (nhánh video giữ nguyên tham số).

## Các thí nghiệm khác

- `ablation_video1.py`: đóng góp của gộp vùng và phục hồi ID trên video1 (`results/video1_ablation.csv`).
- `compare_association.py`: Nearest và Hungarian có chọn phép ghép khác nhau không (A: 0/100 bước, C: 0/100, video1: 1/397).
