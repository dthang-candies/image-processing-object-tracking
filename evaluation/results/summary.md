# Evaluation summary

## video1  (matching: iou)

| Method | Precision | Recall | F1 | RMSE(px) | MOTA | MOTP | IDF1 | IDSW | FM | MT | PT | ML | GT tracks | Pred tracks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Python-Nearest | 0.519 | 0.482 | 0.500 | 11.24 | 0.020 | 0.596 | 0.342 | 21 | 41 | 1 | 3 | 1 | 5 | 38 |
| Python-Hungarian | 0.519 | 0.482 | 0.500 | 11.65 | 0.022 | 0.590 | 0.378 | 19 | 41 | 1 | 3 | 1 | 5 | 38 |

## track_2D_A_0000  (matching: distance)

| Method | Precision | Recall | F1 | RMSE(px) | MOTA | MOTP | IDF1 | IDSW | FM | MT | PT | ML | GT tracks | Pred tracks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Python-Nearest | 0.995 | 0.963 | 0.979 | 0.65 | 0.957 | 0.445 | 0.978 | 1 | 8 | 18 | 1 | 1 | 20 | 19 |
| Python-Hungarian | 0.995 | 0.963 | 0.979 | 0.65 | 0.957 | 0.445 | 0.978 | 1 | 8 | 18 | 1 | 1 | 20 | 19 |
| Fiji-TrackMate | 0.996 | 0.982 | 0.989 | 0.61 | 0.974 | 0.538 | 0.897 | 3 | 4 | 20 | 0 | 0 | 20 | 23 |
| TrackMate-dataset-XML | 0.996 | 0.975 | 0.986 | 0.59 | 0.967 | 0.474 | 0.854 | 4 | 5 | 20 | 0 | 0 | 20 | 22 |

## track_2D_C_0002  (matching: distance)

| Method | Precision | Recall | F1 | RMSE(px) | MOTA | MOTP | IDF1 | IDSW | FM | MT | PT | ML | GT tracks | Pred tracks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Python-Nearest | 0.927 | 0.879 | 0.902 | 2.57 | 0.808 | 2.239 | 0.889 | 3 | 63 | 17 | 3 | 0 | 20 | 21 |
| Python-Hungarian | 0.927 | 0.879 | 0.902 | 2.57 | 0.808 | 2.239 | 0.889 | 3 | 63 | 17 | 3 | 0 | 20 | 21 |
| TrackMate-dataset-XML | 1.000 | 0.956 | 0.978 | 0.69 | 0.955 | 0.516 | 0.967 | 3 | 3 | 18 | 2 | 0 | 20 | 23 |
