# Evaluation summary

## video1  (matching: iou)

| Method | Precision | Recall | F1 | RMSE(px) | MOTA | MOTP | IDF1 | IDSW | FM | MT | PT | ML | GT tracks | Pred tracks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Python-Nearest | 0.519 | 0.482 | 0.500 | 11.24 | 0.020 | 0.596 | 0.342 | 21 | 41 | 1 | 3 | 1 | 5 | 38 |
| Python-Hungarian | 0.519 | 0.482 | 0.500 | 11.65 | 0.022 | 0.590 | 0.378 | 19 | 41 | 1 | 3 | 1 | 5 | 38 |

## track_2D_A_0000  (matching: distance)

| Method | Precision | Recall | F1 | RMSE(px) | MOTA | MOTP | IDF1 | IDSW | FM | MT | PT | ML | GT tracks | Pred tracks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Python-Nearest | 0.994 | 0.470 | 0.639 | 1.75 | 0.466 | 1.153 | 0.635 | 3 | 9 | 5 | 8 | 7 | 20 | 20 |
| Python-Hungarian | 0.994 | 0.470 | 0.639 | 1.75 | 0.466 | 1.153 | 0.635 | 3 | 9 | 5 | 8 | 7 | 20 | 20 |
| Fiji-TrackMate | 1.000 | 0.481 | 0.650 | 0.62 | 0.480 | 0.540 | 0.590 | 3 | 4 | 6 | 8 | 6 | 20 | 23 |
| TrackMate-dataset-XML | 1.000 | 0.478 | 0.647 | 0.59 | 0.477 | 0.476 | 0.561 | 4 | 5 | 6 | 8 | 6 | 20 | 22 |

## track_2D_C_0002  (matching: distance)

| Method | Precision | Recall | F1 | RMSE(px) | MOTA | MOTP | IDF1 | IDSW | FM | MT | PT | ML | GT tracks | Pred tracks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Python-Nearest | 0.829 | 0.754 | 0.790 | 2.70 | 0.597 | 2.366 | 0.781 | 3 | 73 | 9 | 10 | 1 | 20 | 21 |
| Python-Hungarian | 0.829 | 0.754 | 0.790 | 2.70 | 0.597 | 2.366 | 0.781 | 3 | 73 | 9 | 10 | 1 | 20 | 21 |
| TrackMate-dataset-XML | 1.000 | 0.918 | 0.957 | 0.69 | 0.916 | 0.516 | 0.947 | 3 | 3 | 17 | 2 | 1 | 20 | 23 |
