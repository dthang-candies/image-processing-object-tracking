from pathlib import Path

import pytest

from loaders import TrackSet, detect_format, load_tracks

ROOT = Path(__file__).resolve().parents[2]


def test_gt_csv_and_dataset_xml():
    gt = load_tracks(ROOT / "dataset/data3/track_2D_A_0000.csv")
    assert len(gt) == 2000 and len(set(gt.track_id)) == 20
    xml = load_tracks(ROOT / "dataset/data3/track_2D_A_0000.tif.xml")
    assert xml.x.max() <= 350 and len(xml) > 900  # converted from um to pixels


def test_caviar_boxes():
    gt = load_tracks(ROOT / "dataset/data1/ground_truth_video1.xml")
    assert gt.has_boxes and gt.frame_range == (0, 382)


def test_trackmate_csv_scales_um_to_pixels():
    path = ROOT / "Báo cáo XLA/TrackMate_GUI_A/results/track_2D_A_0000_spots.csv"
    if not path.exists():
        pytest.skip("Fiji export not present")
    tracks = load_tracks(path)
    assert detect_format(path) == "trackmate-csv" and 0 <= tracks.x.max() <= 350


def test_restrict_to_frame_drops_points_outside_the_image():
    ts = TrackSet.from_rows("gt", [(0, 1, 10.0, 10.0), (0, 2, -3.0, 5.0), (1, 2, 350.0, 5.0), (1, 3, 349.4, 0.0)])
    inside = ts.restrict_to_frame(350, 350)
    assert sorted(zip(inside.track_id.tolist(), inside.x.tolist())) == [(1, 10.0), (3, 349.4)]
