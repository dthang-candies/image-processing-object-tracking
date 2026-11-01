from pathlib import Path

import pytest

from loaders import detect_format, load_tracks

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
