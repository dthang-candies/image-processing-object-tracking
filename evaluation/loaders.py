"""Readers that turn every result/ground-truth format into one ``TrackSet``.

Supported formats
-----------------
``gt-csv``        ``TrackID,x,y,frame`` (point ground truth, pixels) - data3/data4
``caviar``        CAVIAR XML ground truth with boxes - data1
``python``        ``trajectories_*.csv`` written by ``python/object_tracking/export.py``
``trackmate-csv`` Fiji TrackMate "Export to CSV" (``*_spots.csv`` / ``*_all_spots.csv``)
``trackmate-xml`` Fiji TrackMate session file (``*.xml``)

Every loader returns coordinates in **pixels**, frames **0-based**.
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMATS = ("auto", "gt-csv", "caviar", "python", "trackmate-csv", "trackmate-xml")


@dataclass
class TrackSet:
    """Observations ``(frame, track id, x, y, [w, h])`` of one result or GT.

    ``w``/``h`` are NaN when the source only provides points.
    """

    name: str
    frame: np.ndarray
    track_id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    h: np.ndarray

    @classmethod
    def from_rows(cls, name: str, rows: list[tuple]) -> "TrackSet":
        """Build from ``(frame, id, x, y)`` or ``(frame, id, x, y, w, h)`` rows."""
        if not rows:
            empty = np.empty(0)
            return cls(name, empty.astype(int), empty.astype(int), empty, empty, empty, empty)
        padded = [r if len(r) == 6 else (*r, np.nan, np.nan) for r in rows]
        arr = np.array(padded, dtype=float)
        order = np.lexsort((arr[:, 1], arr[:, 0]))
        arr = arr[order]
        return cls(
            name,
            arr[:, 0].astype(int),
            arr[:, 1].astype(int),
            arr[:, 2],
            arr[:, 3],
            arr[:, 4],
            arr[:, 5],
        )

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def has_boxes(self) -> bool:
        return len(self) > 0 and bool(np.isfinite(self.w).all() and np.isfinite(self.h).all())

    @property
    def frame_range(self) -> tuple[int, int]:
        return (int(self.frame.min()), int(self.frame.max())) if len(self) else (0, -1)

    def restrict_frames(self, first: int, last: int) -> "TrackSet":
        keep = (self.frame >= first) & (self.frame <= last)
        return TrackSet(
            self.name,
            self.frame[keep],
            self.track_id[keep],
            self.x[keep],
            self.y[keep],
            self.w[keep],
            self.h[keep],
        )

    def restrict_to_frame(self, width: int, height: int) -> "TrackSet":
        """Keep only points whose centre lies inside a ``width`` x ``height`` image.

        Pixel centres run from 0 to ``width - 1``; the image edge is at -0.5 and
        ``width - 0.5``. Ground-truth objects outside the field of view cannot be
        observed by any method and would otherwise count as missed detections.
        """
        keep = (self.x >= -0.5) & (self.x <= width - 0.5) & (self.y >= -0.5) & (self.y <= height - 0.5)
        return TrackSet(self.name, self.frame[keep], self.track_id[keep], self.x[keep],
                        self.y[keep], self.w[keep], self.h[keep])

    def shifted(self, dx: float, dy: float) -> "TrackSet":
        return TrackSet(self.name, self.frame, self.track_id, self.x + dx, self.y + dy, self.w, self.h)

    def by_frame(self) -> dict[int, np.ndarray]:
        """Return ``frame -> row indices``."""
        groups: dict[int, list[int]] = {}
        for index, frame in enumerate(self.frame):
            groups.setdefault(int(frame), []).append(index)
        return {f: np.array(i) for f, i in groups.items()}


# ----------------------------------------------------------------- GT readers
def load_gt_csv(path: Path) -> TrackSet:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for record in csv.DictReader(stream):
            rows.append(
                (
                    int(float(record["frame"])),
                    int(float(record["TrackID"])),
                    float(record["x"]),
                    float(record["y"]),
                )
            )
    return TrackSet.from_rows(path.stem, rows)


def load_caviar_xml(path: Path) -> TrackSet:
    """CAVIAR annotation: ``<box h w xc yc>`` is the box centre and size."""
    rows = []
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "frame":
            continue
        frame = int(element.attrib["number"])
        for obj in element.iter("object"):
            box = obj.find("box")
            if box is None:
                continue
            rows.append(
                (
                    frame,
                    int(obj.attrib["id"]),
                    float(box.attrib["xc"]),
                    float(box.attrib["yc"]),
                    float(box.attrib["w"]),
                    float(box.attrib["h"]),
                )
            )
        element.clear()
    return TrackSet.from_rows(path.stem, rows)


# ------------------------------------------------------------- Python results
def load_python_csv(path: Path, *, include_predicted: bool = False) -> TrackSet:
    """Load ``trajectories_*.csv``.

    Rows with ``matched=0`` are Kalman predictions made while the object was
    not detected; they are skipped unless ``include_predicted`` is set.
    """
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            if not include_predicted and record.get("matched", "1") in ("0", "False"):
                continue
            rows.append(
                (
                    int(record["frame"]),
                    int(record["track_id"]),
                    float(record["x"]),
                    float(record["y"]),
                    float(record["bbox_width"]),
                    float(record["bbox_height"]),
                )
            )
    return TrackSet.from_rows(path.stem, rows)


# ------------------------------------------------------------ TrackMate (Fiji)
def _pixel_size_from_xml(path: Path) -> float | None:
    try:
        for _, element in ET.iterparse(path, events=("start",)):
            if element.tag == "ImageData":
                return float(element.attrib["pixelwidth"])
    except (ET.ParseError, KeyError, OSError):
        pass
    return None


def _guess_pixel_size(csv_path: Path) -> float | None:
    """Look for a TrackMate session XML next to the CSV that has ``pixelwidth``."""
    stem = csv_path.stem
    for suffix in ("_spots", "_all_spots", "_tracks", "_edges"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    for candidate in (
        csv_path.with_name(f"{stem}_TrackMate.xml"),
        csv_path.with_name(f"{stem}.xml"),
        *sorted(csv_path.parent.glob("*.xml")),
    ):
        if candidate.exists():
            size = _pixel_size_from_xml(candidate)
            if size:
                return size
    return None


def load_trackmate_csv(
    path: Path,
    *,
    pixel_size: float | None = None,
    keep_untracked: bool = False,
) -> TrackSet:
    """TrackMate "Export to CSV".

    The file has 4 header rows (names, long names, short names, units). If the
    unit row says ``µm`` the coordinates are divided by ``pixel_size``
    (auto-read from a sibling TrackMate XML when not given).
    """
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        index = {name: i for i, name in enumerate(header)}
        data_rows = list(reader)
    units = next((r for r in data_rows[:4] if r and r[0] == ""), [])
    in_pixels = bool(units) and "pixel" in units[index["POSITION_X"]].lower()
    scale = 1.0
    if not in_pixels:
        pixel_size = pixel_size or _guess_pixel_size(path)
        if not pixel_size:
            raise ValueError(
                f"{path.name}: coordinates are in physical units; pass --pixel-size "
                "(e.g. 0.04) or keep the TrackMate XML next to the CSV."
            )
        scale = 1.0 / pixel_size
    rows, next_singleton = [], -1
    for record in data_rows:
        if len(record) <= index["POSITION_X"]:
            continue
        try:
            frame = int(float(record[index["FRAME"]]))
            x = float(record[index["POSITION_X"]])
            y = float(record[index["POSITION_Y"]])
        except ValueError:
            continue  # the three descriptive header rows
        track_text = record[index["TRACK_ID"]].strip()
        if track_text:
            track_id = int(float(track_text))
        elif keep_untracked:
            track_id, next_singleton = next_singleton, next_singleton - 1
        else:
            continue
        rows.append((frame, track_id, x * scale, y * scale))
    return TrackSet.from_rows(path.stem, rows)


def load_trackmate_xml(path: Path, *, keep_untracked: bool = False) -> TrackSet:
    """TrackMate session: spots joined to tracks through the edge list."""
    pixel_size = _pixel_size_from_xml(path)
    spots: dict[int, tuple[int, float, float]] = {}
    spot_track: dict[int, int] = {}
    kept_tracks: set[int] | None = None
    current_track: int | None = None
    section_filtered = False
    for event, element in ET.iterparse(path, events=("start", "end")):
        tag = element.tag
        if event == "start":
            if tag == "FilteredTracks":
                section_filtered, kept_tracks = True, set()
            elif tag == "Track":
                current_track = int(element.attrib["TRACK_ID"])
        else:
            if tag == "Spot":
                spots[int(element.attrib["ID"])] = (
                    int(float(element.attrib["FRAME"])),
                    float(element.attrib["POSITION_X"]),
                    float(element.attrib["POSITION_Y"]),
                )
            elif tag == "Edge" and current_track is not None:
                spot_track[int(element.attrib["SPOT_SOURCE_ID"])] = current_track
                spot_track[int(element.attrib["SPOT_TARGET_ID"])] = current_track
            elif tag == "Track":
                current_track = None
            elif tag == "TrackID" and section_filtered and kept_tracks is not None:
                kept_tracks.add(int(element.attrib["TRACK_ID"]))
    scale = 1.0 / pixel_size if pixel_size else 1.0
    rows, next_singleton = [], -1
    for spot_id, (frame, x, y) in spots.items():
        track_id = spot_track.get(spot_id)
        if track_id is not None and kept_tracks is not None and track_id not in kept_tracks:
            track_id = None  # track removed by the track filters
        if track_id is None:
            if not keep_untracked:
                continue
            track_id, next_singleton = next_singleton, next_singleton - 1
        rows.append((frame, track_id, x * scale, y * scale))
    return TrackSet.from_rows(path.stem, rows)


# ----------------------------------------------------------------- dispatcher
def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".xml":
        with path.open("rb") as stream:
            head = stream.read(600).decode("utf-8", errors="ignore")
        return "trackmate-xml" if "<TrackMate" in head else "caviar"
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig") as stream:
            first = stream.readline().strip().split(",")
        if first[:2] == ["TrackID", "x"]:
            return "gt-csv"
        if "track_id" in first and "bbox_width" in first:
            return "python"
        if "POSITION_X" in first:
            return "trackmate-csv"
    raise ValueError(f"Cannot detect the format of {path}; pass an explicit format.")


def load_tracks(
    path: Path | str,
    fmt: str = "auto",
    *,
    pixel_size: float | None = None,
    include_predicted: bool = False,
    keep_untracked: bool = False,
) -> TrackSet:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    fmt = detect_format(path) if fmt == "auto" else fmt
    if fmt == "gt-csv":
        return load_gt_csv(path)
    if fmt == "caviar":
        return load_caviar_xml(path)
    if fmt == "python":
        return load_python_csv(path, include_predicted=include_predicted)
    if fmt == "trackmate-csv":
        return load_trackmate_csv(path, pixel_size=pixel_size, keep_untracked=keep_untracked)
    if fmt == "trackmate-xml":
        return load_trackmate_xml(path, keep_untracked=keep_untracked)
    raise ValueError(f"Unknown format {fmt!r}; choose from {FORMATS}")
