from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import pipeline as p


def _downsample_points(points: list[tuple[float, float]], max_points: int = 320) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return [points[i] for i in idx]


def prepare_case(image_path: Path) -> dict:
    if not image_path.exists():
        return {"ok": False, "reason": f"image_not_found:{image_path}"}

    try:
        rgb = np.array(Image.open(image_path).convert("RGB"))
    except Exception as exc:
        return {"ok": False, "reason": f"image_open_error:{exc}"}

    gray = rgb.mean(axis=2).astype(np.uint8)
    axis_pairs = p._detect_axis_pairs(gray, max_pairs=1)
    if not axis_pairs:
        return {"ok": False, "reason": "no_axis_pair"}
    axis = axis_pairs[0]
    x0, y0, x1, y1 = axis.plot_bbox

    if x1 - x0 < 20 or y1 - y0 < 20:
        return {"ok": False, "reason": "invalid_bbox"}

    color_series = p._extract_color_series(rgb, axis.plot_bbox)
    traced_xy: tuple[list[float], list[float]] | None = None
    source = "none"

    if color_series:
        best = max(color_series, key=lambda s: len(s[0]))
        traced_xy = (best[0], best[1])
        source = "color"
    else:
        traced_xy = p._extract_dark_fallback(gray, axis.plot_bbox)
        if traced_xy is not None:
            source = "dark"

    if traced_xy is None:
        return {"ok": False, "reason": "trace_failed"}

    x_norm, y_norm = traced_xy
    if len(x_norm) < 10:
        return {"ok": False, "reason": "too_few_points"}

    px_points: list[tuple[float, float]] = []
    for xn, yn in zip(x_norm, y_norm):
        px = x0 + float(xn) * max(1.0, (x1 - x0 - 1))
        py = (y1 - 1) - float(yn) * max(1.0, (y1 - y0 - 1))
        px_points.append((float(px), float(py)))

    px_points = _downsample_points(px_points, max_points=320)

    # Calibrate to normalized [0,1]x[0,1] using detected axis rectangle.
    origin_x = float(axis.y_axis_col)
    origin_y = float(axis.x_axis_row)
    x_end_x = float(axis.x_right)
    y_top_y = float(axis.y_top)

    calib = {
        "p0": [origin_x, origin_y, 0.0, 0.0],
        "p1": [x_end_x, origin_y, 1.0, 0.0],
        "p2": [origin_x, origin_y, 0.0, 0.0],
        "p3": [x_end_x, y_top_y, 1.0, 1.0],
    }

    return {
        "ok": True,
        "image_path": str(image_path),
        "trace_source": source,
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "axis_pair_score": float(axis.score),
        "point_count": int(len(px_points)),
        "pixel_points": [[float(x), float(y)] for x, y in px_points],
        "calibration": calib,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one image case for WebPlotDigitizer automation")
    parser.add_argument("--image", required=True, help="Image path")
    args = parser.parse_args()

    report = prepare_case(Path(args.image))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
