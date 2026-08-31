from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_ROOT = Path(
    os.environ.get(
        "EXTERNAL_PROFILE_IMAGE_ROOT",
        Path(__file__).with_name("external_profile_source_images"),
    )
)


@dataclass(frozen=True)
class PlotCalibration:
    image: str
    x_left: int
    x_right: int
    y_top: int
    y_bottom: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def x_value(self, pixel_x: float) -> float:
        return self.x_min + (pixel_x - self.x_left) * (self.x_max - self.x_min) / (
            self.x_right - self.x_left
        )

    def y_pixel(self, value: float) -> float:
        return self.y_top + (value - self.y_min) * (self.y_bottom - self.y_top) / (
            self.y_max - self.y_min
        )


def colour_mask(rgb: np.ndarray, colour: str) -> np.ndarray:
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    if colour == "red":
        return (red > 175) & (green < 145) & (blue < 145) & ((red - green) > 65)
    if colour == "blue":
        return (blue > 110) & (red < 145) & (green < 190) & ((blue - red) > 35)
    if colour == "black":
        return (red < 65) & (green < 65) & (blue < 65)
    if colour == "orange":
        return (red > 170) & (green > 75) & (green < 190) & (blue < 120)
    if colour == "green":
        return (green > 90) & (red < 180) & (blue < 150) & ((green - red) > 5)
    raise ValueError(colour)


def marker_x(
    calibration: PlotCalibration,
    depth: float,
    colour: str,
    expected_x: float | None = None,
    y_half_window: int = 16,
    x_half_window: int = 45,
) -> tuple[float, float, int]:
    image = np.asarray(Image.open(IMAGE_ROOT / calibration.image).convert("RGB"))
    mask = colour_mask(image, colour)
    y_centre = int(round(calibration.y_pixel(depth)))
    y0 = max(calibration.y_top + 1, y_centre - y_half_window)
    y1 = min(calibration.y_bottom - 1, y_centre + y_half_window)
    x0, x1 = calibration.x_left + 2, calibration.x_right - 2
    if expected_x is not None:
        expected_pixel = calibration.x_left + (
            (expected_x - calibration.x_min)
            * (calibration.x_right - calibration.x_left)
            / (calibration.x_max - calibration.x_min)
        )
        x0 = max(x0, int(expected_pixel - x_half_window))
        x1 = min(x1, int(expected_pixel + x_half_window))
    local = mask[y0 : y1 + 1, x0 : x1 + 1]
    if not local.any():
        raise ValueError(
            f"No {colour} pixels for {calibration.image}, depth={depth}, expected_x={expected_x}"
        )
    column_counts = local.sum(axis=0)
    peak_x = int(np.argmax(column_counts)) + x0
    focus0, focus1 = max(x0, peak_x - 18), min(x1, peak_x + 18)
    focused = mask[y0 : y1 + 1, focus0 : focus1 + 1]
    yy, xx = np.where(focused)
    pixel_x = float(np.mean(xx + focus0))
    spread = float(np.std(xx + focus0))
    return calibration.x_value(pixel_x), spread, int(len(xx))


def trace_series(
    calibration: PlotCalibration,
    source: str,
    case: str,
    colour: str,
    first_row: int,
    last_row: int,
    initial_pixel_x: float,
    physical_depth_unit: str,
    physical_depth_max: float,
    samples: int = 101,
) -> pd.DataFrame:
    image = np.asarray(Image.open(IMAGE_ROOT / calibration.image).convert("RGB"))
    mask = colour_mask(image, colour)
    tracked_y: list[int] = []
    tracked_x: list[float] = []
    previous = initial_pixel_x
    for pixel_y in range(first_row, last_row + 1):
        candidates = np.flatnonzero(mask[pixel_y, calibration.x_left + 2 : calibration.x_right - 1])
        candidates = candidates + calibration.x_left + 2
        nearby = candidates[np.abs(candidates - previous) <= 20]
        if len(nearby):
            current = float(np.median(nearby))
            tracked_y.append(pixel_y)
            tracked_x.append(current)
            previous = current
    if len(tracked_y) < 0.45 * (last_row - first_row + 1):
        raise ValueError(f"Insufficient trace coverage for {case}: {len(tracked_y)} rows")
    target_y = np.linspace(first_row, last_row, samples)
    pixel_x = np.interp(target_y, np.asarray(tracked_y), np.asarray(tracked_x))
    load = np.asarray([calibration.x_value(value) for value in pixel_x])
    normalised_depth = (target_y - first_row) / (last_row - first_row)
    return pd.DataFrame(
        {
            "source": source,
            "case": case,
            "figure_image": calibration.image,
            "normalised_depth": normalised_depth,
            "physical_depth": normalised_depth * physical_depth_max,
            "depth_unit": physical_depth_unit,
            "load": load,
            "load_unit": "kips",
            "digitisation_kind": "published measured trace",
        }
    )


def extract_series(
    calibration: PlotCalibration,
    source: str,
    case: str,
    colour: str,
    depths: list[float],
    expected_loads: list[float],
    units: str = "kips",
    depth_unit: str = "ft",
) -> pd.DataFrame:
    records = []
    for depth, expected in zip(depths, expected_loads, strict=True):
        value, pixel_spread, pixels = marker_x(
            calibration, depth, colour, expected_x=expected
        )
        records.append(
            {
                "source": source,
                "case": case,
                "figure_image": calibration.image,
                "depth": depth,
                "depth_unit": depth_unit,
                "load": value,
                "load_unit": units,
                "marker_colour": colour,
                "digitised_pixel_x_sd": pixel_spread,
                "digitised_pixels": pixels,
            }
        )
    return pd.DataFrame(records)


def build_result() -> pd.DataFrame:
    combined = PlotCalibration(
        "byu_p158.png", 906, 1642, 1102, 2593, 0.0, 250.0, 0.0, 90.0
    )
    h_individual = PlotCalibration(
        "byu_p123.png", 849, 1724, 469, 1969, 0.0, 200.0, 0.0, 100.0
    )
    pipe_individual = PlotCalibration(
        "byu_p138.png", 894, 1665, 443, 1722, 0.0, 300.0, 0.0, 80.0
    )
    concrete_individual = PlotCalibration(
        "byu_p152.png", 886, 1657, 787, 2082, 0.0, 200.0, 0.0, 80.0
    )

    frames = [
        extract_series(
            combined,
            "Kevan (2017) BYU thesis",
            "Turrell H-pile post-blast",
            "blue",
            [0, 4, 16, 25, 32, 36, 55, 62, 72, 85],
            [116, 119, 147, 156, 159, 160, 173, 159, 113, 64],
        ),
        extract_series(
            combined,
            "Kevan (2017) BYU thesis",
            "Turrell pipe pile post-blast",
            "red",
            [0, 4, 20, 40, 52, 60, 76],
            [118, 132, 168, 190, 194, 210, 132],
        ),
        extract_series(
            combined,
            "Kevan (2017) BYU thesis",
            "Turrell concrete pile post-blast",
            "black",
            [0, 4, 12, 20, 28, 36, 52, 60, 68, 72],
            [116, 119, 123, 127, 127, 144, 158, 170, 148, 112],
        ),
        extract_series(
            h_individual,
            "Kevan (2017) BYU thesis",
            "Turrell H-pile post-blast QA",
            "red",
            [0, 16, 25, 32, 36, 55, 62, 72, 85],
            [118, 148, 157, 160, 161, 174, 160, 114, 64],
        ),
        extract_series(
            pipe_individual,
            "Kevan (2017) BYU thesis",
            "Turrell pipe pile post-blast QA",
            "red",
            [0, 4, 20, 40, 52, 60, 76],
            [118, 132, 168, 190, 194, 210, 132],
        ),
        extract_series(
            concrete_individual,
            "Kevan (2017) BYU thesis",
            "Turrell concrete pile post-blast QA",
            "black",
            [0, 4, 12, 20, 28, 36, 52, 60, 68, 72],
            [116, 119, 123, 127, 127, 144, 158, 170, 148, 112],
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def build_fhwa_traces() -> pd.DataFrame:
    twelve_metre = PlotCalibration(
        "fhwa_p10.png", 588, 1214, 622, 1972, 0.0, 300.0, 0.0, 50.0
    )
    fourteen_metre = PlotCalibration(
        "fhwa_p10.png", 1315, 1968, 622, 1972, 0.0, 300.0, 0.0, 14.0
    )
    return pd.concat(
        [
            trace_series(
                twelve_metre,
                "FHWA-HRT-17-060",
                "Christchurch 12-m CFA pile post-blast",
                "orange",
                625,
                1484,
                1085.5,
                "m",
                9.7,
            ),
            trace_series(
                fourteen_metre,
                "FHWA-HRT-17-060",
                "Christchurch 14-m CFA pile post-blast",
                "green",
                622,
                1823,
                1879.0,
                "m",
                12.5,
            ),
        ],
        ignore_index=True,
    )


def build_cambridge_profile() -> pd.DataFrame:
    calibration = PlotCalibration(
        "cam_p200.png", 355, 1106, 1801, 2150, 0.0, 500.0, 0.0, 10.0
    )
    return extract_series(
        calibration,
        "Stringer (2012) Cambridge thesis",
        "MS09 bored pile, Leg 1, 2500 s after shaking",
        "black",
        [0.0, 2.2, 4.5, 6.7, 8.6],
        [330, 330, 335, 340, 375],
        units="kN",
        depth_unit="m",
    )


def main() -> None:
    print(build_result().to_csv(index=False))


if __name__ == "__main__":
    main()
