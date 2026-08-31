from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PRJ2828_DATA_ROOT", OUT / "source_data"))
FULL_GAGES = [f"Gage{i}[MN]" for i in range(8, 0, -1)]
Z_FULL = np.linspace(0.0, 1.0, 8)
PRESENT_INDICES = np.array([0, 1, 2, 3, 5, 7], dtype=int)
CANDIDATE = (0, 1, 3)  # selected from SKS02 for two added stations
UNIFORM = (0, 3, 7)


def event_number(path: Path) -> str:
    match = re.search(r"EQM_\d+", path.name)
    if match is None:
        raise ValueError(path)
    return match.group(0)


def load_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skipinitialspace=True)
    frame = frame.rename(columns={c: str(c).replace("Gage_", "Gage") for c in frame.columns})
    time_col = next(c for c in frame.columns if str(c).strip().endswith("Time[min]"))
    present = [FULL_GAGES[i] for i in PRESENT_INDICES]
    frame = frame[[time_col, *present]].apply(pd.to_numeric, errors="coerce")
    t = frame[time_col].to_numpy(float)
    dt_s = float(np.nanmedian(np.diff(t)) * 60.0)
    window = max(1, int(round(6.0 / dt_s))) if dt_s > 0 else 1
    values = frame[present].rolling(window, center=True, min_periods=max(1, window // 3)).mean()
    valid = np.isfinite(t) & np.isfinite(values.to_numpy(float)).all(axis=1)
    valid_idx = np.flatnonzero(valid)
    selected = valid_idx[
        np.unique(np.linspace(0, len(valid_idx) - 1, min(120, len(valid_idx))).round().astype(int))
    ]
    return values.iloc[selected].reset_index(drop=True)


def score(y_present: np.ndarray, sensors: tuple[int, ...]) -> float:
    position_to_value = dict(zip(PRESENT_INDICES.tolist(), y_present.tolist()))
    sensor_values = np.array([position_to_value[i] for i in sensors], dtype=float)
    prediction = np.interp(Z_FULL[PRESENT_INDICES], Z_FULL[list(sensors)], sensor_values)
    scale = max(float(np.max(np.abs(y_present))), 0.05)
    return float(np.sqrt(np.mean((prediction - y_present) ** 2)) / scale)


def main() -> None:
    rows: list[dict] = []
    paths = sorted(
        path
        for path in DATA_ROOT.rglob("EQM_*_Pile_1_Axial_Load.txt")
        if "SKS03" in path.parts
    )
    if len(paths) != 5:
        raise RuntimeError(f"Expected five SKS03 Pile 1 earthquake files; found {len(paths)}")
    for path in paths:
        event = event_number(path)
        values = load_file(path).to_numpy(float)
        candidate_scores = np.array([score(y, CANDIDATE) for y in values])
        uniform_scores = np.array([score(y, UNIFORM) for y in values])
        candidate_p90 = float(np.quantile(candidate_scores, 0.90))
        uniform_p90 = float(np.quantile(uniform_scores, 0.90))
        rows.append(
            {
                "test": "SKS03",
                "pile": 1,
                "event": event,
                "available_reference_gages_top_to_bottom": "0;1;2;3;5;7",
                "available_reference_normalised_depths": ";".join(f"{Z_FULL[i]:.3f}" for i in PRESENT_INDICES),
                "candidate_gage_indices": ";".join(map(str, CANDIDATE)),
                "uniform_gage_indices": ";".join(map(str, UNIFORM)),
                "sampled_snapshots": len(values),
                "candidate_p90_nrmse": candidate_p90,
                "uniform_p90_nrmse": uniform_p90,
                "candidate_minus_uniform": candidate_p90 - uniform_p90,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "prj2828_sks03_pile1_partial_grid_results.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "test": "SKS03",
                "pile": 1,
                "earthquake_events": len(result),
                "reference_gages": len(PRESENT_INDICES),
                "added_stations": 2,
                "median_candidate_minus_uniform": float(result["candidate_minus_uniform"].median()),
                "minimum_candidate_minus_uniform": float(result["candidate_minus_uniform"].min()),
                "maximum_candidate_minus_uniform": float(result["candidate_minus_uniform"].max()),
                "fraction_events_candidate_better": float(np.mean(result["candidate_minus_uniform"] < 0)),
                "claim_limit": "Error is computed only at six actually measured gage levels; missing Gage 4 and Gage 2 values are not imputed.",
            }
        ]
    )
    summary.to_csv(OUT / "prj2828_sks03_pile1_partial_grid_summary.csv", index=False)
    print(result.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
