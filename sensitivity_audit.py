from __future__ import annotations

import itertools
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PRJ2828_DATA_ROOT", OUT / "source_data"))
Z = np.linspace(0.0, 1.0, 8)
GAGES = [f"Gage{i}[MN]" for i in range(8, 0, -1)]
MAX_SNAPSHOTS = 120
FLOORS_MN = (0.025, 0.05, 0.10)
SMOOTHING = {
    "none": 0.0,
    "half_reported": 0.5,
    "reported": 1.0,
    "double_reported": 2.0,
}


def identity(path: Path) -> tuple[str, str, int]:
    test = "SKS02" if "SKS02" in path.parts else "SKS03"
    event = re.search(r"EQM_\d+", path.name)
    pile = re.search(r"Pile_(\d+)_Axial_Load", path.name)
    if event is None or pile is None:
        raise ValueError(f"Cannot parse earthquake identity: {path}")
    return test, event.group(0), int(pile.group(1))


def load_benchmark(multiplier: float) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(DATA_ROOT.rglob("*_Pile_*_Axial_Load.txt")):
        if "EQM_" not in path.name:
            continue
        test, event, pile = identity(path)
        header = pd.read_csv(path, nrows=0, skipinitialspace=True)
        header = header.rename(columns={c: str(c).replace("Gage_", "Gage") for c in header.columns})
        time_col = next((c for c in header.columns if str(c).strip().endswith("Time[min]")), None)
        if time_col is None or any(c not in header.columns for c in GAGES):
            continue
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            first = stream.readline().rstrip("\r\n").split(",")
            second = stream.readline().rstrip("\r\n").split(",")
        if len(first) != len(second):
            continue
        df = pd.read_csv(path, skipinitialspace=True)
        df = df.rename(columns={c: str(c).replace("Gage_", "Gage") for c in df.columns})
        df = df[[time_col, *GAGES]].apply(pd.to_numeric, errors="coerce")
        t = df[time_col].to_numpy(float)
        values = df[GAGES]
        if multiplier > 0 and len(t) >= 3:
            dt_s = float(np.nanmedian(np.diff(t)) * 60.0)
            reported_s = 6.0 if pile == 1 else 3.0
            rows = max(1, int(round(reported_s * multiplier / dt_s))) if dt_s > 0 else 1
            if rows > 1:
                values = values.rolling(rows, center=True, min_periods=max(1, rows // 3)).mean()
        valid = np.isfinite(t) & np.isfinite(values.to_numpy(float)).all(axis=1)
        valid_idx = np.flatnonzero(valid)
        if not len(valid_idx):
            continue
        selected = valid_idx[
            np.unique(
                np.linspace(0, len(valid_idx) - 1, min(MAX_SNAPSHOTS, len(valid_idx)))
                .round()
                .astype(int)
            )
        ]
        frame = values.iloc[selected].copy()
        frame.insert(0, "time_min", t[selected])
        frame.insert(0, "pile", pile)
        frame.insert(0, "event", event)
        frame.insert(0, "test", test)
        frames.append(frame)
    if not frames:
        raise RuntimeError("No usable earthquake profiles found")
    return pd.concat(frames, ignore_index=True)


def layouts(additional: int) -> list[tuple[int, ...]]:
    return [(0, *combo) for combo in itertools.combinations(range(1, 8), additional)]


def uniform(additional: int) -> tuple[int, ...]:
    wanted = np.linspace(0.0, 1.0, additional + 1)[1:]
    return tuple(sorted({0, *(int(np.argmin(np.abs(Z - d))) for d in wanted)}))


def nrmse(y: np.ndarray, indices: tuple[int, ...], floor_mn: float) -> float:
    yhat = np.interp(Z, Z[list(indices)], y[list(indices)])
    scale = max(float(np.max(np.abs(y))), floor_mn)
    return float(np.sqrt(np.mean((yhat - y) ** 2)) / scale)


def choose(train: pd.DataFrame, additional: int, floor_mn: float) -> tuple[int, ...]:
    best: tuple[int, ...] | None = None
    best_key = (np.inf, np.inf, np.inf)
    for candidate in layouts(additional):
        pile_event: list[float] = []
        for _, cluster in train.groupby(["event", "pile"]):
            values = [nrmse(y, candidate, floor_mn) for y in cluster[GAGES].to_numpy(float)]
            pile_event.append(float(np.quantile(values, 0.90)))
        key = (
            float(np.quantile(pile_event, 0.90)),
            float(np.median(pile_event)),
            float(np.max(pile_event)),
        )
        if key < best_key:
            best_key = key
            best = candidate
    if best is None:
        raise RuntimeError("No layout selected")
    return best


def event_results(
    evaluation: pd.DataFrame,
    candidate: tuple[int, ...],
    baseline: tuple[int, ...],
    floor_mn: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (event, pile), cluster in evaluation.groupby(["event", "pile"]):
        ys = cluster[GAGES].to_numpy(float)
        candidate_p90 = float(np.quantile([nrmse(y, candidate, floor_mn) for y in ys], 0.90))
        uniform_p90 = float(np.quantile([nrmse(y, baseline, floor_mn) for y in ys], 0.90))
        rows.append(
            {
                "event": event,
                "pile": int(pile),
                "candidate_p90_nrmse": candidate_p90,
                "uniform_p90_nrmse": uniform_p90,
                "candidate_minus_uniform": candidate_p90 - uniform_p90,
            }
        )
    pile_rows = pd.DataFrame(rows)
    return (
        pile_rows.groupby("event", as_index=False)
        .agg(
            piles_aggregated=("pile", "nunique"),
            candidate_event_p90_nrmse=("candidate_p90_nrmse", "median"),
            uniform_event_p90_nrmse=("uniform_p90_nrmse", "median"),
            candidate_minus_uniform=("candidate_minus_uniform", "median"),
        )
        .sort_values("event")
    )


def bootstrap(values: np.ndarray, seed: int, draws: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for i in range(draws):
        estimates[i] = np.median(rng.choice(values, size=len(values), replace=True))
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def main() -> None:
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    seed = 202608210
    for smoothing_name, multiplier in SMOOTHING.items():
        print(f"Loading smoothing specification: {smoothing_name}", flush=True)
        benchmark = load_benchmark(multiplier)
        for floor_mn in FLOORS_MN:
            for training_test, evaluation_test in (("SKS02", "SKS03"), ("SKS03", "SKS02")):
                train = benchmark[benchmark["test"].eq(training_test)]
                evaluation = benchmark[benchmark["test"].eq(evaluation_test)]
                for additional in (2, 3, 4):
                    selected = choose(train, additional, floor_mn)
                    by_event = event_results(evaluation, selected, uniform(additional), floor_mn)
                    delta = by_event["candidate_minus_uniform"].to_numpy(float)
                    lo, hi = bootstrap(delta, seed)
                    seed += 1
                    specification = f"{smoothing_name}; floor={floor_mn:.3f} MN"
                    for _, row in by_event.iterrows():
                        detail_rows.append(
                            {
                                "training_test": training_test,
                                "evaluation_test": evaluation_test,
                                "direction": f"{training_test} to {evaluation_test}",
                                "additional_stations_b": additional,
                                "smoothing": smoothing_name,
                                "smoothing_multiplier": multiplier,
                                "normalisation_floor_mn": floor_mn,
                                "specification": specification,
                                "selected_gage_indices_top_to_bottom": ";".join(map(str, selected)),
                                "selected_normalised_depths": ";".join(f"{Z[i]:.3f}" for i in selected),
                                **row.to_dict(),
                            }
                        )
                    summary_rows.append(
                        {
                            "training_test": training_test,
                            "evaluation_test": evaluation_test,
                            "direction": f"{training_test} to {evaluation_test}",
                            "additional_stations_b": additional,
                            "smoothing": smoothing_name,
                            "smoothing_multiplier": multiplier,
                            "normalisation_floor_mn": floor_mn,
                            "specification": specification,
                            "selected_gage_indices_top_to_bottom": ";".join(map(str, selected)),
                            "selected_normalised_depths": ";".join(f"{Z[i]:.3f}" for i in selected),
                            "earthquake_clusters": len(by_event),
                            "median_candidate_minus_uniform": float(np.median(delta)),
                            "bootstrap_95_low": lo,
                            "bootstrap_95_high": hi,
                            "fraction_events_candidate_better": float(np.mean(delta < 0)),
                        }
                    )
    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    stability = (
        summary.groupby(["direction", "additional_stations_b"], as_index=False)
        .agg(
            specifications=("specification", "size"),
            unique_selected_layouts=("selected_normalised_depths", "nunique"),
            minimum_median_delta=("median_candidate_minus_uniform", "min"),
            maximum_median_delta=("median_candidate_minus_uniform", "max"),
            fraction_specifications_median_better=("median_candidate_minus_uniform", lambda x: float(np.mean(x < 0))),
            fraction_specifications_interval_below_zero=("bootstrap_95_high", lambda x: float(np.mean(x < 0))),
            minimum_fraction_events_better=("fraction_events_candidate_better", "min"),
        )
        .sort_values(["direction", "additional_stations_b"])
    )
    detail.to_csv(OUT / "prj2828_sensitivity_event_results.csv", index=False)
    summary.to_csv(OUT / "prj2828_sensitivity_summary.csv", index=False)
    stability.to_csv(OUT / "prj2828_sensitivity_stability.csv", index=False)
    print(stability.to_string(index=False))


if __name__ == "__main__":
    main()
