from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from transfer_audit import (
    GAGE_COLS_TOP_TO_BOTTOM,
    Z,
    choose_robust_layout,
    load_static_layouts,
    prediction,
    profile_metrics,
    uniform_layout,
)


ROOT = Path(__file__).resolve().parent


def event_number(value: str) -> int:
    match = re.search(r"(\d+)$", value)
    if match is None:
        raise ValueError(value)
    return int(match.group(1))


def event_p90(profiles: pd.DataFrame, indices: tuple[int, ...]) -> tuple[float, int]:
    pile_values: list[float] = []
    for _, pile in profiles.groupby("pile"):
        errors = []
        for y in pile[GAGE_COLS_TOP_TO_BOTTOM].to_numpy(float):
            errors.append(profile_metrics(y, prediction(y, indices))["nrmse"])
        pile_values.append(float(np.quantile(errors, 0.90)))
    return float(np.median(pile_values)), len(pile_values)


def main() -> None:
    profiles = pd.read_csv(ROOT / "prj2828_eqm_benchmark_profiles.csv")
    static = load_static_layouts()
    rows: list[dict] = []
    layout_rows: list[dict] = []
    for test, test_profiles in profiles.groupby("test"):
        events = sorted(test_profiles["event"].unique(), key=event_number)
        first_event = events[0]
        for budget in (2, 3, 4):
            first_frozen = choose_robust_layout(
                test_profiles[test_profiles["event"].eq(first_event)], budget
            )
            for evaluation_event in events[1:]:
                preceding = [event for event in events if event_number(event) < event_number(evaluation_event)]
                expanding = choose_robust_layout(
                    test_profiles[test_profiles["event"].isin(preceding)], budget
                )
                evaluation = test_profiles[test_profiles["event"].eq(evaluation_event)]
                uniform_error, piles = event_p90(evaluation, uniform_layout(budget))
                candidates = {
                    "first_event_frozen": first_frozen,
                    "expanding_history": expanding,
                    "static_PRJ4809": static[budget],
                }
                for strategy, layout in candidates.items():
                    candidate_error, _ = event_p90(evaluation, layout)
                    rows.append(
                        {
                            "test": test,
                            "training_events": ";".join(preceding if strategy == "expanding_history" else ([first_event] if strategy == "first_event_frozen" else ["PRJ4809"])),
                            "evaluation_event": evaluation_event,
                            "additional_stations_b": budget,
                            "strategy": strategy,
                            "piles_aggregated": piles,
                            "selected_gage_indices_top_to_bottom": ";".join(map(str, layout)),
                            "selected_normalised_depths": ";".join(f"{Z[i]:.3f}" for i in layout),
                            "candidate_event_p90_nrmse": candidate_error,
                            "uniform_event_p90_nrmse": uniform_error,
                            "candidate_minus_uniform": candidate_error - uniform_error,
                        }
                    )
                layout_rows.extend(
                    [
                        {
                            "test": test,
                            "evaluation_event": evaluation_event,
                            "additional_stations_b": budget,
                            "strategy": "first_event_frozen",
                            "training_events": first_event,
                            "layout": ";".join(f"{Z[i]:.3f}" for i in first_frozen),
                        },
                        {
                            "test": test,
                            "evaluation_event": evaluation_event,
                            "additional_stations_b": budget,
                            "strategy": "expanding_history",
                            "training_events": ";".join(preceding),
                            "layout": ";".join(f"{Z[i]:.3f}" for i in expanding),
                        },
                    ]
                )
    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["test", "additional_stations_b", "strategy"], as_index=False)
        .agg(
            future_earthquakes=("evaluation_event", "nunique"),
            median_candidate_minus_uniform=("candidate_minus_uniform", "median"),
            worst_candidate_minus_uniform=("candidate_minus_uniform", "max"),
            best_candidate_minus_uniform=("candidate_minus_uniform", "min"),
            fraction_future_events_better=("candidate_minus_uniform", lambda x: (x < 0).mean()),
        )
        .sort_values(["additional_stations_b", "test", "median_candidate_minus_uniform"])
    )
    results.to_csv(ROOT / "prj2828_forward_sequence_results.csv", index=False)
    summary.to_csv(ROOT / "prj2828_forward_sequence_summary.csv", index=False)
    pd.DataFrame(layout_rows).drop_duplicates().to_csv(
        ROOT / "prj2828_forward_sequence_layouts.csv", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
