from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import transfer_audit as ta


OUT = Path(__file__).resolve().parent
Z = ta.Z
BUDGETS = (2, 3, 4)


def load_benchmark() -> pd.DataFrame:
    samples: list[pd.DataFrame] = []
    for path in sorted(ta.DATA_ROOT.rglob("*_Pile_*_Axial_Load.txt")):
        _, sample = ta.load_profiles(path)
        if sample is not None:
            samples.append(sample)
    if not samples:
        raise FileNotFoundError(f"No usable axial-load profiles under {ta.DATA_ROOT}")
    profiles = pd.concat(samples, ignore_index=True)
    return profiles[profiles["event"].str.startswith("EQM_")].copy()


def layout_key(profiles: pd.DataFrame, indices: tuple[int, ...]) -> tuple[float, float, float]:
    pile_event_p90: list[float] = []
    for _, group in profiles.groupby(["event", "pile"]):
        scores = [
            ta.profile_metrics(y, ta.prediction(y, indices))["nrmse"]
            for y in group[ta.GAGE_COLS_TOP_TO_BOTTOM].to_numpy(float)
        ]
        pile_event_p90.append(float(np.quantile(scores, 0.90)))
    return (
        float(np.quantile(pile_event_p90, 0.90)),
        float(np.median(pile_event_p90)),
        float(np.max(pile_event_p90)),
    )


def boundary_candidates(additional: int) -> list[tuple[int, ...]]:
    return [
        (0, *combo, 7)
        for combo in itertools.combinations(range(1, 7), additional - 1)
    ]


def choose_boundary_layout(profiles: pd.DataFrame, additional: int) -> tuple[int, ...]:
    return min(boundary_candidates(additional), key=lambda layout: layout_key(profiles, layout))


def event_errors(profiles: pd.DataFrame, indices: tuple[int, ...]) -> pd.DataFrame:
    records: list[dict] = []
    for (event, pile), group in profiles.groupby(["event", "pile"]):
        scores = [
            ta.profile_metrics(y, ta.prediction(y, indices))["nrmse"]
            for y in group[ta.GAGE_COLS_TOP_TO_BOTTOM].to_numpy(float)
        ]
        records.append({"event": event, "pile": pile, "pile_event_p90": np.quantile(scores, 0.90)})
    pile_event = pd.DataFrame(records)
    return (
        pile_event.groupby("event", as_index=False)["pile_event_p90"]
        .median()
        .rename(columns={"pile_event_p90": "event_p90_nrmse"})
    )


def leave_one_event_out(benchmark: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for test in ("SKS02", "SKS03"):
        source = benchmark[benchmark["test"].eq(test)]
        events = sorted(source["event"].unique())
        for budget in BUDGETS:
            full = ta.choose_robust_layout(source, budget)
            uniform = ta.uniform_layout(budget)
            for heldout in events:
                training = source[~source["event"].eq(heldout)]
                chosen = ta.choose_robust_layout(training, budget)
                heldout_profiles = source[source["event"].eq(heldout)]
                candidate_error = float(event_errors(heldout_profiles, chosen)["event_p90_nrmse"].iloc[0])
                uniform_error = float(event_errors(heldout_profiles, uniform)["event_p90_nrmse"].iloc[0])
                full_set, chosen_set = set(full), set(chosen)
                records.append(
                    {
                        "source_test": test,
                        "additional_stations_b": budget,
                        "total_retained_stations": budget + 1,
                        "heldout_event": heldout,
                        "full_data_layout": ";".join(map(str, full)),
                        "loo_layout": ";".join(map(str, chosen)),
                        "exact_layout_match": chosen == full,
                        "jaccard_similarity": len(full_set & chosen_set) / len(full_set | chosen_set),
                        "heldout_candidate_nrmse": candidate_error,
                        "heldout_uniform_nrmse": uniform_error,
                        "heldout_delta": candidate_error - uniform_error,
                    }
                )
    return pd.DataFrame(records)


def station_stability(loo: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (test, budget), group in loo.groupby(["source_test", "additional_stations_b"]):
        layouts = [tuple(map(int, value.split(";"))) for value in group["loo_layout"]]
        for index in range(8):
            records.append(
                {
                    "source_test": test,
                    "additional_stations_b": budget,
                    "total_retained_stations": budget + 1,
                    "gage_index_top_to_bottom": index,
                    "normalised_depth": Z[index],
                    "selection_frequency": np.mean([index in layout for layout in layouts]),
                }
            )
    return pd.DataFrame(records)


def admissible_layout_rank(benchmark: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for source_test, target_test in (("SKS02", "SKS03"), ("SKS03", "SKS02")):
        source = benchmark[benchmark["test"].eq(source_test)]
        target = benchmark[benchmark["test"].eq(target_test)]
        for budget in BUDGETS:
            selected = ta.choose_robust_layout(source, budget)
            candidates = ta.candidate_layouts(budget)
            scores = []
            for layout in candidates:
                score = float(event_errors(target, layout)["event_p90_nrmse"].median())
                scores.append((layout, score))
            scores.sort(key=lambda item: (item[1], item[0]))
            selected_score = next(score for layout, score in scores if layout == selected)
            rank = 1 + sum(score < selected_score - 1e-12 for _, score in scores)
            records.append(
                {
                    "selection_test": source_test,
                    "evaluation_test": target_test,
                    "additional_stations_b": budget,
                    "total_retained_stations": budget + 1,
                    "selected_layout": ";".join(map(str, selected)),
                    "admissible_layout_count": len(scores),
                    "selected_target_median_nrmse": selected_score,
                    "best_target_median_nrmse": scores[0][1],
                    "median_admissible_target_nrmse": float(np.median([score for _, score in scores])),
                    "worst_target_median_nrmse": scores[-1][1],
                    "rank_1_is_best": rank,
                    "fraction_admissible_layouts_outperformed": float(
                        np.mean([selected_score < score - 1e-12 for _, score in scores])
                    ),
                }
            )
    return pd.DataFrame(records)


def boundary_transfer(benchmark: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, int], tuple[int, ...]]]:
    records: list[dict] = []
    layouts: dict[tuple[str, int], tuple[int, ...]] = {}
    for source_test, target_test in (("SKS02", "SKS03"), ("SKS03", "SKS02")):
        source = benchmark[benchmark["test"].eq(source_test)]
        target = benchmark[benchmark["test"].eq(target_test)]
        for budget in BUDGETS:
            candidate = choose_boundary_layout(source, budget)
            layouts[(source_test, budget)] = candidate
            uniform = ta.uniform_layout(budget)
            cand_events = event_errors(target, candidate).rename(columns={"event_p90_nrmse": "candidate"})
            unif_events = event_errors(target, uniform).rename(columns={"event_p90_nrmse": "uniform"})
            paired = cand_events.merge(unif_events, on="event", validate="one_to_one")
            paired["delta"] = paired["candidate"] - paired["uniform"]
            records.append(
                {
                    "selection_test": source_test,
                    "evaluation_test": target_test,
                    "additional_stations_b": budget,
                    "total_retained_stations": budget + 1,
                    "boundary_constrained_layout": ";".join(map(str, candidate)),
                    "normalised_depths": ";".join(f"{Z[i]:.3f}" for i in candidate),
                    "median_candidate_nrmse": paired["candidate"].median(),
                    "median_uniform_nrmse": paired["uniform"].median(),
                    "median_paired_delta": paired["delta"].median(),
                    "events_candidate_better": int((paired["delta"] < 0).sum()),
                    "events": len(paired),
                }
            )
    return pd.DataFrame(records), layouts


def curvature_summary(benchmark: pd.DataFrame, loo: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for test in ("SKS02", "SKS03"):
        source = benchmark[benchmark["test"].eq(test)]
        values = source[ta.GAGE_COLS_TOP_TO_BOTTOM].to_numpy(float)
        scale = np.maximum(np.max(np.abs(values), axis=1), 0.05)
        curvature = np.abs(values[:, :-2] - 2 * values[:, 1:-1] + values[:, 2:]) / scale[:, None]
        for internal_index in range(1, 7):
            records.append(
                {
                    "test": test,
                    "gage_index_top_to_bottom": internal_index,
                    "normalised_depth": Z[internal_index],
                    "median_normalised_absolute_second_difference": float(
                        np.median(curvature[:, internal_index - 1])
                    ),
                    "p90_normalised_absolute_second_difference": float(
                        np.quantile(curvature[:, internal_index - 1], 0.90)
                    ),
                    "loo_selection_frequency_b2_to_b4": float(
                        loo[
                            loo["source_test"].eq(test)
                            & loo["additional_stations_b"].between(2, 4)
                        ]["loo_layout"].map(
                            lambda value: internal_index in tuple(map(int, value.split(";")))
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(records)


def reconstruct(x: np.ndarray, y: np.ndarray, indices: np.ndarray, method: str) -> np.ndarray:
    xs, ys = x[indices], y[indices]
    if method == "linear":
        return np.interp(x, xs, ys)
    if method != "pchip":
        raise ValueError(method)
    estimate = np.empty_like(y, dtype=float)
    inside = (x >= xs.min()) & (x <= xs.max())
    estimate[inside] = pchip_values(xs, ys, x[inside])
    estimate[x < xs.min()] = ys[0]
    estimate[x > xs.max()] = ys[-1]
    return estimate


def pchip_values(xs: np.ndarray, ys: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Shape-preserving cubic Hermite interpolation (Fritsch-Carlson slopes)."""
    n = len(xs)
    if n == 2:
        return np.interp(xq, xs, ys)
    h = np.diff(xs)
    delta = np.diff(ys) / h
    slope = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] > 0:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            slope[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    slope[0] = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(slope[0]) != np.sign(delta[0]):
        slope[0] = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(slope[0]) > abs(3 * delta[0]):
        slope[0] = 3 * delta[0]
    slope[-1] = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(slope[-1]) != np.sign(delta[-1]):
        slope[-1] = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(slope[-1]) > abs(3 * delta[-1]):
        slope[-1] = 3 * delta[-1]

    segment = np.searchsorted(xs, xq, side="right") - 1
    segment = np.clip(segment, 0, n - 2)
    x0 = xs[segment]
    x1 = xs[segment + 1]
    interval = x1 - x0
    t = (xq - x0) / interval
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return (
        h00 * ys[segment]
        + h10 * interval * slope[segment]
        + h01 * ys[segment + 1]
        + h11 * interval * slope[segment + 1]
    )


def nrmse(y: np.ndarray, estimate: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(y))), np.finfo(float).eps)
    return float(np.sqrt(np.mean((estimate - y) ** 2)) / scale)


def external_profiles() -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    profiles: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    supplied = pd.read_csv(OUT / "external_digitised_profiles.csv")
    primary = supplied[
        supplied["quality_class"].eq("primary field trace")
        | supplied["case"].str.contains("Christchurch", na=False)
    ]
    for case, group in primary.groupby("case"):
        group = group.sort_values("normalised_depth")
        evidence = "published CFA trace" if "Christchurch" in case else "marked driven-pile profile"
        profiles[case] = (
            evidence,
            group["normalised_depth"].to_numpy(float),
            group["load"].to_numpy(float),
        )
    return profiles


def map_indices(x: np.ndarray, targets: list[float]) -> tuple[np.ndarray, np.ndarray]:
    indices = np.unique([int(np.argmin(np.abs(x - target))) for target in targets])
    return indices, x[indices]


def external_protocol_sensitivity(
    original_layouts: dict[tuple[str, int], tuple[int, ...]],
    boundary_layouts: dict[tuple[str, int], tuple[int, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    for case, (evidence, x, y) in external_profiles().items():
        for budget in BUDGETS:
            uniform_targets = np.linspace(0.0, 1.0, budget + 1).tolist()
            uniform_indices, uniform_realised = map_indices(x, uniform_targets)
            if len(uniform_indices) != budget + 1:
                continue
            for source in ("SKS02", "SKS03"):
                for layout_family, layout_lookup in (
                    ("original", original_layouts),
                    ("boundary_constrained", boundary_layouts),
                ):
                    targets = [Z[i] for i in layout_lookup[(source, budget)]]
                    candidate_indices, candidate_realised = map_indices(x, targets)
                    if len(candidate_indices) != budget + 1:
                        continue
                    for method in ("linear", "pchip"):
                        cand = nrmse(y, reconstruct(x, y, candidate_indices, method))
                        unif = nrmse(y, reconstruct(x, y, uniform_indices, method))
                        records.append(
                            {
                                "case": case,
                                "evidence_class": evidence,
                                "layout_family": layout_family,
                                "interpolator": method,
                                "layout_source": source,
                                "additional_stations_b": budget,
                                "total_retained_stations": budget + 1,
                                "candidate_requested_depths": ";".join(f"{v:.3f}" for v in targets),
                                "candidate_realised_depths": ";".join(f"{v:.3f}" for v in candidate_realised),
                                "candidate_unique_levels": len(candidate_indices),
                                "uniform_requested_depths": ";".join(f"{v:.3f}" for v in uniform_targets),
                                "uniform_realised_depths": ";".join(f"{v:.3f}" for v in uniform_realised),
                                "uniform_unique_levels": len(uniform_indices),
                                "candidate_nrmse": cand,
                                "uniform_nrmse": unif,
                                "delta_nrmse": cand - unif,
                                "candidate_retains_deepest_level": int(candidate_indices[-1] == len(x) - 1),
                            }
                        )
    results = pd.DataFrame(records)
    summary_records: list[dict] = []
    for keys, group in results.groupby(["layout_family", "interpolator"]):
        family, method = keys
        for tolerance in (0.0025, 0.005, 0.010):
            delta = group["delta_nrmse"].to_numpy(float)
            summary_records.append(
                {
                    "layout_family": family,
                    "interpolator": method,
                    "practical_equivalence_tolerance": tolerance,
                    "comparisons": len(group),
                    "candidate_lower_error": int((delta < -tolerance).sum()),
                    "practically_indistinguishable": int((np.abs(delta) <= tolerance).sum()),
                    "candidate_higher_error": int((delta > tolerance).sum()),
                    "median_delta_nrmse": float(np.median(delta)),
                    "min_delta_nrmse": float(np.min(delta)),
                    "max_delta_nrmse": float(np.max(delta)),
                }
            )
    return results, pd.DataFrame(summary_records)


def main() -> None:
    benchmark = load_benchmark()
    loo = leave_one_event_out(benchmark)
    station = station_stability(loo)
    ranks = admissible_layout_rank(benchmark)
    boundary, boundary_layouts = boundary_transfer(benchmark)
    original_layouts = {
        (test, budget): ta.choose_robust_layout(benchmark[benchmark["test"].eq(test)], budget)
        for test in ("SKS02", "SKS03")
        for budget in BUDGETS
    }
    curvature = curvature_summary(benchmark, loo)

    loo.to_csv(OUT / "prj2828_leave_one_event_out_layout_stability.csv", index=False)
    station.to_csv(OUT / "prj2828_loo_station_selection_frequency.csv", index=False)
    ranks.to_csv(OUT / "prj2828_admissible_layout_target_rank.csv", index=False)
    boundary.to_csv(OUT / "prj2828_boundary_constrained_transfer.csv", index=False)
    curvature.to_csv(OUT / "prj2828_curvature_selection_diagnostic.csv", index=False)

    external_results, external_summary = external_protocol_sensitivity(
        original_layouts, boundary_layouts
    )

    external_results.to_csv(OUT / "external_protocol_sensitivity_results.csv", index=False)
    external_summary.to_csv(OUT / "external_protocol_sensitivity_summary.csv", index=False)

    manifest = {
        "purpose": "Sampling-stability, exhaustive-layout, boundary-coverage, interpolation, mapping and practical-equivalence sensitivity audit.",
        "independent_unit": "earthquake event after aggregation across repeated piles",
        "source_selection_stability": "leave one complete source earthquake out before layout selection",
        "practical_baseline": "target performance ranked against every admissible discrete layout; this deterministically subsumes a random-layout sample on the same grid",
        "boundary_sensitivity": "source selection repeated with upper and lower measured boundaries fixed while retaining the same total station count",
        "interpolation_sensitivity": "piecewise linear and monotone PCHIP, both with endpoint persistence outside retained support",
        "external_mapping": "nearest published level; duplicate mappings omitted; requested and realised depths recorded",
        "equivalence_thresholds": [0.0025, 0.005, 0.01],
        "claim_boundary": "Sensitivity analyses diagnose dependence on sampling, comparator population, boundary coverage, mapping and interpolation; they do not isolate a single physical cause or establish field performance.",
    }
    (OUT / "methodological_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("Leave-one-event-out exact match rates")
    print(loo.groupby(["source_test", "additional_stations_b"])["exact_layout_match"].mean())
    print("\nTarget ranks among all admissible layouts")
    print(ranks.to_string(index=False))
    print("\nBoundary-constrained transfer")
    print(boundary.to_string(index=False))
    print("\nExternal protocol sensitivity at tolerance 0.005")
    print(
        external_summary[
            external_summary["practical_equivalence_tolerance"].eq(0.005)
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
