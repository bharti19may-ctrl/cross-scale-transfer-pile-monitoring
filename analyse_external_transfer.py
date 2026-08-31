from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from digitize_external_profiles import build_cambridge_profile as _digitise_cambridge
from digitize_external_profiles import build_fhwa_traces as _digitise_fhwa, build_result

REDIGITISE = False
PROFILE_CSV = Path(__file__).with_name("external_digitised_profiles.csv")


def supplied_profiles() -> pd.DataFrame:
    """Read archived coordinates; source-page images are optional inputs."""
    data = pd.read_csv(PROFILE_CSV)
    required = {"source", "case", "depth", "load", "normalised_depth"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing coordinate columns: {required - set(data.columns)}")
    if data.duplicated(["case", "normalised_depth"]).any():
        raise ValueError("Duplicate levels in supplied reference profiles")
    return data


def build_fhwa_traces() -> pd.DataFrame:
    if REDIGITISE:
        return _digitise_fhwa()
    return supplied_profiles().query("source == 'FHWA-HRT-17-060'").copy()


def build_cambridge_profile() -> pd.DataFrame:
    if REDIGITISE:
        return _digitise_cambridge()
    data = supplied_profiles()
    return data[data["case"].str.startswith("MS09")].copy()


LAYOUTS = {
    ("SKS02", 2): [0.000, 0.143, 0.429],
    ("SKS02", 3): [0.000, 0.143, 0.714, 0.857],
    ("SKS02", 4): [0.000, 0.143, 0.714, 0.857, 1.000],
    ("SKS03", 2): [0.000, 0.143, 0.714],
    ("SKS03", 3): [0.000, 0.143, 0.429, 1.000],
    ("SKS03", 4): [0.000, 0.143, 0.571, 0.857, 1.000],
}


def uniform_layout(budget: int) -> list[float]:
    return np.linspace(0.0, 1.0, budget + 1).tolist()


def reconciled_byu_profiles() -> pd.DataFrame:
    if not REDIGITISE:
        data = supplied_profiles()
        return data[data["case"].str.startswith("Turrell")].copy()
    raw = build_result()
    raw = raw[~((raw["case"] == "Turrell H-pile post-blast") & (raw["depth"] == 4))]
    raw["canonical_case"] = raw["case"].str.replace(" QA", "", regex=False)
    reconciled = (
        raw.groupby(["source", "canonical_case", "depth", "depth_unit", "load_unit"], as_index=False)
        .agg(
            load=("load", "mean"),
            digitisation_replicates=("load", "size"),
            digitisation_range=("load", lambda values: float(values.max() - values.min())),
        )
        .rename(columns={"canonical_case": "case"})
    )
    reconciled["quality_class"] = np.where(
        reconciled["case"].str.contains("concrete", case=False),
        "qualified sensitivity",
        "primary field trace",
    )
    return reconciled.sort_values(["case", "depth"]).reset_index(drop=True)


def map_layout(observed_depths: np.ndarray, targets: list[float]) -> tuple[np.ndarray, np.ndarray]:
    normalised = (observed_depths - observed_depths.min()) / (
        observed_depths.max() - observed_depths.min()
    )
    indices = np.array([int(np.argmin(np.abs(normalised - target))) for target in targets])
    indices = np.unique(indices)
    return indices, normalised[indices]


def metrics(depth: np.ndarray, load: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    estimate = np.interp(depth, depth[indices], load[indices])
    error = estimate - load
    scale = max(float(np.max(np.abs(load))), np.finfo(float).eps)
    peak = int(np.argmax(load))
    estimated_peak = int(np.argmax(estimate))
    depth_span = float(depth.max() - depth.min())
    return {
        "nrmse": float(np.sqrt(np.mean(error**2)) / scale),
        "mae_relative": float(np.mean(np.abs(error)) / scale),
        "peak_load_relative_error": float(abs(estimate.max() - load.max()) / scale),
        "peak_depth_error_normalised": float(abs(depth[estimated_peak] - depth[peak]) / depth_span),
    }


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = reconciled_byu_profiles()
    records: list[dict] = []
    for case, group in profiles.groupby("case"):
        group = group.sort_values("depth")
        depth = group["depth"].to_numpy(float)
        load = group["load"].to_numpy(float)
        quality = group["quality_class"].iloc[0]
        for budget in (2, 3, 4):
            for source in ("SKS02", "SKS03", "uniform"):
                targets = uniform_layout(budget) if source == "uniform" else LAYOUTS[(source, budget)]
                indices, realised = map_layout(depth, targets)
                if len(indices) != len(targets):
                    continue
                record = {
                    "case": case,
                    "quality_class": quality,
                    "budget_additional_stations": budget,
                    "layout_source": source,
                    "requested_normalised_depths": ";".join(f"{value:.3f}" for value in targets),
                    "realised_normalised_depths": ";".join(f"{value:.3f}" for value in realised),
                    "realised_depths_ft": ";".join(f"{value:.1f}" for value in depth[indices]),
                    "retained_levels": len(indices),
                    "reference_levels": len(depth),
                }
                record.update(metrics(depth, load, indices))
                records.append(record)
    results = pd.DataFrame(records)
    paired = results[results["layout_source"] != "uniform"].merge(
        results[results["layout_source"] == "uniform"],
        on=["case", "quality_class", "budget_additional_stations"],
        suffixes=("_candidate", "_uniform"),
        validate="many_to_one",
    )
    for metric in [
        "nrmse",
        "mae_relative",
        "peak_load_relative_error",
        "peak_depth_error_normalised",
    ]:
        paired[f"delta_{metric}"] = paired[f"{metric}_candidate"] - paired[f"{metric}_uniform"]
    return profiles, paired


def evaluate_fhwa_traces() -> pd.DataFrame:
    profiles = build_fhwa_traces()
    records: list[dict] = []
    for case, group in profiles.groupby("case"):
        group = group.sort_values("normalised_depth")
        depth = group["normalised_depth"].to_numpy(float)
        load = group["load"].to_numpy(float)
        for budget in (2, 3, 4):
            baseline_indices = np.array(
                [int(np.argmin(np.abs(depth - target))) for target in uniform_layout(budget)]
            )
            baseline = metrics(depth, load, np.unique(baseline_indices))
            for source in ("SKS02", "SKS03"):
                candidate_indices = np.array(
                    [int(np.argmin(np.abs(depth - target))) for target in LAYOUTS[(source, budget)]]
                )
                candidate = metrics(depth, load, np.unique(candidate_indices))
                records.append(
                    {
                        "case": case,
                        "quality_class": "published field-trace sensitivity",
                        "budget_additional_stations": budget,
                        "layout_source": source,
                        "delta_nrmse": candidate["nrmse"] - baseline["nrmse"],
                        "delta_peak_load_relative_error": candidate["peak_load_relative_error"]
                        - baseline["peak_load_relative_error"],
                        "delta_peak_depth_error_normalised": candidate[
                            "peak_depth_error_normalised"
                        ]
                        - baseline["peak_depth_error_normalised"],
                    }
                )
    return pd.DataFrame(records)


def evaluate_cambridge_profile() -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = build_cambridge_profile().sort_values("depth").reset_index(drop=True)
    depth = profile["depth"].to_numpy(float)
    load = profile["load"].to_numpy(float)
    records: list[dict] = []
    for budget in (2, 3, 4):
        baseline_indices, baseline_realised = map_layout(depth, uniform_layout(budget))
        if len(baseline_indices) != budget + 1:
            continue
        baseline = metrics(depth, load, baseline_indices)
        for source in ("SKS02", "SKS03"):
            candidate_indices, candidate_realised = map_layout(depth, LAYOUTS[(source, budget)])
            if len(candidate_indices) != budget + 1:
                continue
            candidate = metrics(depth, load, candidate_indices)
            records.append(
                {
                    "case": profile["case"].iloc[0],
                    "quality_class": "resolution-limited centrifuge sensitivity",
                    "budget_additional_stations": budget,
                    "layout_source": source,
                    "candidate_realised_depths": ";".join(
                        f"{value:.3f}" for value in candidate_realised
                    ),
                    "uniform_realised_depths": ";".join(
                        f"{value:.3f}" for value in baseline_realised
                    ),
                    "delta_nrmse": candidate["nrmse"] - baseline["nrmse"],
                    "delta_peak_load_relative_error": candidate["peak_load_relative_error"]
                    - baseline["peak_load_relative_error"],
                    "delta_peak_depth_error_normalised": candidate[
                        "peak_depth_error_normalised"
                    ]
                    - baseline["peak_depth_error_normalised"],
                }
            )
    return profile, pd.DataFrame(records)


def combined_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    byu_profiles, byu_paired = evaluate()
    byu = pd.DataFrame(
        {
            "case": byu_paired["case"],
            "evidence_class": byu_paired["quality_class"],
            "budget_additional_stations": byu_paired["budget_additional_stations"],
            "layout_source": byu_paired["layout_source_candidate"],
            "candidate_realised_depths": byu_paired[
                "realised_normalised_depths_candidate"
            ],
            "uniform_realised_depths": byu_paired["realised_normalised_depths_uniform"],
            "delta_nrmse": byu_paired["delta_nrmse"],
            "delta_peak_load_relative_error": byu_paired[
                "delta_peak_load_relative_error"
            ],
            "delta_peak_depth_error_normalised": byu_paired[
                "delta_peak_depth_error_normalised"
            ],
        }
    )
    fhwa = evaluate_fhwa_traces().rename(columns={"quality_class": "evidence_class"})
    fhwa["candidate_realised_depths"] = fhwa["budget_additional_stations"].map(
        lambda budget: ";".join(f"{value:.3f}" for value in LAYOUTS[("SKS02", budget)])
    )
    for index in fhwa.index:
        source = fhwa.at[index, "layout_source"]
        budget = int(fhwa.at[index, "budget_additional_stations"])
        points = build_fhwa_traces()
        depth = points.loc[points["case"].eq(fhwa.at[index, "case"]), "normalised_depth"].sort_values().to_numpy(float)
        _, realised = map_layout(depth, LAYOUTS[(source, budget)])
        fhwa.at[index, "candidate_realised_depths"] = ";".join(
            f"{value:.3f}" for value in realised
        )
        _, uniform_realised = map_layout(depth, uniform_layout(budget))
        fhwa.at[index, "uniform_realised_depths"] = ";".join(
            f"{value:.3f}" for value in uniform_realised
        )
    cambridge_profile, cambridge = evaluate_cambridge_profile()
    cambridge = cambridge.rename(columns={"quality_class": "evidence_class"})
    all_results = pd.concat([byu, fhwa, cambridge], ignore_index=True, sort=False)
    all_results["comparison_class"] = np.select(
        [
            all_results["delta_nrmse"] < -0.005,
            all_results["delta_nrmse"] > 0.005,
        ],
        ["candidate lower error", "candidate higher error"],
        default="practically indistinguishable (|delta| <= 0.005)",
    )
    all_results["claim_role"] = np.select(
        [
            all_results["evidence_class"].eq("primary field trace"),
            all_results["evidence_class"].eq("published field-trace sensitivity"),
        ],
        ["primary external field check", "secondary trace-level field check"],
        default="sensitivity only",
    )
    return byu_profiles, cambridge_profile, all_results


def write_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    byu_profiles, cambridge_profile, results = combined_results()
    fhwa_profiles = build_fhwa_traces()
    profile_points = pd.concat(
        [
            byu_profiles.assign(normalised_depth=lambda frame: frame.groupby("case")["depth"].transform(lambda values: (values - values.min()) / (values.max() - values.min()))),
            fhwa_profiles,
            cambridge_profile.assign(normalised_depth=lambda frame: (frame["depth"] - frame["depth"].min()) / (frame["depth"].max() - frame["depth"].min())),
        ],
        ignore_index=True,
        sort=False,
    )
    summary = (
        results.groupby(["claim_role", "evidence_class"], as_index=False)
        .agg(
            comparisons=("delta_nrmse", "size"),
            candidate_lower_error=("comparison_class", lambda values: int((values == "candidate lower error").sum())),
            candidate_higher_error=("comparison_class", lambda values: int((values == "candidate higher error").sum())),
            practically_indistinguishable=("comparison_class", lambda values: int(values.str.startswith("practically").sum())),
            median_delta_nrmse=("delta_nrmse", "median"),
            minimum_delta_nrmse=("delta_nrmse", "min"),
            maximum_delta_nrmse=("delta_nrmse", "max"),
        )
        .sort_values(["claim_role", "evidence_class"])
    )
    qa = (
        byu_profiles.groupby("case", as_index=False)
        .agg(
            digitised_points=("load", "size"),
            median_between_figure_range=("digitisation_range", "median"),
            maximum_between_figure_range=("digitisation_range", "max"),
            maximum_load=("load", "max"),
        )
    )
    qa["maximum_range_percent_of_peak"] = 100.0 * qa["maximum_between_figure_range"] / qa["maximum_load"]
    if REDIGITISE or (output_dir / "external_digitised_profiles.csv").resolve() != PROFILE_CSV.resolve():
        profile_points.to_csv(output_dir / "external_digitised_profiles.csv", index=False)
    results.to_csv(output_dir / "external_transfer_results.csv", index=False)
    summary.to_csv(output_dir / "external_transfer_summary.csv", index=False)
    qa.to_csv(output_dir / "external_digitisation_qa.csv", index=False)
    manifest = {
        "analysis_scope": "Frozen SKS02- and SKS03-selected layouts compared with equal-budget uniform placement on independent published profiles.",
        "primary_field_cases": [
            "Turrell H-pile post-blast",
            "Turrell pipe pile post-blast",
        ],
        "secondary_field_cases": [
            "Christchurch 12-m CFA pile post-blast",
            "Christchurch 14-m CFA pile post-blast",
        ],
        "qualified_sensitivity_cases": [
            "Turrell concrete pile post-blast",
            "MS09 bored pile, Leg 1, 2500 s after shaking",
        ],
        "equivalence_threshold_delta_nrmse": 0.005,
        "mapping_rule": "Each frozen normalised target depth is mapped to the nearest published measured level; a comparison is omitted if two targets map to the same level.",
        "reconstruction_rule": "Piecewise-linear interpolation between retained levels, with endpoint persistence outside the retained range, applied consistently to candidate and comparator.",
        "claim_boundary": "The comparison is an independent profile-reconstruction consistency check. It does not validate field instrumentation, safety, failure probability, installation cost, or a universal layout.",
        "statistical_boundary": "Published profiles are case studies, not independent population samples; no p-value or population-level confidence interval is reported.",
    }
    (output_dir / "external_consistency_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


def main() -> None:
    profiles, paired = evaluate()
    print("RECONCILED PROFILES")
    print(profiles.to_string(index=False))
    print("\nPAIRED RESULTS")
    columns = [
        "case",
        "quality_class",
        "budget_additional_stations",
        "layout_source_candidate",
        "realised_normalised_depths_candidate",
        "realised_normalised_depths_uniform",
        "delta_nrmse",
        "delta_peak_load_relative_error",
        "delta_peak_depth_error_normalised",
    ]
    print(paired[columns].to_string(index=False))
    print("\nFHWA TRACE-LEVEL RESULTS")
    print(evaluate_fhwa_traces().to_string(index=False))
    print("\nCAMBRIDGE PROFILE")
    cambridge_profile, cambridge_results = evaluate_cambridge_profile()
    print(cambridge_profile.to_string(index=False))
    print(cambridge_results.to_string(index=False))
    print("\nOUTPUT SUMMARY")
    write_outputs(Path(__file__).resolve().parent / "output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce frozen-layout external comparisons from archived coordinates.")
    parser.add_argument("--redigitise", action="store_true", help="Requires separately obtained source-page images")
    parser.add_argument("--profiles", type=Path, default=PROFILE_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    REDIGITISE = args.redigitise
    PROFILE_CSV = args.profiles
    write_outputs(args.output_dir)
