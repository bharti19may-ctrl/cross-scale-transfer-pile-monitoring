from __future__ import annotations

import itertools
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PRJ2828_DATA_ROOT", OUT / "source_data"))
STATIC_LAYOUT_CSV = Path(os.environ.get("PRJ4809_LAYOUT_CSV", OUT / "negative_control_static_layouts.csv"))

# Nine bridges were installed, but Gage 9 is absent from every SKS03 file.
# The cross-test benchmark therefore uses the eight-level common measured grid
# Gage 8 ... Gage 1.  Pile 1 in SKS03 has two additional missing bridges and is
# retained in the inventory but excluded from the common-grid benchmark.
# z/L=0 and 1 denote the top and bottom of this common measured span, not the
# full pile length.
Z = np.linspace(0.0, 1.0, 8)
GAGE_COLS_TOP_TO_BOTTOM = [f"Gage{i}[MN]" for i in range(8, 0, -1)]
MAX_SNAPSHOTS_PER_FILE = 120


def parse_identity(path: Path) -> tuple[str, str, int]:
    test = "SKS02" if "SKS02" in path.parts else "SKS03"
    event_match = re.search(r"(Day_\d+_Spin_\d+|EQM_\d+|SWM_\d+)", path.name)
    event = event_match.group(1) if event_match else path.parent.name
    pile_match = re.search(r"Pile_(\d+)_Axial_Load", path.name)
    if not pile_match:
        raise ValueError(f"Cannot identify pile: {path}")
    return test, event, int(pile_match.group(1))


def rolling_window_rows(time_min: np.ndarray, pile: int, is_eqm: bool) -> int:
    if not is_eqm or len(time_min) < 3:
        return 1
    dt = float(np.nanmedian(np.diff(time_min)))
    if not np.isfinite(dt) or dt <= 0:
        return 1
    window_min = 0.10 if pile == 1 else 0.05  # 6 s and 3 s, source report
    return max(1, int(round(window_min / dt)))


def load_profiles(path: Path) -> tuple[dict, pd.DataFrame | None]:
    test, event, pile = parse_identity(path)
    header = pd.read_csv(path, nrows=0, skipinitialspace=True)
    header = header.rename(columns={c: str(c).replace("Gage_", "Gage") for c in header.columns})
    time_col = next((c for c in header.columns if str(c).strip().endswith("Time[min]")), None)
    all_present_gages = [c for c in header.columns if re.fullmatch(r"Gage\d+\[MN\]", str(c))]
    present_gages = [c for c in GAGE_COLS_TOP_TO_BOTTOM if c in header.columns]
    missing = [c for c in GAGE_COLS_TOP_TO_BOTTOM if c not in header.columns]
    if time_col is None:
        missing.insert(0, "*Time[min]")
    if missing:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            row_count = max(0, sum(1 for _ in stream) - 1)
        inventory = {
            "test": test,
            "event": event,
            "pile": pile,
            "source_file": str(path),
            "bytes": path.stat().st_size,
            "rows": row_count,
            "complete_rows": 0,
            "time_min_start": np.nan,
            "time_min_end": np.nan,
            "median_dt_s": np.nan,
            "moving_mean_window_rows": 0,
            "moving_mean_window_s": 0.0,
            "sampled_snapshots": 0,
            "missing_fraction": np.nan,
            "gage_count": len(all_present_gages),
            "gage_columns_present": ";".join(all_present_gages),
            "missing_required_columns": ";".join(missing),
            "usable_common_eight_gage_reference": False,
            "exclusion_reason": "incomplete common eight-gage grid",
            "time_status": "not assessed",
        }
        return inventory, None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        first_two = [stream.readline(), stream.readline()]
    header_fields = len(first_two[0].rstrip("\r\n").split(","))
    data_fields = len(first_two[1].rstrip("\r\n").split(",")) if first_two[1] else 0
    if data_fields != header_fields:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            row_count = max(0, sum(1 for _ in stream) - 1)
        inventory = {
            "test": test,
            "event": event,
            "pile": pile,
            "source_file": str(path),
            "bytes": path.stat().st_size,
            "rows": row_count,
            "complete_rows": 0,
            "time_min_start": np.nan,
            "time_min_end": np.nan,
            "median_dt_s": np.nan,
            "moving_mean_window_rows": 0,
            "moving_mean_window_s": 0.0,
            "sampled_snapshots": 0,
            "missing_fraction": np.nan,
            "gage_count": len(all_present_gages),
            "gage_columns_present": ";".join(all_present_gages),
            "missing_required_columns": "",
            "usable_common_eight_gage_reference": False,
            "exclusion_reason": f"row field count {data_fields} does not match header field count {header_fields}",
            "time_status": "time field omitted from data rows",
        }
        return inventory, None
    df = pd.read_csv(path, skipinitialspace=True)
    df = df.rename(columns={c: str(c).replace("Gage_", "Gage") for c in df.columns})
    df = df[[time_col, *GAGE_COLS_TOP_TO_BOTTOM]].apply(pd.to_numeric, errors="coerce")
    time_min = df[time_col].to_numpy(float)
    values = df[GAGE_COLS_TOP_TO_BOTTOM]
    window = rolling_window_rows(time_min, pile, event.startswith("EQM_"))
    if window > 1:
        values = values.rolling(window=window, center=True, min_periods=max(1, window // 3)).mean()

    valid = np.isfinite(time_min) & np.isfinite(values.to_numpy(float)).all(axis=1)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        raise ValueError(f"No complete axial-load rows in {path}")
    choose = valid_idx[np.unique(np.linspace(0, len(valid_idx) - 1, min(MAX_SNAPSHOTS_PER_FILE, len(valid_idx))).round().astype(int))]
    sample = values.iloc[choose].copy()
    sample.insert(0, "time_min", time_min[choose])
    sample.insert(0, "pile", pile)
    sample.insert(0, "event", event)
    sample.insert(0, "test", test)
    sample.insert(0, "source_file", str(path))

    inventory = {
        "test": test,
        "event": event,
        "pile": pile,
        "source_file": str(path),
        "bytes": path.stat().st_size,
        "rows": int(len(df)),
        "complete_rows": int(valid.sum()),
        "time_min_start": float(np.nanmin(time_min)),
        "time_min_end": float(np.nanmax(time_min)),
        "median_dt_s": float(np.nanmedian(np.diff(time_min)) * 60.0) if len(time_min) > 1 else np.nan,
        "moving_mean_window_rows": int(window),
        "moving_mean_window_s": 6.0 if event.startswith("EQM_") and pile == 1 else (3.0 if event.startswith("EQM_") else 0.0),
        "sampled_snapshots": int(len(sample)),
        "missing_fraction": float(1.0 - valid.mean()),
        "gage_count": len(all_present_gages),
        "gage_columns_present": ";".join(all_present_gages),
        "missing_required_columns": "",
        "usable_common_eight_gage_reference": True,
        "exclusion_reason": "",
        "time_status": "provided",
    }
    return inventory, sample


def mapped_indices(depths: list[float]) -> tuple[int, ...]:
    # Gage 8 is the upper boundary of the eight-level cross-test grid.
    idx = {0}
    for depth in depths:
        idx.add(int(np.argmin(np.abs(Z - depth))))
    return tuple(sorted(idx))


def prediction(y: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return np.interp(Z, Z[list(indices)], y[list(indices)])


def profile_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    scale = max(float(np.nanmax(np.abs(y))), 0.05)
    err = yhat - y
    peak_i = int(np.nanargmax(y))
    peak_hat_i = int(np.nanargmax(yhat))
    return {
        "nrmse": float(np.sqrt(np.mean(err**2)) / scale),
        "mae_mn": float(np.mean(np.abs(err))),
        "peak_load_relative_error": float(abs(np.nanmax(yhat) - np.nanmax(y)) / scale),
        "peak_depth_error_z_over_l": float(abs(Z[peak_hat_i] - Z[peak_i])),
    }


def evaluate_layouts(profiles: pd.DataFrame, layouts: dict[str, tuple[int, ...]]) -> pd.DataFrame:
    records: list[dict] = []
    ycols = GAGE_COLS_TOP_TO_BOTTOM
    for row_id, row in profiles.iterrows():
        y = row[ycols].to_numpy(float)
        for name, indices in layouts.items():
            rec = {
                "row_id": int(row_id),
                "test": row["test"],
                "event": row["event"],
                "pile": int(row["pile"]),
                "time_min": float(row["time_min"]),
                "layout": name,
                "gage_indices_top_to_bottom": ";".join(str(i) for i in indices),
                "normalised_depths": ";".join(f"{Z[i]:.3f}" for i in indices),
                "total_observations_including_top": len(indices),
                "additional_stations_b": len(indices) - 1,
            }
            rec.update(profile_metrics(y, prediction(y, indices)))
            records.append(rec)
    return pd.DataFrame(records)


def candidate_layouts(additional: int) -> list[tuple[int, ...]]:
    # The top reading (index 0) is fixed. Candidate internal locations are the
    # remaining seven measured gage levels.
    return [(0, *combo) for combo in itertools.combinations(range(1, 8), additional)]


def choose_robust_layout(train_profiles: pd.DataFrame, additional: int) -> tuple[int, ...]:
    best_layout: tuple[int, ...] | None = None
    best_key = (np.inf, np.inf, np.inf)
    for indices in candidate_layouts(additional):
        cluster_p90: list[float] = []
        for _, cluster in train_profiles.groupby(["event", "pile"]):
            scores = []
            for y in cluster[GAGE_COLS_TOP_TO_BOTTOM].to_numpy(float):
                scores.append(profile_metrics(y, prediction(y, indices))["nrmse"])
            cluster_p90.append(float(np.quantile(scores, 0.90)))
        # Each pile-event history contributes one unit, preventing dense time
        # sampling from masquerading as independent replication.
        key = (
            float(np.quantile(cluster_p90, 0.90)),
            float(np.median(cluster_p90)),
            float(np.max(cluster_p90)),
        )
        if key < best_key:
            best_key = key
            best_layout = indices
    if best_layout is None:
        raise RuntimeError("No candidate layout evaluated")
    return best_layout


def uniform_layout(additional: int) -> tuple[int, ...]:
    depths = np.linspace(0.0, 1.0, additional + 1)[1:]
    return mapped_indices(list(depths))


def load_static_layouts() -> dict[int, tuple[int, ...]]:
    df = pd.read_csv(STATIC_LAYOUT_CSV)
    out: dict[int, tuple[int, ...]] = {}
    rows = df[df["placement"].eq("empirical-p90")]
    for _, row in rows.iterrows():
        additional = int(row["sensor_count"])
        depths = [float(x) for x in str(row["normalised_depths"]).split(";")]
        out[additional] = mapped_indices(depths)
    return out


def summarise_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["test", "layout", "additional_stations_b"], as_index=False)
        .agg(
            snapshots=("nrmse", "size"),
            median_nrmse=("nrmse", "median"),
            p90_nrmse=("nrmse", lambda x: x.quantile(0.90)),
            median_mae_mn=("mae_mn", "median"),
            p90_peak_load_relative_error=("peak_load_relative_error", lambda x: x.quantile(0.90)),
            median_peak_depth_error_z_over_l=("peak_depth_error_z_over_l", "median"),
            p90_peak_depth_error_z_over_l=("peak_depth_error_z_over_l", lambda x: x.quantile(0.90)),
        )
        .sort_values(["additional_stations_b", "test", "p90_nrmse"])
    )


def cluster_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(
            ["test", "event", "pile", "layout", "additional_stations_b"],
            as_index=False,
        )
        .agg(
            snapshots=("nrmse", "size"),
            cluster_median_nrmse=("nrmse", "median"),
            cluster_p90_nrmse=("nrmse", lambda x: x.quantile(0.90)),
            cluster_median_mae_mn=("mae_mn", "median"),
            cluster_p90_peak_load_relative_error=(
                "peak_load_relative_error", lambda x: x.quantile(0.90)
            ),
            cluster_p90_peak_depth_error_z_over_l=(
                "peak_depth_error_z_over_l", lambda x: x.quantile(0.90)
            ),
        )
    )


def summarise_clusters(clusters: pd.DataFrame) -> pd.DataFrame:
    return (
        clusters.groupby(["test", "layout", "additional_stations_b"], as_index=False)
        .agg(
            independent_pile_event_clusters=("cluster_p90_nrmse", "size"),
            median_cluster_p90_nrmse=("cluster_p90_nrmse", "median"),
            p90_cluster_p90_nrmse=("cluster_p90_nrmse", lambda x: x.quantile(0.90)),
            max_cluster_p90_nrmse=("cluster_p90_nrmse", "max"),
            median_cluster_mae_mn=("cluster_median_mae_mn", "median"),
        )
        .sort_values(["additional_stations_b", "test", "p90_cluster_p90_nrmse"])
    )


def bootstrap_median_ci(values: np.ndarray, seed: int, draws: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    for i in range(draws):
        boot[i] = np.median(rng.choice(values, size=len(values), replace=True))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def independent_transfer_comparisons(clusters: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    seed = 20260821
    for additional in (2, 3, 4, 5):
        for test, source in (("SKS02", "SKS03"), ("SKS03", "SKS02")):
            baseline_name = f"uniform_{additional}"
            candidates = {
                "independent_static_PRJ4809": f"static_transfer_{additional}",
                f"cross_regime_trained_{source}": f"trained_{source}_applied_all_{additional}",
            }
            base = clusters[
                clusters["test"].eq(test) & clusters["layout"].eq(baseline_name)
            ][["event", "pile", "cluster_p90_nrmse"]].rename(
                columns={"cluster_p90_nrmse": "uniform_cluster_p90_nrmse"}
            )
            for evidence_label, candidate_name in candidates.items():
                candidate = clusters[
                    clusters["test"].eq(test) & clusters["layout"].eq(candidate_name)
                ][["event", "pile", "cluster_p90_nrmse"]].rename(
                    columns={"cluster_p90_nrmse": "candidate_cluster_p90_nrmse"}
                )
                paired = base.merge(candidate, on=["event", "pile"], validate="one_to_one")
                paired["delta"] = (
                    paired["candidate_cluster_p90_nrmse"]
                    - paired["uniform_cluster_p90_nrmse"]
                )
                # The same piles recur across the shaking sequence.  Aggregate
                # piles within earthquake before resampling so the interval is
                # not artificially narrowed by treating repeat pile histories
                # as independent experiments.
                event_level = (
                    paired.groupby("event", as_index=False)
                    .agg(
                        piles=("pile", "nunique"),
                        candidate_event_p90_nrmse=("candidate_cluster_p90_nrmse", "median"),
                        uniform_event_p90_nrmse=("uniform_cluster_p90_nrmse", "median"),
                        paired_event_delta=("delta", "median"),
                    )
                )
                delta = event_level["paired_event_delta"].to_numpy(float)
                lo, hi = bootstrap_median_ci(delta, seed)
                seed += 1
                records.append(
                    {
                        "evaluation_test": test,
                        "training_source": "PRJ-4809" if "static" in evidence_label else source,
                        "evidence_type": evidence_label,
                        "additional_stations_b": additional,
                        "earthquake_event_clusters": len(event_level),
                        "pile_event_histories": len(paired),
                        "median_candidate_cluster_p90_nrmse": float(
                            event_level["candidate_event_p90_nrmse"].median()
                        ),
                        "median_uniform_cluster_p90_nrmse": float(
                            event_level["uniform_event_p90_nrmse"].median()
                        ),
                        "median_paired_delta_candidate_minus_uniform": float(np.median(delta)),
                        "bootstrap_95_ci_low": lo,
                        "bootstrap_95_ci_high": hi,
                        "fraction_events_candidate_better": float(np.mean(delta < 0)),
                        "interpretation": (
                            "candidate better when delta and its interval are below zero; "
                            "interval resamples earthquake-level pile medians and remains descriptive "
                            "because only 5-6 sequential events are available"
                        ),
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_ROOT.rglob("*_Pile_*_Axial_Load.txt"))
    if not files:
        raise FileNotFoundError(f"No axial-load files under {DATA_ROOT}")

    inventories: list[dict] = []
    samples: list[pd.DataFrame] = []
    for number, path in enumerate(files, start=1):
        print(f"[{number:02d}/{len(files):02d}] {path.name}", flush=True)
        inv, sample = load_profiles(path)
        inventories.append(inv)
        if sample is not None:
            samples.append(sample)
    inventory = pd.DataFrame(inventories)
    profiles = pd.concat(samples, ignore_index=True)
    benchmark = profiles[profiles["event"].str.startswith("EQM_")].copy()
    inventory.to_csv(OUT / "prj2828_axial_profile_inventory.csv", index=False)
    profiles.to_csv(OUT / "prj2828_sampled_reference_profiles.csv", index=False)
    benchmark.to_csv(OUT / "prj2828_eqm_benchmark_profiles.csv", index=False)

    static = load_static_layouts()
    all_results: list[pd.DataFrame] = []
    selected_rows: list[dict] = []
    for additional in (2, 3, 4, 5):
        layouts: dict[str, tuple[int, ...]] = {
            f"uniform_{additional}": uniform_layout(additional),
            f"static_transfer_{additional}": static[additional],
        }
        for train_test, apply_test in (("SKS02", "SKS03"), ("SKS03", "SKS02")):
            chosen = choose_robust_layout(benchmark[benchmark["test"].eq(train_test)], additional)
            name = f"trained_{train_test}_applied_all_{additional}"
            layouts[name] = chosen
            selected_rows.append(
                {
                    "training_test": train_test,
                    "evaluation_scope": "both tests",
                    "additional_stations_b": additional,
                    "layout": name,
                    "gage_indices_top_to_bottom": ";".join(map(str, chosen)),
                    "normalised_depths": ";".join(f"{Z[i]:.3f}" for i in chosen),
                }
            )
        pooled = choose_robust_layout(benchmark, additional)
        layouts[f"pooled_robust_{additional}"] = pooled
        selected_rows.append(
            {
                "training_test": "SKS02+SKS03",
                "evaluation_scope": "both tests",
                "additional_stations_b": additional,
                "layout": f"pooled_robust_{additional}",
                "gage_indices_top_to_bottom": ";".join(map(str, pooled)),
                "normalised_depths": ";".join(f"{Z[i]:.3f}" for i in pooled),
            }
        )
        all_results.append(evaluate_layouts(benchmark, layouts))

    results = pd.concat(all_results, ignore_index=True)
    summary = summarise_results(results)
    clusters = cluster_results(results)
    cluster_summary = summarise_clusters(clusters)
    comparisons = independent_transfer_comparisons(clusters)
    results.to_csv(OUT / "prj2828_layout_transfer_results.csv", index=False)
    summary.to_csv(OUT / "prj2828_layout_transfer_summary.csv", index=False)
    clusters.to_csv(OUT / "prj2828_pile_event_cluster_results.csv", index=False)
    cluster_summary.to_csv(OUT / "prj2828_pile_event_cluster_summary.csv", index=False)
    comparisons.to_csv(OUT / "prj2828_independent_transfer_comparisons.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT / "prj2828_robust_layouts.csv", index=False)

    manifest = {
        "source_project": "DesignSafe PRJ-2828",
        "source_dois": [
            "10.17603/DS2-D25M-GG48",
            "10.17603/DS2-WJGX-TB78",
            "10.17603/DS2-BEZ6-4812",
        ],
        "files": len(files),
        "tests": sorted(inventory["test"].unique().tolist()),
        "events": int(inventory[["test", "event"]].drop_duplicates().shape[0]),
        "piles": int(inventory[["test", "pile"]].drop_duplicates().shape[0]),
        "sampled_reference_profiles": int(len(profiles)),
        "earthquake_benchmark_profiles": int(len(benchmark)),
        "earthquake_pile_event_clusters": int(
            benchmark[["test", "event", "pile"]].drop_duplicates().shape[0]
        ),
        "coordinate_definition": "z/L=0 at Gage 8 (top of the common measured span), z/L=1 at Gage 1 (deepest bridge)",
        "reference_limit": "Eight common measured gage stations are treated as the reference grid; no continuous-profile truth is claimed. SKS03 Pile 1 is excluded because only six of these levels are present.",
        "dynamic_filter": "For EQM files, centred moving means follow the source report: 6 s for Pile 1 and 3 s for Piles 2/3.",
        "independence_rule": "Time snapshots and repeated observations of the same pile are not treated as independent replicates; paired differences are aggregated across piles within each earthquake and bootstrap resampling uses earthquake clusters.",
        "primary_validation_rule": "Layouts are selected on one physical test and evaluated on the other. Pooled and same-test results are exploratory only.",
    }
    (OUT / "prj2828_transfer_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(cluster_summary.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
