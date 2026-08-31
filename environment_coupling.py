from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PRJ2828_DATA_ROOT", OUT / "source_data"))


def identity(path: Path) -> tuple[str, str]:
    test = "SKS02" if "SKS02" in path.parts else "SKS03"
    match = re.search(r"(EQM_\d+)", path.name)
    if match is None:
        raise ValueError(path)
    return test, match.group(1)


def numeric_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, skipinitialspace=True).apply(pd.to_numeric, errors="coerce")


def event_environment() -> pd.DataFrame:
    rows: dict[tuple[str, str], dict] = {}
    for path in DATA_ROOT.rglob("EQM_*_Input_Motion.txt"):
        test, event = identity(path)
        df = numeric_file(path)
        acc = df.iloc[:, 1].to_numpy(float)
        rows[(test, event)] = {
            "test": test,
            "event": event,
            "pga_g": float(np.nanmax(np.abs(acc))),
            "input_motion_samples": int(np.isfinite(acc).sum()),
            "input_motion_source_file": str(path),
        }

    for path in DATA_ROOT.rglob("EQM_*_Keller_Excess_Pore_Pressure_Ratio.txt"):
        test, event = identity(path)
        df = numeric_file(path)
        values = df.iloc[:, 1:].to_numpy(float)
        # One peak per transducer is formed first; the median prevents one
        # local sensor or spike from defining the entire event regime.
        per_sensor_peak = np.nanmax(values, axis=0)
        row = rows.setdefault((test, event), {"test": test, "event": event})
        row.update(
            {
                "keller_sensor_count": int(values.shape[1]),
                "median_sensor_peak_ru": float(np.nanmedian(per_sensor_peak)),
                "p90_sensor_peak_ru": float(np.nanquantile(per_sensor_peak, 0.90)),
                "maximum_observed_ru": float(np.nanmax(values)),
                "keller_source_file": str(path),
            }
        )

    for path in DATA_ROOT.rglob("EQM_*_Settlement.txt"):
        test, event = identity(path)
        df = numeric_file(path)
        names = [str(c).strip() for c in df.columns]
        soil_cols = [i for i, name in enumerate(names) if name.startswith("SM_")]
        pile_cols = [i for i, name in enumerate(names) if "Pile_" in name and "LP" in name]
        row = rows.setdefault((test, event), {"test": test, "event": event})
        if soil_cols:
            soil = df.iloc[:, soil_cols].to_numpy(float)
            row["median_absolute_final_soil_settlement_mm"] = float(
                np.nanmedian(np.abs(soil[-1] - soil[0]))
            )
        if pile_cols:
            pile = df.iloc[:, pile_cols].to_numpy(float)
            row["median_absolute_final_pile_settlement_mm"] = float(
                np.nanmedian(np.abs(pile[-1] - pile[0]))
            )
        row["settlement_source_file"] = str(path)
    return pd.DataFrame(rows.values()).sort_values(["test", "event"])


def coupled_errors(environment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clusters = pd.read_csv(OUT / "prj2828_pile_event_cluster_results.csv")
    records: list[pd.DataFrame] = []
    for budget in (2, 3, 4):
        for test, source in (("SKS02", "SKS03"), ("SKS03", "SKS02")):
            names = {
                "uniform": f"uniform_{budget}",
                "cross_regime": f"trained_{source}_applied_all_{budget}",
                "static_PRJ4809": f"static_transfer_{budget}",
            }
            subset = clusters[
                clusters["test"].eq(test) & clusters["layout"].isin(names.values())
            ].copy()
            subset["strategy"] = subset["layout"].map({v: k for k, v in names.items()})
            event = (
                subset.groupby(["test", "event", "strategy"], as_index=False)
                .agg(
                    piles=("pile", "nunique"),
                    event_median_cluster_p90_nrmse=("cluster_p90_nrmse", "median"),
                )
            )
            wide = event.pivot(index=["test", "event", "piles"], columns="strategy", values="event_median_cluster_p90_nrmse").reset_index()
            wide.columns.name = None
            wide["additional_stations_b"] = budget
            wide["training_test_for_cross_regime"] = source
            wide["cross_regime_minus_uniform"] = wide["cross_regime"] - wide["uniform"]
            wide["static_PRJ4809_minus_uniform"] = wide["static_PRJ4809"] - wide["uniform"]
            records.append(wide)
    coupled = pd.concat(records, ignore_index=True).merge(
        environment, on=["test", "event"], how="left", validate="many_to_one"
    )

    correlations: list[dict] = []
    for (test, budget), group in coupled.groupby(["test", "additional_stations_b"]):
        for driver in ("pga_g", "median_sensor_peak_ru"):
            valid = group[[driver, "cross_regime_minus_uniform"]].dropna()
            ranked = valid.rank(method="average")
            rho = float(np.corrcoef(ranked.iloc[:, 0], ranked.iloc[:, 1])[0, 1])
            correlations.append(
                {
                    "test": test,
                    "additional_stations_b": budget,
                    "driver": driver,
                    "events": len(valid),
                    "spearman_rho_descriptive": rho,
                    "evidence_limit": "exploratory only; 5-6 events, no significance test and no causal interpretation",
                }
            )
    return coupled, pd.DataFrame(correlations)


def main() -> None:
    environment = event_environment()
    coupled, correlations = coupled_errors(environment)
    environment.to_csv(OUT / "prj2828_event_environment_descriptors.csv", index=False)
    coupled.to_csv(OUT / "prj2828_event_regime_transfer.csv", index=False)
    correlations.to_csv(OUT / "prj2828_regime_spearman_exploratory.csv", index=False)
    print(environment.to_string(index=False))
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
