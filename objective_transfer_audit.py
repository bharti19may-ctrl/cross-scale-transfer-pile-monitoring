"""Objective-specific transfer and budget-saturation audit.

This script uses only the event/pile reconstruction results produced by
``transfer_audit.py``.  It compares a layout selected in one centrifuge test
with an equal-budget uniform layout in the other test.  Repeated piles are
first aggregated within an earthquake, after which uncertainty is quantified
by resampling earthquakes.  No continuous neutral-plane location is inferred:
the peak-depth response is the discrete measured-grid level containing the
maximum reconstructed axial load.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "prj2828_pile_event_cluster_results.csv"
EVENT_OUTPUT = HERE / "prj2828_objective_transfer_event_results.csv"
SUMMARY_OUTPUT = HERE / "prj2828_objective_transfer_summary.csv"
SATURATION_OUTPUT = HERE / "prj2828_budget_saturation_summary.csv"
FIGURE_PNG = HERE / "Fig_objective_transfer_1200dpi.png"
FIGURE_SVG = HERE / "Fig_objective_transfer.svg"
MANIFEST = HERE / "objective_transfer_audit_manifest.json"

SEED = 20260822
BOOTSTRAP_REPLICATES = 10_000

METRICS = {
    "cluster_p90_nrmse": "p90 profile NRMSE",
    "cluster_p90_peak_load_relative_error": "p90 peak-load relative error",
    "cluster_p90_peak_depth_error_z_over_l": "p90 discrete peak-depth error, z/L",
}

DIRECTIONS = (
    ("SKS03", "SKS02", "SKS03→SKS02"),
    ("SKS02", "SKS03", "SKS02→SKS03"),
)


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(BOOTSTRAP_REPLICATES, values.size), replace=True)
    medians = np.median(draws, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def build_event_results(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for training_test, evaluation_test, direction in DIRECTIONS:
        subset = df.loc[df["test"].eq(evaluation_test)].copy()
        for budget in (2, 3, 4, 5):
            candidate_name = f"trained_{training_test}_applied_all_{budget}"
            uniform_name = f"uniform_{budget}"
            for metric, metric_label in METRICS.items():
                values = (
                    subset.loc[subset["layout"].isin([candidate_name, uniform_name])]
                    .groupby(["event", "layout"], as_index=False)[metric]
                    .median()
                    .pivot(index="event", columns="layout", values=metric)
                    .dropna()
                )
                for event, record in values.iterrows():
                    candidate = float(record[candidate_name])
                    uniform = float(record[uniform_name])
                    rows.append(
                        {
                            "selection_test": training_test,
                            "evaluation_test": evaluation_test,
                            "direction": direction,
                            "event": event,
                            "additional_stations_b": budget,
                            "metric": metric,
                            "metric_label": metric_label,
                            "candidate_event_value": candidate,
                            "uniform_event_value": uniform,
                            "paired_delta_candidate_minus_uniform": candidate - uniform,
                        }
                    )
    return pd.DataFrame(rows)


def summarise(event_results: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    keys = ["selection_test", "evaluation_test", "direction", "additional_stations_b", "metric", "metric_label"]
    for key, group in event_results.groupby(keys, sort=False):
        deltas = group["paired_delta_candidate_minus_uniform"].to_numpy(float)
        low, high = bootstrap_median_ci(deltas, rng)
        rows.append(
            dict(
                zip(keys, key),
                earthquake_events=int(group["event"].nunique()),
                median_candidate_event_value=float(group["candidate_event_value"].median()),
                median_uniform_event_value=float(group["uniform_event_value"].median()),
                median_paired_delta_candidate_minus_uniform=float(np.median(deltas)),
                bootstrap_95_ci_low=low,
                bootstrap_95_ci_high=high,
                events_candidate_better=int(np.sum(deltas < -1e-12)),
                events_tied=int(np.sum(np.abs(deltas) <= 1e-12)),
                events_candidate_worse=int(np.sum(deltas > 1e-12)),
                fraction_events_candidate_better=float(np.mean(deltas < -1e-12)),
            )
        )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame) -> None:
    """Write a publication-sized SVG; the companion Node script rasterises it."""
    import html

    width, height = 3600, 1420
    panel_w, panel_h = 1010, 820
    panel_lefts = [280, 1330, 2380]
    panel_top = 350
    plot_left_pad, plot_right_pad = 120, 45
    plot_top_pad, plot_bottom_pad = 70, 130
    colours = {"SKS03→SKS02": "#1f77b4", "SKS02→SKS03": "#d97706"}
    panel_titles = (
        "(a) Full-profile reconstruction",
        "(b) Peak axial-load magnitude",
        "(c) Discrete peak-load level",
    )
    metric_order = list(METRICS)
    all_parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827;font-weight:600}.muted{fill:#374151}.grid{stroke:#d8dee8;stroke-width:3}.axis{stroke:#111827;stroke-width:5}.zero{stroke:#111827;stroke-width:5}.blue{stroke:#1f77b4;fill:#1f77b4}.orange{stroke:#d97706;fill:#d97706}</style>',
        '<text x="1800" y="105" text-anchor="middle" font-size="62" font-weight="700">Objective-specific cross-regime transfer and five-added-station saturation</text>',
        '<circle cx="1230" cy="215" r="19" class="blue"/><text x="1270" y="232" font-size="38">SKS03→SKS02</text>',
        '<rect x="1840" y="196" width="38" height="38" fill="white" stroke="#d97706" stroke-width="10"/><text x="1905" y="232" font-size="38">SKS02→SKS03</text>',
        '<text x="138" y="770" text-anchor="middle" font-size="38" font-weight="700" transform="rotate(-90 138 770)">Median paired error change (candidate − uniform)</text>',
    ]

    for panel_no, (left, metric, title) in enumerate(zip(panel_lefts, metric_order, panel_titles)):
        part_all = summary.loc[summary["metric"].eq(metric)]
        low_bound = float(min(part_all["bootstrap_95_ci_low"].min(), -0.02))
        high_bound = float(max(part_all["bootstrap_95_ci_high"].max(), 0.02))
        pad = max(0.015, 0.08 * (high_bound - low_bound))
        ymin, ymax = low_bound - pad, high_bound + pad
        x0 = left + plot_left_pad
        x1 = left + panel_w - plot_right_pad
        y0 = panel_top + plot_top_pad
        y1 = panel_top + panel_h - plot_bottom_pad

        def sx(value: float) -> float:
            return x0 + (value - 2) / 3 * (x1 - x0)

        def sy(value: float) -> float:
            return y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)

        all_parts.append(f'<text x="{left + 15}" y="{panel_top + 12}" font-size="39" font-weight="700">{html.escape(title)}</text>')
        for tick in np.linspace(ymin, ymax, 5):
            y = sy(float(tick))
            all_parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" class="grid"/>')
            all_parts.append(f'<text x="{x0 - 20}" y="{y + 13:.1f}" text-anchor="end" font-size="30" class="muted">{tick:+.2f}</text>')
        zero_y = sy(0.0)
        all_parts.append(f'<line x1="{x0}" y1="{zero_y:.1f}" x2="{x1}" y2="{zero_y:.1f}" class="zero"/>')
        all_parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/>')
        all_parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/>')
        for tick in (2, 3, 4, 5):
            x = sx(tick)
            all_parts.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y1 + 12}" class="axis"/>')
            all_parts.append(f'<text x="{x:.1f}" y="{y1 + 55}" text-anchor="middle" font-size="31" class="muted">{tick}</text>')
        all_parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1 + 104}" text-anchor="middle" font-size="34" font-weight="700">Additional stations, b</text>')
        all_parts.append(f'<text x="{x0 + 12}" y="{y1 - 22}" font-size="25" class="muted">negative favours transfer</text>')

        for direction in colours:
            part = part_all.loc[part_all["direction"].eq(direction)].sort_values("additional_stations_b")
            coords = [(sx(float(row.additional_stations_b)), sy(float(row.median_paired_delta_candidate_minus_uniform))) for row in part.itertuples()]
            all_parts.append('<polyline points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in coords) + f'" fill="none" stroke="{colours[direction]}" stroke-width="8"/>')
            for row, (x, y) in zip(part.itertuples(), coords):
                y_low = sy(float(row.bootstrap_95_ci_low))
                y_high = sy(float(row.bootstrap_95_ci_high))
                colour = colours[direction]
                all_parts.extend([
                    f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="{colour}" stroke-width="7"/>',
                    f'<line x1="{x-14:.1f}" y1="{y_low:.1f}" x2="{x+14:.1f}" y2="{y_low:.1f}" stroke="{colour}" stroke-width="7"/>',
                    f'<line x1="{x-14:.1f}" y1="{y_high:.1f}" x2="{x+14:.1f}" y2="{y_high:.1f}" stroke="{colour}" stroke-width="7"/>',
                ])
                if direction == "SKS03→SKS02":
                    all_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" fill="{colour}" stroke="white" stroke-width="5"/>')
                else:
                    all_parts.append(f'<rect x="{x-15:.1f}" y="{y-15:.1f}" width="30" height="30" fill="white" stroke="{colour}" stroke-width="8"/>')

    all_parts.extend([
        '<text x="1800" y="1320" text-anchor="middle" font-size="31" class="muted">Earthquake-level medians; 95% intervals resample earthquakes (SKS02 n=6; SKS03 n=5).</text>',
        '<text x="1800" y="1365" text-anchor="middle" font-size="31" class="muted">Peak depth is a discrete measured-grid level, not a continuous neutral-plane estimate.</text>',
        '</svg>',
    ])
    FIGURE_SVG.write_text("\n".join(all_parts), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(INPUT)
    numeric = list(METRICS) + ["additional_stations_b"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
    event_results = build_event_results(df)
    summary = summarise(event_results)
    event_results.to_csv(EVENT_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    summary.loc[summary["additional_stations_b"].eq(5)].to_csv(SATURATION_OUTPUT, index=False)
    make_figure(summary)
    manifest = {
        "input": INPUT.name,
        "outputs": [EVENT_OUTPUT.name, SUMMARY_OUTPUT.name, SATURATION_OUTPUT.name, FIGURE_PNG.name, FIGURE_SVG.name],
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "aggregation": "median across repeated piles within each earthquake; bootstrap resampling of earthquakes",
        "scope_limit": "peak depth is a discrete measured-grid descriptor; no continuous neutral-plane inference",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
