from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_external_transfer import LAYOUTS, metrics, uniform_layout


HERE = Path(__file__).resolve().parent
OUT = HERE
INK = "#172033"
MUTED = "#374151"
GRID = "#DCE2EA"
BLUE = "#1769AA"
BLUE_PALE = "#DCECF7"
ORANGE = "#D97706"
ORANGE_PALE = "#F7E4C7"
NEUTRAL = "#E7E9ED"


def text(x, y, value, size=24, anchor="start", weight=600, colour=INK):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial,Helvetica,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{colour}">{html.escape(str(value))}</text>'
    )


def polyline(points, colour, width=5, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{colour}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
    )


def main() -> None:
    results = pd.read_csv(OUT / "external_transfer_results.csv")
    width, height = 1800, 1220
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        text(70, 58, "External profile transfer of frozen centrifuge layouts", 36, weight=700),
        text(70, 99, "Change in reconstruction NRMSE relative to equal-budget uniform placement; negative values favour the frozen layout", 24, colour=MUTED),
    ]

    # Panel A: field-case matrix.
    svg.append(text(70, 155, "A  Field-profile comparison matrix", 28, weight=700))
    cases = [
        ("Turrell H-pile post-blast", "Turrell H-pile"),
        ("Turrell pipe pile post-blast", "Turrell pipe pile"),
        ("Christchurch 12-m CFA pile post-blast", "Christchurch 12-m CFA"),
        ("Christchurch 14-m CFA pile post-blast", "Christchurch 14-m CFA"),
    ]
    cols = [(budget, source) for budget in (2, 3, 4) for source in ("SKS02", "SKS03")]
    x0, y0, label_w, cell_w, cell_h = 70, 235, 260, 108, 104
    for j, (budget, source) in enumerate(cols):
        xx = x0 + label_w + j * cell_w + cell_w / 2
        svg.append(text(xx, 190, f"b = {budget}", 21, anchor="middle", weight=700))
        svg.append(text(xx, 220, source, 19, anchor="middle", colour=MUTED))
    for i, (case, short) in enumerate(cases):
        yy = y0 + i * cell_h
        svg.append(text(x0, yy + 44, short, 22, weight=600))
        evidence = "primary" if "Turrell" in case else "trace-level"
        svg.append(text(x0, yy + 73, evidence, 17, colour=MUTED))
        for j, (budget, source) in enumerate(cols):
            row = results[
                results["case"].eq(case)
                & results["budget_additional_stations"].eq(budget)
                & results["layout_source"].eq(source)
            ].iloc[0]
            delta = float(row["delta_nrmse"])
            if delta < -0.005:
                fill, stroke, marker, dash = BLUE_PALE, BLUE, "−", ""
            elif delta > 0.005:
                fill, stroke, marker, dash = ORANGE_PALE, ORANGE, "+", ""
            else:
                fill, stroke, marker, dash = NEUTRAL, MUTED, "≈", "8 5"
            xx = x0 + label_w + j * cell_w
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            svg.append(
                f'<rect x="{xx+5}" y="{yy+5}" width="{cell_w-10}" height="{cell_h-10}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="3"{dash_attr}/>'
            )
            svg.append(text(xx + 19, yy + 31, marker, 22, anchor="middle", weight=700, colour=stroke))
            svg.append(text(xx + cell_w / 2, yy + 60, f"{delta:+.3f}", 23, anchor="middle", weight=700, colour=INK))

    legend_y = y0 + len(cases) * cell_h + 30
    for offset, fill, stroke, marker, label in [
        (0, BLUE_PALE, BLUE, "−", "lower error"),
        (205, NEUTRAL, MUTED, "≈", "|change| <= 0.005"),
        (485, ORANGE_PALE, ORANGE, "+", "higher error"),
    ]:
        svg.append(f'<rect x="{x0+offset}" y="{legend_y}" width="29" height="22" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        svg.append(text(x0 + offset + 14.5, legend_y + 18, marker, 17, anchor="middle", weight=700, colour=stroke))
        svg.append(text(x0 + offset + 40, legend_y + 19, label, 19, colour=MUTED))

    # Panel B: H-pile reconstruction example.
    panel_x, panel_y, panel_w, panel_h = 1040, 175, 680, 610
    svg.append(text(panel_x, 155, "B  H-pile reconstruction at b = 4", 28, weight=700))
    plot_left, plot_right = panel_x + 105, panel_x + panel_w - 25
    plot_top, plot_bottom = panel_y + 65, panel_y + panel_h - 75
    for tick in (50, 100, 150, 200):
        xx = plot_left + tick / 220.0 * (plot_right - plot_left)
        svg.append(f'<line x1="{xx:.1f}" y1="{plot_top}" x2="{xx:.1f}" y2="{plot_bottom}" stroke="{GRID}" stroke-width="2"/>')
        svg.append(text(xx, plot_top - 18, tick, 18, anchor="middle", colour=MUTED))
    for tick in np.linspace(0, 1, 5):
        yy = plot_top + tick * (plot_bottom - plot_top)
        svg.append(f'<line x1="{plot_left}" y1="{yy:.1f}" x2="{plot_right}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="2"/>')
        svg.append(text(plot_left - 18, yy + 7, f"{tick:.2f}", 18, anchor="end", colour=MUTED))
    svg.append(text((plot_left + plot_right) / 2, plot_top - 48, "Axial load (kips)", 21, anchor="middle", weight=600))
    axis_x = plot_left - 72
    axis_y = (plot_top + plot_bottom) / 2
    svg.append(
        f'<text x="{axis_x:.1f}" y="{axis_y:.1f}" text-anchor="middle" '
        f'font-family="Arial,Helvetica,sans-serif" font-size="21" font-weight="600" '
        f'fill="{INK}" transform="rotate(-90 {axis_x:.1f} {axis_y:.1f})">Normalised depth</text>'
    )
    profile = pd.read_csv(OUT / "external_digitised_profiles.csv")
    profile = profile[profile["case"].eq("Turrell H-pile post-blast")].sort_values("depth")
    depth = profile["depth"].to_numpy(float)
    z = (depth - depth.min()) / (depth.max() - depth.min())
    load = profile["load"].to_numpy(float)

    def realised_indices(targets):
        return np.unique([int(np.argmin(np.abs(z - target))) for target in targets])

    uniform_idx = realised_indices(uniform_layout(4))
    candidate_idx = realised_indices(LAYOUTS[("SKS03", 4)])
    uniform_estimate = np.interp(z, z[uniform_idx], load[uniform_idx])
    candidate_estimate = np.interp(z, z[candidate_idx], load[candidate_idx])

    def map_points(values):
        return [
            (
                plot_left + value / 220.0 * (plot_right - plot_left),
                plot_top + zz * (plot_bottom - plot_top),
            )
            for value, zz in zip(values, z, strict=True)
        ]

    svg.append(polyline(map_points(uniform_estimate), MUTED, 5, "13 10"))
    svg.append(polyline(map_points(candidate_estimate), BLUE, 6))
    svg.append(polyline(map_points(load), INK, 3))
    for xx, yy in map_points(load):
        svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="7" fill="#FFFFFF" stroke="{INK}" stroke-width="3"/>')
    for idx in candidate_idx:
        xx, yy = map_points(load)[idx]
        svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="11" fill="{BLUE}" stroke="#FFFFFF" stroke-width="3"/>')
    legend_items = [
        (INK, None, "published points"),
        (BLUE, None, "SKS03-frozen layout"),
        (MUTED, "13 10", "uniform layout"),
    ]
    for i, (colour, dash, label) in enumerate(legend_items):
        yy = plot_bottom + 35 + i * 31
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="{plot_left}" y1="{yy}" x2="{plot_left+55}" y2="{yy}" stroke="{colour}" stroke-width="5"{dash_attr}/>' )
        svg.append(text(plot_left + 68, yy + 7, label, 19, colour=MUTED))

    # Panel C: descriptive outcome count.
    svg.append(text(1040, 865, "C  Descriptive count across four field profiles", 28, weight=700))
    field = results[results["claim_role"].isin(["primary external field check", "secondary trace-level field check"])]
    counts = field["comparison_class"].value_counts()
    lower = int(counts.get("candidate lower error", 0))
    higher = int(counts.get("candidate higher error", 0))
    tie = int(counts.get("practically indistinguishable (|delta| <= 0.005)", 0))
    total = lower + higher + tie
    bar_x, bar_y, bar_w, bar_h = 1080, 935, 590, 65
    cursor = bar_x
    for count, fill, label in [(lower, BLUE, "lower"), (tie, MUTED, "indistinguishable"), (higher, ORANGE, "higher")]:
        segment = bar_w * count / total
        svg.append(f'<rect x="{cursor:.1f}" y="{bar_y}" width="{segment:.1f}" height="{bar_h}" fill="{fill}"/>')
        if segment > 48:
            svg.append(text(cursor + segment / 2, bar_y + 43, count, 24, anchor="middle", weight=700, colour="#FFFFFF"))
        cursor += segment
    svg.append(text(bar_x, 1040, f"Frozen layout: {lower} lower-error, {tie} indistinguishable and", 21, colour=INK))
    svg.append(text(bar_x, 1072, f"{higher} higher-error comparisons (n = {total}).", 21, colour=INK))
    svg.append(text(bar_x, 1107, "Counts describe these cases only; they are not population frequencies.", 19, colour=MUTED))

    svg.append(text(70, 1170, "b denotes the number of added stations beyond the fixed upper reference. Mapping uses the nearest published level; no target profile was used to retune a layout.", 20, colour=MUTED))
    svg.append("</svg>")
    (OUT / "Fig07_External_Profile_Transfer.svg").write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
