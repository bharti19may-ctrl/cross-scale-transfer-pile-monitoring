"""Generate a publication-ready SVG for the methodological sensitivity audit."""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BLUE = "#3569A8"
ORANGE = "#D78B29"
NAVY = "#183A5A"
GREY = "#B8C0C8"
INK = "#1F2933"
GRID = "#D7DDE3"
WHITE = "#FFFFFF"


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def t(x: float, y: float, value: str, size: int = 17, anchor: str = "start", weight: int = 600, fill: str = INK) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}">'
        f'{html.escape(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = INK) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'


def line(x1: float, y1: float, x2: float, y2: float, colour: str = GRID, width: float = 1.0, dash: str = "") -> str:
    pattern = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}" stroke-width="{width}"{pattern}/>'


loo = read_csv("prj2828_loo_summary.csv")
ranks = read_csv("prj2828_admissible_layout_target_rank.csv")
protocol = [row for row in read_csv("external_protocol_sensitivity_summary.csv") if abs(float(row["practical_equivalence_tolerance"]) - 0.005) < 1e-12]

W, H = 1200, 650
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<defs><pattern id="orangeHatch" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#F4D7A6"/><path d="M0 0 L8 8" stroke="#9A5B00" stroke-width="1.5"/></pattern><pattern id="greyDots" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#DDE1E5"/><circle cx="4" cy="4" r="1.2" fill="#59636D"/></pattern></defs>',
    rect(0, 0, W, H, WHITE, WHITE),
]
parts.append(t(W / 2, 38, "Sampling, comparator-population and protocol sensitivity", 27, "middle", 700))
panel_lefts = [20, 420, 820]
panel_w = 360
plot_top, plot_bottom = 145, 450


def axes(left: float, ymax: float, ticks: list[float], ylabel: str) -> None:
    for val in ticks:
        y = plot_bottom - (val / ymax) * (plot_bottom - plot_top)
        parts.append(line(left + 55, y, left + panel_w - 10, y))
        parts.append(t(left + 48, y + 6, f"{val:g}", 15, "end", 600))
    parts.append(line(left + 55, plot_top, left + 55, plot_bottom, INK, 1.3))
    parts.append(line(left + 55, plot_bottom, left + panel_w - 10, plot_bottom, INK, 1.3))
    centre = (plot_top + plot_bottom) / 2
    parts.append(f'<text x="{left + 12:.1f}" y="{centre:.1f}" text-anchor="middle" transform="rotate(-90 {left + 12:.1f} {centre:.1f})" font-family="Arial, Helvetica, sans-serif" font-size="17" font-weight="700" fill="{INK}">{html.escape(ylabel)}</text>')


# Panel (a): grouped bars.
left = panel_lefts[0]
parts.append(t(left, 78, "(a) Source-selection stability", 20, "start", 700))
axes(left, 110, [0, 25, 50, 75, 100], "Exact layout match (%)")
budgets = [2, 3, 4]
centres = [left + 105, left + 205, left + 305]
for i, budget in enumerate(budgets):
    for j, (project, colour) in enumerate([("SKS02", BLUE), ("SKS03", "url(#orangeHatch)")]):
        value = 100 * float(next(row["exact_layout_match_rate"] for row in loo if row["source_project"] == project and int(row["added_budget_b"]) == budget))
        x = centres[i] - 31 + j * 34
        h = (value / 110) * (plot_bottom - plot_top)
        parts.append(rect(x, plot_bottom - h, 30, h, colour))
        parts.append(t(x + 15, plot_bottom - h - 8, f"{value:.0f}%", 15, "middle", 700))
    parts.append(t(centres[i], 474, str(budget), 16, "middle", 700))
parts.append(t(left + 205, 506, "Added stations, b  (total nₛ = b + 1)", 16, "middle", 700))
parts.append(rect(left + 92, 531, 20, 14, BLUE)); parts.append(t(left + 120, 544, "SKS02", 15, "start", 700))
parts.append(rect(left + 220, 531, 20, 14, "url(#orangeHatch)")); parts.append(t(left + 248, 544, "SKS03", 15, "start", 700))

# Panel (b): exhaustive target-grid rank.
left = panel_lefts[1]
parts.append(t(left, 78, "(b) Target-grid comparator rank", 20, "start", 700))
axes(left, 100, [0, 25, 50, 75, 100], "Admissible layouts outperformed (%)")
parts.append(line(left + 55, plot_bottom - 0.5 * (plot_bottom - plot_top), left + panel_w - 10, plot_bottom - 0.5 * (plot_bottom - plot_top), NAVY, 1.5, "7 5"))
for i, budget in enumerate(budgets):
    for j, (source, target, colour) in enumerate([("SKS02", "SKS03", BLUE), ("SKS03", "SKS02", "url(#orangeHatch)")]):
        value = 100 * float(next(row["fraction_admissible_layouts_outperformed"] for row in ranks if row["selection_test"] == source and row["evaluation_test"] == target and int(row["additional_stations_b"]) == budget))
        x = centres[i] - panel_lefts[0] + left - 31 + j * 34
        h = (value / 100) * (plot_bottom - plot_top)
        parts.append(rect(x, plot_bottom - h, 30, h, colour))
        parts.append(t(x + 15, plot_bottom - h - 8, f"{value:.0f}%", 15, "middle", 700))
    parts.append(t(centres[i] - panel_lefts[0] + left, 474, str(budget), 16, "middle", 700))
parts.append(t(left + 205, 506, "Added stations, b  (total nₛ = b + 1)", 16, "middle", 700))
parts.append(rect(left + 42, 531, 19, 14, BLUE)); parts.append(t(left + 68, 544, "SKS02→SKS03", 13, "start", 700))
parts.append(rect(left + 196, 531, 19, 14, "url(#orangeHatch)")); parts.append(t(left + 222, 544, "SKS03→SKS02", 13, "start", 700))
parts.append(line(left + 42, 570, left + 62, 570, NAVY, 1.8, "7 5")); parts.append(t(left + 68, 576, "Median rank", 13, "start", 700))

# Panel (c): stacked classification counts.
left = panel_lefts[2]
parts.append(t(left, 78, "(c) Full-scale protocol sensitivity", 20, "start", 700))
axes(left, 24, [0, 6, 12, 18, 24], "Profile comparisons (total = 24)")
protocol_order = [("original", "linear", "Original", "linear"), ("original", "pchip", "Original", "PCHIP"), ("boundary_constrained", "linear", "Boundary", "linear"), ("boundary_constrained", "pchip", "Boundary", "PCHIP")]
centres3 = [left + 90, left + 170, left + 250, left + 330]
for i, (family, method, label1, label2) in enumerate(protocol_order):
    row = next(r for r in protocol if r["layout_family"] == family and r["interpolator"] == method)
    counts = [int(row["candidate_lower_error"]), int(row["practically_indistinguishable"]), int(row["candidate_higher_error"])]
    colours = [BLUE, "url(#greyDots)", "url(#orangeHatch)"]
    y = plot_bottom
    for count, colour in zip(counts, colours):
        h = (count / 24) * (plot_bottom - plot_top)
        y -= h
        parts.append(rect(centres3[i] - 24, y, 48, h, colour))
    parts.append(t(centres3[i], plot_top - 10, "/".join(map(str, counts)), 15, "middle", 700))
    parts.append(t(centres3[i], 474, label1, 13, "middle", 700))
    parts.append(t(centres3[i], 491, label2, 13, "middle", 700))
parts.append(t(left + 210, 519, "Layout constraint and interpolation", 16, "middle", 700))
parts.append(rect(left + 30, 538, 18, 13, BLUE)); parts.append(t(left + 55, 550, "Lower", 13, "start", 700))
parts.append(rect(left + 119, 538, 18, 13, "url(#greyDots)")); parts.append(t(left + 144, 550, "Equivalent", 13, "start", 700))
parts.append(rect(left + 238, 538, 18, 13, "url(#orangeHatch)")); parts.append(t(left + 263, 550, "Higher", 13, "start", 700))
parts.append(t(left + 30, 582, "Counts above bars: lower/equivalent/higher", 13, "start", 600))
parts.append(t(left + 30, 603, "Equivalence tolerance: ±0.005 NRMSE", 13, "start", 600))
parts.append(t(20, 638, "Selection uses source responses only; target responses are used solely for the frozen-layout audit.", 15, "start", 600))
parts.append("</svg>")

target = ROOT / "Fig08_Methodological_Sensitivity.svg"
target.write_text("\n".join(parts), encoding="utf-8")
print(target)
