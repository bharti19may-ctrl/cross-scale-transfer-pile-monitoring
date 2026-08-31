from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
INK = "#172033"
MUTED = "#374151"
GRID = "#DCE2EA"
BLUE = "#1769AA"
ORANGE = "#D97706"
OLIVE = "#71852A"
PALE_BLUE = "#EAF3FA"


def text(x, y, value, size=24, anchor="start", weight=600, colour=INK):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial,Helvetica,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{colour}">{html.escape(str(value))}</text>'
    )


def svg_open(width, height, title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        text(80, 62, title, 34, weight=700),
        text(80, 103, subtitle, 23, colour=MUTED),
    ]


def figure_transfer():
    df = pd.read_csv(Path(__file__).with_name("prj2828_independent_transfer_comparisons.csv"))
    df = df[df["additional_stations_b"].isin([2, 3, 4])].copy()
    width, height = 1800, 1120
    svg = svg_open(
        width,
        height,
        "Independent layout transfer relative to uniform placement",
        "Paired earthquake-level change in p90 NRMSE; negative values favour transfer; intervals resample earthquake clusters",
    )
    svg += [
        '<circle cx="90" cy="147" r="10" fill="#1769AA"/><text x="112" y="155" font-family="Arial" font-size="22" fill="#172033">cross-regime dynamic layout</text>',
        '<rect x="470" y="137" width="19" height="19" fill="#FFFFFF" stroke="#D97706" stroke-width="4"/><text x="501" y="155" font-family="Arial" font-size="22" fill="#172033">PRJ-4809 static layout</text>',
    ]
    xmin, xmax = -0.26, 0.14
    panel_top, panel_h, panel_w = 235, 690, 710
    for panel_i, test_name in enumerate(["SKS02: uniform deposit", "SKS03: interbedded deposit"]):
        test = test_name.split(":")[0]
        x0 = 90 + panel_i * 855
        plot_left, plot_right = x0 + 205, x0 + panel_w - 35
        plot_w = plot_right - plot_left
        svg.append(text(x0, 218, test_name, 27, weight=700))
        for tick in np.arange(-0.2, 0.101, 0.1):
            xx = plot_left + (tick - xmin) / (xmax - xmin) * plot_w
            svg.append(f'<line x1="{xx:.1f}" y1="{panel_top}" x2="{xx:.1f}" y2="{panel_top+panel_h}" stroke="{GRID}" stroke-width="2"/>')
            svg.append(text(xx, panel_top + panel_h + 38, f"{tick:+.1f}", 21, anchor="middle", colour=MUTED))
        zero = plot_left + (0 - xmin) / (xmax - xmin) * plot_w
        svg.append(f'<line x1="{zero:.1f}" y1="{panel_top}" x2="{zero:.1f}" y2="{panel_top+panel_h}" stroke="{INK}" stroke-width="3"/>')
        sub = df[df["evaluation_test"].eq(test)]
        row_y = panel_top + 70
        for budget in (2, 3, 4):
            svg.append(text(x0 + 5, row_y + 18, f"{budget} added stations", 23, weight=600))
            for offset, evidence, label, colour, shape in [
                (-15, f"cross_regime_trained_{'SKS03' if test == 'SKS02' else 'SKS02'}", "cross", BLUE, "circle"),
                (35, "independent_static_PRJ4809", "static", ORANGE, "square"),
            ]:
                row = sub[(sub["additional_stations_b"].eq(budget)) & sub["evidence_type"].eq(evidence)].iloc[0]
                yy = row_y + offset
                lo = plot_left + (row.bootstrap_95_ci_low - xmin) / (xmax - xmin) * plot_w
                hi = plot_left + (row.bootstrap_95_ci_high - xmin) / (xmax - xmin) * plot_w
                xx = plot_left + (row.median_paired_delta_candidate_minus_uniform - xmin) / (xmax - xmin) * plot_w
                svg.append(text(x0 + 185, yy + 8, label, 19, anchor="end", colour=MUTED))
                svg.append(f'<line x1="{lo:.1f}" y1="{yy:.1f}" x2="{hi:.1f}" y2="{yy:.1f}" stroke="{colour}" stroke-width="5"/>')
                svg.append(f'<line x1="{lo:.1f}" y1="{yy-9:.1f}" x2="{lo:.1f}" y2="{yy+9:.1f}" stroke="{colour}" stroke-width="4"/><line x1="{hi:.1f}" y1="{yy-9:.1f}" x2="{hi:.1f}" y2="{yy+9:.1f}" stroke="{colour}" stroke-width="4"/>')
                if shape == "circle":
                    svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="10" fill="{colour}" stroke="#FFFFFF" stroke-width="3"/>')
                else:
                    svg.append(f'<rect x="{xx-9:.1f}" y="{yy-9:.1f}" width="18" height="18" fill="#FFFFFF" stroke="{colour}" stroke-width="4"/>')
            if budget < 4:
                svg.append(f'<line x1="{x0}" y1="{row_y+84}" x2="{x0+panel_w}" y2="{row_y+84}" stroke="#EEF1F5" stroke-width="2"/>')
            row_y += 220
        svg.append(f'<line x1="{plot_left}" y1="{panel_top+panel_h}" x2="{plot_right}" y2="{panel_top+panel_h}" stroke="{INK}" stroke-width="3"/>')
    svg.append(text(width / 2, 1012, "Median paired change in p90 NRMSE (candidate − uniform)", 25, anchor="middle", weight=600))
    svg.append(text(80, 1070, "Repeated piles are aggregated within each earthquake (SKS02 n=6; SKS03 n=5). Intervals crossing zero do not establish an improvement.", 21, colour=MUTED))
    svg.append("</svg>")
    (OUT / "Fig_transfer_vs_uniform.svg").write_text("\n".join(svg), encoding="utf-8")


def figure_regime():
    df = pd.read_csv(Path(__file__).with_name("prj2828_event_regime_transfer.csv"))
    df = df[df["additional_stations_b"].eq(3)].copy()
    width, height = 1800, 1080
    svg = svg_open(
        width,
        height,
        "Measured pore-pressure regime and cross-regime transfer",
        "Three added stations; event values aggregate the available piles; negative reconstruction-error change favours transfer",
    )
    panel_top, panel_h, panel_w = 190, 690, 720
    ymin, ymax = -0.16, 0.08
    for panel_i, test_name in enumerate(["SKS02: uniform deposit", "SKS03: interbedded deposit"]):
        test = test_name.split(":")[0]
        sub = df[df["test"].eq(test)].sort_values("median_sensor_peak_ru")
        x0 = 110 + panel_i * 850
        plot_left, plot_right = x0 + 90, x0 + panel_w
        plot_w = plot_right - plot_left
        xmin = 0.0
        xmax = max(1.1, float(sub["median_sensor_peak_ru"].max()) * 1.08)
        svg.append(text(x0, 172, test_name, 27, weight=700))
        for tick in np.arange(0, 1.01, 0.2):
            xx = plot_left + (tick - xmin) / (xmax - xmin) * plot_w
            svg.append(f'<line x1="{xx:.1f}" y1="{panel_top}" x2="{xx:.1f}" y2="{panel_top+panel_h}" stroke="{GRID}" stroke-width="2"/>')
            svg.append(text(xx, panel_top + panel_h + 36, f"{tick:.1f}", 21, anchor="middle", colour=MUTED))
        for tick in np.arange(-0.15, 0.051, 0.05):
            yy = panel_top + (ymax - tick) / (ymax - ymin) * panel_h
            svg.append(f'<line x1="{plot_left}" y1="{yy:.1f}" x2="{plot_right}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="2"/>')
            svg.append(text(plot_left - 18, yy + 7, f"{tick:+.2f}", 20, anchor="end", colour=MUTED))
        zero_y = panel_top + (ymax - 0) / (ymax - ymin) * panel_h
        svg.append(f'<line x1="{plot_left}" y1="{zero_y:.1f}" x2="{plot_right}" y2="{zero_y:.1f}" stroke="{INK}" stroke-width="3"/>')
        label_offsets = {
            ("SKS02", "EQM_1"): (18, -48, "start"),
            ("SKS02", "EQM_2"): (20, -52, "start"),
            ("SKS02", "EQM_3"): (34, 62, "start"),
            ("SKS02", "EQM_4"): (-22, 62, "end"),
            ("SKS02", "EQM_5"): (20, 66, "start"),
            ("SKS02", "EQM_6"): (-28, -52, "end"),
            ("SKS03", "EQM_1"): (24, -52, "start"),
            ("SKS03", "EQM_2"): (-24, 62, "end"),
            ("SKS03", "EQM_3"): (24, -52, "start"),
            ("SKS03", "EQM_4"): (-24, -52, "end"),
            ("SKS03", "EQM_5"): (24, 62, "start"),
        }
        for _, row in sub.reset_index(drop=True).iterrows():
            xx = plot_left + (row.median_sensor_peak_ru - xmin) / (xmax - xmin) * plot_w
            yy = panel_top + (ymax - row.cross_regime_minus_uniform) / (ymax - ymin) * panel_h
            svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="12" fill="{BLUE}" stroke="#FFFFFF" stroke-width="3"/>')
            dx, dy, anchor = label_offsets[(test, row.event)]
            label_x, label_y = xx + dx, yy + dy
            line_end_x = label_x - 7 if anchor == "start" else label_x + 7
            line_end_y = label_y + (9 if dy < 0 else -18)
            svg.append(f'<line x1="{xx:.1f}" y1="{yy:.1f}" x2="{line_end_x:.1f}" y2="{line_end_y:.1f}" stroke="#8090A3" stroke-width="2"/>')
            svg.append(text(label_x, label_y, row.event.replace("_", " "), 19, anchor=anchor, weight=600, colour=INK))
        svg.append(f'<line x1="{plot_left}" y1="{panel_top}" x2="{plot_left}" y2="{panel_top+panel_h}" stroke="{INK}" stroke-width="3"/><line x1="{plot_left}" y1="{panel_top+panel_h}" x2="{plot_right}" y2="{panel_top+panel_h}" stroke="{INK}" stroke-width="3"/>')
    svg.append(text(width / 2, 990, "Median of sensor-wise peak excess pore-pressure ratio, rᵤ", 25, anchor="middle", weight=600))
    svg.append(text(63, 545, "Δ p90 NRMSE (cross-regime − uniform)", 23, anchor="middle", weight=600).replace('<text ', '<text transform="rotate(-90 63 545)" '))
    svg.append(text(80, 1040, "Descriptive event coupling only (six SKS02 and five SKS03 earthquakes); it is not a causal or significance claim.", 21, colour=MUTED))
    svg.append("</svg>")
    (OUT / "Fig_environment_coupling.svg").write_text("\n".join(svg), encoding="utf-8")


def figure_layouts():
    layouts = [
        ("Uniform", [0.000, 0.286, 0.714, 1.000], INK, "circle"),
        ("PRJ-4809 static", [0.000, 0.429, 0.857, 1.000], ORANGE, "square"),
        ("Selected on SKS02", [0.000, 0.143, 0.714, 0.857], BLUE, "circle"),
        ("Selected on SKS03", [0.000, 0.143, 0.429, 1.000], BLUE, "diamond"),
    ]
    width, height = 1800, 950
    svg = svg_open(
        width,
        height,
        "Candidate layouts on the common measured span",
        "Illustration for three added stations plus the fixed upper boundary; depths are discrete measured gage levels",
    )
    top, bottom = 205, 760
    for i, (name, depths, colour, shape) in enumerate(layouts):
        x = 260 + i * 420
        svg.append(text(x, 165, name, 25, anchor="middle", weight=700))
        svg.append(f'<rect x="{x-42}" y="{top}" width="84" height="{bottom-top}" rx="36" fill="{PALE_BLUE}" stroke="#8090A3" stroke-width="3"/>')
        for d in depths:
            y = top + d * (bottom - top)
            if shape == "square":
                svg.append(f'<rect x="{x-13}" y="{y-13:.1f}" width="26" height="26" fill="#FFFFFF" stroke="{colour}" stroke-width="5"/>')
            elif shape == "diamond":
                svg.append(f'<rect x="{x-12}" y="{y-12:.1f}" width="24" height="24" fill="{colour}" stroke="#FFFFFF" stroke-width="3" transform="rotate(45 {x} {y:.1f})"/>')
            else:
                svg.append(f'<circle cx="{x}" cy="{y:.1f}" r="14" fill="{colour}" stroke="#FFFFFF" stroke-width="3"/>')
            svg.append(text(x + 64, y + 8, f"{d:.3f}", 21, colour=MUTED))
    svg.append(text(105, 185, "z", 23, anchor="middle", weight=700))
    svg.append(f'<line x1="105" y1="{top}" x2="105" y2="{bottom}" stroke="{INK}" stroke-width="3"/>')
    svg.append(f'<path d="M95 {bottom-16} L105 {bottom} L115 {bottom-16}" fill="none" stroke="{INK}" stroke-width="3"/>')
    svg.append(text(130, top + 8, "0 (upper)", 21, colour=MUTED))
    svg.append(text(130, bottom + 8, "1 (deeper)", 21, colour=MUTED))
    svg.append(text(90, 870, "The layouts are hypotheses for reconstruction on the eight-gage reference grid; they are not universal field-design prescriptions.", 22, colour=MUTED))
    svg.append("</svg>")
    (OUT / "Fig_layout_comparison.svg").write_text("\n".join(svg), encoding="utf-8")


def figure_forward_sequence():
    df = pd.read_csv(Path(__file__).with_name("prj2828_forward_sequence_results.csv"))
    df = df[df["strategy"].eq("first_event_frozen")].copy()
    width, height = 1800, 1080
    svg = svg_open(
        width,
        height,
        "Forward-sequence reconstruction after freezing the first-event layout",
        "The layout is selected using EQM 1 only and evaluated on every later earthquake; negative values favour it over uniform placement",
    )
    svg += [
        '<circle cx="95" cy="148" r="10" fill="#1769AA"/><text x="118" y="156" font-family="Arial" font-size="22" fill="#172033">2 added stations</text>',
        '<rect x="355" y="138" width="19" height="19" fill="#FFFFFF" stroke="#D97706" stroke-width="4"/><text x="389" y="156" font-family="Arial" font-size="22" fill="#172033">3 added stations</text>',
        '<path d="M665 159 L677 137 L689 159 Z" fill="#71852A"/><text x="704" y="156" font-family="Arial" font-size="22" fill="#172033">4 added stations</text>',
    ]
    ymin, ymax = -0.24, 0.06
    panel_top, panel_h, panel_w = 240, 650, 700
    styles = {
        2: (BLUE, "", "circle"),
        3: (ORANGE, "14 9", "square"),
        4: (OLIVE, "4 9", "triangle"),
    }
    for panel_i, test_name in enumerate(["SKS02: uniform deposit", "SKS03: interbedded deposit"]):
        test = test_name.split(":")[0]
        sub = df[df["test"].eq(test)].copy()
        events = sorted(sub["evaluation_event"].unique(), key=lambda value: int(value.split("_")[-1]))
        x0 = 115 + panel_i * 850
        plot_left, plot_right = x0 + 95, x0 + panel_w
        plot_w = plot_right - plot_left
        svg.append(text(x0, 218, test_name, 27, weight=700))
        for tick in np.arange(-0.2, 0.051, 0.05):
            yy = panel_top + (ymax - tick) / (ymax - ymin) * panel_h
            svg.append(f'<line x1="{plot_left}" y1="{yy:.1f}" x2="{plot_right}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="2"/>')
            svg.append(text(plot_left - 18, yy + 7, f"{tick:+.2f}", 20, anchor="end", colour=MUTED))
        zero_y = panel_top + (ymax - 0.0) / (ymax - ymin) * panel_h
        svg.append(f'<line x1="{plot_left}" y1="{zero_y:.1f}" x2="{plot_right}" y2="{zero_y:.1f}" stroke="{INK}" stroke-width="3"/>')
        positions = {
            event: plot_left + i * plot_w / max(1, len(events) - 1)
            for i, event in enumerate(events)
        }
        for event, xx in positions.items():
            svg.append(text(xx, panel_top + panel_h + 38, event.replace("_", " "), 21, anchor="middle", colour=MUTED))
        for budget, (colour, dash, marker) in styles.items():
            series = sub[sub["additional_stations_b"].eq(budget)].sort_values(
                "evaluation_event", key=lambda col: col.map(lambda value: int(value.split("_")[-1]))
            )
            points = []
            for _, row in series.iterrows():
                xx = positions[row.evaluation_event]
                yy = panel_top + (ymax - row.candidate_minus_uniform) / (ymax - ymin) * panel_h
                points.append((xx, yy))
            coords = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{colour}" stroke-width="5"{dash_attr}/>' )
            for xx, yy in points:
                if marker == "circle":
                    svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="10" fill="{colour}" stroke="#FFFFFF" stroke-width="3"/>')
                elif marker == "square":
                    svg.append(f'<rect x="{xx-9:.1f}" y="{yy-9:.1f}" width="18" height="18" fill="#FFFFFF" stroke="{colour}" stroke-width="4"/>')
                else:
                    svg.append(f'<path d="M{xx:.1f} {yy-11:.1f} L{xx+11:.1f} {yy+10:.1f} L{xx-11:.1f} {yy+10:.1f} Z" fill="{colour}" stroke="#FFFFFF" stroke-width="2"/>')
        svg.append(f'<line x1="{plot_left}" y1="{panel_top}" x2="{plot_left}" y2="{panel_top+panel_h}" stroke="{INK}" stroke-width="3"/><line x1="{plot_left}" y1="{panel_top+panel_h}" x2="{plot_right}" y2="{panel_top+panel_h}" stroke="{INK}" stroke-width="3"/>')
    svg.append(text(65, 565, "Δ p90 NRMSE (frozen layout − uniform)", 23, anchor="middle", weight=600).replace('<text ', '<text transform="rotate(-90 65 565)" '))
    svg.append(text(80, 1005, "This is temporal extrapolation within each physical test, not an independent field experiment or an adaptive installation protocol.", 21, colour=MUTED))
    svg.append("</svg>")
    (OUT / "Fig_forward_sequence.svg").write_text("\n".join(svg), encoding="utf-8")


def figure_sensitivity():
    df = pd.read_csv(Path(__file__).with_name("prj2828_sensitivity_summary.csv"))
    width, height = 1800, 1080
    svg = svg_open(
        width,
        height,
        "Sensitivity to smoothing and NRMSE normalisation floor",
        "Twelve prespecified combinations per direction and station budget; each layout is reselected under its own specification",
    )
    svg += [
        '<circle cx="95" cy="150" r="7" fill="#FFFFFF" stroke="#8090A3" stroke-width="3"/><text x="116" y="158" font-family="Arial" font-size="22" fill="#172033">alternative specification</text>',
        '<path d="M500 138 L512 150 L500 162 L488 150 Z" fill="#1769AA" stroke="#FFFFFF" stroke-width="2"/><text x="530" y="158" font-family="Arial" font-size="22" fill="#172033">nominal smoothing; 0.05 MN floor</text>',
        '<line x1="970" y1="150" x2="1025" y2="150" stroke="#172033" stroke-width="5"/><text x="1042" y="158" font-family="Arial" font-size="22" fill="#172033">full specification range</text>',
    ]
    ymin, ymax = -0.14, 0.05
    panel_top, panel_h, panel_w = 245, 650, 710
    directions = ["SKS03 to SKS02", "SKS02 to SKS03"]
    panel_titles = ["Selected on SKS03 → evaluated on SKS02", "Selected on SKS02 → evaluated on SKS03"]
    smoothing_order = {name: i for i, name in enumerate(["none", "half_reported", "reported", "double_reported"])}
    floor_order = {value: i for i, value in enumerate([0.025, 0.05, 0.10])}
    jitter = np.linspace(-24, 24, 12)
    for panel_i, (direction, panel_title) in enumerate(zip(directions, panel_titles)):
        sub = df[df["direction"].eq(direction)].copy()
        x0 = 105 + panel_i * 855
        plot_left, plot_right = x0 + 105, x0 + panel_w
        plot_w = plot_right - plot_left
        svg.append(text(x0, 220, panel_title, 26, weight=700))
        for tick in np.arange(-0.125, 0.051, 0.025):
            yy = panel_top + (ymax - tick) / (ymax - ymin) * panel_h
            svg.append(f'<line x1="{plot_left}" y1="{yy:.1f}" x2="{plot_right}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="2"/>')
            svg.append(text(plot_left - 18, yy + 7, f"{tick:+.3f}", 19, anchor="end", colour=MUTED))
        zero_y = panel_top + (ymax - 0.0) / (ymax - ymin) * panel_h
        svg.append(f'<line x1="{plot_left}" y1="{zero_y:.1f}" x2="{plot_right}" y2="{zero_y:.1f}" stroke="{INK}" stroke-width="3"/>')
        for budget_i, budget in enumerate((2, 3, 4)):
            xx = plot_left + (budget_i + 0.5) * plot_w / 3
            rows = sub[sub["additional_stations_b"].eq(budget)].copy()
            rows["order"] = rows.apply(
                lambda row: smoothing_order[row.smoothing] * 3 + floor_order[float(row.normalisation_floor_mn)], axis=1
            )
            rows = rows.sort_values("order")
            lo = float(rows["median_candidate_minus_uniform"].min())
            hi = float(rows["median_candidate_minus_uniform"].max())
            y_lo = panel_top + (ymax - lo) / (ymax - ymin) * panel_h
            y_hi = panel_top + (ymax - hi) / (ymax - ymin) * panel_h
            svg.append(f'<line x1="{xx:.1f}" y1="{y_hi:.1f}" x2="{xx:.1f}" y2="{y_lo:.1f}" stroke="{INK}" stroke-width="5"/>')
            svg.append(f'<line x1="{xx-13:.1f}" y1="{y_hi:.1f}" x2="{xx+13:.1f}" y2="{y_hi:.1f}" stroke="{INK}" stroke-width="4"/><line x1="{xx-13:.1f}" y1="{y_lo:.1f}" x2="{xx+13:.1f}" y2="{y_lo:.1f}" stroke="{INK}" stroke-width="4"/>')
            for j, (_, row) in enumerate(rows.iterrows()):
                yy = panel_top + (ymax - row.median_candidate_minus_uniform) / (ymax - ymin) * panel_h
                xj = xx + jitter[j]
                if row.smoothing == "reported" and abs(float(row.normalisation_floor_mn) - 0.05) < 1e-9:
                    svg.append(f'<path d="M{xj:.1f} {yy-11:.1f} L{xj+11:.1f} {yy:.1f} L{xj:.1f} {yy+11:.1f} L{xj-11:.1f} {yy:.1f} Z" fill="{BLUE}" stroke="#FFFFFF" stroke-width="2"/>')
                else:
                    svg.append(f'<circle cx="{xj:.1f}" cy="{yy:.1f}" r="7" fill="#FFFFFF" stroke="#8090A3" stroke-width="3"/>')
            svg.append(text(xx, panel_top + panel_h + 38, f"b = {budget}", 21, anchor="middle", colour=MUTED))
            label_y = min(panel_top + panel_h - 10, y_lo + 34)
            svg.append(text(xx, label_y, f"{lo:+.3f} to {hi:+.3f}", 18, anchor="middle", colour=MUTED))
        svg.append(f'<line x1="{plot_left}" y1="{panel_top}" x2="{plot_left}" y2="{panel_top+panel_h}" stroke="{INK}" stroke-width="3"/><line x1="{plot_left}" y1="{panel_top+panel_h}" x2="{plot_right}" y2="{panel_top+panel_h}" stroke="{INK}" stroke-width="3"/>')
    svg.append(text(64, 570, "Median Δ p90 NRMSE (candidate − uniform)", 23, anchor="middle", weight=600).replace('<text ', '<text transform="rotate(-90 64 570)" '))
    svg.append(text(80, 1015, "Specification ranges are robustness envelopes, not confidence intervals. Direction is invariant: the three-retained-station SKS03-to-SKS02 candidate remains worse.", 21, colour=MUTED))
    svg.append("</svg>")
    (OUT / "Fig_sensitivity_envelope.svg").write_text("\n".join(svg), encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    figure_transfer()
    figure_regime()
    figure_layouts()
    figure_forward_sequence()
    figure_sensitivity()
    print(f"Wrote 5 SVG figures to {OUT}")


if __name__ == "__main__":
    main()
