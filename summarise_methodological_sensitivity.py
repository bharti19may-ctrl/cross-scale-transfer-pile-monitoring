"""Create compact, reproducible summaries for the methodological sensitivity audit."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict[str, object]], fields: list[str]) -> None:
    with (ROOT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


loo = read_csv("prj2828_leave_one_event_out_layout_stability.csv")
loo_groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
for row in loo:
    loo_groups[(row["source_test"], int(row["additional_stations_b"]))].append(row)

loo_summary: list[dict[str, object]] = []
for (source, budget), rows in sorted(loo_groups.items()):
    deltas = [float(row["heldout_delta"]) for row in rows]
    exact = [row["exact_layout_match"].strip().lower() == "true" for row in rows]
    loo_summary.append(
        {
            "source_project": source,
            "added_budget_b": budget,
            "total_retained_stations_ns": budget + 1,
            "heldout_events": len(rows),
            "exact_layout_match_rate": sum(exact) / len(exact),
            "candidate_better_events": sum(delta < 0 for delta in deltas),
            "median_heldout_delta_candidate_minus_uniform": statistics.median(deltas),
            "min_heldout_delta": min(deltas),
            "max_heldout_delta": max(deltas),
        }
    )

write_csv(
    "prj2828_loo_summary.csv",
    loo_summary,
    [
        "source_project",
        "added_budget_b",
        "total_retained_stations_ns",
        "heldout_events",
        "exact_layout_match_rate",
        "candidate_better_events",
        "median_heldout_delta_candidate_minus_uniform",
        "min_heldout_delta",
        "max_heldout_delta",
    ],
)

protocol = read_csv("external_protocol_sensitivity_summary.csv")
protocol_005 = [row for row in protocol if abs(float(row["practical_equivalence_tolerance"]) - 0.005) < 1e-12]

rank_rows = read_csv("prj2828_admissible_layout_target_rank.csv")
boundary_rows = read_csv("prj2828_boundary_constrained_transfer.csv")
curvature_rows = read_csv("prj2828_curvature_selection_diagnostic.csv")

key_results = {
    "leave_one_event_out": loo_summary,
    "all_admissible_layout_target_rank": rank_rows,
    "boundary_constrained_transfer": boundary_rows,
    "external_protocols_at_equivalence_tolerance_0_005": protocol_005,
    "curvature_diagnostic": curvature_rows,
    "interpretation_boundary": (
        "These summaries quantify selection stability, comparator-population rank, and protocol "
        "sensitivity. They do not establish a universal layout or field-performance validation."
    ),
}
with (ROOT / "methodological_sensitivity_key_results.json").open("w", encoding="utf-8") as stream:
    json.dump(key_results, stream, indent=2)

print(json.dumps(key_results, indent=2))
