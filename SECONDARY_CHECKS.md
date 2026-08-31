# Secondary Checks

These checks are kept outside the primary benchmark because their measured grids, resolution, or data quality differ from the common eight-level centrifuge reference and the primary full-scale evidence. They are not statistically pooled with the principal comparisons.

## Six-level SKS03 Pile 1 check

Five additional earthquake histories are available for SKS03 Pile 1 on a six-level measured grid. Missing Gage 4 and Gage 2 values are not imputed. For the `b = 2` comparison, both candidate and equal-budget uniform layouts are directly realisable. The SKS02-selected candidate has lower p90 NRMSE in all five histories; candidate-minus-uniform differences range from `−0.366` to `−0.071`, with a median of `−0.180`. This result supports the forward smallest-budget transfer on a different measured grid, but it is not treated as an additional common-grid experiment.

Reproducible values are supplied in `prj2828_sks03_pile1_partial_grid_results.csv` and `prj2828_sks03_pile1_partial_grid_summary.csv`; the calculation is implemented in `partial_grid_validation.py`.

## Quality-qualified Turrell concrete-pile profile

The concrete-pile profile is retained as a sensitivity because the source thesis reports unrealistic static strain readings following rainfall. Across the six mapped budget-layout comparisons, three have lower error than uniform placement, one is practically equivalent, and two have higher error at the declared `±0.005` NRMSE tolerance. Median candidate-minus-uniform NRMSE is `−0.016`. The result is not combined with the primary H- and pipe-pile evidence.

## Five-level Cambridge profile

The Cambridge MS09 bored-pile profile at 2,500 s after shaking contains five published levels. Three realised comparisons have higher error than uniform placement and one is practically equivalent. Two budget-layout combinations are omitted because nearest-level mapping produces duplicate stations and the intended station count is not realised. This case is a resolution-limited centrifuge sensitivity rather than a full-scale validation.

The external-profile coordinates, requested and realised station depths, duplicate-station checks, and comparison results are supplied in `external_digitised_profiles.csv`, `external_transfer_results.csv`, `external_transfer_summary.csv`, and `external_protocol_sensitivity_results.csv`.
