# Supplemental Materials

## Article

Target-Response-Blind Portability Audit of Sparse Axial-Load Monitoring Layouts for Piles under Liquefaction-Induced Downdrag

## Contents and reproduction order

1. `transfer_audit.py` inventories and qualifies the processed axial-load files, selects discrete layouts, evaluates held-out transfer, aggregates repeated piles within each earthquake, and performs the earthquake-cluster bootstrap.
2. `forward_sequence.py` selects each layout from EQM1 and evaluates it on later earthquakes without re-optimisation.
3. `sensitivity_audit.py` repeats the transfer analysis for four smoothing choices and three NRMSE floors.
4. `partial_grid_validation.py` evaluates the SKS03 Pile 1 six-level secondary check without imputing missing targets.
5. `environment_coupling.py` derives co-located PGA and pore-pressure descriptors for descriptive physical-regime interpretation.
6. `objective_transfer_audit.py` evaluates peak-load magnitude, discrete peak-load level, and the five-added-station saturation check using the same earthquake-level pairing.
7. `digitize_external_profiles.py` records the graphical-extraction procedure for the published Turrell, Christchurch, and Cambridge profile coordinates. The supplied `external_digitised_profiles.csv` is the fixed input used by the external-profile calculation, and `external_digitisation_qa.csv` records the between-figure extraction check for the Turrell profiles.
8. `analyse_external_transfer.py` maps the frozen SKS02 and SKS03 layouts to the nearest published levels, applies the equal-budget uniform comparator, and writes the independent-profile results and evidence-class summary.
9. `methodological_sensitivity_audit.py` performs leave-one-earthquake-out source selection, exhaustive target-grid ranking, lower-boundary-constrained selection, measured-grid second-difference diagnostics, and full-scale interpolation, boundary, mapping, and equivalence-tolerance checks. `summarise_methodological_sensitivity.py` writes the compact summaries used in the article.
10. `SECONDARY_CHECKS.md` records the six-level SKS03 Pile 1, quality-qualified Turrell concrete-pile, and five-level Cambridge results that are kept outside the primary benchmark.
11. The figure builders listed below reproduce all main and supplementary figures in SVG and 1200-dpi PNG format.
12. `field_profile_transfer.py` evaluates the selected digital Delft reference profiles; `extraction_resolution_sensitivity.py` evaluates graphical-resolution scenarios and error-coverage decomposition.
13. `phase_transfer.py` evaluates the frozen layouts separately during shaking and before/after an operational pressure-decay boundary. `verify_phase.py` independently recomputes all phase metrics using interpolation-weight matrices.

The CSV files contain the evidence architecture, qualification inventory, selected layouts, paired event-level comparisons, cluster summaries, forward-sequence checks, specification sensitivity, six-level check, environmental descriptors, objective-specific results, external digitised coordinates, extraction-quality values, frozen-layout external-profile comparisons, leave-one-earthquake-out stability, exhaustive target-grid ranks, boundary-constrained transfer, second-difference descriptors, and full-scale protocol sensitivity. The JSON manifests record deterministic settings, compact key results, and evidence boundaries.

## Public source data

Raw source files and reports are not duplicated. They are publicly preserved at the following cited records:

- SKS02 centrifuge data: https://doi.org/10.17603/DS2-D25M-GG48
- SKS03 centrifuge data: https://doi.org/10.17603/DS2-WJGX-TB78
- PRJ-4809 static pile database: https://doi.org/10.17603/DS2-BEZ6-4812
- Turrell driven-pile thesis: https://scholarsarchive.byu.edu/etd/6966
- Christchurch CFA-pile TechBrief: https://www.fhwa.dot.gov/publications/research/infrastructure/structures/bridge/17060/17060.pdf
- Cambridge thesis: https://doi.org/10.17863/CAM.14025

Set `PRJ2828_DATA_ROOT` to the extracted PRJ-2828 data directory before running the centrifuge scripts. The included `negative_control_static_layouts.csv` records layouts derived from the cited PRJ-4809 database and used only as a static negative control. Time samples are deterministic computational points and are not independent replications; statistical resampling is performed at earthquake-event level after repeated piles are aggregated.

The external-profile scripts use frozen layouts and do not use target axial-load responses to relocate or select stations. Target metadata are used only for nearest-level mapping and subsequent scoring. Exact digitised coordinates and requested-versus-realised mapping fields are provided so the graphical extraction and station-count realisation can be inspected directly. The primary calculation uses linear interpolation with endpoint persistence; the declared sensitivity uses the monotone piecewise cubic Hermite construction of Fritsch and Carlson (1980), original or lower-boundary-constrained layouts, and equivalence tolerances of ±0.0025, ±0.005, and ±0.010 NRMSE.

To repeat the graphical extraction independently, render the cited PDF pages at their original scale and name the images `byu_p123.png`, `byu_p138.png`, `byu_p152.png`, `byu_p158.png`, `fhwa_p10.png`, and `cam_p200.png`. Place them in an `external_profile_source_images` folder beside the scripts, or set `EXTERNAL_PROFILE_IMAGE_ROOT` to that folder. The derived coordinates supplied here are sufficient to repeat the frozen-layout calculations and figure generation without redistributing source-page images.

## Evidence boundary

These materials reproduce reconstruction error, discrete peak-response descriptors, layout-selection stability, exhaustive target-grid ranks, and declared protocol sensitivities on measured or published profile grids. The second-difference quantity is a descriptive measured-grid profile-shape indicator, not a constitutive parameter or causal test. The peak-depth response is the measured-grid level of maximum axial load, not a continuous neutral-plane estimate. Counts across published profiles are descriptive comparisons, not field success probabilities. The materials do not establish sensor hardware reliability, installation cost, structural safety, failure probability, causal mechanism, or a universal monitoring layout.


## Executable reproduction commands

Python 3.12 was used. Install the packages listed in `requirements.txt` in a
separate environment. For the raw centrifuge rerun, set `PRJ2828_DATA_ROOT` to
the directory containing the extracted SKS02/SKS03 processed-data trees.
Run the following from this directory, in order:

```text
python transfer_audit.py
python forward_sequence.py
python sensitivity_audit.py
python partial_grid_validation.py
python environment_coupling.py
python objective_transfer_audit.py
python methodological_sensitivity_audit.py
python summarise_methodological_sensitivity.py
```

The following checks run directly from this archive, without raw data or
source-page images:

```text
python analyse_external_transfer.py
python extraction_resolution_sensitivity.py
python field_profile_transfer.py
python phase_transfer.py
python verify_phase.py
```

To recreate the selected Delft profiles from the original download, run
`python field_profile_transfer.py --prepare-from PATH_TO_EXTRACTED_DELFT`
with `PATH_TO_EXTRACTED_DELFT` replaced by the extracted `delft` directory
(the directory containing `load-test`, not its parent).

The default external calculation reads `external_digitised_profiles.csv`.
Only `python analyse_external_transfer.py --redigitise` requires the separately
obtained source images and extraction-image settings described above.

To regenerate all ten figures from the supplied CSV results, run:

```text
python make_figures.py
python make_external_figure.py
python make_methodological_sensitivity_figure.py
python make_field_figure.py
python make_phase_figure.py
python -c "import pandas as pd; import objective_transfer_audit as a; a.make_figure(pd.read_csv(a.SUMMARY_OUTPUT))"
node render_figures_1200dpi.js
```

Node.js with the `sharp` package is required for the last command. The PNGs
are 8400 pixels wide at 1200 dpi. `FIGURE_MAP.csv` gives main/supplement
numbering and the script-generated filenames. The separate main-figure archive
uses `Fig_01_1200dpi.png` to `Fig_06_1200dpi.png`; its `FIGURE_INDEX.csv`
maps these names to the source filenames. Figs. S1–S4 are embedded in
`Supplemental_Methods_and_Tables.docx`; the retained figure-building scripts
regenerate the editable SVG sources when required.

## Delft field-reference data

The additional digital reference data are from Duffy, Gavin, de Lange and Korff
(2025), DOI https://doi.org/10.4121/78720ecb-daf4-4676-b5f4-281e93e4388a,
licensed CC BY 4.0. The derived `delft_selected_stage_profiles.npz` is supplied
under that attribution licence. It contains numeric arrays (no Python pickle).
The original `README.md`, `core.py` and Duffy et al. (2024),
https://doi.org/10.1061/JGGEFK.GTENG-12340, explain source processing.

These are author-interpreted forces, not a new calibration of strain or pile
stiffness. The source interpolated its output at 0.01 m; the instrument's
reported spatial resolution is 0.20 m. Evaluation uses 0.20- and 0.40-m
reference grids and does not count interpolated positions as independent
measurements. The separate head-load coordinate and recorded depth gaps are
retained without adding observations. F3 is excluded because the source does
not supply an interpreted profile following casing-extraction problems.
The five remaining piles belong to one site. Loading stages are repeated
measurements, not independent field experiments. No target response selects,
relocates or retunes a layout. Static compression is not liquefaction validation.

## Resolution and coverage checks

`extraction_resolution_sensitivity.py` applies common perturbations to candidate
and comparator. The 1-, 2- and 4-pixel amplitudes are specified graphical stress
scenarios, not calibrated confidence limits. Turrell load bounds additionally
cover half the between-figure range. Published Turrell gauge depths remain
fixed. FHWA line-location perturbations are smooth and correlated along depth;
101 trace ordinates do not represent 101 independently instrumented levels.
Every scenario and every comparison is retained, including unstable classes.

The error-coverage calculation separates signed contributions to squared
NRMSE inside and outside the candidate's retained span. The identity holds per
reference profile before any aggregation; separate medians need not add to the
median total. It diagnoses where reconstruction error occurs, not why the soil
or pile develops its mechanical response.

## Phase-specific diagnostics

The numeric `phase_selected_samples.npz` contains derived profiles from the cited
Sinha et al. (2020a, b) sources, with full column definitions and settings in
`phase_manifest.json`. It uses no Python pickle. `phase_boundaries.csv` records
the motion-energy interval, pressure-proxy peak and decay bounds;
`phase_sampling_audit.csv` records chronology, eligibility and nearest-time
sampling offsets. `phase_source_hashes.csv` identifies every input file.
To rebuild these derived samples from the original extracted source data, run
`python phase_transfer.py --prepare-from PATH_TO_PRJ2828_EXTRACTED`.

The whole-history benchmark uses file-order row windows, equivalent to 6 s for
Pile 1 and 3 s otherwise at the median sampling interval, and row-index-spaced
snapshots. Its `reported` result-field label denotes that implemented nominal
window, not the SKS03 source prescription of 6 s. The no/half/nominal/double
window sensitivity includes 6 s for the primary SKS03 piles in the double case.
All original errors use a snapshot-specific maximum-absolute-load scale with a
0.05-MN floor; no whole-history maximum is used.

The phase diagnostic instead removes overlapping times at the fast/slow join,
uses actual elapsed-time smoothing and selects the nearest distinct measured
times to evenly spaced targets within each phase. The supplied data include
both unsmoothed and smoothed cases, three pressure-decay fractions and two
sampling caps. A pressure-response floor of 0.10 excludes SKS02 EQM1 from
decay partitioning but not from shaking. The 12 settings reuse the same
experiments and are not independent validation replications. Before/after decay
does not mean before/after complete liquefaction or dissipation.

The phase tables report full-profile, peak-magnitude and discrete peak-depth
errors for b = 2–4. `phase_verification.json` records an independent matrix-based
recomputation. Every adverse result is retained. The boundary-retained layouts
were already available from the whole-history sensitivity; this phase check is
retrospective, not a prospective independent test of a new boundary rule.

## Mirabello source qualification

Lusvardi (2020), https://scholarsarchive.byu.edu/etd/8769/, is a relevant
blast-liquefaction micropile thesis, but it is not scored as a complete measured
dynamic reference. Its strain-derived load is retained only to 6.2 m and
15 min; the final deeper profile includes CAPWAP and settlement-based estimates
(printed pp. 148–155). Those estimates are not independent gauge observations.
The source qualification and file hash are in `mirabello_qualification.json`.
The thesis PDF is not redistributed. No synthetic profiles are added as
experimental evidence.
