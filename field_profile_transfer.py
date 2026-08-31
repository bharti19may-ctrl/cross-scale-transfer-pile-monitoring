"""Frozen-layout checks on author-processed Delft field profiles.

Digital reference curves are not independent centimetre-spaced measurements.
The source reports 0.20-m spatial resolution; 0.20/0.40-m evaluation grids are
therefore used, with the isolated head load and last recorded level retained.
All selections below are fixed without observing target reconstruction error.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd
from methodological_sensitivity_audit import reconstruct

HERE = Path(__file__).resolve().parent
PILES = ('F1', 'F2', 'T1', 'T2', 'T3')
DOI = 'https://doi.org/10.4121/78720ecb-daf4-4676-b5f4-281e93e4388a'


def prepare(root):
    arrays, ledger = {}, []
    for pile in PILES:
        folder = root / 'load-test' / pile
        path = folder / f'{pile}_load-distrib.csv'
        frame = pd.read_csv(path)
        times = pd.to_datetime(frame['timestamp'])
        depth = -np.array(frame.columns[1:], float)
        assert np.all(np.diff(depth) > 0), 'Source coordinates must run downwards'
        values = frame.iloc[:, 1:].to_numpy(float)
        stages = pd.read_csv(folder / f'{pile}_datums.csv')
        rows, labels = [], []
        for row in stages.itertuples():
            if not re.fullmatch(r'Step \d+', str(row.step)):
                continue
            start = pd.to_datetime(row.start, dayfirst=True) + pd.Timedelta(seconds=45)
            end = pd.to_datetime(row.end, dayfirst=True) - pd.Timedelta(seconds=45)
            valid = (times >= start) & (times <= end) & np.isfinite(values).all(axis=1)
            eligible = np.flatnonzero(valid)
            entry = dict(pile=pile, stage=row.step, eligible_complete_rows=len(eligible),
                         source_file=path.relative_to(root).as_posix(),
                         source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if not len(eligible):
                entry['status'] = 'excluded: no complete in-stage profile'
            else:
                chosen = eligible[-1]
                rows.append(values[chosen]); labels.append(str(row.step))
                entry.update(status='included', timestamp=str(times.iloc[chosen]))
            ledger.append(entry)
        arrays[pile + '_depth_m'] = depth
        arrays[pile + '_load_kN'] = np.array(rows)
        arrays[pile + '_stage'] = np.array(labels)
    np.savez_compressed(HERE / 'delft_selected_stage_profiles.npz', **arrays)
    pd.DataFrame(ledger).to_csv(HERE / 'delft_stage_eligibility.csv', index=False)


def frozen_layouts():
    layouts = {}
    frame = pd.read_csv(HERE / 'prj2828_robust_layouts.csv')
    for r in frame.itertuples():
        b = int(r.additional_stations_b)
        if r.training_test in ('SKS02','SKS03') and b in (2,3,4):
            layouts[('original',r.training_test,b)] = np.array(str(r.gage_indices_top_to_bottom).split(';'),float)/7
    for r in pd.read_csv(HERE / 'prj2828_boundary_constrained_transfer.csv').itertuples():
        b = int(r.additional_stations_b)
        if b in (2,3,4):
            layouts[('boundary_constrained',r.selection_test,b)] = np.array(str(r.boundary_constrained_layout).split(';'),float)/7
    assert len(layouts) == 12
    return layouts


def map_nodes(z, targets):
    idx = np.unique([int(np.argmin(np.abs(z-t))) for t in targets])
    assert len(idx) == len(targets), 'Distinct requested stations must remain distinct'
    return idx


def grid_indices(depth, pitch):
    # Column zero is the separately supplied head load; the remaining profile is
    # the source's 0.01-m interpolated DFOS product. Do not fill its upper gap.
    # The final T3 interval is 0.03 m, so use physical coordinates rather
    # than assuming that every stored column is exactly one centimetre apart.
    targets = np.arange(depth[1], depth[-1], pitch)
    selected = [1 + int(np.argmin(np.abs(depth[1:]-t))) for t in targets]
    return np.unique(np.r_[0, selected, len(depth)-1])


def evaluate():
    data = np.load(HERE / 'delft_selected_stage_profiles.npz', allow_pickle=False)
    records = []
    for pile in PILES:
        depth = data[pile+'_depth_m']
        loads = data[pile+'_load_kN']
        stages = data[pile+'_stage']
        for pitch in (.2,.4):
            grid = grid_indices(depth,pitch)
            x = depth[grid]; z = (x-x[0])/(x[-1]-x[0])
            for (boundary,source,b),targets in frozen_layouts().items():
                ci = map_nodes(z,targets); ui = map_nodes(z,np.linspace(0,1,b+1))
                outside = (z < z[ci[0]]) | (z > z[ci[-1]])
                for method in ('linear','pchip'):
                    for stage,yfull in zip(stages,loads,strict=True):
                        y = yfull[grid]; scale = max(np.max(np.abs(y)),np.finfo(float).eps)
                        cp = reconstruct(z,y,ci,method); up = reconstruct(z,y,ui,method)
                        ce = ((cp-y)/scale)**2; ue = ((up-y)/scale)**2
                        rc,ru = np.sqrt(ce.mean()),np.sqrt(ue.mean())
                        delta_inside = (ce[~outside].sum()-ue[~outside].sum())/len(z)
                        delta_outside = (ce[outside].sum()-ue[outside].sum())/len(z)
                        assert np.isclose(rc**2-ru**2,delta_inside+delta_outside,atol=1e-12)
                        # A separate distance-weighted diagnostic addresses the
                        # long head-to-DFOS gap; it is not used to select layouts.
                        wc = np.sqrt(np.trapezoid(ce,z)); wu = np.sqrt(np.trapezoid(ue,z))
                        records.append(dict(pile=pile,stage=stage,reference_pitch_m=pitch,
                            boundary_rule=boundary,interpolator=method,layout_source=source,
                            additional_stations_b=b,total_retained=b+1,reference_levels=len(z),
                            measured_span_m=x[-1]-x[0],head_to_dfos_gap_m=x[1]-x[0],
                            candidate_realised_z=';'.join(f'{v:.6f}' for v in z[ci]),
                            uniform_realised_z=';'.join(f'{v:.6f}' for v in z[ui]),
                            candidate_nrmse=rc,uniform_nrmse=ru,delta_nrmse=rc-ru,
                            delta_distance_weighted_nrmse=wc-wu,
                            candidate_outside_sse_fraction=ce[outside].sum()/ce.sum() if ce.sum()>0 else 0,
                            delta_squared_error_inside=delta_inside,delta_squared_error_outside=delta_outside))
    frame=pd.DataFrame(records)
    frame.to_csv(HERE/'delft_field_stage_comparisons.csv',index=False)
    group=['reference_pitch_m','boundary_rule','interpolator','layout_source','additional_stations_b','pile']
    per=frame.groupby(group,as_index=False).agg(stages=('stage','size'),median_paired_delta=('delta_nrmse','median'),
        median_candidate_nrmse=('candidate_nrmse','median'),median_uniform_nrmse=('uniform_nrmse','median'),
        median_distance_weighted_delta=('delta_distance_weighted_nrmse','median'),
        median_outside_sse_fraction=('candidate_outside_sse_fraction','median'),
        median_delta_squared_error_inside=('delta_squared_error_inside','median'),
        median_delta_squared_error_outside=('delta_squared_error_outside','median'))
    per.to_csv(HERE/'delft_field_pile_comparisons.csv',index=False)
    summary=per.groupby(group[:-1],as_index=False).agg(piles=('pile','size'),
        median_pile_delta=('median_paired_delta','median'),min_pile_delta=('median_paired_delta','min'),
        max_pile_delta=('median_paired_delta','max'),piles_lower=('median_paired_delta',lambda x:int((x<-.005).sum())),
        piles_equivalent=('median_paired_delta',lambda x:int((x.abs()<=.005).sum())),
        piles_higher=('median_paired_delta',lambda x:int((x>.005).sum())))
    summary.to_csv(HERE/'delft_field_transfer_summary.csv',index=False)
    manifest=dict(source_doi=DOI,piles=list(PILES),field_sites=1,
        excluded_pile='F3: source reports casing-extraction problems; no interpreted force file supplied',
        source_processing='Force inferred from DFOS strain using source stiffness processing; 1-cm grid is linearly interpolated, not independent 1-cm measurements',
        source_resolution_m=.2,reference_grids_m=[.2,.4],stage_rule='Last complete record within each numbered Step, after 45-s buffers at both ends',
        profile_counts={p:int(data[p+'_load_kN'].shape[0]) for p in PILES},
        frozen_selection='Original and boundary-constrained SKS02/SKS03 layouts; exact source indices divided by seven; no target-response fitting',
        uncertainty_unit='Five piles from one site, with repeated loading stages; descriptive summaries only',
        coverage_identity='NRMSE_candidate^2 - NRMSE_uniform^2 equals signed inside-span plus outside-span contributions before aggregation',
        limitations='Static compression and author-processed reference profiles; not liquefaction validation, raw strain-to-force calibration validation, independent multi-site performance, or hardware validation')
    (HERE/'delft_field_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(summary.query("reference_pitch_m == .2 and boundary_rule == 'original' and interpolator == 'linear'").to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-from',type=Path,help='Optional raw extracted delft directory; otherwise use the archived selected profiles')
    args=parser.parse_args()
    if args.prepare_from:prepare(args.prepare_from)
    evaluate()
