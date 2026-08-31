"""Bounded digitisation stress tests; draws are not experimental replications.

One-, two- and four-pixel scenarios are declared stress amplitudes, not calibrated
confidence bounds. Turrell load bounds also cover half the between-figure range.
Published gauge depths are fixed. Smooth perturbations of the FHWA traced curves
represent coherent graphical uncertainty, not 101 independent measured gauges.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyse_external_transfer import LAYOUTS, map_layout, metrics, uniform_layout

HERE=Path(__file__).resolve().parent
DRAWS=2000
SEED=20260827
CASES={
 'Turrell H-pile post-blast': (250/736,0),
 'Turrell pipe pile post-blast': (300/771,0),
 'Christchurch 12-m CFA pile post-blast': (300/626,1/859),
 'Christchurch 14-m CFA pile post-blast': (300/653,1/1201),
}


def classify(x):
    return np.where(x < -.005,-1,np.where(x > .005,1,0))


def run():
    data=pd.read_csv(HERE/'external_digitised_profiles.csv')
    output=[]; decomposition=[]
    for case,(load_per_pixel,depth_per_pixel) in CASES.items():
        group=data[data.case.eq(case)].sort_values('normalised_depth')
        x=group.normalised_depth.to_numpy(float); y=group.load.to_numpy(float)
        spread=group.digitisation_range.fillna(0).to_numpy(float)/2
        pairs=[]
        for b in (2,3,4):
            ui,_=map_layout(x,uniform_layout(b))
            for source in ('SKS02','SKS03'):
                ci,_=map_layout(x,LAYOUTS[(source,b)])
                assert len(ci)==len(ui)==b+1
                pairs.append((source,b,ci,ui))
                scale=max(abs(y)); ce=((np.interp(x,x[ci],y[ci])-y)/scale)**2
                ue=((np.interp(x,x[ui],y[ui])-y)/scale)**2
                outside=x>x[ci[-1]]
                a=(ce[~outside].sum()-ue[~outside].sum())/len(x)
                b_out=(ce[outside].sum()-ue[outside].sum())/len(x)
                assert np.isclose(ce.mean()-ue.mean(),a+b_out,atol=1e-12)
                decomposition.append(dict(case=case,layout_source=source,additional_stations_b=b,
                    outside_reference_fraction=float(outside.mean()),
                    candidate_outside_sse_fraction=ce[outside].sum()/ce.sum() if ce.sum()>0 else 0,
                    delta_squared_error_inside=a,delta_squared_error_outside=b_out,
                    delta_nrmse=np.sqrt(ce.mean())-np.sqrt(ue.mean())))
        for pixels in (1,2,4):
            rng=np.random.default_rng(SEED)
            changes=np.zeros((DRAWS,len(pairs)))
            mapping_changes=np.zeros(len(pairs),int)
            bound=np.maximum(pixels*load_per_pixel,spread)
            for draw in range(DRAWS):
                if depth_per_pixel:
                    # Smooth line-location errors are shared by both comparators.
                    anchors=np.linspace(0,1,11)
                    ey=np.interp(x,anchors,rng.uniform(-1,1,11))
                    ex=np.interp(x,anchors,np.r_[0,rng.uniform(-1,1,9),0])
                    xx=x+pixels*depth_per_pixel*ex
                else:
                    ey=rng.uniform(-1,1,len(x)); xx=x.copy()
                assert np.all(np.diff(xx)>0)
                yy=y+bound*ey
                for j,(source,b,ci0,ui0) in enumerate(pairs):
                    ci,_=map_layout(xx,LAYOUTS[(source,b)])
                    ui,_=map_layout(xx,uniform_layout(b))
                    assert len(ci)==len(ui)==b+1
                    mapping_changes[j]+=int(not np.array_equal(ci,ci0) or not np.array_equal(ui,ui0))
                    changes[draw,j]=metrics(xx,yy,ci)['nrmse']-metrics(xx,yy,ui)['nrmse']
            for j,(source,b,ci,ui) in enumerate(pairs):
                nominal=metrics(x,y,ci)['nrmse']-metrics(x,y,ui)['nrmse']
                vals=changes[:,j]
                output.append(dict(case=case,layout_source=source,additional_stations_b=b,pixel_scenario=pixels,
                    draws=DRAWS,seed=SEED,nominal_delta_nrmse=nominal,min_delta_nrmse=vals.min(),
                    max_delta_nrmse=vals.max(),median_delta_nrmse=np.median(vals),
                    draws_lower=int((vals<-.005).sum()),draws_equivalent=int((abs(vals)<=.005).sum()),
                    draws_higher=int((vals>.005).sum()),draws_same_class=int((classify(vals)==classify(nominal)).sum()),
                    draws_changed_mapping=int(mapping_changes[j]),load_bound_max=float(bound.max()),
                    normalised_depth_bound=pixels*depth_per_pixel))
    result=pd.DataFrame(output)
    result.to_csv(HERE/'external_extraction_resolution_sensitivity.csv',index=False)
    pd.DataFrame(decomposition).to_csv(HERE/'external_error_coverage_decomposition.csv',index=False)
    manifest=dict(seed=SEED,draws_per_scenario=DRAWS,pixel_scenarios=[1,2,4],
        source='PlotCalibration values in digitize_external_profiles.py; between-figure ranges in external_digitised_profiles.csv',
        interpretation='Assumption-based graphical-resolution stress tests, not confidence intervals or new experimental data',
        claims='Only sign/classification stability under the stated perturbations; no calibrated measurement uncertainty or field success probability')
    (HERE/'extraction_resolution_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(result.groupby('pixel_scenario').agg(comparisons=('case','size'),
        fully_stable_classifications=('draws_same_class',lambda s:int((s==DRAWS).sum())),
        minimum_same_class_fraction=('draws_same_class',lambda s:float(s.min()/DRAWS))).to_string())


if __name__=='__main__':run()
