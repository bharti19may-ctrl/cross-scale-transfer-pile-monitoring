"""Phase-resolved diagnostics of frozen layouts. No target-response selection.
Default: reproduce from supplied numeric samples. --prepare-from: source rebuild.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import transfer_audit as base

HERE=Path(__file__).resolve().parent
PHASES=['shaking','before_decay','after_decay']
ETAS=[.25,.5,.75]
CAPS=[120,240]
COLS=['test_code','event_number','pile','smoothed','eta','sampling_cap','phase_code','time_s']+['Q'+str(i) for i in range(8)]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return pd.read_csv(p,skipinitialspace=True).rename(columns=lambda c:str(c).replace('Gage_','Gage').strip()).apply(pd.to_numeric,errors='coerce')
def monotonic(t):
    # Keep the original fast segment; omit later overlapping times at the join.
    prior=np.r_[-np.inf,np.maximum.accumulate(t[:-1])]
    return np.isfinite(t)&(t>prior)
def smooth(t,y,width):
    return pd.DataFrame(y,index=pd.to_timedelta(t,unit='s')).rolling(pd.Timedelta(seconds=width),center=True,min_periods=1).mean().to_numpy()
def nearest(t,lo,hi,cap):
    eligible=np.flatnonzero((t>=lo)&(t<=hi))
    if len(eligible)<10 or hi<=lo:return np.array([],dtype=int),0.
    v=t[eligible];targets=np.linspace(v[0],v[-1],min(cap,len(v)))
    a=np.clip(np.searchsorted(v,targets),0,len(v)-1);b=np.maximum(0,a-1)
    pick=np.where(abs(v[b]-targets)<abs(v[a]-targets),b,a)
    return eligible[np.unique(pick)],float(np.max(np.minimum(abs(v[a]-targets),abs(v[b]-targets))))
def pressure_boundary(t,R,t95,peak,eta):
    # First observed 60-s interval entirely below eta*reference peak.
    # No claim of complete dissipation or liquefaction threshold is made.
    below=R<=eta*peak
    bad=np.r_[0,np.cumsum(~below)]
    for i in np.flatnonzero((t>=t95)&below):
        j=np.searchsorted(t,t[i]+60,side='left')
        if j>=len(t):break
        if bad[j+1]==bad[i] and np.max(np.diff(t[i:j+1]),initial=0)<=60:
            return float(t[i]),float(t[i]-t[max(i-1,0)])
    return None,None
def prepare(raw):
    rows=[];bounds=[];qa=[];sources=[]
    for test_code,test in enumerate(['SKS02','SKS03']):
        for a in sorted((raw/test).rglob('EQM_*_Input_Motion.txt')):
            event=a.name.split('_Input')[0];en=int(event.split('_')[1]);folder=a.parent
            pp=folder/(event+'_Keller_Excess_Pore_Pressure_Ratio.txt')
            motion=read(a);at=motion.iloc[:,0].to_numpy(float);acc=motion.iloc[:,1].to_numpy(float)
            assert np.all(np.diff(at)>0) and abs(at[0])<.1
            energy=np.r_[0,np.cumsum(.5*(acc[1:]**2+acc[:-1]**2)*np.diff(at))]
            assert energy[-1]>0
            t05,t95=np.interp([.05,.95],energy/energy[-1],at)
            p=read(pp);pt0=p.iloc[:,0].to_numpy(float)*60;pm=monotonic(pt0)
            pt=pt0[pm];values=p.iloc[:,1:].to_numpy(float)[pm]
            variable=(np.nanmax(values,axis=0)-np.nanmin(values,axis=0))>1e-8
            available=np.isfinite(values[:,variable]).sum(axis=1)>=np.ceil(.8*variable.sum())
            incomplete=int((~np.isfinite(values[:,variable])).sum())
            pt=pt[available];values=values[available]
            R=np.nanquantile(np.maximum(values[:,variable],0),.9,axis=1)
            R=smooth(pt,R[:,None],3)[:,0]
            peak=float(np.max(R[(pt>=t05)&(pt<=t95+60)]))
            for path in [a,pp]:
                sources.append({'source_file':str(path.relative_to(raw)).replace('\\','/'),'sha256':digest(path)})
            end=min(float(pt[-1]),float(pt[-1]))
            bs={}
            for eta in ETAS:
                recovery,gap=pressure_boundary(pt,R,t95,peak,eta) if peak>=.1 else (None,None)
                bs[eta]=recovery
                bounds.append(dict(test=test,event=event,eta=eta,shaking_start_s=t05,shaking_end_s=t95,
                    pressure_reference_peak=peak,decay_boundary_s=recovery,boundary_preceding_interval_s=gap,
                    pressure_end_s=end,pressure_channels=int(variable.sum()),flat_channels_excluded=int((~variable).sum()),
                    overlap_rows_removed=int((~pm).sum()),missing_pressure_cells=incomplete,rows_below_80_percent_coverage=int((~available).sum()),
                    status='available' if recovery is not None else 'weak pressure response or no sustained crossing'))
            for f in sorted(folder.glob(event+'_Pile_*_Axial_Load.txt')):
                pile=int(f.name.split('_Pile_')[1].split('_')[0]);d=read(f)
                if not set(base.GAGE_COLS_TOP_TO_BOTTOM).issubset(d.columns):continue
                t0=d.iloc[:,0].to_numpy(float)*60;y0=d[base.GAGE_COLS_TOP_TO_BOTTOM].to_numpy(float)
                assert np.array_equal(t0,pt0),f.name+' is not synchronous with pressure'
                mask=monotonic(t0)&np.isfinite(y0).all(axis=1)
                t=t0[mask];y=y0[mask];end=min(pt[-1],t[-1])
                width=6. if pile==1 else 3.
                sources.append({'source_file':str(f.relative_to(raw)).replace('\\','/'),'sha256':digest(f)})
                for smoothed in [0,1]:
                    yy=smooth(t,y,width) if smoothed else y
                    guard=width/2 if smoothed else 0.
                    for eta in ETAS:
                        recovery=bs[eta]
                        intervals=[(0,t05+guard,t95-guard)]
                        if recovery is not None:
                            intervals.extend([(1,t95+guard,recovery-guard),(2,recovery+guard,end)])
                        for cap in CAPS:
                            for ph,lo,hi in intervals:
                                ix,offset=nearest(t,lo,hi,cap)
                                qa.append(dict(test=test,event=event,pile=pile,smoothed=smoothed,eta=eta,sampling_cap=cap,
                                    phase=PHASES[ph],lower_s=lo,upper_s=hi,snapshots=len(ix),max_nearest_offset_s=offset,
                                    source_overlap_rows_removed=int((~monotonic(t0)).sum())))
                                if len(ix)<10:continue
                                m=np.column_stack([np.full(len(ix),v) for v in [test_code,en,pile,smoothed,eta,cap,ph]]+[t[ix],yy[ix]])
                                rows.append(m)
            print('Prepared',test,event,'pressure peak',round(peak,3),'half recovery',bs[.5],flush=True)
    samples=np.concatenate(rows);np.savez_compressed(HERE/'phase_selected_samples.npz',samples=samples)
    pd.DataFrame(bounds).to_csv(HERE/'phase_boundaries.csv',index=False)
    pd.DataFrame(qa).to_csv(HERE/'phase_sampling_audit.csv',index=False)
    pd.DataFrame(sources).drop_duplicates().to_csv(HERE/'phase_source_hashes.csv',index=False)
    manifest={'columns':COLS,'phase_names':PHASES,'test_names':['SKS02','SKS03'],
      'primary':{'eta':.5,'sampling_cap':120,'smoothed':1},
      'sensitivity':{'eta':ETAS,'sampling_cap':CAPS,'smoothed':[0,1]},
      'phase_rule':'5-95% cumulative squared input acceleration; first 60-s observed interval below eta times the pressure reference peak',
      'pressure_proxy':'90th percentile across available non-flat Keller channels after clipping negative ratios to zero and 3-s elapsed-time moving mean; at least 80% channel availability, no missing-value imputation',
      'peak_window':'from t05 to t95+60 seconds; recovery phases not assigned when peak<0.1',
      'time_rule':'Preserve original order; omit times not exceeding all previous times at fast/slow overlap; never invent missing response values',
      'smoothing':'centred elapsed-time moving mean: 6 s for Pile 1 and 3 s otherwise; half-window phase-edge guard',
      'sampling':'nearest distinct measured times to 120/240 equally spaced targets within each phase; at least 10 samples',
      'normalisation':'snapshot maximum absolute load, floored at 0.05 MN, identical for candidate and uniform',
      'independence':'within-pile phase quantiles; paired differences aggregated over piles within each earthquake; only 6 and 5 sequential earthquakes in two experiments',
      'limits':'Retrospective phase diagnostics, not new experiments, unsmoothed dynamic-peak validation, or complete pore-pressure dissipation identification',
      'sources':['https://doi.org/10.17603/DS2-D25M-GG48','https://doi.org/10.17603/DS2-WJGX-TB78']}
    (HERE/'phase_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
def metrics(Y,indices):
    pred=np.array([base.prediction(y,indices) for y in Y]);s=np.maximum(np.max(abs(Y),axis=1),.05)
    return np.column_stack([np.sqrt(np.mean((pred-Y)**2,axis=1))/s,
         abs(pred.max(axis=1)-Y.max(axis=1))/s,
         abs(base.Z[pred.argmax(axis=1)]-base.Z[Y.argmax(axis=1)])])
def analyse():
    arr=np.load(HERE/'phase_selected_samples.npz',allow_pickle=False)['samples']
    df=pd.DataFrame(arr,columns=COLS)
    original=pd.read_csv(HERE/'prj2828_robust_layouts.csv')
    boundary=pd.read_csv(HERE/'prj2828_boundary_constrained_transfer.csv')
    out=[];metric_names=['profile_nrmse','peak_load_error','peak_depth_error']
    group_cols=COLS[:7]
    for key,g in df.groupby(group_cols,sort=True):
        tc,en,pile,sm,eta,cap,ph=key;test=['SKS02','SKS03'][int(tc)];source='SKS03' if test=='SKS02' else 'SKS02'
        Y=g[COLS[8:]].to_numpy(float)
        for b in [2,3,4]:
            uniform=base.uniform_layout(b);u=np.quantile(metrics(Y,uniform),.9,axis=0)
            oi=original[(original.training_test==source)&(original.additional_stations_b==b)].iloc[0].gage_indices_top_to_bottom
            bi=boundary[(boundary.selection_test==source)&(boundary.additional_stations_b==b)].iloc[0].boundary_constrained_layout
            for policy,st in [('original',oi),('boundary_retained',bi)]:
                idx=tuple(map(int,str(st).split(';')));c=np.quantile(metrics(Y,idx),.9,axis=0)
                for i,name in enumerate(metric_names):
                    out.append(dict(test=test,source=source,event='EQM_'+str(int(en)),pile=int(pile),smoothed=int(sm),eta=eta,
                        sampling_cap=int(cap),phase=PHASES[int(ph)],policy=policy,additional_stations_b=b,metric=name,
                        candidate=c[i],uniform=u[i],delta=c[i]-u[i],snapshots=len(g)))
    pile=pd.DataFrame(out);pile.to_csv(HERE/'phase_pile_results.csv',index=False)
    keys=['test','source','event','smoothed','eta','sampling_cap','phase','policy','additional_stations_b','metric']
    event=pile.groupby(keys,as_index=False).agg(candidate=('candidate','median'),uniform=('uniform','median'),delta=('delta','median'),piles=('pile','nunique'))
    event.to_csv(HERE/'phase_event_results.csv',index=False)
    keys.remove('event')
    summary=event.groupby(keys,as_index=False).agg(events=('event','nunique'),median_delta=('delta','median'),
          min_delta=('delta','min'),max_delta=('delta','max'),events_lower=('delta',lambda s:int((s < -1e-12).sum())),
          events_higher=('delta',lambda s:int((s > 1e-12).sum())))
    summary.to_csv(HERE/'phase_transfer_summary.csv',index=False)
    primary=summary[(summary.smoothed==1)&(summary.eta==.5)&(summary.sampling_cap==120)]
    print(primary[primary.metric=='profile_nrmse'].to_string(index=False))
    assert event.piles.between(1,2).all()
    assert df.groupby(['test_code','event_number','pile']).ngroups==22
    assert df['phase_code'].isin([0,1,2]).all()
    assert all(np.isfinite(x).all() for x in [arr,pile[['candidate','uniform','delta']].to_numpy()])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-from',type=Path);a=ap.parse_args()
    if a.prepare_from:prepare(a.prepare_from)
    analyse()
