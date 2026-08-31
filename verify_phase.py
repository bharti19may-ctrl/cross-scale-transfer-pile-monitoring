"""Independent matrix-weight reconstruction check for phase results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
COLS=['test_code','event_number','pile','smoothed','eta','sampling_cap','phase_code','time_s']+['Q'+str(i) for i in range(8)]
PH=['shaking','before_decay','after_decay']
arr=np.load(P/'phase_selected_samples.npz',allow_pickle=False)['samples']
samples=pd.DataFrame(arr,columns=COLS)
actual=pd.read_csv(P/'phase_pile_results.csv')
original=pd.read_csv(P/'prj2828_robust_layouts.csv')
boundary=pd.read_csv(P/'prj2828_boundary_constrained_transfer.csv')
maxdiff=0.;checks=0
z=np.arange(8)/7
def weights(idx):
    W=np.zeros((8,len(idx)))
    for i,x in enumerate(z):
        if x<=z[idx[0]]:W[i,0]=1
        elif x>=z[idx[-1]]:W[i,-1]=1
        else:
            j=np.searchsorted(z[idx],x)-1
            a=(x-z[idx[j]])/(z[idx[j+1]]-z[idx[j]])
            W[i,j]=1-a;W[i,j+1]=a
    return W
def score(Y,idx):
    pred=Y[:,idx]@weights(idx).T
    scale=np.maximum(np.abs(Y).max(axis=1),.05)
    v=np.array([np.sqrt(((pred-Y)**2).sum(axis=1)/8)/scale,
        np.abs(pred.max(axis=1)-Y.max(axis=1))/scale,
        np.abs(z[pred.argmax(axis=1)]-z[Y.argmax(axis=1)])]).T
    return np.quantile(v,.9,axis=0)
for key,g in samples.groupby(COLS[:7]):
    tc,en,pile,sm,eta,cap,ph=key
    assert np.all(np.diff(g.time_s)>0)
    assert 10<=len(g)<=cap
    test=['SKS02','SKS03'][int(tc)];source='SKS03' if tc==0 else 'SKS02'
    Y=g[COLS[8:]].to_numpy(float)
    for b in [2,3,4]:
        ui=sorted({int(np.argmin(abs(z-x))) for x in np.linspace(0,1,b+1)})
        uniform=score(Y,ui)
        for policy,table,col in [('original',original,'gage_indices_top_to_bottom'),('boundary_retained',boundary,'boundary_constrained_layout')]:
            sourcecol='training_test' if policy=='original' else 'selection_test'
            st=table[(table[sourcecol]==source)&(table.additional_stations_b==b)][col].iloc[0]
            idx=list(map(int,st.split(';')));candidate=score(Y,idx)
            a=actual[(actual.test==test)&(actual.event=='EQM_'+str(int(en)))&(actual.pile==pile)&(actual.smoothed==sm)&(actual.eta==eta)&(actual.sampling_cap==cap)&(actual.phase==PH[int(ph)])&(actual.policy==policy)&(actual.additional_stations_b==b)].set_index('metric').loc[['profile_nrmse','peak_load_error','peak_depth_error']]
            diff=np.max(np.abs(a[['candidate','uniform','delta']].to_numpy()-np.column_stack([candidate,uniform,candidate-uniform])))
            maxdiff=max(maxdiff,float(diff));checks+=len(a)
assert maxdiff<1e-12,maxdiff
assert checks==len(actual)
assert np.isfinite(arr).all()
report={'result_rows_independently_verified':checks,'maximum_absolute_numerical_difference':maxdiff,'sample_rows':len(arr),'distinct_pile_event_histories':samples.groupby(COLS[:3]).ngroups,'chronology_and_sample_cap_checks':'passed','nonfinite_values':int((~np.isfinite(arr)).sum())}
(P/'phase_verification.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
