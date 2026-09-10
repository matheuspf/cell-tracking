"""Post-freeze, label-free HOCT feature/coverage audit; no model or graph changes."""
import sys,time
import pandas as pd
from .common import *


def run():
    sys.path.insert(0,str(WORK/'python'))
    from hoct._api import _MEAN,_STD
    names=['t','z_um','y_um','x_um','equivalent_diameter_um','intensity_min',
        'intensity_max','intensity_mean','intensity_std',
        *[f'inertia_{a}{b}_um2' for a in 'zyx' for b in 'zyx'],'border_dist']
    assert len(names)==len(_MEAN)==len(_STD)==19
    mean=np.asarray(_MEAN);std=np.asarray(_STD)
    totals={e:dict(n=0,sums=np.zeros(19),squares=np.zeros(19),minimum=np.full(19,np.inf),
        maximum=np.full(19,-np.inf),outside=np.zeros(19,np.int64)) for e in ['44b6','6bba','pooled']}
    clips=[];input_hashes={};start=time.monotonic()
    for row in inventory():
        name=row['dataset'];source='6bba' if row['embryo']=='44b6' else '44b6'
        path=OUT/'observations'/f'{name}.npz'
        with np.load(path,allow_pickle=False) as f:
            old=f['oldmask'];nodes=f['nodes'][old];props=f['properties'][old]
            valid=f['valid_region'][old];collisions=f['collisions'][old]
        assert np.isfinite(props[valid]).all()
        values=np.column_stack([nodes[valid,1],nodes[valid,2:]*[1.625,.40625,.40625],props[valid]]).astype(np.float64)
        for embryo in [row['embryo'],'pooled']:
            r=totals[embryo];r['n']+=len(values);r['sums']+=values.sum(0);r['squares']+=(values**2).sum(0)
            r['minimum']=np.minimum(r['minimum'],values.min(0));r['maximum']=np.maximum(r['maximum'],values.max(0))
            r['outside']+=(np.abs((values-mean)/std)>3).sum(0)
        inertia=props[valid,5:14].reshape(-1,3,3).astype(np.float64)
        eigenvalues=np.linalg.eigvalsh(inertia);spread=eigenvalues[:,-1]-eigenvalues[:,0]
        isotropic=spread<=np.maximum(np.abs(eigenvalues[:,-1]),1e-12)*1e-6
        with np.load(OUT/'banks'/source/'P0'/f'{name}.npz',allow_pickle=False) as f:pairs=f['pairs']
        order=np.argsort(nodes[:,0]);index=order[np.searchsorted(nodes[order,0],pairs)]
        assert np.array_equal(nodes[index,0],pairs)
        endpoint_support=valid[index].all(1)
        with np.load(OUT/'model_scores/H'/f'{name}.npz',allow_pickle=False) as f:finite=np.isfinite(f['H0'])
        assert len(finite)==len(pairs) and not (finite&~endpoint_support).any()
        clips.append(dict(dataset=name,embryo=row['embryo'],nodes=len(nodes),valid_regions=int(valid.sum()),
            invalid_regions=int((~valid).sum()),seed_collisions=int(collisions.sum()),
            isotropic_inertia_regions=int(isotropic.sum()),anisotropic_inertia_regions=int((~isotropic).sum()),
            regions_with_nonzero_intensity_std=int((props[valid,4]>0).sum()),
            candidate_edges=len(pairs),both_endpoint_regions_valid=int(endpoint_support.sum()),
            finite_HOCT_edges=int(finite.sum()),missing_region_edges=int((~endpoint_support).sum()),
            valid_endpoint_edges_without_HOCT_score=int((endpoint_support&~finite).sum())))
        input_hashes[name]=dict(observation=sha(path),HOCT_scores=sha(OUT/'model_scores/H'/f'{name}.npz'))
    features=[];coverage=[]
    for embryo,r in totals.items():
        observed_mean=r['sums']/r['n'];observed_std=np.sqrt(np.maximum(0,r['squares']/r['n']-observed_mean**2))
        for i,name in enumerate(names):
            features.append(dict(embryo=embryo,feature=name,valid_regions=r['n'],mean=observed_mean[i],
                std=observed_std[i],minimum=r['minimum'][i],maximum=r['maximum'][i],
                official_mean=mean[i],official_std=std[i],standardized_mean=(observed_mean[i]-mean[i])/std[i],
                fraction_outside_official_3std=r['outside'][i]/r['n']))
        subset=[row for row in clips if embryo=='pooled' or row['embryo']==embryo]
        coverage.append(dict(embryo=embryo,samples=len(subset),
            **{k:sum(row[k] for row in subset) for k in clips[0] if k not in ['dataset','embryo']}))
    pd.DataFrame(features).to_csv(OUT/'HOCT_feature_distribution.csv',index=False)
    pd.DataFrame(clips).to_csv(OUT/'HOCT_feature_coverage_rows.csv',index=False)
    write(OUT/'HOCT_feature_audit.json',dict(at=now(),complete=True,clips=len(clips),seconds=time.monotonic()-start,
        coverage=coverage,feature_order=names,input_hashes=input_hashes,annotations_read=False,
        stage='post-freeze descriptive audit; no source or target calibration changes',
        domain_scope='Standardize uses these exact official constants; outside-three-standard-deviation fractions are descriptive, not validity or biological quality thresholds.',
        morphology_scope='Actual image-supported watershed regions. An isotropic sphere has equal inertia eigenvalues; measured tensor anisotropy and nonzero intensity variation characterize these regions, without creating sphere replacement inputs.',
        missing_score_scope='Valid endpoint regions without a score reflect the frozen tiled temporal/spatial model interface; missing-score incumbent edges are protected.'))
    print('Complete label-free HOCT feature audit',coverage,flush=True)


if __name__=='__main__':run()
