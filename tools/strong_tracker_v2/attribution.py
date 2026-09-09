"""Error substitutions and quality-confounding diagnostics on measured outputs."""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .common import OUT,V1,inventory,load_graph,read_json,write_json
from .evaluate import baseline_fp_set


def run(args):
    rows=[];quality=[]
    variants=['E_hgb_m0.5','E_hgb_m1.5','DE_primary','D_temporal_p0.05']
    for r in inventory():
        name=r['dataset'];b=load_graph(V1/'baseline/public'/f'{name}.npz');n,e=b['nodes'],b['edges']
        old=set(map(tuple,e));m=load_graph(OUT/'evaluation/membership'/f'{name}.npz')
        matches={int(i):int(j) for i,j in zip(n[:,0],m['matched_gt_id']) if j>=0}
        gt=load_graph(V1/'evaluation/gt'/f'{name}.npz');gs=set(map(tuple,gt['edges']))
        tp=set(map(tuple,m['tp_edges']));fp=baseline_fp_set(n,e,gt,matches)
        pred=load_graph(OUT/'predictions'/f'{name}.npz')
        for v in variants:
            new=set(map(tuple,pred[v+'__edges']));ntp={x for x in new if (matches.get(int(x[0])),matches.get(int(x[1]))) in gs}
            changed_fp=sorted(fp-new);restored_tp=sorted((ntp-tp)-old)
            lookup=defaultdict(set)
            for j,(a,d) in enumerate(restored_tp):lookup[a].add(j);lookup[d].add(j)
            rr=[];cc=[]
            for i,(a,d) in enumerate(changed_fp):
                for j in lookup[a]|lookup[d]:rr.append(i);cc.append(j)
            if rr:
                matrix=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(changed_fp),len(restored_tp))).tocsr()
                paired=int(np.sum(maximum_bipartite_matching(matrix,perm_type='column')>=0))
            else:paired=0
            measured=read_json(OUT/'evaluation/scores'/v/f'{name}.json')['result']
            assert len(ntp)==measured['edge_tp'],'Fixed-node matching attribution drift'
            rows.append(dict(dataset=name,embryo=r['embryo'],variant=v,paired_fp_to_tp_link_substitutions=paired,
                deleted_fp_without_paired_restoration=len(changed_fp)-paired,restored_tp_without_paired_fp_deletion=len(restored_tp)-paired,
                lost_baseline_tp=len(tp-ntp),new_evaluated_fp=measured['introduced_fp'],
                added_unknown_or_evaluated_edges=len(new-old),deleted_unknown_or_evaluated_edges=len(old-new)))
        for v in ['F_hgb_quality_tracklet_r0.9','F_hgb_geometry_tracklet_r0.9','F_v1_transfer_control']:
            keep=pred[v+'__keep'];f=b['features'];conf=b['detector_confidence'];missing=b['confidence_missing']
            for label,sel in [('kept',keep),('removed',~keep)]:
                quality.append(dict(dataset=name,embryo=r['embryo'],variant=v,group=label,nodes=int(sel.sum()),
                    matched_annotations=int(m['annotation_label'][sel].sum()),response_sum=float(f[sel,5].sum()),
                    intensity_sum=float(f[sel,6].sum()),track_length_sum=float(f[sel,10].sum()),
                    native_missing=int(missing[sel].sum()),low_response=int((f[sel,5]<.05).sum())))
    pd.DataFrame(rows).to_csv(OUT/'association_changes.csv',index=False)
    df=pd.DataFrame(rows);numeric=list(df.select_dtypes(include='number').columns)
    summary=df.groupby(['variant','embryo'])[numeric].sum().reset_index().to_dict('records')
    pd.DataFrame(quality).to_csv(OUT/'filter_quality_diagnostics.csv',index=False)
    q=pd.DataFrame(quality).groupby(['variant','embryo','group']).sum(numeric_only=True).reset_index()
    for col in ['response','intensity','track_length']:q[col+'_mean']=q[col+'_sum']/q.nodes
    q['annotation_prevalence']=q.matched_annotations/q.nodes;q['missing_fraction']=q.native_missing/q.nodes
    write_json(OUT/'association_attribution.json',dict(rows=summary,
        pairing='Maximum-cardinality matching of actually deleted FP links to newly added TP links sharing a source or target; each link used at most once'))
    write_json(OUT/'filter_quality_summary.json',dict(rows=q.to_dict('records'),
        biological_false_detection_truth_available=False,
        interpretation='Sparse match labels cannot distinguish ordinary false-detection removal from removal of real unannotated cells. Quality profiles test confounding; no claim to isolate latent annotator preference.'))
