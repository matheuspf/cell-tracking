"""Prediction-only local fork hypotheses and temporally supported features."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .common import SCALE, adjacency

FORK_FEATURES = ['p_low','p_high','p_sum','p_difference','native_missing_count','existing_count',
    'distance_low_um','distance_high_um','sister_distance_um','barycenter_distance_um',
    'motion_barycenter_residual_um','distance_asymmetry','angle_cosine','daughter_separation_change_1',
    'daughter_separation_change_2','daughter_separation_change_3','future_missing_count',
    'parent_history_missing','parent_response','daughter_response_low','daughter_response_high',
    'intensity_conservation_ratio','daughter_intensity_ratio','parent_density',
    'daughter_birth_count','competing_parent_distance_low','competing_parent_distance_high',
    'daughter_age_low','daughter_age_high','parent_track_remaining','original_fork',
    'source_native_missing','daughter_native_missing_count','best_continuation_p','third_best_p']


def build(base,native):
    n,e=base['nodes'],base['edges'];ix,pred,succ=adjacency(n,e)
    pos=n[:,2:]*SCALE;f=base['features'];pairs=native['pairs'];ef=native['edge_features']
    by_parent=[[] for _ in n]
    for k,(i,j) in enumerate(pairs):by_parent[i].append(k)
    triples=[];features=[]
    for i,ks in enumerate(by_parent):
        if len(ks)<2:continue
        ranked=sorted(ks,key=lambda k:(not bool(ef[k,2]),ef[k,3]-3*ef[k,0],int(pairs[k,1])))[:4]
        p_by_j={int(pairs[k,1]):k for k in ranked}
        for a,b in combinations(sorted(p_by_j),2):
            ka,kb=p_by_j[a],p_by_j[b]
            ea,eb=ef[ka],ef[kb]
            original=set(succ[i])=={a,b}
            # Alternatives include a retained continuation when one exists.
            if succ[i] and not ({a,b}&set(succ[i])):continue
            if max(ea[3],eb[3])>12. and not original:continue
            sister=np.linalg.norm(pos[a]-pos[b])
            if (sister<1. or sister>16.) and not original:continue
            future=[];ca,cb=a,b;missing=0
            for step in range(3):
                if len(succ[ca])==1 and len(succ[cb])==1 and succ[ca][0]!=succ[cb][0]:
                    ca,cb=succ[ca][0],succ[cb][0]
                    future.append(np.linalg.norm(pos[ca]-pos[cb])-sister)
                else:
                    future.append(0.);missing+=1
            # Keep boundary masks explicit. A new fork must have at least one
            # observed frame of distinct daughter persistence.
            if missing==3 and not original:continue
            velocity=pos[i]-pos[pred[i][0]] if len(pred[i])==1 else np.zeros(3)
            bary=(pos[a]+pos[b])/2-pos[i]
            da,db=pos[a]-pos[i],pos[b]-pos[i]
            owner=[]
            for c in [a,b]:
                owner.append(np.linalg.norm(pos[c]-pos[pred[c][0]]) if pred[c] and pred[c][0]!=i else 0.)
            pp=sorted([float(ef[k,0]) for k in ks],reverse=True)
            values=[min(ea[0],eb[0]),max(ea[0],eb[0]),ea[0]+eb[0],abs(ea[0]-eb[0]),ea[1]+eb[1],ea[2]+eb[2],
                min(ea[3],eb[3]),max(ea[3],eb[3]),sister,np.linalg.norm(bary),np.linalg.norm(bary-.5*velocity),
                abs(ea[3]-eb[3])/max(.1,(ea[3]+eb[3])/2),np.dot(da,db)/max(.01,np.linalg.norm(da)*np.linalg.norm(db)),
                *future,missing,len(pred[i])!=1,f[i,5],min(f[a,5],f[b,5]),max(f[a,5],f[b,5]),
                (f[a,6]+f[b,6])/(f[i,6]+.05),min(f[a,6],f[b,6])/(max(f[a,6],f[b,6])+.05),f[i,8],
                int(not pred[a])+int(not pred[b]),min(owner),max(owner),min(f[a,11],f[b,11]),max(f[a,11],f[b,11]),
                f[i,12],original,native['native_index'][i]<0,
                int(native['native_index'][a]<0)+int(native['native_index'][b]<0),pp[0],pp[2] if len(pp)>2 else 0.]
            triples.append((i,a,b));features.append(values)
    return dict(triples=np.array(triples,np.int64).reshape(-1,3),
                fork_features=np.array(features,np.float32).reshape(-1,len(FORK_FEATURES)))
