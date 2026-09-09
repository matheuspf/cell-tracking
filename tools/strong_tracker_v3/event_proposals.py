"""Prediction-only event proposals on the current canonical graph.

Rows identify a parent, an unordered daughter pair and two explicit one-frame
daughter paths. No annotation imports or crop centers enter this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, product

import numpy as np
from scipy.spatial import cKDTree

from .common import adjacency


@dataclass(frozen=True)
class ProposalConfig:
    neighbors: int = 6
    reverse_neighbors: int = 2
    parent_gate_um: float = 16.
    sister_gate_um: float = 20.
    paths_per_daughter: int = 2
    context_frames: int = 9
    uniform_modulus: int = 64
    second_probability_trigger: float = .08


EVENT_FEATURES = [
    'p_low', 'p_high', 'p_sum', 'p_difference', 'native_missing_count',
    'existing_count', 'distance_low_um', 'distance_high_um', 'sister_distance_um',
    'barycenter_distance_um', 'motion_barycenter_residual_um', 'distance_asymmetry',
    'angle_cosine', 'path_distance_low_um', 'path_distance_high_um',
    'future_separation_change', 'future_missing_count', 'parent_history_missing',
    'parent_response', 'daughter_response_low', 'daughter_response_high',
    'intensity_ratio', 'daughter_intensity_ratio', 'parent_density',
    'daughter_birth_count', 'owner_distance_low', 'owner_distance_high',
    'owner_probability_low', 'owner_probability_high', 'owner_missing_count',
    'original_fork', 'boundary_left', 'boundary_right', 'path_probability_low',
    'path_probability_high', 'candidate_path_count_low', 'candidate_path_count_high',
    'parent_terminates', 'native_parent_missing', 'nearby_birth_trigger',
]


def candidate_neighbors(nodes, edges, native, scale, config):
    """Six forward and two reverse neighbors, plus all incumbent edges."""
    ix, pred, succ = adjacency(nodes, edges)
    pos = nodes[:, 2:] * np.asarray(scale)
    native_p = {tuple(map(int,p)): float(f[0]) for p,f in zip(native['pairs'],native['edge_features']) if not f[1]}
    near = [set(s) for s in succ]
    forward_count = reverse_count = before_gate = 0
    for t in np.unique(nodes[:,1])[:-1]:
        a,b = np.flatnonzero(nodes[:,1]==t),np.flatnonzero(nodes[:,1]==t+1)
        if not len(a) or not len(b): continue
        d,j=cKDTree(pos[b]).query(pos[a], k=min(config.neighbors,len(b)))
        for i,ds,js in zip(a,np.asarray(d).reshape(len(a),-1),np.asarray(j).reshape(len(a),-1)):
            before_gate += len(ds)
            for dd,jj in zip(ds,js):
                if dd<=config.parent_gate_um: near[i].add(int(b[jj])); forward_count+=1
        d,j=cKDTree(pos[a]).query(pos[b],k=min(config.reverse_neighbors,len(a)))
        for i,ds,js in zip(b,np.asarray(d).reshape(len(b),-1),np.asarray(j).reshape(len(b),-1)):
            for dd,jj in zip(ds,js):
                if dd<=config.parent_gate_um: near[a[jj]].add(int(i));reverse_count+=1
    # Native ranking resolves geometric ties, never reads source or target labels.
    rank=lambda i,j: (np.linalg.norm(pos[j]-pos[i])-3*native_p.get((i,j),0),int(nodes[j,0]))
    near=[sorted(x,key=lambda j:rank(i,j)) for i,x in enumerate(near)]
    return near,native_p,(ix,pred,succ),dict(forward_before_gate=before_gate,forward_after_gate=forward_count,reverse_after_gate=reverse_count)


def build(nodes, edges, native, scale, config=ProposalConfig()):
    near, probability,(_,pred,succ),counts=candidate_neighbors(nodes,edges,native,scale,config)
    pos=nodes[:,2:]*np.asarray(scale)
    nf=native.get('node_features',native.get('features',np.zeros((len(nodes),13),np.float32)))
    if nf.shape[1]<13: nf=np.pad(nf,((0,0),(0,13-nf.shape[1])))
    birth=np.array([any(not pred[j] for j in ns) for ns in near])
    anchors=set()
    for i, ns in enumerate(near):
        pp=sorted([probability.get((i,j),0.) for j in ns],reverse=True)
        if len(succ[i])!=1 or birth[i] or (len(pp)>1 and pp[1]>=config.second_probability_trigger) or int(nodes[i,0])%config.uniform_modulus==0:
            anchors.add(i)
    # Explicit -1/0/+1 anchors, using prediction adjacency and candidates.
    expanded=set(anchors)
    for i in anchors:
        expanded.update(pred[i]);expanded.update(near[i][:2])
    paths=[list(ns[:config.paths_per_daughter]) or [-1] for ns in near]
    # An existing trajectory remains an eligible path when ranking prefers another.
    for i in range(len(nodes)):
        for j in succ[i]:
            if j not in paths[i]:
                paths[i]=([j]+paths[i])[:config.paths_per_daughter]
    rows=[]; feats=[]; existing_pool=[]; pair_before=pair_after=0
    end_t=int(nodes[:,1].max())
    mapping=native.get('native_index',np.zeros(len(nodes)))
    for i in sorted(expanded):
        # Keep the complete forward/reverse union. Re-ranking it to the forward
        # count would silently discard reverse/birth candidates in dense regions.
        options=set(near[i])|set(succ[i])
        for a,b in combinations(sorted(options),2):
            pair_before+=1
            original=set(succ[i])=={a,b}
            da,db=pos[a]-pos[i],pos[b]-pos[i]
            dista,distb=np.linalg.norm(da),np.linalg.norm(db)
            sister=float(np.linalg.norm(pos[a]-pos[b]))
            if not original and (max(dista,distb)>config.parent_gate_um or not 1<=sister<=config.sister_gate_um): continue
            pair_after+=1
            pa,pb=probability.get((i,a),0.),probability.get((i,b),0.)
            velocity=pos[i]-pos[pred[i][0]] if len(pred[i])==1 else np.zeros(3)
            bary=(da+db)/2
            owner_dist=[];owner_prob=[];owner_missing=0
            for d in [a,b]:
                if pred[d] and pred[d][0]!=i:
                    o=pred[d][0];owner_dist.append(np.linalg.norm(pos[d]-pos[o]))
                    owner_prob.append(probability.get((o,d),0.));owner_missing+=int((o,d) not in probability)
                else:owner_dist.append(0.);owner_prob.append(0.)
            for qa,qb in product(paths[a],paths[b]):
                if qa>=0 and qa==qb: continue
                dpath=[np.linalg.norm(pos[q]-pos[d]) if q>=0 else 0. for d,q in [(a,qa),(b,qb)]]
                ppath=[probability.get((d,q),0.) for d,q in [(a,qa),(b,qb)]]
                future=float(np.linalg.norm(pos[qa]-pos[qb])-sister) if qa>=0 and qb>=0 else 0.
                rows.append((i,a,b,qa,qb))
                feats.append([min(pa,pb),max(pa,pb),pa+pb,abs(pa-pb),int((i,a) not in probability)+int((i,b) not in probability),
                    int(a in succ[i])+int(b in succ[i]),min(dista,distb),max(dista,distb),sister,np.linalg.norm(bary),
                    np.linalg.norm(bary-.5*velocity),abs(dista-distb)/max(.1,(dista+distb)/2),np.dot(da,db)/max(.01,dista*distb),
                    *sorted(dpath),future,int(qa<0)+int(qb<0),len(pred[i])!=1,nf[i,5],min(nf[a,5],nf[b,5]),max(nf[a,5],nf[b,5]),
                    (nf[a,6]+nf[b,6])/(nf[i,6]+.05),min(nf[a,6],nf[b,6])/(max(nf[a,6],nf[b,6])+.05),nf[i,8],
                    int(not pred[a])+int(not pred[b]),*sorted(owner_dist),*sorted(owner_prob),owner_missing,original,
                    min(4,int(nodes[i,1]))/4,min(4,end_t-int(nodes[i,1]))/4,*sorted(ppath),
                    *sorted([sum(q>=0 for q in paths[a]),sum(q>=0 for q in paths[b])]),not succ[i],
                    mapping[i]<0,birth[i]])
                # Conservative pool reproduces v2 geometric/continuation limits,
                # but uses native incumbent graph and candidate-persistence labels.
                existing_pool.append(original or (bool({a,b}&set(succ[i])) and max(dista,distb)<=12 and sister<=16 and a in near[i][:4] and b in near[i][:4]))
    rows=np.asarray(rows,np.int64).reshape(-1,5)
    parents,inverse=np.unique(rows[:,0],return_inverse=True) if len(rows) else (np.empty(0,np.int64),np.empty(0,np.int64))
    counts.update(anchors=len(anchors),expanded_anchors=len(expanded),pair_before_gate=pair_before,pair_after_gate=pair_after,
                  alternatives=len(rows),crop_parents=len(parents),configuration=asdict(config))
    return dict(events=rows,event_features=np.asarray(feats,np.float32).reshape(-1,len(EVENT_FEATURES)),
                existing_pool=np.asarray(existing_pool,bool),crop_parents=parents,crop_index=inverse),counts


def extract_crops(image, nodes, parents, scale, side=12):
    """Identical source/inference parent-centered nine-frame triplanar sampling.

    Samples a 24um field at 2um spacing. Zero padding and a separate validity
    channel retain spatial/time boundaries. Only one raw frame is loaded at once.
    """
    parents=np.asarray(parents,int)
    result=np.zeros((len(parents),9,4,side,side),np.float32)
    if not len(parents):return result.astype(np.uint8)
    center=nodes[parents,2:].astype(float);pt=nodes[parents,1].astype(int)
    offsets=(np.arange(side)-(side-1)/2)*2.
    yy,xx=np.meshgrid(offsets,offsets,indexing='ij')
    shape=np.asarray(image.shape[1:]);scale=np.asarray(scale)
    grids=[]
    for axes in [(1,2),(0,2),(0,1)]:
        delta=np.zeros((side,side,3));delta[:,:,axes[0]]=yy;delta[:,:,axes[1]]=xx
        coords=np.rint(center[:,None,None,:]+delta/scale).astype(int)
        valid=np.all((coords>=0)&(coords<shape),axis=-1)
        grids.append((np.clip(coords,0,shape-1),valid))
    for t in range(image.shape[0]):
        ids=np.flatnonzero(abs(pt-t)<=4)
        if not len(ids):continue
        frame=np.asarray(image[t])
        tt=t-pt[ids]+4
        for plane,(coords,valid) in enumerate(grids):
            c=coords[ids]
            result[ids,tt,plane]=frame[c[:,:,:,0],c[:,:,:,1],c[:,:,:,2]]*valid[ids]
        result[ids,tt,3]=np.mean([valid[ids] for _,valid in grids],axis=0)
    values=result[:,:,:3]
    lo=np.quantile(values,.02,axis=(1,2,3,4),keepdims=True)
    hi=np.quantile(values,.995,axis=(1,2,3,4),keepdims=True)
    result[:,:,:3]=np.clip((values-lo)/np.maximum(hi-lo,1),0,1)
    return np.rint(result*255).astype(np.uint8)
