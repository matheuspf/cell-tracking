"""Named image/geometry features. No GT, filenames, embryo IDs, or teachers."""
import numpy as np
from scipy.spatial import cKDTree

NAMES=('parent_confidence','daughter_confidence_min','daughter_confidence_max',
       'parent_distance_min_um','parent_distance_max_um','sister_distance_um',
       'barycenter_distance_um','history_velocity_um','history_valid','motion_residual_um',
       'daughter_path_count','future_separation_change_um','future_valid',
       'parent_density_10um','added_clean_logits','removed_clean_logits','donor_count',
       'introduced_birth_count','introduced_termination_count','left_frame_support','right_frame_support',
       'parent_image_mean','parent_image_contrast','daughter_image_mean_min','daughter_image_mean_max',
       'daughter_image_contrast_min','daughter_image_contrast_max','parent_spatial_support',
       'daughter_spatial_support_min','daughter_spatial_support_max')


def action_features(bank,group,confidence,patch=None,valid=None,index=None,*,summaries=None):
    parent=group['parent'];n=bank.nodes;pos=bank.pos
    t=int(n[parent,1]);density=bank.density[parent]
    if summaries is None:
        needed={parent,*[j for d in group['forks'] for j in d.event[1:3]]}
        summaries={i:(float(patch[index[i],1].mean()),float(patch[index[i],1].std()),float(valid[index[i],1,2])) for i in needed}
    history=len(bank.pred[parent])==1
    velocity=pos[parent]-pos[bank.pred[parent][0]] if history else np.zeros(3)
    events=np.asarray([d.event for d in group['forks']],np.int64)
    a,b,qa,qb=events[:,1],events[:,2],events[:,3],events[:,4]
    delta1,delta2=pos[a]-pos[parent],pos[b]-pos[parent]
    d1,d2=np.linalg.norm(delta1,axis=1),np.linalg.norm(delta2,axis=1)
    sister=np.linalg.norm(pos[a]-pos[b],axis=1);future=(qa>=0)&(qb>=0)
    change=np.zeros(len(events));change[future]=np.linalg.norm(pos[qa[future]]-pos[qb[future]],axis=1)-sister[future]
    pm,ps,pv=summaries[parent]
    am,ast,av=np.asarray([summaries[int(i)] for i in a]).T
    bm,bst,bv=np.asarray([summaries[int(i)] for i in b]).T
    full=lambda x:np.full(len(events),x)
    values=np.column_stack([full(confidence[parent]),np.minimum(confidence[a],confidence[b]),np.maximum(confidence[a],confidence[b]),
        np.minimum(d1,d2),np.maximum(d1,d2),sister,np.linalg.norm((delta1+delta2)/2,axis=1),
        full(np.linalg.norm(velocity)),full(int(history)),np.linalg.norm((delta1+delta2)/2-.5*velocity,axis=1),
        (qa>=0).astype(int)+(qb>=0),change,future,full(density),
        [sum(bank.logits[e] for e in d.add) for d in group['forks']],
        [sum(bank.logits[e] for e in d.remove) for d in group['forks']],
        [len(d.owners) for d in group['forks']],
        [sum(n[j,1]>0 for j in d.births) for d in group['forks']],
        [sum(n[j,1]<bank.frames-1 for j in d.terminations) for d in group['forks']],
        full(min(4,t)/4),full(min(4,bank.frames-1-t)/4),full(pm),full(ps),
        np.minimum(am,bm),np.maximum(am,bm),np.minimum(ast,bst),np.maximum(ast,bst),full(pv),np.minimum(av,bv),np.maximum(av,bv)]).astype(np.float32)
    return values,np.r_[values.mean(0),values.max(0),np.log1p(len(values))].astype(np.float32)


def normalize(values,mean,std):
    return np.clip((values-mean)/np.maximum(std,1e-4),-10,10).astype(np.float32)


IDENTITY_NAMES=('parent_confidence','child_confidence','distance_um','dz_um','dy_um','dx_um',
                'clean_edge_logit','left_frame_support','right_frame_support')


def identity_features(bank,pairs,confidence):
    values=[]
    for a,b in pairs:
        delta=bank.pos[b]-bank.pos[a];t=int(bank.nodes[a,1])
        values.append([confidence[a],confidence[b],np.linalg.norm(delta),*delta,bank.logits[(int(a),int(b))],
                       min(4,t)/4,min(4,bank.frames-1-t)/4])
    return np.asarray(values,np.float32).reshape(-1,len(IDENTITY_NAMES))
