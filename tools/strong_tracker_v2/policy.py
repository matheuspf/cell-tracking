"""Prediction-only coherent deletion actions and image/native group features."""
import numpy as np
import pandas as pd

from annotation_selection.filter_graph import tracklets


def group_actions(nodes,edges,mode='tracklet'):
    units,pred,succ=tracklets(nodes,edges)
    if mode=='node':groups=[np.array([i]) for i in range(len(nodes))]
    elif mode=='segment':
        # Non-overlapping five-frame sections of predicted linear tracklets.
        codes=pd.factorize(pd.MultiIndex.from_arrays([units,nodes[:,1]//5]))[0]
        order=np.argsort(codes,kind='stable')
        groups=np.split(order,np.flatnonzero(np.diff(codes[order]))+1)
    else:
        order=np.argsort(units,kind='stable')
        groups=np.split(order,np.flatnonzero(np.diff(units[order]))+1)
    protected=np.zeros(len(nodes),bool)
    if mode=='fork_protected':
        for i,children in enumerate(succ):
            if len(children)>=2:
                context={i,*pred[i],*children}
                context.update(j for c in children for j in succ[c])
                protected[list(context)]=True
    return groups,protected


def ranked_mask(nodes,edges,score,keep_fraction,mode):
    if keep_fraction==1:return np.ones(len(nodes),bool)
    if mode=='node':
        mask=np.ones(len(nodes),bool)
        order=np.lexsort((np.arange(len(nodes)),score))
        mask[order[:int(np.floor((1-keep_fraction)*len(nodes)))]]=False
        return mask
    groups,protected=group_actions(nodes,edges,mode)
    # Mean and maximum are observable group summaries; the risk arm learns its
    # group score separately, rather than using the old 0.9 quantile rule.
    risk=np.array([.5*np.mean(score[g])+.5*np.max(score[g]) for g in groups])
    order=np.lexsort((np.arange(len(groups)),risk))
    budget=int(np.floor((1-keep_fraction)*len(nodes)))
    mask=np.ones(len(nodes),bool);removed=0
    for j in order:
        g=groups[j]
        if protected[g].any():continue
        if removed+len(g)>budget:continue
        mask[g]=False;removed+=len(g)
        if removed>=budget:break
    return mask


def group_features(base,native,groups):
    x=np.column_stack([base['features'][:,5:],native['node_extra']])
    result=[]
    for g in groups:
        # No match labels, estimates, coordinates, event IDs or GT metadata.
        result.append(np.r_[np.mean(x[g],axis=0),np.min(x[g],axis=0),np.max(x[g],axis=0),np.log1p(len(g))])
    return np.asarray(result,np.float32)
