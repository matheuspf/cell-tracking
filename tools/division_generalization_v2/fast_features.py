"""Exact vectorized frozen inference descriptors; training keeps its reference.

Only immutable node descriptors are cached. Ragged means are grouped by length
so every reduction keeps the reference's sorted input order and float32 dtype.
"""
from collections import defaultdict
import numpy as np
from pipeline_error_training.scoring import EVENT_SCALE,edge_features
from .features import FEATURE_DIM,build as reference_build


def prepare(bank):
    if not hasattr(bank,'_v2_frozen_nodes'):
        nf=bank.nf[:,:13].astype(np.float32)
        nf=np.nan_to_num(nf,nan=0.,posinf=10.,neginf=-10.)
        bank._v2_frozen_nodes=np.clip(nf,-10,10)


def ragged_means(groups,values,width):
    result=np.zeros((len(groups),width),np.float32)
    lengths=defaultdict(list)
    for i,g in enumerate(groups):
        if len(g):lengths[len(g)].append(i)
    for n,rows in lengths.items():
        ix=np.asarray([groups[i] for i in rows],np.int64)
        result[rows]=values[ix].mean(axis=1)
    return result


def build(bank,parent,records):
    if not records:return reference_build(bank,parent,records)
    prepare(bank)
    decisions=[d for d,_ in records]
    nodes,native=bank.nodes,bank.native
    used=sorted({parent}|{n for d in decisions for n in d.event if n>=0}|
                {n for d in decisions for e in d.remove|d.add for n in e})
    node_map={n:k for k,n in enumerate(used)}
    pairs=sorted({e for d in decisions for e in d.remove|d.add})
    pair_index={e:i for i,e in enumerate(pairs)}
    ef=edge_features(nodes,native,pairs,np.asarray(bank.pos[0]*0+[1.625,.40625,.40625]))
    escale=np.ones(38,np.float32);escale[3:24]=20
    ef=np.clip(ef/escale,-10,10)
    added=[sorted(d.add) for d in decisions];removed=[sorted(d.remove) for d in decisions]
    owners=[sorted({int(o[0]) for o in d.owners}) for d in decisions]
    events=np.asarray([d.event[:3] for d in decisions],np.int64)
    p,a,b=events.T;nf=bank._v2_frozen_nodes
    descriptors=np.zeros((len(records),FEATURE_DIM),np.float32)
    descriptors[:,:40]=np.clip(np.asarray([f for _,f in records])/EVENT_SCALE,-10,10)
    descriptors[:,40:78]=ragged_means([[pair_index[e] for e in es] for es in added],ef,38)
    descriptors[:,78:116]=ragged_means([[pair_index[e] for e in es] for es in removed],ef,38)
    descriptors[:,116:129]=nf[p]
    descriptors[:,129:142]=nf[a]+nf[b]
    descriptors[:,142:155]=abs(nf[a]-nf[b])
    descriptors[:,155:168]=ragged_means(owners,nf,13)
    descriptors[:,168:]=np.asarray([[len(d.add)/8,len(d.remove)/8,len(d.births)/4,
        len(d.terminations)/4,len(o)/4,float(d.kind=='division'),float(d.kind=='keep'),
        float(len(bank.succ[d.event[0]])==2)] for d,o in zip(decisions,owners)],np.float32)
    incidence=np.zeros((len(records),3,len(used)),np.float32)
    for slot,groups in enumerate((added,removed)):
        rows=[];columns=[]
        for i,es in enumerate(groups):
            rows.extend([i]*(2*len(es)))
            columns.extend(node_map[n] for e in es for n in e)
        if rows:np.add.at(incidence[:,slot],(rows,columns),1.)
        total=incidence[:,slot].sum(axis=1)
        incidence[:,slot]/=np.maximum(total,1.)[:,None]
    for i,o in enumerate(owners):
        for owner in o:incidence[i,2,node_map[owner]]=1/max(1,len(o))
    return dict(features=descriptors,event_index=np.searchsorted(used,events).astype(np.int64),
        incidence=incidence,query_voxels=(nodes[used,2:]-nodes[parent,2:]).astype(np.float32),
        query_time=(nodes[used,1]-nodes[parent,1]).astype(np.float32),
        query_nodes=np.asarray(used,np.int64),keep=np.asarray([d.kind=='keep' for d in decisions],bool))
