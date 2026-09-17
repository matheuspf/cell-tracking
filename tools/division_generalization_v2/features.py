"""Annotation-free descriptors of complete edit graphs and physical queries."""
import numpy as np
from pipeline_error_training.scoring import EVENT_SCALE, edge_features

FEATURE_DIM = 176


def build(bank, parent, records):
    decisions = [d for d, _ in records]
    nodes, native = bank.nodes, bank.native
    used = sorted({parent} | {n for d in decisions for n in d.event if n >= 0} |
                  {n for d in decisions for e in d.remove | d.add for n in e})
    node_map = {n:k for k,n in enumerate(used)}
    all_pairs = sorted({e for d in decisions for e in d.remove | d.add})
    ef = edge_features(nodes, native, all_pairs, np.asarray(bank.pos[0]*0+[1.625,.40625,.40625]))
    # Fixed physical feature scaling. No GT or target-fitted moments enter.
    escale = np.ones(38, np.float32)
    escale[[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]] = 20
    ef = np.clip(ef/escale, -10, 10)
    fmap = {e:ef[k] for k,e in enumerate(all_pairs)}
    nf = bank.nf[:, :13].astype(np.float32)
    nf = np.nan_to_num(nf, nan=0., posinf=10., neginf=-10.)
    nf = np.clip(nf, -10, 10)
    descriptors, event_index, incidence = [], [], []
    for d, event_features in records:
        p, a, b = d.event[:3]
        owners = sorted({int(o[0]) for o in d.owners})
        add = np.mean([fmap[e] for e in sorted(d.add)], axis=0) if d.add else np.zeros(38)
        remove = np.mean([fmap[e] for e in sorted(d.remove)], axis=0) if d.remove else np.zeros(38)
        frozen = np.concatenate([nf[p], nf[a]+nf[b], abs(nf[a]-nf[b]),
                                 nf[owners].mean(0) if owners else np.zeros(13)])
        counts = [len(d.add)/8, len(d.remove)/8, len(d.births)/4, len(d.terminations)/4,
                  len(owners)/4, float(d.kind=='division'), float(d.kind=='keep'),
                  float(len(bank.succ[p])==2)]
        descriptors.append(np.concatenate([np.clip(event_features/EVENT_SCALE,-10,10), add,remove,frozen,counts]))
        event_index.append([node_map[p],node_map[a],node_map[b]])
        # Complete action context includes every changed endpoint and owner, with
        # separate before/after incidence; distant owners retain explicit masks.
        inc = np.zeros((3,len(used)), np.float32)
        for slot, edges in enumerate((d.add,d.remove)):
            for x,y in edges:
                inc[slot,node_map[x]] += 1
                inc[slot,node_map[y]] += 1
            if inc[slot].sum():
                inc[slot] /= inc[slot].sum()
        for owner in owners:
            inc[2,node_map[owner]] = 1/max(1,len(owners))
        incidence.append(inc)
    relative = (nodes[used,2:]-nodes[parent,2:]).astype(np.float32)
    times = (nodes[used,1]-nodes[parent,1]).astype(np.float32)
    return dict(features=np.asarray(descriptors,np.float32).reshape(-1,FEATURE_DIM),
                event_index=np.asarray(event_index,np.int64).reshape(-1,3),
                incidence=np.asarray(incidence,np.float32), query_voxels=relative,
                query_time=times, query_nodes=np.asarray(used,np.int64),
                keep=np.asarray([d.kind=='keep' for d in decisions],bool))
