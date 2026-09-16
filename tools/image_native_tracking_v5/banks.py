"""Common bank: source motion neighbors, all C0 edges, genuine native alternatives."""
import torch
from .common import *
from .candidates import bank
from . import native_adapter as native

def build(model,nodes,base_edges,fs,ft,distance):
    spatial=bank(nodes,base_edges,distance);pairs=set(map(tuple,spatial));values={};native_added=0
    by_time={t:np.flatnonzero(nodes[:,1]==t) for t in range(100)}
    node_time={int(n[0]):int(n[1]) for n in nodes};pair_by_time={}
    for a,b in spatial:pair_by_time.setdefault(node_time[int(a)],[]).append((int(a),int(b)))
    with torch.no_grad():
        for t in range(99):
            ai=by_time[t];bi=by_time[t+1]
            if not len(ai) or not len(bi):continue
            log=native.logits(model,nodes[ai],nodes[bi],features=(fs[ai].astype(np.float32),ft[bi].astype(np.float32)))
            prob=log.softmax(0).clamp(1e-6,1-1e-6).cpu().numpy();odds=np.log(prob/(1-prob))
            am={int(nodes[i,0]):j for j,i in enumerate(ai)};bm={int(nodes[i,0]):j for j,i in enumerate(bi)}
            for a,b in pair_by_time.get(t,[]):values[(a,b)]=float(odds[am[a],bm[b]])
            # At most 19 choices per daughter because incoming probabilities sum to one.
            aa,bb=np.where(prob>.05)
            for i,j in zip(aa,bb):
                edge=(int(nodes[ai[i],0]),int(nodes[bi[j],0]))
                if edge not in pairs:native_added+=1;pairs.add(edge)
                values[edge]=float(odds[i,j])
    ordered=np.array(sorted(pairs),np.int64).reshape(-1,2)
    scores=np.array([values[tuple(p)] for p in ordered],np.float32)
    return ordered,scores,dict(spatial_edges=len(spatial),native_alternatives_added=native_added,
        common_bank='eight incoming/outgoing physical neighbors, every C0 edge, native incoming p>0.05',distance_um=distance,
        features_precision='native float32 encoder samples stored float16; shared deterministic dequantization')

def cached(model,name,source,expanded=False):
    from .candidates import motion_config
    dest=OUT/'banks'/source/('P1' if expanded else 'P0')/f'{name}.npz'
    if dest.exists():
        receipt=read(dest.with_suffix('.json'))
        assert receipt['sha256']==sha(dest)
        assert receipt['observations_sha256']==sha(OUT/'observations'/f'{name}.npz')
        assert receipt['source']==source and receipt['expanded']==expanded
        assert receipt['distance_um']==motion_config()[source]['distance_um']
        return arrays(dest)
    c=arrays(OUT/'observations'/f'{name}.npz');mask=np.ones(len(c['nodes']),bool) if expanded else c['oldmask']
    pairs,logits,receipt=build(model,c['nodes'][mask],graph(name)['edges'],c['features_source'][mask],c['features_target'][mask],motion_config()[source]['distance_um'])
    save(dest,pairs=pairs,native_logits=logits)
    write(dest.with_suffix('.json'),dict(dataset=name,source=source,expanded=expanded,**receipt,sha256=sha(dest),created=now(),
        observations_sha256=sha(OUT/'observations'/f'{name}.npz'),code_sha256=sha(Path(__file__))))
    return dict(pairs=pairs,native_logits=logits)
