"""DeepCenter-confirmed subset of frozen image-only point alternatives."""
import time
import numpy as np
from .common import *

CONFIG=dict(variant='R_heatmap_local',point_confidence=.5,temporal_confidence=.25,
    xy_pool_factor=4,temporal_offsets=[-1,0,1],centroid_peak_neighborhood=[1,1,1],
    source='Existing DeepCenter checkpoint; cached heatmaps read-only; missing frames queried locally',
    policy_frozen_before_this_arm_scores=True)

def apply(ctx,sample,nodes,edges,proposal_nodes,proposal_edges,ledger,namespace=None):
    from .replay import repair_namespace
    ns=namespace or repair_namespace(ctx)
    bundle=ns['_heatmap_state']['bundle'];cache={};frame_cache={};counts=dict(cache_queries=0,model_queries=0)
    name=sample['dataset'];new={int(r[0]):r for r in proposal_nodes};old={int(r[0]):r for r in nodes}
    def score(t,xyz):
        if t<0 or t>=sample['image_shape'][0]:return None
        paths=[ctx.v2/'heatmaps'/name/f'{t}.npy',ctx.out/'heatmaps'/name/f'{t}.npy']
        available=next((p for p in paths if p.exists()),None)
        if available is not None:
            h=np.load(available,mmap_mode='r');counts['cache_queries']+=1
        else:
            h=ns['deepcenter_heatmap_for_frame'](name,t,bundle,frame_cache,cache);counts['model_queries']+=1
        if h is None:return None
        q=np.rint(np.asarray(xyz)/[1,4,4]).astype(int);lo=np.maximum(q-1,0);hi=np.minimum(q+2,h.shape)
        patch=h[tuple(slice(a,b) for a,b in zip(lo,hi))]
        return float(patch.max()) if patch.size else None
    def supported(row):
        t=int(row[1]);xyz=row[2:];center=score(t,xyz)
        if center is None or center<CONFIG['point_confidence']:return False
        neighbors=[score(t+dt,xyz) for dt in [-1,1] if 0<=t+dt<sample['image_shape'][0]]
        return all(s is not None and s>=CONFIG['temporal_confidence'] for s in neighbors)
    result=nodes.copy();ix={int(r[0]):i for i,r in enumerate(nodes)};added=[];add_edges=[];accepted=[]
    for action in ledger:
        if action['kind']=='local_image_centroid':
            k=action['node_id']
            if k in old and supported(new[k]):result[ix[k]]=new[k];accepted.append(action)
        else:
            ids=action['node_ids']
            if all(supported(new[k]) for k in ids):
                added.extend(new[k] for k in ids);add_edges.extend(action['added_edges']);accepted.append(action)
    if added:result=np.vstack([result,np.asarray(added,np.int64)])
    e=np.vstack([edges,np.asarray(add_edges,np.int64)]) if add_edges else edges.copy()
    # Candidate artifacts are generated from the same baseline; links cannot
    # refer to a node rejected by the event-level image-evidence gate.
    ids=set(map(int,result[:,0]));e=np.array([p for p in e if int(p[0]) in ids and int(p[1]) in ids],np.int64).reshape(-1,2)
    validate(result,e,sample['image_shape'])
    return result,e,accepted,dict(**counts,accepted_actions=len(accepted),proposed_actions=len(ledger),inserted_nodes=len(added))

def run(ctx,args=None):
    from .replay import repair_namespace
    old=read_json(ctx.out/'heatmap_rescue_config.json') if (ctx.out/'heatmap_rescue_config.json').exists() else {}
    write_json(ctx.out/'heatmap_rescue_config.json',dict(created=old.get('created',now()),config=CONFIG,code_sha256=sha(__file__),
        bounded_round='Additional image-evidence confirmation arm; implementation requirement, no target-label thresholds'),immutable=True)
    ns=repair_namespace(ctx);records=[]
    for s in ctx.samples():
        name=s['dataset'];dest=ctx.out/'heatmap_rescue'/f'{name}.json';path=ctx.out/'candidate_graphs/R_heatmap_local'/f'{name}.npz'
        source=ctx.out/'candidate_graphs/R_image_local'/f'{name}.npz';meta=ctx.out/'rescue'/f'{name}.json'
        stamp=dict(base=sha(ctx.incumbent(name)),proposals=sha(source),ledger=sha(meta),code=sha(__file__),config=digest(CONFIG))
        if dest.exists():
            r=read_json(dest)
            if r['inputs']!=stamp or r['sha256']!=sha(path):raise ValueError('Heatmap rescue cache drift')
        else:
            start=time.perf_counter();b=load_graph(ctx.incumbent(name));p=load_graph(source);ledger=read_json(meta)['variants']['R_image_local']['ledger']
            n,e,actions,stats=apply(ctx,s,b['nodes'],b['edges'],p['nodes'],p['edges'],ledger,ns)
            save_graph(path,n,e);r=dict(dataset=name,inputs=stamp,sha256=sha(path),stats=stats,actions=actions,seconds=time.perf_counter()-start)
            write_json(dest,r)
        records.append(r);print('heatmap_rescue',name,r['stats'],flush=True)
    keys=set().union(*(r['stats'] for r in records))
    write_json(ctx.out/'heatmap_rescue_summary.json',dict(samples=199,config=CONFIG,
        totals={k:sum(r['stats'].get(k,0) for r in records) for k in keys},seconds=sum(r['seconds'] for r in records)))

if __name__=='__main__':
    from .context import RunContext
    run(RunContext.default())
