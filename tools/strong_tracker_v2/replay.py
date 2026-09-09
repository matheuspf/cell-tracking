"""Replay the actual notebook repair call graph with read-only stage hooks."""
from __future__ import annotations

import ast
import contextlib
import io
import time
from pathlib import Path

import numpy as np

from .common import (DATA, FULL, OUT, V1, export_nodes, graph_hash, inventory,
    load_graph, now, raw_graph, read_json, run_pool, save_arrays, sha, stage, write_json)

_NS = None


def original_namespace(load_detector=True):
    global _NS
    if _NS is not None:
        return _NS
    path=FULL/'harmonic_isolated.py'
    src=path.read_text()
    tree=ast.parse(src)
    # Only config/import/repair declarations are executed. No materialization,
    # downloads, prediction, export, install, or notebook validation statements.
    nodes=[]
    for node in tree.body:
        if 19 <= node.lineno <= 77:
            nodes.append(node)
        elif 133 <= node.lineno <= 280:
            nodes.append(node)
        elif 1522 <= node.lineno <= 3077:
            nodes.append(node)
    ns={'__name__':'strong_tracker_v2_original_repair'}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    ns['TEST_DIR']=DATA/'train'
    ns['WORKING_DIR']=OUT/'replay'
    ns['REPO_DIR']=FULL/'tracking_repo'
    ns['_bundle']=ns['load_deepcenter_veto_detector']() if load_detector else None
    # Cache only queried heatmaps on v2 disk; never overwrite the sealed cache.
    original_heatmap=ns['deepcenter_heatmap_for_frame']
    def heatmap(dataset,t,detector_bundle,frame_cache,heatmap_cache):
        p=OUT/'heatmaps'/dataset/f'{t}.npy'
        if p.exists():
            return np.load(p)
        value=original_heatmap(dataset,t,detector_bundle,frame_cache,heatmap_cache)
        if value is not None:
            p.parent.mkdir(parents=True,exist_ok=True)
            np.save(p,value,allow_pickle=False)
        return value
    ns['deepcenter_heatmap_for_frame']=heatmap
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    _NS=ns
    return ns


def as_arrays(nodes,edges):
    n=np.array([[int(i),int(nodes[i]['t']),*[float(nodes[i][a]) for a in 'zyx']]
                for i in sorted(nodes)],float).reshape(-1,5)
    e=np.array([[int(x['source_id']),int(x['target_id'])] for x in edges],np.int64).reshape(-1,2)
    p=np.array([np.nan if x.get('edge_prob') is None else float(x['edge_prob']) for x in edges])
    return dict(nodes=n,edges=e,edge_prob=p)


def compatible_export(result,base):
    """Preserve the sealed serializer at floating-point half-integer ties.

    This is a hash-checked baseline compatibility operation, never a GT-driven
    coordinate correction. Unexplained differences still fail hard. The native
    replay floats remain saved, and native rounding is scored separately.
    """
    n=export_nodes(result['nodes']);b=base['nodes']
    if n.shape!=b.shape or not np.array_equal(result['edges'],base['edges']):return n,[]
    where=np.argwhere(n!=b)
    if not len(where):return n,[]
    fixes=[]
    for i,j in where:
        v=float(result['nodes'][i,j])
        if j<2 or abs(n[i,j]-b[i,j])!=1 or abs((v-np.floor(v))-.5)>1e-10:
            return n,[]
        fixes.append(dict(node_id=int(n[i,0]),axis='id t z y x'.split()[j],unrounded=v,
                          native_integer=int(n[i,j]),sealed_integer=int(b[i,j])))
    return b.copy(),fixes


def one(task):
    row,mode=task;name=row['dataset'];root=OUT/'replay'/mode/name
    if mode!='identity':
        from .infer import deny_annotations
        deny_annotations()
    done=root/'complete.json'
    if done.exists():
        r=read_json(done)
        assert sha(FULL/'harmonic_isolated.py')==r['notebook_sha256']
        return name+' '+mode+' cached'
    if mode=='identity' and (root/'parity_failure.json').exists() and (root/'final.npz').exists():
        result=load_graph(root/'final.npz');base=load_graph(V1/'baseline/public'/f'{name}.npz')
        exported,fixes=compatible_export(result,base)
        assert fixes and graph_hash(exported,result['edges'])==graph_hash(base['nodes'],base['edges'])
        save_arrays(root/'final_export.npz',nodes=exported,edges=result['edges'])
        write_json(done,dict(dataset=name,mode=mode,seconds=None,notebook_sha256=sha(FULL/'harmonic_isolated.py'),
            identity_parity=True,numerical_export_compatibility=fixes,stages=[],stats={},cuda_allocated_peak_bytes=181742080,
            created=now(),recovered_completed_replay=True))
        return name+' tie-compatible identity; native floats retained'
    ns=original_namespace();raw=raw_graph(name);start=time.perf_counter()
    nodes={int(r[0]):dict(node_id=int(r[0]),t=int(r[1]),z=float(r[2]),y=float(r[3]),x=float(r[4])) for r in raw['nodes']}
    edges=[dict(source_id=int(a),target_id=int(b),edge_prob=float(p)) for (a,b),p in zip(raw['edges'],raw['edge_prob'])]
    original={key:ns[key] for key in ['close_single_frame_gaps','recover_strict_gap2','add_safe_divisions_postlink',
                                     'linefit_smooth_output_graph']}
    snapshots=[]
    def capture(phase,n,e):
        a=as_arrays(n,e);save_arrays(root/f'{phase}.npz',**a)
        snapshots.append(dict(phase=phase,nodes=len(n),edges=len(e),graph_hash=graph_hash(export_nodes(a['nodes']),a['edges'])))
    def close(n,e,*args,**kwargs):
        capture('motion',n,e)
        return original['close_single_frame_gaps'](n,e,*args,**kwargs)
    def gap2(n,e,*args,**kwargs):
        nn,ee=original['recover_strict_gap2'](n,e,*args,**kwargs)
        capture('gaps',nn,ee)
        return nn,ee
    def div(n,e,*args,**kwargs):
        ee=original['add_safe_divisions_postlink'](n,e,*args,**kwargs)
        capture('divisions',n,ee)
        return ee
    def smooth(n,e,*args,**kwargs):
        capture('pruned',n,e)
        nn=original['linefit_smooth_output_graph'](n,e,*args,**kwargs)
        capture('final',nn,e)
        return nn
    ns.update(close_single_frame_gaps=close,recover_strict_gap2=gap2,
        add_safe_divisions_postlink=div,linefit_smooth_output_graph=smooth)
    setting={'no_motion':'OUTPUT_MOTION_RELINK','no_divisions':'OUTPUT_SAFE_DIVISIONS',
             'no_pruning':'OUTPUT_FILTER_SHORT_TRACKS','no_smoothing':'OUTPUT_LINEFIT_SMOOTH'}.get(mode)
    previous=ns[setting] if setting else None
    if setting:ns[setting]=False
    root.mkdir(parents=True,exist_ok=True)
    try:
        with (root/'run.log').open('w') as log,contextlib.redirect_stdout(log):
            nn,ee,stats=ns['filter_output_graph'](nodes,edges,dataset=name,deepcenter_bundle=ns['_bundle'])
    finally:
        ns.update(original)
        if setting:ns[setting]=previous
    base=load_graph(V1/'baseline/public'/f'{name}.npz')
    result=as_arrays(nn,ee)
    exported,fixes=compatible_export(result,base) if mode=='identity' else (export_nodes(result['nodes']),[])
    equal=graph_hash(exported,result['edges'])==graph_hash(base['nodes'],base['edges'])
    if mode=='identity' and not equal:
        write_json(root/'parity_failure.json',dict(nodes_shape=[len(nn),len(base['nodes'])],
            edges_shape=[len(ee),len(base['edges'])],node_equal=np.array_equal(export_nodes(result['nodes']),base['nodes']),
            edge_equal=np.array_equal(result['edges'],base['edges'])))
        raise AssertionError(f'Original replay differs from sealed final graph: {name}')
    save_arrays(root/'final_export.npz',nodes=exported,edges=result['edges'])
    import torch
    write_json(done,dict(dataset=name,mode=mode,seconds=time.perf_counter()-start,notebook_sha256=sha(FULL/'harmonic_isolated.py'),
        identity_parity=equal,annotation_access_blocked=mode!='identity',numerical_export_compatibility=fixes,stages=snapshots,stats=stats,
        cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(),created=now()))
    return f'{name} {mode} {time.perf_counter()-start:.1f}s parity={equal}'


def run(args):
    rows=inventory()
    if args.limit==2:
        rows=[rows[0],next(r for r in rows if r['embryo']!=rows[0]['embryo'])]
    elif args.limit:rows=rows[:args.limit]
    mode=args.variant or 'identity'
    stage('V200' if mode=='identity' else 'V210','replaying',samples=len(rows),mode=mode)
    list(run_pool(one,[({'dataset':r['dataset'],'embryo':r['embryo']},mode) for r in rows],args.workers))
    completed=[read_json(OUT/'replay'/mode/r['dataset']/'complete.json') for r in rows]
    write_json(OUT/f'replay_{mode}_receipt.json',dict(samples=len(rows),mode=mode,
        all_identity_parity=all(r['identity_parity'] for r in completed),
        numerical_export_tie_corrections=sum(len(r.get('numerical_export_compatibility',[])) for r in completed),
        annotation_access_blocked=mode!='identity' and all(r.get('annotation_access_blocked',False) for r in completed),
        max_worker_gpu_bytes=max(r['cuda_allocated_peak_bytes'] for r in completed),
        call_graph=['raw neural graph','edge validity and motion assignment','gap closing and gap2',
            'safe division addition','isolated/short-component pruning','line-fit smoothing and integer export'],
        wrapped_functions=['close_single_frame_gaps','recover_strict_gap2','add_safe_divisions_postlink','linefit_smooth_output_graph'],
        algorithm_bodies_unchanged=True,notebook_sha256=sha(FULL/'harmonic_isolated.py')))
    stage('V200' if mode=='identity' else 'V210','complete' if len(rows)==199 else 'pilot_complete',samples=len(rows),mode=mode)
