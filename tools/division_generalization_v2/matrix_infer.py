"""Frozen checkpoint matrix: one raw scene and one bank enumeration per anchor."""
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import numpy as np
import torch

from strong_tracker_v3.decode import Action,BoundedActionComponents
from pipeline_error_training.actions import fork_support
from .actions import EventBank,canonical_actions,close_resources
from .fast_features import build,prepare as prepare_frozen_features
from .fast_scenes import InferenceScenes as Scenes
from .resources import Lease
from .infer import decode
from .common import save_graph,write_json,graph_hash


def prepared_anchors(bank,row,need_images,max_anchors):
    """Bounded CPU scene prefetch; yield the original anchor order exactly."""
    local=threading.local()
    parents=sorted(bank.expanded)
    if max_anchors is not None:parents=parents[:max_anchors]
    # Warm the immutable graph constructor and native pair lookup before the
    # readers start. All per-anchor decisions and counters remain independent.
    if parents:canonical_actions(bank,parents[0],replace=True)
    if '_pair_index' not in bank.native:
        bank.native['_pair_index']={tuple(map(int,p)):k for k,p in enumerate(bank.native['pairs'])}
    prepare_frozen_features(bank)
    def prepare(parent):
        ts=time.monotonic();records,rejected=canonical_actions(bank,parent,replace=True)
        arrays=build(bank,parent,records) if records else None
        bank_seconds=time.monotonic()-ts;ts=time.monotonic();raw=None
        if need_images and records:
            if not hasattr(local,'reader'):local.reader=Scenes([row],persist=False)
            raw=local.reader.get(row['dataset'],parent,bank.nodes[parent,2:],int(bank.nodes[parent,1]))
            # Every inference anchor is visited once; no raw scene LRU is useful.
            local.reader.memory.clear()
        return parent,records,rejected,arrays,raw,bank_seconds,time.monotonic()-ts
    iterator=iter(parents)
    with ThreadPoolExecutor(max_workers=4,thread_name_prefix='v2-scene-io') as pool:
        pending=deque(pool.submit(prepare,p) for p in [next(iterator,None) for _ in range(min(8,len(parents)))] if p is not None)
        while pending:
            result=pending.popleft().result()
            parent=next(iterator,None)
            if parent is not None:pending.append(pool.submit(prepare,parent))
            yield result


@torch.no_grad()
def predict_matrix(row,graph,native,packages,output,max_anchors=None):
    started=time.monotonic()
    bank=EventBank(graph['nodes'],graph['edges'],native,row['physical_scale'])
    protected=fork_support(graph['nodes'],graph['edges'])
    collectors={key:{a:BoundedActionComponents() for a in ('protected','replacement')} for key in packages}
    counts=Counter();timings=Counter();positive=Counter();maxima={key:[] for key in packages}
    raw_positive=Counter();protected_rejected=Counter();preferred=Counter()
    score_summary={key:dict(raw_min=None,raw_max=None) for key in packages}
    need_images=any(model.image for model,_,_ in packages.values())
    prepared=prepared_anchors(bank,row,need_images,max_anchors);finished=False
    while not finished:
        import contextlib
        context=Lease('frozen_matrix/'+row['dataset'],required_gib=4.) if need_images else contextlib.nullcontext()
        with context as lease:
            for model,_,_ in packages.values():model.to('cuda' if model.image else 'cpu').eval()
            block=time.monotonic()
            while True:
                try:parent,records,rejected,arrays,raw,bank_seconds,loader_seconds=next(prepared)
                except StopIteration:finished=True;break
                counts['anchors']+=1
                counts.update(rejected)
                timings['bank_features_worker_seconds']+=bank_seconds
                timings['loader_worker_seconds']+=loader_seconds
                if not records:continue
                ts=time.monotonic()
                batches={}
                for device in ({'cpu','cuda'} if need_images else {'cpu'}):
                    batches[device]={k:torch.as_tensor(v,device=device) for k,v in arrays.items() if k!='query_nodes'}
                scene=None
                if need_images:
                    scene=torch.as_tensor(raw,device='cuda',dtype=torch.float32)[None]/255.
                if need_images:torch.cuda.synchronize()
                timings['transfer_seconds']+=time.monotonic()-ts
                counts['actions']+=len(records)
                for key,(model,recipe,cal) in packages.items():
                    device='cuda' if model.image else 'cpu';ts=time.monotonic()
                    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=bool(model.image and recipe['amp'])):
                        gains=model([batches[device]],scene if model.image else None)[0]['gain'].cpu().numpy()
                    timings[key+'_model_seconds']+=time.monotonic()-ts
                    raw_positive[key]+=int((gains>1e-9).sum())
                    s=score_summary[key]
                    s['raw_min']=min(float(gains.min()),s['raw_min']) if s['raw_min'] is not None else float(gains.min())
                    s['raw_max']=max(float(gains.max()),s['raw_max']) if s['raw_max'] is not None else float(gains.max())
                    gains=gains/cal['temperature']+cal['intercept']
                    gains=np.where(arrays['keep'],0.,gains)
                    if not np.isfinite(gains).all():raise ValueError('Nonfinite matrix gain')
                    best=int(np.argmax(gains));positive[key]+=int((gains>1e-9).sum())
                    preferred[key]+=int(records[best][0].kind!='keep')
                    maxima[key].append(dict(parent=parent,key=records[best][0].key,gain=float(gains[best]),kind=records[best][0].kind))
                    for i in np.flatnonzero(gains>1e-9):
                        d=records[i][0]
                        if d.kind=='keep':continue
                        d=close_resources(d,bank.pred,bank.succ)
                        touched={n for e in d.remove|d.add for n in e}
                        for app,collector in collectors[key].items():
                            if app=='protected' and touched&protected:
                                protected_rejected[key]+=1;continue
                            collector.add(Action(d.key,float(gains[i]),set(d.remove),set(d.add),set(d.resources),
                                                 [dict(event=d.event),*d.owners],d.kind))
                if counts['anchors']%128==0:
                    write_json(output/'progress.json',dict(dataset=row['dataset'],anchors=counts['anchors'],
                        total_anchors=len(bank.expanded),seconds=time.monotonic()-started,packages=len(packages)))
                if time.monotonic()-block>=30:break
            for model,_,_ in packages.values():model.cpu()
            if need_images:torch.cuda.empty_cache()
        if need_images:
            timings['lease_seconds']+=lease.seconds;timings['wait_seconds']+=lease.wait_seconds
    for key,(_,recipe,cal) in packages.items():
        root=output/key
        for app,collector in collectors[key].items():
            edges,ledger=decode(graph['nodes'],graph['edges'],collector)
            path=root/app/(row['dataset']+'.npz');save_graph(path,graph['nodes'],edges)
            write_json(path.with_suffix('.json'),dict(dataset=row['dataset'],application=app,source=recipe['source'],
                graph_hash=graph_hash(graph['nodes'],edges),ledger=ledger,calibration=cal,
                counts=dict(counts,positive_gains=positive[key],raw_positive_gains=raw_positive[key],
                    anchors_preferring_edit=preferred[key],protected_rejection=protected_rejected[key]),
                timings=dict(timings),score_summary=score_summary[key],
                frozen_matrix_scene_shared=True,trainable_embedding_cache=False,bank_sha256=bank.hash,
                seconds=time.monotonic()-started,partial_fixture=max_anchors is not None))
        write_json(root/'trace.json',dict(counts=dict(counts),maxima=maxima[key],timings=dict(timings),bank_sha256=bank.hash,
            score_summary=score_summary[key],
            seconds=time.monotonic()-started,compute_accounting='Shared bank/scene costs counted once at matrix root'))
    write_json(output/'complete.json',dict(counts=dict(counts),timings=dict(timings),seconds=time.monotonic()-started,
        packages=list(packages),partial_fixture=max_anchors is not None,
        cpu_prefetch_workers=4,pending_anchors_max=8,stable_original_anchor_order=True,
        worker_times_overlap=True))
