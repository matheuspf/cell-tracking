"""Frozen checkpoint matrix: one raw scene and one bank enumeration per anchor."""
from collections import Counter
import time
import numpy as np
import torch

from strong_tracker_v3.decode import Action,BoundedActionComponents
from pipeline_error_training.actions import fork_support
from .actions import EventBank,canonical_actions,close_resources
from .features import build
from .scenes import Scenes
from .resources import Lease
from .infer import decode
from .common import save_graph,write_json,graph_hash


@torch.no_grad()
def predict_matrix(row,graph,native,packages,output,max_anchors=None):
    started=time.monotonic()
    bank=EventBank(graph['nodes'],graph['edges'],native,row['physical_scale'])
    reader=Scenes([row],persist=False);protected=fork_support(graph['nodes'],graph['edges'])
    collectors={key:{a:BoundedActionComponents() for a in ('protected','replacement')} for key in packages}
    counts=Counter();timings=Counter();positive=Counter();maxima={key:[] for key in packages}
    need_images=any(model.image for model,_,_ in packages.values())
    parents=iter(sorted(bank.expanded));finished=False
    while not finished:
        import contextlib
        context=Lease('frozen_matrix/'+row['dataset'],required_gib=4.) if need_images else contextlib.nullcontext()
        with context as lease:
            for model,_,_ in packages.values():model.to('cuda' if model.image else 'cpu').eval()
            block=time.monotonic()
            while True:
                try:parent=next(parents)
                except StopIteration:finished=True;break
                if max_anchors is not None and counts['anchors']>=max_anchors:finished=True;break
                counts['anchors']+=1
                ts=time.monotonic();records,rejected=canonical_actions(bank,parent,replace=True)
                counts.update(rejected)
                if not records:continue
                arrays=build(bank,parent,records)
                timings['bank_features_seconds']+=time.monotonic()-ts
                batches={}
                for device in ({'cpu','cuda'} if need_images else {'cpu'}):
                    batches[device]={k:torch.as_tensor(v,device=device) for k,v in arrays.items() if k!='query_nodes'}
                scene=None;ts=time.monotonic()
                if need_images:
                    raw=reader.get(row['dataset'],parent,bank.nodes[parent,2:],int(bank.nodes[parent,1]))
                    scene=torch.as_tensor(raw,device='cuda',dtype=torch.float32)[None]/255.
                timings['loader_transfer_seconds']+=time.monotonic()-ts
                counts['actions']+=len(records)
                for key,(model,recipe,cal) in packages.items():
                    device='cuda' if model.image else 'cpu';ts=time.monotonic()
                    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=bool(model.image and recipe['amp'])):
                        gains=model([batches[device]],scene if model.image else None)[0]['gain'].cpu().numpy()
                    timings[key+'_model_seconds']+=time.monotonic()-ts
                    gains=gains/cal['temperature']+cal['intercept']
                    gains=np.where(arrays['keep'],0.,gains)
                    if not np.isfinite(gains).all():raise ValueError('Nonfinite matrix gain')
                    best=int(np.argmax(gains));positive[key]+=int((gains>1e-9).sum())
                    maxima[key].append(dict(parent=parent,key=records[best][0].key,gain=float(gains[best]),kind=records[best][0].kind))
                    for i in np.flatnonzero(gains>1e-9):
                        d=records[i][0]
                        if d.kind=='keep':continue
                        d=close_resources(d,bank.pred,bank.succ)
                        touched={n for e in d.remove|d.add for n in e}
                        for app,collector in collectors[key].items():
                            if app=='protected' and touched&protected:continue
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
                counts=dict(counts,positive_gains=positive[key]),timings=dict(timings),
                frozen_matrix_scene_shared=True,trainable_embedding_cache=False,bank_sha256=bank.hash,
                seconds=time.monotonic()-started,partial_fixture=max_anchors is not None))
        write_json(root/'trace.json',dict(counts=dict(counts),maxima=maxima[key],timings=dict(timings),bank_sha256=bank.hash,
            seconds=time.monotonic()-started,compute_accounting='Shared bank/scene costs counted once at matrix root'))
    write_json(output/'complete.json',dict(counts=dict(counts),timings=dict(timings),seconds=time.monotonic()-started,
        packages=list(packages),partial_fixture=max_anchors is not None))
