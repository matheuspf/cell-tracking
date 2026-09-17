"""Source-only pre/post decoder hard negatives; unknowns are diagnostic only."""
from collections import defaultdict,OrderedDict
import gzip
import json
import numpy as np
import torch

from pipeline_error_training.feasibility import from_record
from strong_tracker_v3.decode import Action,BoundedActionComponents
from .common import WORK,read_json,write_json,verified_graph,adjacency
from .infer import decode
from .resources import Lease
from .actions import close_resources


def scored_anchors(model,dataset,keys,device,amp):
    """Yield CPU scores while yielding the shared GPU lease every 30 seconds."""
    import time,contextlib
    iterator=iter(keys);finished=False
    while not finished:
        context=Lease('source_mining/'+dataset.source,required_gib=3.) if model.image else contextlib.nullcontext()
        with context:
            model.to(device);started=time.monotonic()
            while True:
                try:key=next(iterator)
                except StopIteration:finished=True;break
                batch,scene=dataset.batch(dict(key=key),device)
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
                    output=model([batch],scene[None] if scene is not None else None)[0]
                yield key,output['gain'].cpu().numpy(),supported_false(batch).cpu().numpy(),\
                    ~batch['supported'].cpu().numpy()&~batch['keep'].cpu().numpy()
                if time.monotonic()-started>=30:break
            model.cpu()
            if model.image:torch.cuda.empty_cache()


def supported_false(batch):
    return batch['supported'].bool() & (batch['utility']<=1e-12) & ~batch['keep']


@torch.no_grad()
def refresh(model,dataset,device,amp,path):
    if path.exists():
        return read_json(path)['replay_keys']
    model.eval();ranked=defaultdict(list);unknown=[];post=[]
    by_clip=defaultdict(list)
    for key,a in dataset.anchors.items():by_clip[a['dataset']].append(key)
    rows={r['dataset']:r for r in dataset.rows}
    for name,keys in sorted(by_clip.items()):
        graph=verified_graph(rows[name]);_,pred,succ=adjacency(graph['nodes'],graph['edges'])
        with gzip.open(WORK/'source'/dataset.source/name/'anchors.json.gz','rt') as f:
            records={a['anchor']:a for a in json.load(f)}
        collector=BoundedActionComponents();lookup={}
        for key,gains,false,unidentified in scored_anchors(model,dataset,keys,device,amp):
            if false.any():
                score=float(gains[false].max())
                ranked[dataset.anchors[key]['group']].append((score,key))
            if unidentified.any() and gains[unidentified].max()>0:
                unknown.append(dict(key=key,max_gain=float(gains[unidentified].max()),label='unknown_not_negative'))
            a=records[dataset.anchors[key]['anchor']]
            for i in np.flatnonzero(gains>1e-9):
                d=from_record(dict(decision=a['decisions'][i]))
                if d.kind=='keep':continue
                d=close_resources(d,pred,succ)
                lookup[d.key]=(key,bool(false[i]),float(gains[i]),d.kind)
                collector.add(Action(d.key,float(gains[i]),set(d.remove),set(d.add),set(d.resources),
                                     [dict(event=d.event),*d.owners],d.kind))
        _,ledger=decode(graph['nodes'],graph['edges'],collector)
        for edit in ledger['edits']:
            key,false,gain,kind=lookup[edit['key']]
            if false and kind=='division':
                post.append(dict(key=key,gain=gain,kind=kind))
                ranked[dataset.anchors[key]['group']].append((gain,key))
    # Exactly one representative per source lineage/conflict group per refresh.
    representatives=[max(values,key=lambda x:(x[0],x[1])) for _,values in sorted(ranked.items())]
    representatives.sort(key=lambda x:(-x[0],x[1]))
    keys=[key for _,key in representatives]
    write_json(path,dict(source=dataset.source,partition=dataset.partition,replay_keys=keys,
        pre_decoder_group_representatives=[dict(key=k,gain=v) for v,k in representatives],
        selected_supported_false_forks=post,unknown_high_scores=unknown,
        unknown_as_negative_count=0,one_per_biological_group=True,
        scope='Full complete actions on prepared source probability sample plus positive stream; actual bounded decoder',
        target_used=False))
    model.train()
    return keys
