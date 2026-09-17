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


def conflict_representatives(ranked):
    """One hard anchor per biological group and connected resource conflict."""
    candidates=[max(values,key=lambda x:(x['gain'],x['key'])) for values in ranked.values()]
    parent=list(range(len(candidates)));owners={}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(candidates):
        for resource in row['resources']:
            key=(row['dataset'],*resource)
            if key in owners:parent[find(i)]=find(owners[key])
            else:owners[key]=i
    components=defaultdict(list)
    for i,row in enumerate(candidates):components[find(i)].append(row)
    selected=[max(rows,key=lambda r:(r['gain'],r['key'])) for rows in components.values()]
    selected.sort(key=lambda r:(-r['gain'],r['key']))
    return selected,len(candidates)


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
            a=records[dataset.anchors[key]['anchor']]
            if false.any():
                indices=np.flatnonzero(false);i=indices[np.argmax(gains[indices])]
                d=close_resources(from_record(dict(decision=a['decisions'][i])),pred,succ)
                ranked[dataset.anchors[key]['group']].append(dict(gain=float(gains[i]),key=key,
                    dataset=name,action_key=d.key,resources=sorted(d.resources)))
            if unidentified.any() and gains[unidentified].max()>0:
                unknown.append(dict(key=key,max_gain=float(gains[unidentified].max()),label='unknown_not_negative'))
            for i in np.flatnonzero(gains>1e-9):
                d=from_record(dict(decision=a['decisions'][i]))
                if d.kind=='keep':continue
                d=close_resources(d,pred,succ)
                lookup[d.key]=(key,bool(false[i]),float(gains[i]),d.kind,sorted(d.resources))
                collector.add(Action(d.key,float(gains[i]),set(d.remove),set(d.add),set(d.resources),
                                     [dict(event=d.event),*d.owners],d.kind))
        _,ledger=decode(graph['nodes'],graph['edges'],collector)
        for edit in ledger['edits']:
            key,false,gain,kind,resources=lookup[edit['key']]
            if false and kind=='division':
                post.append(dict(key=key,gain=gain,kind=kind))
                ranked[dataset.anchors[key]['group']].append(dict(gain=gain,key=key,
                    dataset=name,action_key=edit['key'],resources=resources))
    # Exactly one representative per source lineage/conflict group per refresh.
    representatives,biological_groups=conflict_representatives(ranked)
    keys=[r['key'] for r in representatives]
    write_json(path,dict(source=dataset.source,partition=dataset.partition,replay_keys=keys,
        pre_decoder_group_representatives=representatives,
        selected_supported_false_forks=post,unknown_high_scores=unknown,
        unknown_as_negative_count=0,one_per_biological_group=True,one_per_resource_conflict_component=True,
        biological_groups_before_conflict_cap=biological_groups,conflict_components=len(keys),
        scope='Full complete actions on prepared source probability sample plus positive stream; actual bounded decoder',
        target_used=False))
    model.train()
    return keys
