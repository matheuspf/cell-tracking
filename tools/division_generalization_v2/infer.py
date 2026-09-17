"""Annotation-free full bank scoring; native evidence enters only learned heads."""
from collections import Counter
import contextlib
import time
import numpy as np
import torch

from strong_tracker_v3.decode import Action,BoundedActionComponents,solve_actions,legal_edges
from pipeline_error_training.actions import fork_support
from pipeline_error_training.serialization import export_csv,export_geff
from .actions import EventBank,canonical_actions,close_resources
from .features import build
from .model import ActionModel
from .fast_scenes import InferenceScenes as Scenes
from .resources import Lease
from .common import save_graph,write_json,read_json,graph_hash,sha,adjacency,RESULTS


def load_checkpoint(path):
    from pathlib import Path
    saved=torch.load(path,map_location='cpu',weights_only=False)
    for filename,key in [('model.py','model_code_sha256'),('features.py','features_code_sha256'),('scenes.py','scenes_code_sha256')]:
        actual=sha(Path(__file__).with_name(filename))
        if saved['recipe'].get(key)!=actual:
            compatibility=RESULTS/'model_code_compatibility.json'
            allowed=False
            if filename=='model.py' and not saved['recipe']['image'] and compatibility.exists():
                c=read_json(compatibility)
                allowed=(c['status']=='measured' and c['old_model_sha256']==saved['recipe'].get(key)
                    and c['new_model_sha256']==actual and c['G30_exact_outputs'])
            if not allowed:raise ValueError('Checkpoint architecture/input implementation drift: '+filename)
    model=ActionModel(image=saved['recipe']['image'])
    model.load_state_dict(saved['model']);model.eval()
    return model,saved['recipe']


def decode(nodes,edges,collector):
    actions,stream=collector.finish()
    chosen,solver=solve_actions(actions,max_changed_edges=int(np.floor(.02*len(edges))),time_limit=2.)
    ix,_,_=adjacency(nodes,edges)
    original={(ix[int(a)],ix[int(b)]) for a,b in edges};result=set(original);ledger=[]
    for index in chosen:
        a=actions[index]
        if not a.remove<=result:raise RuntimeError('Atomic owner removal drift')
        result.difference_update(a.remove);result.update(a.add)
        ledger.append(dict(value=a.value,kind=a.kind,key=a.candidate,owners=a.owners,
            remove=sorted(a.remove),add=sorted(a.add)))
    convert=lambda es:np.asarray([[nodes[a,0],nodes[b,0]] for a,b in sorted(es)],np.int64).reshape(-1,2)
    out=edges.copy() if result==original else convert(result)
    if not legal_edges(nodes,set(map(tuple,out))):raise RuntimeError('Illegal decoded ownership')
    if original^result != {e for a in ledger for e in a['remove']+a['add']}:
        raise RuntimeError('Unexplained edit outside owned components')
    return out,dict(**stream,**solver,edits=ledger,changed_edges=len(original^result),
                    exact_outside_owned_actions=True)


@torch.no_grad()
def score_anchor(model,bank,parent,records,scenes,row,device,amp):
    arrays=build(bank,parent,records)
    b={k:torch.as_tensor(v,device=device) for k,v in arrays.items() if k!='query_nodes'}
    s=None
    if model.image:
        raw=scenes.get(row['dataset'],parent,bank.nodes[parent,2:],int(bank.nodes[parent,1]))
        s=torch.as_tensor(raw,device=device,dtype=torch.float32)[None]/255.
    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
        output=model([b],s)[0]
    return output['gain'].cpu().numpy(),arrays


def predict(row,graph,native,model,recipe,destination=None,*,calibration=None,literal_zero=False,
            disable=False,applications=('protected','replacement'),max_anchors=None):
    started=time.monotonic()
    bank=EventBank(graph['nodes'],graph['edges'],native,row['physical_scale'])
    scene_reader=Scenes([row],persist=False)
    protected=fork_support(graph['nodes'],graph['edges'])
    collectors={a:BoundedActionComponents() for a in applications}
    counts=Counter();timings=Counter();maxima=[];score_summary=dict(raw_min=None,raw_max=None)
    device='cuda' if model.image else 'cpu';amp=bool(model.image and recipe.get('amp',False))
    calibration=calibration or dict(temperature=1.,intercept=0.,status='calibration_unestablished')
    if literal_zero and (calibration['temperature']!=1 or calibration['intercept']!=0):
        raise ValueError('Literal zero identity fixture requires disabled calibration')
    parents=iter(sorted(bank.expanded));finished=False
    while not finished and not disable:
        context=Lease('inference/'+row['dataset'],required_gib=3.) if model.image else contextlib.nullcontext()
        with context as lease:
            model.to(device);model.eval();block=time.monotonic()
            while True:
                try:parent=next(parents)
                except StopIteration:finished=True;break
                if max_anchors is not None and counts['anchors']>=max_anchors:
                    finished=True;break
                counts['anchors']+=1
                if destination is not None and counts['anchors']%128==0:
                    write_json(destination/'progress.json',dict(dataset=row['dataset'],status='running',
                        anchors=counts['anchors'],total_anchors=len(bank.expanded),
                        seconds=time.monotonic()-started,counts=dict(counts)))
                ts=time.monotonic();records,rejections=canonical_actions(bank,parent,replace=True)
                timings['bank_seconds']+=time.monotonic()-ts;counts.update(rejections)
                if not records:continue
                ts=time.monotonic()
                gains,_=score_anchor(model,bank,parent,records,scene_reader,row,device,amp)
                timings['scene_and_model_seconds']+=time.monotonic()-ts
                if literal_zero and np.count_nonzero(gains):
                    raise ValueError('Literal-zero fixture received a nonzero model score')
                counts['raw_positive_gains']+=int((gains>1e-9).sum())
                score_summary['raw_min']=min(float(gains.min()),score_summary['raw_min']) if score_summary['raw_min'] is not None else float(gains.min())
                score_summary['raw_max']=max(float(gains.max()),score_summary['raw_max']) if score_summary['raw_max'] is not None else float(gains.max())
                gains=gains/calibration['temperature']+calibration['intercept']
                gains=np.where([d.kind=='keep' for d,_ in records],0.,gains)
                if not np.isfinite(gains).all():raise ValueError('Nonfinite learned action gain')
                counts['actions']+=len(records)
                counts['positive_gains']+=int((gains>1e-9).sum())
                best=int(np.argmax(gains))
                counts['anchors_preferring_edit']+=int(records[best][0].kind!='keep')
                maxima.append(dict(parent=parent,key=records[best][0].key,gain=float(gains[best]),
                                   kind=records[best][0].kind))
                for (d,_),gain in zip(records,gains):
                    if d.kind=='keep' or gain<=1e-9:continue
                    d=close_resources(d,bank.pred,bank.succ)
                    touched={n for e in d.remove|d.add for n in e}
                    for application,collector in collectors.items():
                        if application=='protected' and touched&protected:
                            counts['protected_rejection']+=1;continue
                        collector.add(Action(d.key,float(gain),set(d.remove),set(d.add),set(d.resources),
                            [dict(event=d.event),*d.owners],d.kind))
                if time.monotonic()-block>=30:break
            model.cpu()
            if model.image:torch.cuda.empty_cache()
        if model.image:
            timings['lease_seconds']+=lease.seconds;timings['lease_wait_seconds']+=lease.wait_seconds
    outputs={}
    for application,collector in collectors.items():
        edges,ledger=decode(graph['nodes'],graph['edges'],collector)
        outputs[application]=dict(nodes=graph['nodes'],edges=edges,ledger=ledger)
        if destination is not None:
            path=destination/application/(row['dataset']+'.npz')
            save_graph(path,graph['nodes'],edges)
            write_json(path.with_suffix('.json'),dict(dataset=row['dataset'],application=application,
                graph_hash=graph_hash(graph['nodes'],edges),counts=dict(counts),timings=dict(timings),
                score_summary=score_summary,
                seconds=time.monotonic()-started,ledger=ledger,calibration=calibration,
                source=recipe.get('source'),literal_zero_active_path=literal_zero,disable_switch=disable,
                bank_sha256=bank.hash,partial_fixture=max_anchors is not None))
    return outputs,dict(counts=dict(counts),timings=dict(timings),seconds=time.monotonic()-started,
                        maxima=maxima,bank_sha256=bank.hash,score_summary=score_summary)
