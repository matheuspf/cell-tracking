"""Complete deployment denominators, frozen image embeddings and atomic edits."""
from collections import Counter
import heapq
from pathlib import Path
import time
import numpy as np
import torch
from .common import Blocked,read,write,sha
from .features import NAMES,IDENTITY_NAMES,action_features,normalize
from .actions import Bank,utilities,apply_decisions
from .models import CompactPolicy


def observation_evidence(bank,image_path,resources,model=None):
    from pipeline_error_training.crops import Images
    from pipeline_error_training.compact_inference_crops import sample
    images=Images(image_path,max_frames=9);n=len(bank.nodes)
    summaries=np.empty((n,3),np.float32)
    embedding=np.empty((n,128),np.float32) if model is not None else None
    parity=None
    for start in range(0,n,4096):
        indices=range(start,min(n,start+4096));patch,valid=sample(images,bank.nodes,bank.pred,bank.succ,indices)
        values=patch.astype(np.float32)/255
        summaries[start:start+len(patch),0]=values[:,1].mean((1,2,3))
        summaries[start:start+len(patch),1]=values[:,1].std((1,2,3))
        summaries[start:start+len(patch),2]=valid[:,1,2]
        if model is not None:
            with resources.lease(2*2**30) as lease:
                lease['operation']='compact_embeddings'
                model.cuda().eval()
                with torch.inference_mode():
                    x=torch.from_numpy(values).cuda();mask=torch.from_numpy(valid).cuda()
                    z=model.encoder(x,mask)
                    embedding[start:start+len(patch)]=z.cpu().numpy()
                    if parity is None:
                        reference=model.encoder(x[:1],mask[:1])
                        difference=float((z[:1]-reference).abs().max())
                        if difference>1e-5:raise Blocked('Frozen compact embedding batch parity failed')
                        parity=dict(maximum_absolute_difference=difference,tolerance=1e-5,passed=True)
                    del x,mask,z
                model.cpu();torch.cuda.empty_cache()
    return summaries,embedding,parity


def score_bank(bank,confidence,image_path,normalizer,resources,*,model=None,linear=None,progress=None,_profile_parents=None):
    """Yield every nonempty complete parent exactly once, without row pruning."""
    summaries,z,parity=observation_evidence(bank,image_path,resources,model)
    pending=[];rows=0;counts=Counter()
    def flush():
        if not pending:return []
        if linear is not None:
            a=np.asarray(linear['occurrence']);b=np.asarray(linear['action'])
            return [(g,float(p@a[:-1]+a[-1]),x@b[:-1]+b[-1]) for g,x,p in pending]
        occ=[];action=[];lengths=[]
        for g,x,p in pending:
            parent=g['parent'];ds=sorted({j for d in g['forks'] for j in d.event[1:3]})
            occ.append(np.r_[z[parent],z[ds].mean(0),z[ds].max(0),p])
            pairs=np.array([d.event[1:3] for d in g['forks']],np.int64)
            action.append(np.c_[np.broadcast_to(z[parent],(len(x),128)),z[pairs[:,0]]+z[pairs[:,1]],np.abs(z[pairs[:,0]]-z[pairs[:,1]]),x])
            lengths.append(len(x))
        with resources.lease(2*2**30) as lease:
            lease['operation']='compact_heads'
            model.cuda().eval()
            with torch.inference_mode():
                aa=model.occurrence(torch.from_numpy(np.stack(occ)).cuda()).flatten().cpu().numpy()
                bb=model.action(torch.from_numpy(np.concatenate(action)).cuda()).flatten().cpu().numpy()
            model.cpu();torch.cuda.empty_cache()
        end=np.cumsum([0,*lengths])
        return [(g,float(a),bb[end[i]:end[i+1]]) for i,((g,_,_),a) in enumerate(zip(pending,aa))]
    for index,parent in enumerate(sorted(bank.expanded)):
        if _profile_parents is not None and parent not in _profile_parents:continue
        g=bank.parent(parent)
        if not g['complete']:
            counts[g['reason']]+=1
            counts['incomplete_unique_forks_lower_bound']+=g.get('unique_forks_lower_bound',len(g['forks']))
            continue
        if not g['forks']:counts['empty']+=1;continue
        x,p=action_features(bank,g,confidence,summaries=summaries)
        x=normalize(x,*normalizer['action']);p=normalize(p,*normalizer['parent'])
        if pending and (len(pending)>=4096 or rows+len(x)>262144):
            yield from flush();pending=[];rows=0
        pending.append((g,x,p));rows+=len(x);counts['complete_nonempty']+=1;counts['complete_group_unique_forks']+=len(x)
        if progress is not None and index%1000==0:write(progress,dict(parents=index,total_parents=len(bank.expanded),census=dict(counts),embedding_parity=parity))
    yield from flush()
    if progress is not None:write(progress,dict(status='complete',total_parents=len(bank.expanded),census=dict(counts),embedding_parity=parity,
        fork_count_status='lower_bound' if counts['parent_unique_action_cap'] or counts['missing_score'] else 'exact',
        construction_rejections=dict(bank.alternatives.rejections)))


def load_model(folder):
    model=CompactPolicy(2*len(NAMES)+1,len(NAMES),len(IDENTITY_NAMES))
    model.load_state_dict(torch.load(Path(folder)/'compact.pt',map_location='cpu',weights_only=True)['model'])
    return model.eval()


def apply_package(spec,package,images,output,resources,baseline):
    from .graphs import export_csv,read_csv,graph_hash
    with np.load(output/'graph.npz') as f:arrays={k:f[k] for k in f.files}
    nodes,edges,confidence,scores=(arrays[k] for k in ('nodes','edges','confidence','edge_scores'))
    bank=Bank(nodes,edges,scores,100)
    norm=read(package/'normalizer.json')['values'];cal=read(package/'calibration.json')
    model=load_model(package) if spec['arm']=='C11' else None
    linear=read(package/'linear.json')['parameters'] if spec['arm']=='C01' else None
    tail=[];count=0;started=time.monotonic()
    parent_ids=[];occurrence_logits=[];conditional_logits=[];offset=[0]
    def proposals():
        nonlocal count
        for group,occurrence,conditional in score_bank(bank,confidence,images,norm,resources,model=model,linear=linear,
                                                      progress=output/'policy_progress.json'):
            parent_ids.append(group['parent']);occurrence_logits.append(occurrence)
            conditional_logits.append(np.asarray(conditional).copy());offset.append(offset[-1]+len(conditional))
            raw=utilities(bank,group,occurrence,conditional,0.)
            calibrated=utilities(bank,group,occurrence/cal['temperature']+cal['intercept'],conditional,cal['margin'] or 0)
            count+=1;entry=(float(raw.max()),int(group['parent']),float(occurrence),len(raw))
            if len(tail)<50:heapq.heappush(tail,entry)
            elif entry>tail[0]:heapq.heapreplace(tail,entry)
            if not cal['disabled_policy']:
                for d,value in zip(group['forks'],calibrated):
                    if value>0:yield d,float(value)
    # tee buffers at most one pair: zip advances both iterators together. The
    # inherited collector sees the complete stream and retains bounded components.
    from itertools import tee
    a,b=tee(proposals())
    result,solver=apply_decisions(nodes,edges,(d for d,v in a),(v for d,v in b),max_fraction=.02,time_limit=2.)
    actual=export_csv(output/'submission.csv',spec.get('dataset_name',images.name.removesuffix('.zarr')),nodes,result)
    restored=read_csv(output/'submission.csv',spec.get('dataset_name',images.name.removesuffix('.zarr')))
    if not all(np.array_equal(a,b) for a,b in zip(actual,restored)):raise Blocked('Policy integer CSV roundtrip failed')
    arrays['nodes'],arrays['edges']=actual
    np.savez_compressed(output/'graph.npz',**arrays)
    # Label-free raw scores permit post-freeze attribution without another model
    # pass or GT-directed inference. Complete legal denominators are preserved.
    np.savez_compressed(output/'policy_logits.npz',parents=np.asarray(parent_ids,np.int64),
        occurrence=np.asarray(occurrence_logits,np.float64),offset=np.asarray(offset,np.int64),
        conditional=np.concatenate(conditional_logits) if conditional_logits else np.empty(0,np.float64))
    write(output/'policy_trace.json',dict(raw_top_groups=sorted(tail,reverse=True),groups=count,solver=solver,
                                        disabled_policy=cal['disabled_policy'],policy_wall_seconds=time.monotonic()-started))
    return {**baseline,'nodes':len(nodes),'edges':len(result),'graph_hash':graph_hash(*actual),
            'graph_sha256':sha(output/'graph.npz'),'csv_sha256':sha(output/'submission.csv'),
            'disabled_policy':cal['disabled_policy'],'changed_edges':solver['changed_edges'],
            'accepted_actions':len(solver['edits']),'C00_graph_hash':baseline['graph_hash'],
            'policy_logits_sha256':sha(output/'policy_logits.npz'),
            'policy_code_sha256':sha(Path(__file__)),'action_bank_code_sha256':sha(Path(__file__).with_name('actions.py'))}
