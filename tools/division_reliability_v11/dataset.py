"""Explicit complete source clip banks and supported-group fitting arrays."""
from pathlib import Path
import sys
import time
from .common import DATA, REPO, WORK, Blocked, read, write, sha, now


def prepare_clip(prediction,image_path,label_path,folder,*,diagnostic=False):
    """Caller has already installed its source-only pre-import read guard."""
    import numpy as np
    from .actions import Bank
    from .data import labels
    from .features import action_features,identity_features
    from pipeline_error_training.labels import SourceLabels,supported_incoming
    from pipeline_error_training.crops import Images
    from pipeline_error_training.compact_inference_crops import sample
    folder.mkdir(parents=True,exist_ok=True)
    parent_hash=sha(prediction/'graph.npz')
    if (folder/'receipt.json').exists():
        receipt=read(folder/'receipt.json')
        if receipt['graph_sha256']!=parent_hash:raise Blocked('Source bank parent graph changed')
        if sha(folder/'training.npz')!=receipt['training_sha256']:raise Blocked('Source bank arrays changed')
        return receipt
    with np.load(prediction/'graph.npz') as f:
        nodes,edges,scores,confidence=(f[k] for k in ('nodes','edges','edge_scores','confidence'))
    bank=Bank(nodes,edges,scores,100);gt,ge=labels(label_path)
    lab=SourceLabels(nodes,edges,gt,ge,[1.625,.40625,.40625])
    from .risk_labels import Risks
    risk_adapter=Risks(bank,lab)
    tic=time.monotonic();kept=[];count=dict(positive=0,negative=0,unknown=0,incomplete=0,empty=0)
    multiplicity={};total=0;incomplete_lower_bound=0;events={}
    for i,p in enumerate(sorted(bank.expanded)):
        group=bank.parent(p)
        if not group['complete']:
            count['incomplete']+=1
            incomplete_lower_bound+=group.get('unique_forks_lower_bound',len(group['forks']))
            continue
        if not group['forks']:count['empty']+=1;continue
        risks,es=risk_adapter.group(group)
        target=1 if (risks==1).any() else 0 if (risks==0).all() else -1
        total+=len(risks);bucket=str(min(4096,2**int(np.ceil(np.log2(len(risks))))))
        multiplicity[bucket]=multiplicity.get(bucket,0)+1
        count[{1:'positive',0:'negative',-1:'unknown'}[target]]+=1
        if target>=0:
            for e in es:events.setdefault(str(e),[]).append(len(kept))
            kept.append((group,risks,es))
        if i%2000==0:write(folder/'progress.json',dict(parents=i,total_parents=len(bank.expanded),census=count))
    # Independent incoming identity labels; unmatched proposals never negative.
    pairs=np.array(sorted(bank.logits),np.int64).reshape(-1,2)
    ids=nodes[:,0]
    iy=supported_incoming(ids[pairs],lab.persistent_matches,ge)
    pairs=pairs[iy>=0];iy=iy[iy>=0]
    incoming=sorted(set(pairs[:,1].tolist()))
    needed=sorted({j for group,_,_ in kept for a in group['forks'] for j in a.event[:3]}|set(pairs.flatten()))
    images=Images(image_path,max_frames=9)
    patch,valid=sample(images,nodes,bank.pred,bank.succ,needed)
    index={p:i for i,p in enumerate(needed)}
    rows=[];parents=[];risks=[];event_rows=[];offset=[0];parent_ids=[]
    # Each group uses the complete legal denominator, including its unknown rows.
    image_scalars=patch.astype(np.float32)/255
    for group,y,_ in kept:
        x,p=action_features(bank,group,confidence,image_scalars,valid,index)
        rows.append(x);parents.append(p);risks.extend(y);parent_ids.append(index[group['parent']])
        event_rows.extend([[index[a.event[k]] for k in (0,1,2)] for a in group['forks']]);offset.append(len(risks))
    from .features import NAMES
    mapped_pairs=np.array([[index[int(a)],index[int(b)]] for a,b in pairs],np.int64).reshape(-1,2)
    identity=identity_features(bank,pairs,confidence)
    np.savez_compressed(folder/'training.npz',patch=patch,valid=valid,
        action_x=np.concatenate(rows) if rows else np.empty((0,len(NAMES)),np.float32),
        parent_x=np.asarray(parents,np.float32).reshape(-1,2*len(NAMES)+1),
        risks=np.asarray(risks,np.int8),events=np.asarray(event_rows,np.int64).reshape(-1,3),offset=np.asarray(offset,np.int64),
        parent=np.asarray(parent_ids,np.int64),identity_pairs=mapped_pairs,identity_x=identity,identity_y=iy,
        original_indices=np.asarray(needed,np.int64))
    receipt=dict(status='prepared',graph_sha256=parent_hash,training_sha256=sha(folder/'training.npz'),
                 source_only=True,diagnostic=diagnostic,census=count,complete_group_fork_count=total,
                 all_group_fork_count_lower_bound=total+incomplete_lower_bound,
                 fork_count_status='lower_bound' if count['incomplete'] else 'exact',
                 complete_parent_count=len(bank.expanded)-count['incomplete'],total_parent_count=len(bank.expanded),
                 multiplicity_upper_buckets=multiplicity,positive_events=events,
                 risk_label_shortcuts=risk_adapter.stats,
                 group_parents=[g['parent'] for g,_,_ in kept],identity_groups=len(incoming),
                 identity_positive=int((iy==1).sum()),identity_negative=int((iy==0).sum()),
                 wall_seconds=time.monotonic()-tic,finished_utc=now())
    if not diagnostic:
        from .diagnostics import bank_funnel,detection_audit
        from .evaluation import score,finite
        official,detail=score(image_path.name.removesuffix('.zarr'),nodes,edges,label_path)
        receipt.update(official_metrics=finite(official),official_events=detail,
                       funnel=bank_funnel(bank,lab,events),detection=detection_audit(nodes,edges,gt,ge,lab),
                       action_rejections=dict(bank.alternatives.rejections),
                       preparation_code_sha256=sha(Path(__file__)))
    receipt['wall_seconds']=time.monotonic()-tic
    write(folder/'receipt.json',receipt,immutable=True);return receipt


def run(source,seed,clip,*,part='fit',pilot=False):
    split=read(WORK/'source_partitions.json')[source]
    if part not in ('fit','calibration') or clip not in split[part]:raise Blocked('Incorrect source bank population')
    prediction=WORK/('source_inference-overfit' if pilot else 'predictions/C00')/source/str(seed)/clip
    folder=WORK/('pilot_banks' if pilot else 'banks')/source/str(seed)/part/clip
    folder.mkdir(parents=True,exist_ok=True)
    import tempfile
    (folder/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((folder/'tmp').resolve())
    from .guard import install
    sys.dont_write_bytecode=True
    image=DATA/'train'/f'{clip}.zarr';label=DATA/'train'/f'{clip}.geff'
    guard=install(inputs=[prediction,image,label],outputs=[folder],code_roots=[REPO/'tools',REPO/'handover',
        REPO/'work/annotation-selection-v1/official',Path(sys.prefix),Path(sys.base_prefix),
        *[Path(p) for p in sys.path if 'site-packages' in p]])
    result=prepare_clip(prediction,image,label,folder,diagnostic=pilot)
    write(folder/'worker.json',dict(guard=guard,source=source,seed=seed,clip=clip,partition=part,
        receipt_sha256=sha(folder/'receipt.json'),finished_utc=now()))
    return dict(**result,guard=guard)
