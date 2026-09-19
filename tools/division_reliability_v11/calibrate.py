"""Source-calibration census, occurrence fit, and fresh graph safety selection."""
from pathlib import Path
import sys,tempfile,time
from .common import DATA,REPO,WORK,RESULTS,Blocked,read,write,sha,now


def run(source,seed,arm):
    from .readiness import require_production
    require_production('calibrate')
    if arm not in ('C01','C11'):raise Blocked('Fork policy calibration requires C01 or C11')
    clips=read(WORK/'source_partitions.json')[source]['calibration'];fit=WORK/'fits'/source/str(seed)
    folder=fit/'calibration'/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'calibration.json').exists():return read(folder/'calibration.json')
    (folder/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((folder/'tmp').resolve())
    from .resources import Resources,ROOT
    resources=Resources('inference',source,seed)
    from .guard import install
    sys.dont_write_bytecode=True
    bank_root=WORK/'banks'/source/str(seed)/'calibration'
    guard=install(inputs=[fit/'linear',fit/'compact',bank_root,
        *[WORK/'decoded_inventory'/f'{n}.json' for n in clips],
        *[WORK/'predictions/C00'/source/str(seed)/n for n in clips],
        *[DATA/'train'/f'{n}{suffix}' for n in clips for suffix in ('.zarr','.geff')]],outputs=[folder,ROOT],
        code_roots=[REPO/'tools',REPO/'handover',REPO/'work/annotation-selection-v1/official',Path(sys.prefix),Path(sys.base_prefix),
                    *[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    import torch
    from scipy.special import logsumexp
    from itertools import tee
    from .features import NAMES,IDENTITY_NAMES
    from .models import CompactPolicy
    from .policy import score_bank
    from .actions import Bank,utilities,apply_decisions
    from .calibration import fit_occurrence,select_margin,reliability
    from .evaluation import score,finite
    from .graphs import export_csv,read_csv
    from .provenance import digest
    from .calibration_cache import signature,compute,cached_utility
    from annotation_selection.metric_adapter import aggregate
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    model=None;linear=None
    if arm=='C11':
        model=CompactPolicy(2*len(NAMES)+1,len(NAMES),len(IDENTITY_NAMES))
        model.load_state_dict(torch.load(fit/'compact/final.pt',map_location='cpu',weights_only=True)['model']);model.eval()
        head_hash=sha(fit/'compact/final.pt')
    else:linear=read(fit/'linear/model.json')['parameters'];head_hash=sha(fit/'linear/model.json')
    normalizer=read(fit/'linear/normalizer.json')['values'];xs=[];ys=[];positive_events=set();parents={};base_rows=[];base_events={}
    from collections import Counter
    census=Counter();ranks=Counter()
    for clip in clips:
        prediction=WORK/'predictions/C00'/source/str(seed)/clip;dest=folder/clip;dest.mkdir(exist_ok=True)
        with np.load(prediction/'graph.npz') as f:nodes,edges,scores,confidence=(f[k] for k in ('nodes','edges','edge_scores','confidence'))
        base,events=score(clip,nodes,edges,DATA/'train'/f'{clip}.geff');base_rows.append(base);base_events[clip]=events
        bank=Bank(nodes,edges,scores,100)
        bank_receipt=read(bank_root/clip/'receipt.json')
        census.update(bank_receipt['census'])
        if bank_receipt['graph_sha256']!=sha(prediction/'graph.npz') or bank_receipt['training_sha256']!=sha(bank_root/clip/'training.npz'):
            raise Blocked('Calibration label bank differs from its frozen C00 observations')
        with np.load(bank_root/clip/'training.npz') as f:
            offsets=f['offset'];risks=f['risks']
            known={p:1 if (risks[a:b]==1).any() else 0 for p,a,b in zip(bank_receipt['group_parents'],offsets[:-1],offsets[1:])}
            labels_by_parent={p:risks[a:b] for p,a,b in zip(bank_receipt['group_parents'],offsets[:-1],offsets[1:])}
        positive_events.update(bank_receipt['positive_events'])
        compute(bank,confidence,DATA/'train'/f'{clip}.zarr',normalizer,resources,model,linear,dest,arm,signature(prediction,fit,arm))
        with np.load(dest/'logits.npz') as f:
            conditional=f['conditional']
            for p,a,start,end in zip(f['parents'],f['occurrence'],f['offset'][:-1],f['offset'][1:]):
                if p in known:xs.append(float(a));ys.append(known[p])
                if known.get(p)==1:
                    y=labels_by_parent[p];b=conditional[start:end]
                    if len(b)!=len(y):raise Blocked('Calibration conditional diagnostic ordering differs')
                    best=b[y==1].max();ranks['positive_groups']+=1
                    ranks['top1_all_legal']+=int(best>=b.max()-1e-12)
                    ranks['top1_supported_only']+=int(best>=b[y>=0].max()-1e-12)
                    ranks['legal_actions_in_positive_groups']+=len(y);ranks['unknown_actions_in_positive_groups']+=int((y<0).sum())
        parents[clip]=dict(raw_images=sha(WORK/'decoded_inventory'/f'{clip}.json'),
            raw_labels=digest({str(p.relative_to(DATA)):sha(p) for p in (DATA/'train'/f'{clip}.geff').rglob('*') if p.is_file()}))
    calibrated=fit_occurrence(xs,ys,len(positive_events));baseline=aggregate(base_rows,clips)
    trials=[]
    for margin in calibrated['margins']:
        rows=[];introduced=recovered=lost=accepted=0
        for clip in clips:
            prediction=WORK/'predictions/C00'/source/str(seed)/clip;dest=folder/clip
            with np.load(prediction/'graph.npz') as f:nodes,edges,scores=(f[k] for k in ('nodes','edges','edge_scores'))
            bank=Bank(nodes,edges,scores,100)
            with np.load(dest/'logits.npz') as f:cache={k:f[k] for k in f.files}
            def stream():
                for p,a,start,end,competitor in zip(cache['parents'],cache['occurrence'],cache['offset'][:-1],cache['offset'][1:],cache['competitor']):
                    b=cache['conditional'][start:end]
                    gains=cached_utility(float(a),b,cache['structural'][start:end],competitor,calibrated['temperature'],calibrated['intercept'],margin)
                    selected=np.flatnonzero(gains>0)
                    # The cached denominator and structural terms cover every
                    # unique legal action. Reconstruct edits only when the exact
                    # calibrated complete utility is positive.
                    if len(selected):
                        g=bank.parent(int(p))
                        check=utilities(bank,g,float(a)/calibrated['temperature']+calibrated['intercept'],b,margin)
                        if not np.array_equal(gains,check):raise Blocked('Cached complete utility parity failed')
                        for i in selected:yield g['forks'][int(i)],float(gains[i])
            aa,bb=tee(stream());edited,trace=apply_decisions(nodes,edges,(d for d,v in aa),(v for d,v in bb))
            csv=dest/f'margin-{margin}.csv';export_csv(csv,clip,nodes,edited);nn,ee=read_csv(csv,clip)
            row,detail=score(clip,nn,ee,DATA/'train'/f'{clip}.geff');rows.append(row)
            before=base_events[clip]
            introduced+=len(set(detail['fp_predicted_forks'])-set(before['fp_predicted_forks']))
            recovered+=len(set(detail['recovered_gt_events'])-set(before['recovered_gt_events']))
            lost+=len(set(before['recovered_gt_events'])-set(detail['recovered_gt_events']));accepted+=len(trace['edits'])
            write(dest/f'margin-{margin}.json',finite(dict(metrics=row,events=detail,solver=trace)))
        summary=aggregate(rows,clips)
        trials.append(dict(margin=margin,score=summary['score'],combined_delta=summary['score']-baseline['score'],
            adjusted_edge_delta=summary['adj_edge_jaccard']-baseline['adj_edge_jaccard'],introduced_fp=introduced,
            newly_recovered_tp=recovered,lost_tp=lost,accepted_actions=accepted,summary=finite(summary)))
    selection=select_margin(trials)
    result=dict(status='calibrated',arm=arm,source=source,seed=seed,guard=guard,**calibrated,**selection,
        baseline=finite(baseline),trials=trials,known_parent_groups=len(xs),negative_groups=sum(y==0 for y in ys),
        distinct_positive_events=len(positive_events),parent_manifests=parents,head_sha256=head_hash,finished_utc=now(),
        raw_occurrence_reliability=reliability(xs,ys),
        calibrated_occurrence_reliability=reliability(np.asarray(xs)/calibrated['temperature']+calibrated['intercept'],ys),
        full_parent_census=dict(census),conditional_rank=dict(ranks),
        implementation_sha256={n:sha(Path(__file__).with_name(n)) for n in ('calibrate.py','calibration_cache.py','calibration.py','policy.py','actions.py','features.py','models.py')})
    write(folder/'calibration.json',result,immutable=True);resources.close();return result
