"""Complete source-calibration raw scores in a worker without label access."""
from pathlib import Path
import sys,tempfile
from .common import REPO,DATA,WORK,Blocked,read,write,sha,now


def cached_utility(occurrence,conditional,structural,competitor,temperature,intercept,margin):
    from scipy.special import logsumexp
    return occurrence/temperature+intercept+conditional-logsumexp(conditional)+structural-competitor-margin


def signature(prediction,fit,arm):
    head=fit/('compact/final.pt' if arm=='C11' else 'linear/model.json')
    return dict(graph=sha(prediction/'graph.npz'),head=sha(head),normalizer=sha(fit/'linear/normalizer.json'),
        policy=sha(Path(__file__).with_name('policy.py')),bank=sha(Path(__file__).with_name('actions.py')),
        calibration=sha(Path(__file__).with_name('calibrate.py')),cache=sha(Path(__file__)))


def compute(bank,confidence,image,normalizer,resources,model,linear,dest,arm,parents):
    import numpy as np
    from .policy import score_bank
    if (dest/'logits.json').exists():
        old=read(dest/'logits.json')
        if old['signature']!=parents or old['sha256']!=sha(dest/'logits.npz'):raise Blocked('Calibration raw score cache parents changed')
        return old
    pp=[];occ=[];conditional=[];structural=[];competitors=[];offset=[0]
    for g,a,b in score_bank(bank,confidence,image,normalizer,resources,model=model,linear=linear,progress=dest/'progress.json'):
        if not np.isfinite(a) or not np.isfinite(b).all():raise Blocked('Nonfinite raw calibration logits')
        pp.append(g['parent']);occ.append(a);conditional.append(np.asarray(b).copy());offset.append(offset[-1]+len(b))
        competitors.append(max([0.,*[bank.structural(d) for d in g['nofork']]]))
        structural.extend(bank.structural(d) for d in g['forks'])
    np.savez_compressed(dest/'logits.npz',parents=np.asarray(pp,np.int64),occurrence=np.asarray(occ,np.float64),
        conditional=np.concatenate(conditional) if conditional else np.empty(0,np.float32 if arm=='C11' else np.float64),
        offset=np.asarray(offset,np.int64),structural=np.asarray(structural,np.float64),competitor=np.asarray(competitors,np.float64))
    result=dict(signature=parents,sha256=sha(dest/'logits.npz'),complete_legal_denominators=True,finished_utc=now())
    write(dest/'logits.json',result,immutable=True);return result


def run(source,seed,arm,clip):
    from .readiness import require_production
    require_production('calibration-predict')
    if arm not in ('C01','C11') or clip not in read(WORK/'source_partitions.json')[source]['calibration']:
        raise Blocked('Invalid source calibration population')
    fit=WORK/'fits'/source/str(seed);prediction=WORK/'predictions/C00'/source/str(seed)/clip
    dest=fit/'calibration'/arm/clip;dest.mkdir(parents=True,exist_ok=True)
    (dest/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((dest/'tmp').resolve())
    parents=signature(prediction,fit,arm)
    from .resources import Resources,ROOT
    resources=Resources('inference',source,seed)
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[fit/'linear',fit/'compact',prediction,DATA/'train'/f'{clip}.zarr'],outputs=[dest,ROOT],
        code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np,torch
    from .actions import Bank
    from .models import CompactPolicy
    from .features import NAMES,IDENTITY_NAMES
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    with np.load(prediction/'graph.npz') as f:nodes,edges,scores,confidence=(f[k] for k in ('nodes','edges','edge_scores','confidence'))
    bank=Bank(nodes,edges,scores,100);model=linear=None
    if arm=='C11':
        model=CompactPolicy(2*len(NAMES)+1,len(NAMES),len(IDENTITY_NAMES))
        model.load_state_dict(torch.load(fit/'compact/final.pt',map_location='cpu',weights_only=True)['model']);model.eval()
    else:linear=read(fit/'linear/model.json')['parameters']
    result=compute(bank,confidence,DATA/'train'/f'{clip}.zarr',read(fit/'linear/normalizer.json')['values'],resources,model,linear,dest,arm,parents)
    resources.close()
    receipt=dict(**result,guard=guard,status='cached')
    write(dest/'worker.json',receipt)
    return receipt
