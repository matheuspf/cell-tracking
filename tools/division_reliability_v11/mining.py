"""One immutable midpoint mining pass, resumable by complete source-fit clip."""
from pathlib import Path
import sys,tempfile,shutil,heapq
from .common import REPO,DATA,WORK,RESULTS,Blocked,read,write,sha,now


def prepare(source,seed):
    from .readiness import require_production
    require_production('mining-prepare')
    compact=WORK/'fits'/source/str(seed)/'compact';folder=compact/'mining';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'signature.json').exists():
        sig=read(folder/'signature.json')
        if sig['checkpoint_sha256']!=sha(folder/'frozen_checkpoint.pt'):raise Blocked('Midpoint snapshot changed')
        if sig['mining_code_sha256']!=sha(Path(__file__)):raise Blocked('Midpoint mining implementation changed')
        return sig
    clips=read(WORK/'source_partitions.json')[source]['fit'];bank=WORK/'banks'/source/str(seed)/'fit'
    lock=read(RESULTS/'execution_lock.json')
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[compact/'resume.pt',bank,WORK/'fits'/source/str(seed)/'linear/normalizer.json'],outputs=[folder],
        code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import torch
    state=torch.load(compact/'resume.pt',map_location='cpu',weights_only=True)
    if state['source']!=source or state['seed']!=seed or state['step']!=lock['event_updates']//2 or state['mined']:
        raise Blocked('Mining requires the unmined registered midpoint state')
    if state['lock_identity']!=lock['identity']:raise Blocked('Wrong midpoint lock')
    shutil.copy2(compact/'resume.pt',folder/'frozen_checkpoint.pt')
    signature=dict(source=source,seed=seed,step=state['step'],guard=guard,checkpoint_sha256=sha(folder/'frozen_checkpoint.pt'),
        source_bank_receipts={n:sha(bank/n/'receipt.json') for n in clips},
        policy_code_sha256=sha(Path(__file__).with_name('policy.py')),mining_code_sha256=sha(Path(__file__)),
        normalizer_sha256=sha(WORK/'fits'/source/str(seed)/'linear/normalizer.json'))
    write(folder/'signature.json',signature,immutable=True);return signature


def run(source,seed,clip):
    from .readiness import require_production
    require_production('mine')
    if clip not in read(WORK/'source_partitions.json')[source]['fit']:raise Blocked('Mining excludes calibration and target')
    fit=WORK/'fits'/source/str(seed);folder=fit/'compact/mining';sig=read(folder/'signature.json')
    output=folder/clip;output.mkdir(exist_ok=True)
    if sig['mining_code_sha256']!=sha(Path(__file__)):raise Blocked('Midpoint mining implementation changed')
    if (output/'receipt.json').exists():
        old=read(output/'receipt.json')
        if old['signature_sha256']!=sha(folder/'signature.json'):raise Blocked('Mining clip belongs to another snapshot')
        return old
    (output/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((output/'tmp').resolve())
    prediction=WORK/'predictions/C00'/source/str(seed)/clip;bankfolder=WORK/'banks'/source/str(seed)/'fit'/clip
    from .resources import Resources,ROOT
    resources=Resources('event_mining',source,seed)
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[folder/'frozen_checkpoint.pt',folder/'signature.json',fit/'linear/normalizer.json',prediction,bankfolder,
        DATA/'train'/f'{clip}.zarr'],outputs=[output,ROOT],code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np,torch
    from .actions import Bank,utilities
    from .models import CompactPolicy
    from .features import NAMES,IDENTITY_NAMES
    from .policy import score_bank
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    if sha(folder/'frozen_checkpoint.pt')!=sig['checkpoint_sha256'] or sha(bankfolder/'receipt.json')!=sig['source_bank_receipts'][clip]:
        raise Blocked('Frozen mining parents changed')
    receipt=read(bankfolder/'receipt.json')
    if sha(prediction/'graph.npz')!=receipt['graph_sha256'] or sha(bankfolder/'training.npz')!=receipt['training_sha256']:
        raise Blocked('Mining observation/label bank changed')
    if sha(fit/'linear/normalizer.json')!=sig['normalizer_sha256'] or sha(Path(__file__).with_name('policy.py'))!=sig['policy_code_sha256']:
        raise Blocked('Mining normalization/scoring code changed')
    with np.load(bankfolder/'training.npz') as f:
        offsets=f['offset'];risk=f['risks']
        negative={receipt['group_parents'][i]:i for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])) if b>a and (risk[a:b]==0).all()}
    known=set(receipt['group_parents'])
    with np.load(prediction/'graph.npz') as f:nodes,edges,scores,confidence=(f[k] for k in ('nodes','edges','edge_scores','confidence'))
    bank=Bank(nodes,edges,scores,100)
    model=CompactPolicy(2*len(NAMES)+1,len(NAMES),len(IDENTITY_NAMES))
    model.load_state_dict(torch.load(folder/'frozen_checkpoint.pt',map_location='cpu',weights_only=True)['model']);model.eval()
    normalizer=read(fit/'linear/normalizer.json')['values'];hard=[];unknown=[]
    for g,a,b in score_bank(bank,confidence,DATA/'train'/f'{clip}.zarr',normalizer,resources,model=model,progress=output/'progress.json'):
        score=float(utilities(bank,g,a,b,0).max());p=g['parent']
        if p in negative:
            item=(score,-p,negative[p])
            if len(hard)<256:heapq.heappush(hard,item)
            elif item>hard[0]:heapq.heapreplace(hard,item)
        elif p not in known:
            item=(score,-p)
            if len(unknown)<20:heapq.heappush(unknown,item)
            elif item>unknown[0]:heapq.heapreplace(unknown,item)
    result=dict(selected_parent_groups=len(hard),supported_negative_pool=len(negative),
        selected_groups=[g for _,_,g in sorted(hard,reverse=True)],
        unknown_top_groups=[dict(raw_gain=s,parent=-p) for s,p in sorted(unknown,reverse=True)],unknown_used_as_negative=False,
        signature_sha256=sha(folder/'signature.json'),guard=guard,source=source,seed=seed,clip=clip,finished_utc=now())
    write(output/'receipt.json',result,immutable=True);resources.close();return result


def finish(source,seed):
    folder=WORK/'fits'/source/str(seed)/'compact/mining';sig=read(folder/'signature.json')
    if (folder/'receipt.json').exists():return read(folder/'receipt.json')
    pool={};summary={}
    for clip in sig['source_bank_receipts']:
        r=read(folder/clip/'receipt.json')
        if r['signature_sha256']!=sha(folder/'signature.json'):raise Blocked('Partial mining clip belongs to another snapshot')
        summary[clip]=r
        if r['selected_groups']:pool[clip]=r['selected_groups']
    result=dict(pool=pool,summary=summary,criterion='maximum raw complete-fork gain per parent',per_clip_parent_cap=256,
        passes=1,source_fit_only=True,signature=sig,finished_utc=now())
    write(folder/'receipt.json',result,immutable=True);return result
