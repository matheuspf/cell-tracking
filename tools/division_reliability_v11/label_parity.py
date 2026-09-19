"""Real source comparison against every per-action official risk call."""
from pathlib import Path
import sys,tempfile,time
from .common import DATA,REPO,WORK,Blocked,read,write,now


def run(source,seed,clip):
    if clip not in read(WORK/'source_partitions.json')[source]['fit']:raise Blocked('Source fit only')
    prediction=WORK/'source_inference-overfit'/source/str(seed)/clip;folder=WORK/'label_parity'/source/str(seed)
    folder.mkdir(parents=True,exist_ok=True);(folder/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((folder/'tmp').resolve())
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[prediction,DATA/'train'/f'{clip}.geff'],outputs=[folder],code_roots=[REPO/'tools',REPO/'handover',
        REPO/'work/annotation-selection-v1/official',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    from .actions import Bank
    from .risk_labels import Risks
    from .data import labels
    from pipeline_error_training.labels import SourceLabels
    with np.load(prediction/'graph.npz') as f:n,e,s=(f[k] for k in ('nodes','edges','edge_scores'))
    gt,ge=labels(DATA/'train'/f'{clip}.geff');bank=Bank(n,e,s,100)
    lab=SourceLabels(n,e,gt,ge,[1.625,.40625,.40625]);adapter=Risks(bank,lab)
    pp=sorted(bank.expanded);chosen={pp[int(i)] for i in np.linspace(0,len(pp)-1,min(1024,len(pp)))}
    chosen|=set(lab.matches);chosen|=adapter.parent_roles
    # Include incoming-parent neighborhoods around every source-matched child.
    chosen|={p for p in pp if any(d in lab.matches for d in bank.near[p])}
    tic=time.monotonic();groups=actions=0
    for p in sorted(chosen&bank.expanded):
        g=bank.parent(p)
        if not g['complete']:continue
        fast,events=adapter.group(g);reference=[lab.decision(d) for d in g['forks']]
        if not np.array_equal(fast,[r['metric_fork_target'] for r in reference]) or events!=sorted({e for r in reference for e in r['compatible_events']}):
            raise Blocked('Optimized risk labels differ from official per-action helper')
        groups+=1;actions+=len(fast)
    result=dict(status='passed',source=source,seed=seed,clip=clip,guard=guard,groups=groups,actions=actions,
                selection='uniform prediction index grid plus all matched parents, children and local GT-window parent roles',
                shortcuts=adapter.stats,wall_seconds=time.monotonic()-tic,finished_utc=now())
    write(folder/'receipt.json',result);return result
