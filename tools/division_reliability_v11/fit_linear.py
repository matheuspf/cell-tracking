"""C01 on the complete supported source-fit census, with event-unit weights."""
from pathlib import Path
import sys
from .common import REPO,WORK,Blocked,read,write,sha,now


def run(source,seed):
    from .readiness import require_production
    require_production('fit-linear')
    clips=read(WORK/'source_partitions.json')[source]['fit'];bank=WORK/'banks'/source/str(seed)/'fit'
    folder=WORK/'fits'/source/str(seed)/'linear';folder.mkdir(parents=True,exist_ok=True)
    parents={n:sha(bank/n/'receipt.json') for n in clips}
    if (folder/'model.json').exists():
        result=read(folder/'model.json')
        if result['source_bank_receipts']!=parents or result['normalizer_sha256']!=sha(folder/'normalizer.json'):
            raise Blocked('Existing linear fit has changed parents')
        if result['status']!='fitted':raise Blocked('Existing linear fit did not converge')
        return result
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[bank],outputs=[folder],code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),
                 *[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    from .event_training import fit_normalizer
    from .features import normalize,NAMES,IDENTITY_NAMES
    from .linear import fit
    normalization=fit_normalizer(bank,clips);write(folder/'normalizer.json',normalization,immutable=True)
    event_anchors={}
    for clip in clips:
        for event,anchors in read(bank/clip/'receipt.json')['positive_events'].items():
            event_anchors.setdefault(event,[]).extend((clip,g) for g in anchors)
    event_weights={}
    for anchors in event_anchors.values():
        for key in anchors:event_weights[key]=event_weights.get(key,0)+1/len(anchors)
    px=[];ax=[];ys=[];weights=[];census=dict(positive_events=len(event_anchors),positive_groups=0,negative_groups=0)
    for clip in clips:
        receipt=read(bank/clip/'receipt.json')
        if sha(bank/clip/'training.npz')!=receipt['training_sha256']:raise Blocked('Source bank hash differs')
        with np.load(bank/clip/'training.npz') as d:
            parent=normalize(d['parent_x'],*normalization['values']['parent'])
            action=normalize(d['action_x'],*normalization['values']['action'])
            for i,(a,b) in enumerate(zip(d['offset'][:-1],d['offset'][1:])):
                y=d['risks'][a:b];px.append(parent[i]);ax.append(action[a:b]);ys.append(y);weights.append(event_weights.get((clip,i),1.))
                census['positive_groups' if (y==1).any() else 'negative_groups']+=1
    if not census['positive_groups']:raise Blocked('No positive source action group; C01/C11 head training blocked')
    fitted=fit(np.stack(px),ax,ys,weights,[True]*len(ys))
    result=dict(status='fitted' if all(d['converged'] for d in fitted['diagnostics']) else 'nonconverged',
                source=source,seed=seed,guard=guard,parameters=fitted,source_bank_receipts=parents,census=census,
                objective='complete supported census; one unit per distinct positive event, one per negative group',
                normalizer_sha256=sha(folder/'normalizer.json'),action_schema=list(NAMES),identity_schema=list(IDENTITY_NAMES),finished_utc=now(),
                implementation_sha256={n:sha(Path(__file__).with_name(n)) for n in ('fit_linear.py','linear.py','features.py','event_training.py')})
    write(folder/'model.json',result,immutable=True)
    if result['status']!='fitted':raise Blocked('C01 CPU fit did not converge; see optimizer diagnostics')
    return result
