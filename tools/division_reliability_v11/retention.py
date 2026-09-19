"""Resolve every registered arm explicitly before any target metric access."""
from .common import WORK,RESULTS,Blocked,read,write,sha,now
from .provenance import digest,unseal
from .stage_provenance import validate_stages


def block(source,seed,arms,stage,reason,evidence):
    value=dict(source=source,seed=seed,arms=arms,stage=stage,status='blocked',reason=reason,
        evidence_path=str(evidence.relative_to(WORK)),evidence_sha256=sha(evidence),recorded_utc=now(),target_metrics_opened=False)
    path=WORK/'blocked_cells'/f'{source}-{seed}-{stage}.json'
    if path.exists():return read(path)
    write(path,value,immutable=True);return value


def resolve():
    lock=read(RESULTS/'execution_lock.json');matrix={}
    blockers=[read(p) for p in (WORK/'blocked_cells').glob('*.json')]
    for cell in lock['schedule']:
        source,seed=cell['source'],cell['seed']
        for arm in lock['arms']:
            key=f'{source}/{seed}/{arm}';package=WORK/'packages'/source/str(seed)/arm
            if (package/'manifest.json').exists():
                spec=unseal(package);validate_stages(spec['ancestry'],spec['root_artifact'],source)
                matrix[key]=dict(status='retained',package_identity=spec['identity'])
            else:
                causes=[b for b in blockers if b['source']==source and b['seed']==seed and arm in b['arms']]
                if not causes:raise Blocked('Unresolved registered cell; target stage remains locked: '+key)
                matrix[key]=dict(status='blocked',reasons=causes)
    value=dict(status='resolved',matrix=matrix,all_registered_arms_retained=all(v['status']=='retained' for v in matrix.values()),
        resolved_before_target_inference=True,resolved_utc=now(),execution_lock_identity=lock['identity'])
    value['identity']=digest(value)
    path=WORK/'retention.json'
    if path.exists():
        old=read(path)
        if old['matrix']!=matrix:raise Blocked('Retained matrix changed after target inference unlocked')
        return old
    write(path,value,immutable=True);return value
