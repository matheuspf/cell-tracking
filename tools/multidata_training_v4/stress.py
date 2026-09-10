"""Frozen held-out observation stress; diagnostics never select target thresholds."""
from .common import *
from .adapters import synthetic_record, temporal_patches
from .index import supervision
from .proposals import build

CASES=['clean','localization','dropout','duplicates','clipped_daughter','short_history']

def fixture(raw,lab,case,seed):
    image,points,spacing=synthetic_record(raw,lab,'sequence')
    t=lab['t'];mapping=np.arange(len(t));rng=np.random.default_rng(seed)
    extent=np.maximum(np.quantile(points,.75,axis=0)-np.quantile(points,.25,axis=0),1e-5)
    if case=='localization':points=points+rng.normal(0,.004,points.shape)*extent
    elif case in ['dropout','clipped_daughter','short_history']:
        keep=np.ones(len(t),bool)
        if case=='dropout':keep=rng.random(len(t))>.15
        elif case=='short_history':keep=t>=2
        else:
            parents,counts=np.unique(lab['edges'][:,0],return_counts=True)
            for parent in parents[counts==2][::2]:
                child=lab['edges'][lab['edges'][:,0]==parent,1][0];keep[child]=False
        t,points,mapping=t[keep],points[keep],mapping[keep]
    elif case=='duplicates':
        duplicate=np.flatnonzero(rng.random(len(t))<.10)
        points=np.vstack([points,points[duplicate]+rng.normal(0,.002,(len(duplicate),3))*extent])
        t=np.r_[t,t[duplicate]];mapping=np.r_[mapping,np.full(len(duplicate),-1)]
    possible=np.flatnonzero((mapping>=0)&(t<t.max()))
    # Uniform parent observations: hard negatives are not balanced away here.
    anchors=np.sort(rng.choice(possible,min(256,len(possible)),replace=False))
    a,c,x=build(t,points,anchors)
    y,links,w=supervision(a,c,mapping,lab['edges'],lab['t'],dense=True)
    ids=np.c_[a,c];used=np.unique(ids[ids>=0]);p,v=temporal_patches(image,t[used],points[used],spacing)
    reverse=np.full(len(t),-1,int);reverse[used]=np.arange(len(used));lookup=reverse[np.maximum(ids,0)]
    pp=p[lookup];vv=v[lookup];vv[ids<0]=0
    return dict(x=x,target=y,patch=pp,valid=vv),dict(observations=len(t),anchors=len(a),
        positives=int((y>0).sum()),continuations=int((y==0).sum()),censored=int((y<0).sum()),
        duplicate_observations=int((mapping<0).sum()),short_history_parents=int((x[:,0,12]==0).sum()))

def prepare():
    records=[json.loads(s) for s in (OUT/'runtime_dataset_index.jsonl').read_text().splitlines()]
    selected=sorted([r for r in records if r['source']=='synthetic_sequence' and r['partition']=='test'],key=lambda r:r['sample'])[::6][:16]
    names=[r['sample'] for r in selected]
    lock=dict(created=now(),samples=names,seed=271828,cases=CASES,
        procedure='Six fixed observation corruptions; uniform held-out parents; no training or threshold selection',
        temporal_grid='original pooled sequence, never repooled',target_results_used=False)
    lock_path=OUT/'stress_lock.json'
    if lock_path.exists():assert read(lock_path)['samples']==names
    else:write(lock_path,lock)
    manifest={r['sample_id']:r for r in read(PREPARED/'synthetic_manifest.json')};stats=[]
    for i,name in enumerate(names):
        r=manifest[name];raw=arrays(PREPARED.parent/r['source']);lab=arrays(PREPARED.parent/r['labels'])
        for case in CASES:
            path=OUT/'stress_fixtures'/case/f'{name}.npz';meta=path.with_suffix('.json')
            if meta.exists():stats.append(read(meta));continue
            data,counts=fixture(raw,lab,case,271828+i)
            assert np.isfinite(data['x']).all() and data['valid'].sum(2).max()<=3
            save(path,**data);entry=dict(sample=name,case=case,partition='generator_test',source_sha256=sha(PREPARED.parent/r['source']),
                fixture_sha256=sha(path),**counts);write(meta,entry);stats.append(entry)
    import pandas as pd
    pd.DataFrame(stats).to_csv(OUT/'stress_fixture_summary.csv',index=False)
    write(OUT/'stress_fixture_receipt.json',dict(completed=True,source_examples=len(names),cases=CASES,fixtures=len(stats),
        lock_sha256=sha(lock_path),no_training_partition_access=True))

def measure():
    import torch
    import pandas as pd
    from .calibrate import load_model,scores,diagnostics
    torch.set_num_threads(2);rows=[]
    names=['G_synthetic','G_fish','G_combined','G_multispecies','G_real_44b6','G_real_6bba',
        'I_synthetic','I_real_44b6','I_real_6bba','I_randomized_44b6','I_randomized_6bba']
    for case in CASES:
        parts=[arrays(p) for p in sorted((OUT/'stress_fixtures'/case).glob('*.npz'))]
        data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
        for name in names:
            model=load_model(name);d={k:v for k,v in data.items() if name.startswith('I_') or k in ['x','target']}
            diag=diagnostics(scores(model,d),d['target'],1.)
            rows.append(dict(case=case,model=name,temperature=1.,checkpoint_sha256=sha(OUT/'models'/f'{name}.pt'),**{k:v for k,v in diag.items() if k!='bins'}))
            del model
        print('held-out stress',case,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'stress_diagnostics.csv',index=False)
    write(OUT/'stress_receipt.json',dict(completed=True,models=names,cases=CASES,comparisons=len(rows),
        training_updates=0,threshold_changes=0,selection_input=False,scope='generator engineering diagnostic, no biological independence claim'))

def run():
    prepare();measure()
