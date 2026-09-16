"""Model-only batch predictions. Training/evaluation modules are never imported here."""
import time,functools
from concurrent.futures import ProcessPoolExecutor
import torch
from .common import *
from . import native_adapter as native
from .banks import cached as bank_cached
from .score_models import encode_at_nodes,native_scores,hoct_scores
from .hoct_adapter import load as hoct_load
from .calibrate import apply
from .temporal_decode import decode

VARIANTS={
 'J_native_frozen':dict(family='N0',seed=20260910,pop='P0'),
 'H_general_J':dict(family='H0',seed=20260910,pop='P0'),
 'H_probe_J':dict(family='H1',seed=20260910,pop='P0'),
 'H_probe_native_J':dict(family='H2',seed=20260910,pop='P0'),
 'N_head_J':dict(family='N1',seed=20260910,pop='P0'),
 'N_backbone_J':dict(family='N2',seed=20260910,pop='P0'),
 'P_union_native_J':dict(family='N0',seed=20260910,pop='P1'),
 'P_union_N_J':dict(family='N2',seed=20260910,pop='P1'),
 'P_DC_native_J':dict(family='N0',seed=20260910,pop='PDC'),
 'P_DC_N_J':dict(family='N2',seed=20260910,pop='PDC'),
 'P_image_ablation':dict(family='N0',seed=20260910,pop='Pnoimage'),
 'N_final_seed2':dict(family='N2',seed=314159,pop='P0'),
 'H_final_seed2':dict(family='H2',seed=314159,pop='P0'),
 'P_final_seed2':dict(family='N2',seed=314159,pop='P1'),
 'P_DC_final_seed2':dict(family='N2',seed=314159,pop='PDC'),
}

def wait_for(paths):
    while any(not p.exists() for p in paths):time.sleep(15)

def score_native(stage,seed=20260910):
    torch.set_num_threads(2)
    wait_for([OUT/'models/native'/f'{s}_{stage}_{seed}.json' for s in ['44b6','6bba']])
    models={s:native.load(OUT/'models/native'/f'{s}_{stage}_{seed}.pt') for s in ['44b6','6bba']};frozen=native.load()
    for row in inventory():
        name=row['dataset'];source='6bba' if row['embryo']=='44b6' else '44b6';dest=OUT/'model_scores'/f'{stage}_{seed}'/f'{name}.npz'
        if dest.exists():continue
        wait_for([OUT/'observations'/f'{name}.npz']);c=arrays(OUT/'observations'/f'{name}.npz');nodes=c['nodes'];m=c['oldmask'];model=models[source]
        bank0=bank_cached(frozen,name,source);bank1=bank_cached(frozen,name,source,True)
        start=time.monotonic()
        if stage=='N2':fs,ft=encode_at_nodes(model,nodes,DATA/'train'/f'{name}.zarr')
        else:fs,ft=c['features_source'].astype(np.float32),c['features_target'].astype(np.float32)
        p0=native_scores(model,nodes[m],bank0['pairs'],fs[m],ft[m]);p1=native_scores(model,nodes,bank1['pairs'],fs,ft) if stage=='N2' else np.empty(0)
        save(dest,P0=p0,P1=p1)
        write(dest.with_suffix('.json'),dict(dataset=name,source=source,stage=stage,seed=seed,seconds=time.monotonic()-start,
            checkpoint_sha256=sha(OUT/'models/native'/f'{source}_{stage}_{seed}.pt'),sha256=sha(dest),
            P0_bank_sha256=sha(OUT/'banks'/source/'P0'/f'{name}.npz'),P1_bank_sha256=sha(OUT/'banks'/source/'P1'/f'{name}.npz'),
            native_image_network_executed=stage=='N2',features_source='actual source-adapted temporal 3D encoder' if stage=='N2' else 'unchanged native encoder samples'))
        print('native prediction',stage,seed,name,round(time.monotonic()-start,1),flush=True)

def score_hoct():
    torch.set_num_threads(2)
    wait_for([OUT/'models/hoct'/f'{s}_probe_{seed}.json' for s in ['44b6','6bba'] for seed in [20260910,314159]])
    general=hoct_load();frozen=native.load()
    heads={s:[(f'H1_{seed}',torch.load(OUT/'models/hoct'/f'{s}_probe_{seed}.pt',weights_only=False,map_location='cpu')) for seed in [20260910,314159]] for s in ['44b6','6bba']}
    for row in inventory():
        name=row['dataset'];source='6bba' if row['embryo']=='44b6' else '44b6';dest=OUT/'model_scores/H'/f'{name}.npz'
        if dest.exists():continue
        c=arrays(OUT/'observations'/f'{name}.npz');m=c['oldmask'];nodes=c['nodes'][m];common=bank_cached(frozen,name,source)
        start=time.monotonic();pred,r=hoct_scores(general,nodes,common['pairs'],c['properties'][m],c['valid_region'][m],heads[source])
        save(dest,**pred)
        write(dest.with_suffix('.json'),dict(dataset=name,source=source,seconds=time.monotonic()-start,general=r,sha256=sha(dest),
            bank_sha256=sha(OUT/'banks'/source/'P0'/f'{name}.npz'),checkpoints={p.name:sha(p) for p in (OUT/'models/hoct').glob('*.pt')}))
        print('HOCT prediction',name,round(time.monotonic()-start,1),flush=True)

def prerequisites(v):
    cfg=VARIANTS[v];f=cfg['family'];seed=cfg['seed'];p=[]
    for source in ['44b6','6bba']:
        p.append(OUT/'calibration'/f'{source}_J.json')
        if f in ['N1','N2']:p.append(OUT/'calibration'/f'{source}_{f}_{seed}.json')
        elif f=='Hctc':p.append(OUT/'calibration'/f'{source}_Hctc.json')
        elif f.startswith('H'):p.append(OUT/'calibration'/f'{source}_H_{seed}.json')
    return p

def transform(c,base,common,model_scores,config,joint,calibration,dc=None):
    expanded=config['pop']!='P0';f=config['family'];seed=config['seed'];m=np.ones(len(c['nodes']),bool) if expanded else c['oldmask']
    native_log=common['native_logits']
    if f=='N0':scores=joint['native_scale']*native_log+joint['native_offset']
    elif f in ['N1','N2']:scores=apply(calibration,model_scores['P1' if expanded else 'P0'])
    elif f=='Hctc':scores=apply(calibration,model_scores['Hctc'])
    elif f=='H0':scores=apply(calibration,model_scores['H0'])
    elif f=='H1':scores=apply(calibration,model_scores[f'H1_{seed}'])
    elif f=='H2':scores=apply(calibration,model_scores[f'H1_{seed}'],native_log)
    else:raise ValueError(f)
    confidence=c['confidence'][m].copy()
    if config['pop']=='PDC':assert dc is not None;confidence=np.sqrt(np.clip(confidence*dc[m],1e-8,1-1e-6))
    if config['pop']=='Pnoimage':confidence[:]=.5
    return decode(c['nodes'][m],common['pairs'],scores,base['nodes'],base['edges'],confidence,
        c['split_owner'][m] if expanded else None,config=joint['config'])

def decode_one(task):
    variant,row=task;name=row['dataset'];source='6bba' if row['embryo']=='44b6' else '44b6';config=VARIANTS[variant];f=config['family'];seed=config['seed'];expanded=config['pop']!='P0'
    dest=OUT/'deltas'/variant/f'{name}.npz';receipt=OUT/'prediction_receipts'/variant/f'{name}.json'
    if receipt.exists():assert sha(dest)==read(receipt)['delta_sha256'];return
    c=arrays(OUT/'observations'/f'{name}.npz');base=graph(name);common=arrays(OUT/'banks'/source/('P1' if expanded else 'P0')/f'{name}.npz')
    model_scores={};calibration=None;joint=read(OUT/'calibration'/f'{source}_J.json')
    if f in ['N1','N2']:
        model_scores=arrays(OUT/'model_scores'/f'{f}_{seed}'/f'{name}.npz');calibration=read(OUT/'calibration'/f'{source}_{f}_{seed}.json')['calibration']
    elif f.startswith('H'):
        model_scores=arrays(OUT/'model_scores/H'/f'{name}.npz')
        calibration=read(OUT/'calibration'/f'{source}_Hctc.json')['calibration'] if f=='Hctc' else read(OUT/'calibration'/f'{source}_H_{seed}.json')['models'][f]
    dc=arrays(OUT/'deepcenter'/f'{name}.npz')['confidence'] if config['pop']=='PDC' else None
    start=time.monotonic();n,e,r=transform(c,base,common,model_scores,config,joint,calibration,dc)
    from strong_tracker_v3.common import validate,graph_hash
    validate(n,e,row['image_shape']);save_delta(name,variant,n,e)
    write(receipt,dict(dataset=name,source=source,variant=variant,config=config,created=now(),seconds=time.monotonic()-start,
        graph_hash=graph_hash(n,e),delta_sha256=sha(dest),decode=r,code_sha256=sha(Path(__file__))))

def run_decode(variants,workers=10):
    # Both directional source configurations are frozen before their complete batch is decoded/scored.
    wait_for([p for v in variants for p in prerequisites(v)])
    paths={str(p.relative_to(OUT)):sha(p) for v in variants for p in prerequisites(v)}
    write(OUT/'configuration_locks'/f'{digest(variants)[:16]}.json',dict(variants={v:VARIANTS[v] for v in variants},created=now(),
        both_directions_frozen=True,calibrations=paths,code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    tasks=[]
    for v in variants:
        cfg=VARIANTS[v];f=cfg['family']
        for r in inventory():
            name=r['dataset'];source='6bba' if r['embryo']=='44b6' else '44b6';required=[OUT/'banks'/source/('P0' if cfg['pop']=='P0' else 'P1')/f'{name}.npz']
            if f in ['N1','N2']:required.append(OUT/'model_scores'/f"{f}_{cfg['seed']}"/f'{name}.npz')
            elif f.startswith('H'):required.append(OUT/'model_scores/H'/f'{name}.npz')
            if cfg['pop']=='PDC':required.append(OUT/'deepcenter'/f'{name}.npz')
            wait_for(required);tasks.append((v,r))
    with cpu_batch(),ProcessPoolExecutor(max_workers=min(workers,10)) as pool:
        for i,_ in enumerate(pool.map(decode_one,tasks,chunksize=1)):
            if i%20==0:print('joint decode',i+1,len(tasks),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['native','hoct','decode']);p.add_argument('--stage',choices=['N1','N2']);p.add_argument('--seed',type=int,default=20260910);p.add_argument('--variants',nargs='+')
    a=p.parse_args()
    if a.action=='native':score_native(a.stage,a.seed)
    elif a.action=='hoct':score_hoct()
    else:run_decode(a.variants)
