"""Global retained-model and complete target-prediction freeze before labels."""
from pathlib import Path
from .common import REPO,WORK,RESULTS,Blocked,read,write,sha,now
from .provenance import digest,unseal,validate


def run():
    import sys
    if (WORK/'freeze/manifest.json').exists():
        verify();return read(WORK/'freeze/public_summary.json')
    split=read(WORK/'source_partitions.json');specs=[];models={}
    from .retention import resolve
    retention=resolve()
    for source in ('44b6','6bba'):
        target='6bba' if source=='44b6' else '44b6'
        clips=sorted(split[target]['fit']+split[target]['calibration'])
        for seed in (20260918,314159):
            for arm in ('C00','C01','C11'):
                if retention['matrix'][f'{source}/{seed}/{arm}']['status']!='retained':continue
                package=WORK/'packages'/source/str(seed)/arm
                from .stage_provenance import validate_stages
                spec=unseal(package);validate_stages(spec['ancestry'],spec['root_artifact'],source)
                models[f'{source}/{seed}/{arm}']=dict(package_identity=spec['identity'],files=spec['files'])
                for clip in clips:
                    path=WORK/'predictions'/arm/source/str(seed)/clip
                    receipt=read(path/'receipt.json')
                    if receipt['status']!='predicted_unscored' or receipt['package_identity']!=spec['identity']:
                        raise Blocked('Target prediction package mismatch')
                    if receipt['frames']!=100 or len(receipt['frame_hashes'])!=100:raise Blocked('Incomplete prediction frame population')
                    specs.append((source,target,seed,arm,clip,path,receipt))
    folder=WORK/'freeze';folder.mkdir(parents=True,exist_ok=True)
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[p for *_,p,_ in specs],outputs=[folder,RESULTS/'prediction_manifest.json'],
        code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    from .graphs import graph_hash,read_csv
    predictions={}
    for source,target,seed,arm,clip,path,receipt in specs:
        if sha(path/'graph.npz')!=receipt['graph_sha256'] or sha(path/'submission.csv')!=receipt['csv_sha256']:
            raise Blocked('Prediction changed before freeze')
        if arm!='C00' and sha(path/'policy_logits.npz')!=receipt['policy_logits_sha256']:
            raise Blocked('Raw policy denominator scores changed before freeze')
        with np.load(path/'graph.npz') as f:nodes,edges=f['nodes'],f['edges']
        nn,ee=read_csv(path/'submission.csv',clip)
        if not np.array_equal(nodes,nn) or not np.array_equal(edges,ee):raise Blocked('CSV differs from retained graph')
        times={int(n[0]):int(n[1]) for n in nodes};edge_times=np.array([times[int(a)] for a,b in edges])
        frames={str(t):dict(nodes=int((nodes[:,1]==t).sum()),edges=int((edge_times==t).sum()),
            sha256=graph_hash(nodes[nodes[:,1]==t],edges[edge_times==t])) for t in range(100)}
        key=f'{source}/{seed}/{arm}/{clip}';details=folder/'frames'/f'{source}-{seed}-{arm}-{clip}.json'
        write(details,frames,immutable=True)
        predictions[key]=dict(source=source,target=target,seed=seed,arm=arm,clip=clip,frames=100,
            package_identity=receipt['package_identity'],graph_sha256=receipt['graph_sha256'],csv_sha256=receipt['csv_sha256'],
            policy_logits_sha256=receipt.get('policy_logits_sha256'),
            per_frame_manifest_sha256=sha(details),per_frame_manifest=str(details.relative_to(REPO)))
    manifest=dict(status='frozen',created_utc=now(),models=models,predictions=predictions,guard=guard,
                  matrix_resolved=True,retention=retention,
                  all_four_cells_complete=retention['all_registered_arms_retained'],
                  all_three_arms_complete=retention['all_registered_arms_retained'],target_metrics_opened=False)
    manifest['identity']=digest(manifest)
    write(folder/'manifest.json',manifest,immutable=True)
    # A compact public index; detailed hashes remain a portable local artifact.
    summary=dict(status='frozen',identity=manifest['identity'],models=models,predictions=predictions,
                 retention=retention,
                 full_manifest_sha256=sha(folder/'manifest.json'),full_manifest_path=str((folder/'manifest.json').relative_to(REPO)))
    # The result directory was opened narrowly above; write via an owned folder
    # first, then the CLI report stage copies its verified public summary.
    write(folder/'public_summary.json',summary,immutable=True)
    return summary


def verify(key=None):
    path=WORK/'freeze/manifest.json'
    if not path.exists():raise Blocked('No global target prediction freeze; label access forbidden')
    value=read(path);identity=value['identity']
    if digest({k:v for k,v in value.items() if k!='identity'})!=identity:raise Blocked('Freeze identity changed')
    if value['status']!='frozen' or not value['matrix_resolved']:raise Blocked('Global matrix is not resolved and frozen')
    if key is not None:
        item=value['predictions'].get(key)
        if item is None:raise Blocked('Cell/clip absent from freeze')
        folder=WORK/'predictions'/item['arm']/item['source']/str(item['seed'])/item['clip']
        if sha(folder/'graph.npz')!=item['graph_sha256'] or sha(folder/'submission.csv')!=item['csv_sha256']:
            raise Blocked('Frozen target prediction bytes changed')
        if item.get('policy_logits_sha256') and sha(folder/'policy_logits.npz')!=item['policy_logits_sha256']:
            raise Blocked('Frozen target raw policy scores changed')
    return value
