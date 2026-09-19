"""Complete source pilot policy/solver integration; never a retained fit."""
from pathlib import Path
import sys,tempfile,time,shutil
from .common import REPO,DATA,WORK,Blocked,read,write,sha,now


def run(source,seed,clip,arm):
    if clip not in read(WORK/'source_partitions.json')[source]['fit']:raise Blocked('Source-fit only proof')
    prediction=WORK/'source_inference-overfit'/source/str(seed)/clip
    bankroot=WORK/'pilot_banks'/source/str(seed)/'fit'
    fit=WORK/'mixed_profile'/source/str(seed)
    folder=WORK/'policy_proof'/source/str(seed)/arm;folder.mkdir(parents=True,exist_ok=True)
    (folder/'tmp').mkdir(exist_ok=True);tempfile.tempdir=str((folder/'tmp').resolve())
    from .resources import Resources,ROOT
    resources=Resources('pilot',source,seed)
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[prediction,bankroot,fit,DATA/'train'/f'{clip}.zarr'],outputs=[folder,ROOT],code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np,torch
    from .features import normalize
    from .linear import fit as linear_fit
    from .policy import apply_package
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    normalization=read(fit/'normalizer.json');write(folder/'normalizer.json',normalization)
    if arm=='C01':
        with np.load(bankroot/clip/'training.npz') as d:
            parent=normalize(d['parent_x'],*normalization['values']['parent'])
            action=normalize(d['action_x'],*normalization['values']['action'])
            risks=[d['risks'][a:b] for a,b in zip(d['offset'][:-1],d['offset'][1:])]
            xs=[action[a:b] for a,b in zip(d['offset'][:-1],d['offset'][1:])]
            weights={}
            for anchors in read(bankroot/clip/'receipt.json')['positive_events'].values():
                for a in anchors:weights[a]=weights.get(a,0)+1/len(anchors)
            result=linear_fit(parent,xs,risks,[weights.get(i,1.) for i in range(len(parent))],[True]*len(parent))
        write(folder/'linear.json',dict(parameters=result,diagnostic=True))
    else:shutil.copy2(fit/'final.pt',folder/'compact.pt')
    write(folder/'calibration.json',dict(temperature=1.,intercept=0.,margin=6,disabled_policy=False,diagnostic_fixed_not_fitted=True))
    shutil.copy2(prediction/'graph.npz',folder/'graph.npz')
    shutil.copy2(prediction/'submission.csv',folder/'submission.csv')
    tick=time.monotonic()
    result=apply_package(dict(arm=arm,source=source,seed=seed),folder,DATA/'train'/f'{clip}.zarr',folder,resources,read(prediction/'receipt.json'))
    result.update(status='source_policy_proof',arm=arm,source=source,seed=seed,retained_fit=False,guard=guard,
                  gpu_lease_seconds=resources.seconds,wall_seconds=time.monotonic()-tick,finished_utc=now())
    write(folder/'receipt.json',result);resources.close();return result
