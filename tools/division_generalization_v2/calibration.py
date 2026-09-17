"""Held-source max-selected edit evidence; intercept is never regularized."""
from collections import defaultdict
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

from .common import RESULTS,WORK,read_json,write_json
from .resources import Lease


def fit(rows):
    good_groups={r['group'] for r in rows if r['target']==1}
    bad_groups={r['group'] for r in rows if r['target']==0}
    result=dict(status='calibration_unestablished',temperature=1.,intercept=0.,
        decision_units=len(rows),positive_groups=len(good_groups),negative_groups=len(bad_groups),
        scope='Sparse-supported max-selected source edits, not biological prevalence',
        intercept_penalized=False,threshold=0.,source_groups_independently_certified=False)
    if len(good_groups)<5 or len(bad_groups)<5:
        result['reason']='Fewer than five distinct source overlap groups in one class; fixed zero relative gains retained'
        return result
    x=np.asarray([r['gain'] for r in rows]);y=np.asarray([r['target'] for r in rows])
    w=np.asarray([r['weight'] for r in rows],float);w/=w.sum()
    def objective(theta):
        inverse=np.exp(theta[0]);z=x*inverse+theta[1]
        residual=(expit(z)-y)*w
        # One fixed weak slope regularizer and bounds handle separation. The
        # intercept is unpenalized, with an explicit numerical finite bound.
        value=np.sum(w*(np.logaddexp(0,z)-y*z))+.001*theta[0]**2
        gradient=np.array([np.sum(residual*x*inverse)+.002*theta[0],residual.sum()])
        return value,gradient
    fitted=minimize(objective,[0.,0.],jac=True,method='L-BFGS-B',bounds=[(-4.,4.),(-30.,30.)])
    if not fitted.success or not np.isfinite(fitted.x).all():
        result['reason']='Finite bounded source fit failed; no resubstitution fallback'
        return result
    result.update(status='held_source_fitted',temperature=float(np.exp(-fitted.x[0])),intercept=float(fitted.x[1]),
        optimizer_message=str(fitted.message),finite_separation_bounds=[[-4,4],[-30,30]],
        boundary_reached=bool(abs(fitted.x[0])>3.999 or abs(fitted.x[1])>29.999))
    return result


@torch.no_grad()
def run(model,dataset,path,amp,checkpoint_sha256=None):
    if path.exists():return read_json(path)
    # Match the training and frozen prediction runtime, including small-head
    # fallback fitting when invoked from a standalone source-screen process.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    device='cuda' if model.image else 'cpu'
    split=read_json(RESULTS/'split_manifest.json')['directions'][dataset.source]
    rows=[];keys=iter(sorted(k for k,a in dataset.anchors.items() if a['random_included']))
    finished=False
    import contextlib,time
    while not finished:
        context=Lease('calibration/'+dataset.source,required_gib=3.) if model.image else contextlib.nullcontext()
        with context:
            model.to(device);model.eval();started=time.monotonic()
            while True:
                try:key=next(keys)
                except StopIteration:finished=True;break
                b,s=dataset.batch(dict(key=key),device)
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
                    out=model([b],s[None] if s is not None else None)[0]
                gain=out['gain'].cpu().numpy();edits=~b['keep'].cpu().numpy()
                # Maximize BEFORE support filtering; unsupported maxima remain
                # unknown instead of replacing them with a labeled runner-up.
                if not edits.any():continue
                indices=np.flatnonzero(edits);i=indices[np.argmax(gain[indices])]
                if bool(b['supported'][i]):
                    a=dataset.anchors[key]
                    rows.append(dict(key=key,group=split[a['dataset']]['overlap_group'],
                        gain=float(gain[i]),target=int(b['utility'][i]>1e-12),
                        weight=1/a['random_probability'],selected_action=int(i),null_gain=0.))
                if time.monotonic()-started>=30:break
            model.cpu()
            if model.image:torch.cuda.empty_cache()
    result=fit(rows)
    result.update(source=dataset.source,partition=dataset.partition,rows=rows,post_maximization=True,target_used=False)
    if result['status']=='calibration_unestablished' and checkpoint_sha256 is not None:
        from .head_oof import run as small_head_oof
        original={k:v for k,v in result.items() if k!='rows'}
        with torch.enable_grad():
            result=small_head_oof(model,dataset,path.parent/'small_head_oof',amp,checkpoint_sha256)
        result['original_held_source_support']=original
    write_json(path,result)
    return result
