"""Deadline-gated W470 treatment, independently frozen after required controls."""
import subprocess
import sys
import time
import os
import signal
from concurrent.futures import ThreadPoolExecutor

def jobs(seed2=False):
    from .common import SEED,OUT,read
    seed=314159 if seed2 else SEED;suffix='_seed2' if seed2 else '';pre=[];adapt=[]
    gbase='G_combined'
    if seed2:
        gbase='G_external_seed2'
        if read(OUT/'models/G_external_seed2.json')['config']['domains']!=['synthetic','zebrafish']:
            gbase='G_rendered_base_seed2';pre.append(dict(name=gbase,component='G',domains=['synthetic','zebrafish'],steps=20000,seed=seed))
    for source in ['44b6','6bba']:
        domain=f'rendered_zoo_{source}';iname=f'I_rendered_{source}{suffix}'
        pre.append(dict(name=iname,component='I',domains=['synthetic',domain],steps=20000,init='D_synthetic'+suffix,seed=seed,calibration_source=source))
        adapt.append(dict(name=f'G_C7{suffix}_{source}',component='G',domains=[source],steps=8000,init=gbase,source=source,replay=['synthetic','zebrafish'],seed=seed))
        adapt.append(dict(name=f'I_C7{suffix}_{source}',component='I',domains=[source],steps=8000,init=iname,source=source,replay=['synthetic',domain],seed=seed,calibration_source=source))
    return pre,adapt

def worker(seed2,name):
    from .train import fit
    spec=next(j for j in sum(jobs(seed2),[]) if j['name']==name);fit(**spec)

def sanity():
    import numpy as np
    import torch
    from .common import OUT,arrays,write
    from .render_zoo import render
    from .models import ImageEvent
    from .train import loss
    t=np.arange(3);p=np.zeros((3,3));stats=dict(source='44b6',amplitude=1.,sigma_pixels=1.,background=0.,noise_std=0.)
    patch,valid=render(t,p,np.array([2]),p[:1],stats)
    assert np.array_equal(valid,[[1,1,0]]) and patch.max()>0 and not patch[:,2].any()
    # Random per-cell brightness is fixed independently of links; all planes
    # of an isolated isotropic point must agree numerically.
    np.testing.assert_array_equal(patch[0,1,0],patch[0,1,1]);np.testing.assert_array_equal(patch[0,1,0],patch[0,1,2])
    d=arrays(OUT/'cache/rendered_zoo_44b6/train.npz');ids=np.r_[np.flatnonzero(d['target']>0)[:4],np.flatnonzero(d['target']==0)[:4]]
    b={k:torch.as_tensor(d[k][ids],device='cuda') for k in ['x','patch','valid','target','link_y','weight']};b['patch']=b['patch'].float()/255
    torch.manual_seed(1729);model=ImageEvent().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    initial=float(loss(model,b,'I').detach())
    for i in range(800):
        optimizer.zero_grad();v=loss(model,b,'I');v.backward()
        if i==0:gradient=float(sum(p.grad.norm() for p in model.encoder.parameters()))
        optimizer.step()
    final=float(loss(model,b,'I').detach());assert final<initial*.4 and gradient>0,(initial,final,gradient)
    write(OUT/'rendering_sanity.json',dict(passed=True,actual_rendered_bags=len(ids),extra_sanity_updates=800,
        initial_loss=initial,final_loss=final,encoder_gradient_norm=gradient,isolated_point_plane_parity=True,missing_future_mask=True))

def run(phase=None):
    if phase and phase.startswith('infer_'):
        from . import offline_guard
        from .infer import transform,GUARD,EXTERNAL_GUARD
    from .common import OUT,SEED,read,write,now,sha,arrays,save,inputs,V3,Path
    if phase=='prepare':
        from .render_zoo import run as prepare
        prepare();sanity();return
    if phase and phase.startswith('train_'):
        seed2=phase.endswith('seed2')
        def launch(job):
            with (OUT/'logs'/f'rendering_{job["name"]}.log').open('a') as f:
                r=subprocess.run([sys.executable,'-m','multidata_training_v4.rendering_trial',str(int(seed2)),job['name']],stdout=f,stderr=subprocess.STDOUT)
            assert r.returncode==0,job
        for batch in jobs(seed2):
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,batch))
        return
    if phase and phase.startswith('calibrate_'):
        import torch
        from scipy.special import logsumexp
        from scipy.optimize import minimize_scalar
        from .calibrate import sample,load_model,scores,diagnostics
        from .provenance import model_manifest
        seed2=phase.endswith('seed2');suffix='_seed2' if seed2 else '';result={};torch.set_num_threads(2)
        path=OUT/'calibration_rendering.json'
        if path.exists():result=read(path)['models']
        drift=[]
        for source in ['44b6','6bba']:
            a=f'G_C7{suffix}_{source}';b=f'G_C4{suffix}_{source}'
            if not (OUT/'models'/f'{b}.json').exists():continue
            ca=read(OUT/'models'/f'{a}.json')['config'];cb=read(OUT/'models'/f'{b}.json')['config']
            assert {k:v for k,v in ca.items() if k!='name'}=={k:v for k,v in cb.items() if k!='name'}
            x=torch.load(OUT/'models'/f'{a}.pt',map_location='cpu',weights_only=False)['model']
            y=torch.load(OUT/'models'/f'{b}.pt',map_location='cpu',weights_only=False)['model']
            keys=[k for k in x if x[k].is_floating_point()]
            delta=sum(float((x[k].double()-y[k].double()).square().sum()) for k in keys)
            norm=sum(float(y[k].double().square().sum()) for k in keys)
            drift.append(dict(rendering_model=a,control_model=b,configs_equal_except_name=True,
                bitwise_equal=all(torch.equal(x[k],y[k]) for k in x),relative_parameter_l2=(delta/max(norm,1e-30))**.5,
                measured_before_this_round_target_outcomes=True))
        drift_path=OUT/'rendering_geometry_drift.json';old=read(drift_path) if drift_path.exists() else []
        merged={r['rendering_model']:r for r in old};merged.update({r['rendering_model']:r for r in drift})
        write(drift_path,list(merged.values()))
        for comp in ['G','I']:
            val=sample('synthetic',comp);test=sample('synthetic',comp,'test')
            for source in ['44b6','6bba']:
                name=f'{comp}_C7{suffix}_{source}'
                if name in result:assert result[name]['model_sha256']==sha(OUT/'models'/f'{name}.pt');continue
                m=load_model(name);l=scores(m,val);y=val['target'];known=y>=0
                temp=float(minimize_scalar(lambda t:float((logsumexp(l[known]/t,axis=1)-l[known,y[known]]/t).mean()),bounds=(.5,8.),method='bounded').x)
                result[name]=dict(temperature=temp,calibration=diagnostics(l,y,temp),generator_test=diagnostics(scores(m,test),test['target'],temp),model_sha256=sha(OUT/'models'/f'{name}.pt'));del m
        write(path,dict(models=result,primary_margin=4.,same_primary_procedure=True,rendering_plan_sha256=sha(OUT/'rendering_plan.json')))
        variant='C7_seed2' if seed2 else 'C7'
        frozen_cal=dict(models={k:v for k,v in result.items() if ('seed2' in k)==seed2},primary_margin=4.,same_primary_procedure=True)
        frozen_path=OUT/f'calibration_rendering_{variant}.json'
        if frozen_path.exists():assert read(frozen_path)==frozen_cal
        else:write(frozen_path,frozen_cal)
        subset=[p.stem for p in (OUT/'models').glob('*.json') if ('_C7' in p.stem or '_rendered_' in p.stem) and ('seed2' in p.stem)==seed2]
        manifest=model_manifest(subset=subset,destination=OUT/f'checkpoint_manifest_rendering_{variant}.json')
        aggregate=read(OUT/'checkpoint_manifest_rendering.json') if (OUT/'checkpoint_manifest_rendering.json').exists() else {}
        aggregate.update(manifest);write(OUT/'checkpoint_manifest_rendering.json',aggregate);return
    if phase and phase.startswith('infer_'):
        import torch
        import zarr
        from strong_tracker_v3.common import validate
        parts=phase.split('_');torch.set_num_threads(2);variant='C7_seed2' if parts[1]=='seed2' else 'C7';cal=read(OUT/f'calibration_rendering_{variant}.json')
        frozen=dict(variant=variant,calibration_sha256=sha(OUT/f'calibration_rendering_{variant}.json'),
            checkpoint_manifest_sha256=sha(OUT/f'checkpoint_manifest_rendering_{variant}.json'),before_this_round_outcomes=True,
            driver_code_sha256=sha(Path(__file__)),source_workers=['44b6','6bba'],
            inference_code_sha256={n:sha(Path(__file__).parent/n) for n in ['infer.py','models.py','proposals.py','adapters.py','decode.py']})
        config=OUT/f'rendering_inference_config_{variant}.json'
        if config.exists():assert read(config)==frozen
        else:write(config,frozen)
        if len(parts)==2:
            processes=[subprocess.Popen([sys.executable,'-m','multidata_training_v4','rendering_trial','--phase',phase+'_'+s]) for s in ['44b6','6bba']]
            codes=[p.wait() for p in processes];assert not any(codes),codes
            write(OUT/f'rendering_prediction_receipt_{variant}.json',dict(completed=True,annotation_guard=GUARD,external_guard=EXTERNAL_GUARD,both_directions=True,
                workers=[read(OUT/f'rendering_prediction_worker_{variant}_{s}.json') for s in ['44b6','6bba']]))
            return
        worker_source=parts[2];assert worker_source in ['44b6','6bba']
        for row in inputs():
            if row['embryo']==worker_source:continue
            name=row['dataset'];source='6bba' if row['embryo']=='44b6' else '44b6';dest=OUT/'predictions'/variant/f'{name}.npz';proof=OUT/'rendering_prediction_receipts'/variant/f'{name}.json'
            if proof.exists():assert sha(dest)==read(proof)['prediction_sha256'];continue
            g=arrays(V3/'selected_predictions'/f'{name}.npz');image=zarr.open_group(row['image_path'],mode='r')['0'][:]
            n,e,r=transform(g['nodes'],g['edges'],image,row['physical_scale'],source,variant,OUT/'models',cal,
                evidence_callback=lambda evidence:save(OUT/'additional_candidate_evidence'/variant/f'{name}.npz',**evidence))
            validate(n,e,row['image_shape']);save(dest,nodes=n,edges=e)
            write(proof,dict(dataset=name,variant=variant,prediction_sha256=sha(dest),**r));print('rendering frozen',variant,name,flush=True)
        write(OUT/f'rendering_prediction_worker_{variant}_{worker_source}.json',dict(completed=True,source=worker_source,annotation_guard=GUARD,external_guard=EXTERNAL_GUARD,
            offline_guard=dict(installed_before_dependencies=True,blocked_network_attempts=len(offline_guard.BLOCKED_NETWORK_ATTEMPTS),
                blocked_dataset_reads=len(offline_guard.BLOCKED_EXTERNAL_DATA_READS),includes_rendered_datasets=True)));return
    import pandas as pd
    from datetime import datetime
    deadline=datetime.fromisoformat(read(OUT/'authorized_run_window.json')['deadline_utc'].replace('Z','+00:00')).timestamp()
    plan=OUT/'rendering_plan.json';receipt=OUT/'rendering_trial_receipt.json'
    if receipt.exists():return
    scores=pd.read_csv(OUT/'ablation_scores.csv')
    def qualifies(variant):
        rows=scores[scores.variant==variant]
        return len(rows)==3 and (rows.delta_v3>=-1e-10).all() and rows[rows.embryo=='pooled'].iloc[0].delta_v3>1e-12
    cal=read(OUT/'calibration.json')['models'];utility=max(r['generator_test']['positive_pair_correct']/max(r['generator_test']['positive'],1) for n,r in cal.items() if '_C4_' in n or '_C6_' in n)
    if not plan.exists():
        eligible=deadline-time.time()>=5700 and utility>.5 and not any(qualifies(a) for a in ['C4','C6'])
        write(plan,dict(created=now(),eligible=bool(eligible),generator_pair_recall_max=utility,remaining_seconds=deadline-time.time(),
            primary_C4_qualifies=bool(qualifies('C4')),primary_C6_qualifies=bool(qualifies('C6')),
            lock_sha256=sha(OUT/'rendering_followup_lock.json'),source_stats_use_opposite_target=False))
    if not read(plan)['eligible']:
        write(receipt,dict(completed=False,optional_skipped=True,reason='Predeclared utility/transfer-gap/time condition not satisfied',plan=read(plan)));return
    stages=['prepare','train_primary','calibrate_primary','infer_primary','evaluate_primary'];completed=[]
    for stage in stages:
        remaining=deadline-time.time()-1500
        if remaining<=0:break
        variant='C7_seed2' if stage.endswith('seed2') else 'C7'
        command=[sys.executable,'-m','multidata_training_v4','rendering_trial','--phase',stage]
        if stage.startswith('evaluate_'):command=[sys.executable,'-m','multidata_training_v4','evaluate','--variants',variant,'--workers','8']
        start=time.monotonic()
        with (OUT/'logs'/f'rendering_stage_{stage}.log').open('a') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM);process.wait();code=124
        completed.append(dict(stage=stage,returncode=code,seconds=time.monotonic()-start));write(OUT/'rendering_progress.json',completed)
        if code:break
        if stage=='evaluate_primary':
            scores=pd.read_csv(OUT/'ablation_scores.csv')
            repeat_estimate=1.3*sum(r['seconds'] for r in completed if r['stage']!='prepare')+1500
            if qualifies('C7') and deadline-time.time()>=max(3900,repeat_estimate):stages.extend(['train_seed2','calibrate_seed2','infer_seed2','evaluate_seed2'])
    if any(r['stage']=='evaluate_primary' and r['returncode']==0 for r in completed):
        ledger=pd.read_csv(OUT/'dataset_use.csv').fillna('');add=[]
        for source in ['44b6','6bba']:
            domain=f'rendered_zoo_{source}'
            if domain not in set(ledger.source):add.append(dict(source=domain,available=True,hashes_verified=True,images=True,selected=True,
                role='explicit rendered Zoo-fish optical pretraining/replay',label_quality='weak Zoo trajectories; rendered images',permitted_use='derived from eligible organizer-cleared Zoo-fish',
                calibration_exposure=f'{source} source-only image statistics; shared Zoo acquisition',exclusion='',alias_group='zoo_zebrafish'))
        if add:pd.concat([ledger,pd.DataFrame(add)],ignore_index=True).to_csv(OUT/'dataset_use.csv',index=False)
    write(receipt,dict(completed=len(completed)==len(stages) and all(r['returncode']==0 for r in completed),optional_skipped=False,
        jobs=completed,primary_measured=any(r['stage']=='evaluate_primary' and r['returncode']==0 for r in completed),
        second_seed_measured=any(r['stage']=='evaluate_seed2' and r['returncode']==0 for r in completed),
        calibration_and_thresholds_unchanged=True,operational_exploratory=True))

if __name__=='__main__':worker(bool(int(sys.argv[1])),sys.argv[2])
