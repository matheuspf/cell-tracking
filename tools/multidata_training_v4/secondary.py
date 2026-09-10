"""Queued second-seed replication, preserving budgets and both source directions."""
def run(phase=None):
    if phase and phase.startswith('infer'):
        # Guard precedes any numerical dependency imports in this subprocess.
        from .infer import transform,GUARD,EXTERNAL_GUARD
    from .common import OUT,SEED,read,write,now,sha
    plan_path=OUT/'secondary_plan.json'
    if phase=='train':
        from .secondary_training import run as train
        train();return
    if phase=='calibrate':
        existing=OUT/'calibration_secondary.json'
        if existing.exists():
            old=read(existing)
            for name,record in old['models'].items():assert record['model_sha256']==sha(OUT/'models'/f'{name}.pt')
            assert old['plan_sha256']==sha(plan_path)
            return
        from .provenance import model_manifest
        model_manifest(secondary=True)
        import torch
        from scipy.special import logsumexp
        from scipy.optimize import minimize_scalar
        from .calibrate import sample,load_model,scores,diagnostics
        plan=read(plan_path);result={};torch.set_num_threads(2)
        for comp in ['G','I']:
            val=sample('synthetic',comp);test=sample('synthetic',comp,'test')
            for source in ['44b6','6bba']:
                for arm in ['C1',plan['arm']]+(['C1short'] if plan.get('repeat_short_control') else []):
                    name=f'{comp}_{arm}_seed2_{source}';m=load_model(name);l=scores(m,val);y=val['target'];known=y>=0
                    temp=float(minimize_scalar(lambda t:float((logsumexp(l[known]/t,axis=1)-l[known,y[known]]/t).mean()),bounds=(.5,8.),method='bounded').x)
                    result[name]=dict(temperature=temp,calibration=diagnostics(l,y,temp),generator_test=diagnostics(scores(m,test),test['target'],temp),model_sha256=sha(OUT/'models'/f'{name}.pt'))
                    del m
        write(OUT/'calibration_secondary.json',dict(created=now(),models=result,primary_margin=4.,same_primary_procedure=True,plan_sha256=sha(plan_path)))
        return
    if phase and phase.startswith('infer'):
        import torch
        import zarr
        import numpy as np
        from .common import arrays,save,V3,inputs
        from strong_tracker_v3.common import validate
        plan=read(plan_path);cal=read(OUT/'calibration_secondary.json');torch.set_num_threads(2)
        if phase=='infer':
            import subprocess,sys
            from pathlib import Path
            frozen=dict(created_policy='both source directions frozen before second-round outcomes',
                calibration_sha256=sha(OUT/'calibration_secondary.json'),models=read(OUT/'checkpoint_manifest_secondary.json'),
                inference_code_sha256={n:sha(Path(__file__).parent/n) for n in ['infer.py','models.py','proposals.py','adapters.py','decode.py','secondary.py']},
                source_workers=['44b6','6bba'])
            config=OUT/'secondary_inference_config.json'
            if config.exists():assert read(config)==frozen
            else:write(config,frozen)
            processes=[subprocess.Popen([sys.executable,'-m','multidata_training_v4','secondary','--phase','infer_'+s]) for s in ['44b6','6bba']]
            codes=[p.wait() for p in processes];assert not any(codes),codes
            write(OUT/'secondary_prediction_receipt.json',dict(completed=True,annotation_reads=0,both_directions=True,
                annotation_guard=GUARD,external_guard=EXTERNAL_GUARD,
                workers=[read(OUT/f'secondary_prediction_worker_{s}.json') for s in ['44b6','6bba']],
                model_hashes=read(OUT/'checkpoint_manifest_secondary.json'),calibration_sha256=sha(OUT/'calibration_secondary.json')))
            return
        worker_source=phase.removeprefix('infer_')
        assert worker_source in ['44b6','6bba']
        for row in inputs():
            if row['embryo']==worker_source:continue
            name=row['dataset'];source='6bba' if row['embryo']=='44b6' else '44b6';g=arrays(V3/'selected_predictions'/f'{name}.npz')
            image=None
            for variant in ['C1_seed2',plan['arm']+'_seed2']+(['C1short_seed2'] if plan.get('repeat_short_control') else []):
                dest=OUT/'predictions'/variant/f'{name}.npz'
                proof=OUT/'secondary_prediction_receipts'/variant/f'{name}.json'
                if proof.exists():assert sha(dest)==read(proof)['prediction_sha256'];continue
                if image is None:image=zarr.open_group(row['image_path'],mode='r')['0'][:]
                n,e,receipt=transform(g['nodes'],g['edges'],image,row['physical_scale'],source,variant,OUT/'models',cal,
                    evidence_callback=lambda evidence:save(OUT/'additional_candidate_evidence'/variant/f'{name}.npz',**evidence))
                validate(n,e,row['image_shape']);save(dest,nodes=n,edges=e)
                write(proof,dict(dataset=name,variant=variant,source_model=source,prediction_sha256=sha(dest),**receipt))
            print('second-seed frozen',name,flush=True)
        write(OUT/f'secondary_prediction_worker_{worker_source}.json',dict(completed=True,source=worker_source,annotation_guard=GUARD,external_guard=EXTERNAL_GUARD))
        return
    import os,sys,subprocess,time,signal
    from datetime import datetime
    import pandas as pd
    deadline=datetime.fromisoformat(read(OUT/'authorized_run_window.json')['deadline_utc'].replace('Z','+00:00')).timestamp()
    while True:
        p=OUT/'scheduled_tests_receipt.json'
        if p.exists():
            jobs=read(p)['jobs']
            if all(j['state']=='complete' for j in jobs):break
        if time.time()>deadline-7200:
            write(OUT/'secondary_verification.json',dict(completed=False,reason='Insufficient remaining time for full second-seed fits and graph evaluation'));return
        time.sleep(30)
    if not plan_path.exists():
        scores=pd.read_csv(OUT/'ablation_scores.csv');qual=[]
        for arm in ['C2','C3','C4','C5','C6']:
            rr=scores[scores.variant==arm]
            if (rr.delta_v3>=-1e-10).all() and rr[rr.embryo=='pooled'].iloc[0].delta_v3>1e-12:qual.append((rr[rr.embryo=='pooled'].iloc[0].score,arm))
        arm=max(qual)[1] if qual else 'C4'
        short=scores[scores.variant=='C1short']
        repeat_short=bool(len(short)==3 and (short.delta_v3>=-1e-10).all() and short[short.embryo=='pooled'].iloc[0].delta_v3>1e-12 and deadline-time.time()>10800)
        write(plan_path,dict(created=now(),arm=arm,seed=314159,reason='strongest qualifying primary external arm' if qual else 'prespecified C4 replication despite no qualifying external arm',
            controls=['C1'],repeat_short_control=repeat_short,target_threshold_changes=False,primary_scores_sha256=sha(OUT/'ablation_scores.csv')))
    plan=read(plan_path);arm=plan['arm'];jobs=[]
    for ph in ['train','calibrate','infer','evaluate']:
        remain=deadline-time.time()-600
        if remain<=0:break
        variants=f'C1_seed2,{arm}_seed2'+(',C1short_seed2' if plan.get('repeat_short_control') else '')
        cmd=[sys.executable,'-m','multidata_training_v4','secondary','--phase',ph] if ph!='evaluate' else [sys.executable,'-m','multidata_training_v4','evaluate','--variants',variants]
        log=OUT/'logs'/f'secondary_{ph}.log';start=time.monotonic()
        with log.open('a') as f:
            process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=process.wait(timeout=remain)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM);process.wait();code=124
        jobs.append(dict(phase=ph,returncode=code,seconds=time.monotonic()-start));write(OUT/'secondary_progress.json',jobs)
        if code:break
    complete=len(jobs)==4 and all(j['returncode']==0 for j in jobs)
    write(OUT/'secondary_verification.json',dict(completed=complete,arm=arm,seed=314159,jobs=jobs,
        no_lucky_seed_promotion=True,both_directions_frozen_before_second_round=complete))
