"""Concise honest receipts, curves and resumable commands; no invented results."""
import csv
import json
from pathlib import Path
import psutil

from .common import WORK,RESULTS,REPO,read_json,write_json,write_csv,sha,now


def run(args=None):
    RESULTS.mkdir(parents=True,exist_ok=True)
    fits=[];curves=[];source_scores=[];calibrations=[]
    for path in sorted((WORK/'training').glob('*/*/*/training_receipt.json')):
        row=read_json(path);fits.append(dict(row,receipt_sha256=sha(path)))
        folder=path.parent
        if (folder/'diagnostics.jsonl').exists():
            for line in (folder/'diagnostics.jsonl').read_text().splitlines():
                d=json.loads(line)
                for split in ('fit','calibration'):
                    curves.append(dict(arm=row['arm'],source=row['source'],seed=row['seed'],step=d['step'],partition=split,**d[split]))
    for path in sorted((WORK/'screens').glob('*/*/*/*/summary.json')):
        r=read_json(path)
        for app,score in r['applications'].items():
            source_scores.append(dict(arm=r['arm'],source=r['source'],seed=r['seed'],step=r['step'],application=app,
                score=score['score'],delta=score['delta'],lost_supported_edges=score['lost_supported_edges'],clips=score['clips'],
                division_tp=score['division_tp'],division_fp=score['division_fp'],division_fn=score['division_fn']))
        calibrations.append({k:v for k,v in r['calibration'].items() if k!='rows'})
    write_json(RESULTS/'training_receipts.json',dict(fits=fits,actual_receipts_required=True))
    write_csv(RESULTS/'training_curves.csv',curves)
    write_csv(RESULTS/'source_scores.csv',source_scores)
    write_json(RESULTS/'calibration_audit.json',dict(status='measured' if calibrations else 'pending',fits=calibrations,
        reason=None if calibrations else 'Adequate training and full source screens have not completed'))
    ledger=WORK/'resources/leases.jsonl';events=[]
    if ledger.exists():events=[json.loads(line) for line in ledger.read_text().splitlines()]
    resource=dict(exclusive_lease_seconds=sum(e.get('lease_seconds',0.) for e in events if e['event']=='end'),
        wait_seconds=sum(e.get('wait_seconds',0.) for e in events if e['event']=='end'),
        open_leases=[e for e in events if e['event']=='begin' and e['token'] not in {x['token'] for x in events if x['event']=='end'}],
        scope='Full exclusive lease intervals, including loader and CPU work; not kernel time',
        study_disk_gib=sum(p.stat().st_size for p in WORK.rglob('*') if p.is_file())/2**30)
    monitors=[]
    for p in WORK.rglob('*.json'):
        if p.name=='resources.json' or p.name.endswith('.resources.json'):
            try:monitors.append(read_json(p))
            except (OSError,ValueError):pass
    resource.update(peak_total_gpu_gib=max((r.get('peak_total_gpu_gib',0.) for r in monitors),default=None),
                    peak_process_tree_rss_gib=max((r.get('peak_process_tree_rss_gib',0.) for r in monitors),default=None))
    write_json(RESULTS/'resource.json',resource)
    active=[]
    for p in psutil.process_iter(['cmdline','create_time']):
        try:
            cmd=p.info['cmdline'] or []
            if any(x in ('division_generalization_v2','division_generalization_v2.prediction_entry') for x in cmd):
                active.append(dict(pid=p.pid,command=cmd,started=p.info['create_time']))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    complete=(WORK/'queue/complete.json').exists()
    target=read_json(RESULTS/'target_evaluation.json') if (RESULTS/'target_evaluation.json').exists() else None
    baseline=read_json(RESULTS/'baseline_validation.json') if (RESULTS/'baseline_validation.json').exists() else None
    status=dict(study='division-generalization-v2',updated=now(),status='complete' if complete else 'incomplete_resumable',
        production_default='P0',target_met=False,new_target_scores=target,
        completed_directional_fits=len([r for r in fits if r['status']=='complete' and r['arm']!='prefix']),
        required_directional_fits=10,active_processes=active,
        prepared_clips={s:len(list((WORK/'source'/s).glob('*/receipt.json'))) for s in ('44b6','6bba')},
        reason=None if complete else 'Full training, full source screens, frozen target comparison and fresh proof remain required')
    write_json(RESULTS/'STATUS.json',status)
    replication=dict(status='measured' if target else 'pending',seeds=[20260916,314159],
                     paired_shared_prefix=True,independent_experiments=False,recommended=False,
                     reason='Recommendation requires all complete graphs, both seeds and fresh gates')
    recommendation=None
    if target and (RESULTS/'target_freeze.json').exists():
        freeze=read_json(RESULTS/'target_freeze.json');arm=freeze['nominee']
        pooled=[r for r in target['pooled'] if r['arm']==arm]
        embryos=[r for r in target['embryos'] if r['arm']==arm]
        bp={r['embryo']:r['score'] for r in baseline['embryos'] if r['arm']=='P0'}
        ready=len(pooled)==2 and len(embryos)==4 and (RESULTS/'fresh_image_validation.json').exists()
        if ready and all(r['score']>.934864986413134 for r in pooled) and all(r['score']>=bp[r['embryo']] for r in embryos) \
                and next(r['score'] for r in pooled if r['seed']==20260916)>.935178370257:
            recommendation=arm
        replication.update(recommended=bool(recommendation),nominee=arm,
            target_met_by_seed={str(r['seed']):r['score']>=.95 for r in pooled},
            robust_target_met=bool(recommendation and all(r['score']>=.95 for r in pooled)))
        status['target_met']=replication['robust_target_met'];write_json(RESULTS/'STATUS.json',status)
    write_json(RESULTS/'replication.json',replication)
    if not (RESULTS/'stage_attribution.json').exists():
        write_json(RESULTS/'stage_attribution.json',dict(status='pending',value=None,
            reason='Requires complete frozen source and target inference ledgers'))
    if not (RESULTS/'error_transitions.csv').exists():write_csv(RESULTS/'error_transitions.csv',[])
    report=[f'Candidate recommended: {recommendation}' if recommendation else 'P0 retained','',
        f'Execution: **{status["status"]}**. Requested replicated score ≥0.95: **{"achieved" if status["target_met"] else "not established"}**.','',
        '| Arm | Pooled score | Delta vs P0 | Delta vs C4_m6 | Status |',
        '|---|---:|---:|---:|---|']
    measured=(baseline['pooled'] if baseline else [])+(target['pooled'] if target else [])
    for r in measured:
        report.append(f'| {r["arm"]}{" / "+str(r["seed"]) if "seed" in r else ""} | {r["score"]:.12f} | {r["score"]-.934864986413134:+.12f} | {r["score"]-.935178370257:+.12f} | {r["status"]} |')
    for arm in ('G30','J_uniform','J_mined'):
        if not any(r['arm']==arm for r in measured):
            reason='source qualification pending' if not target else 'source failed; target export not qualified'
            report.append(f'| {arm} | null | null | null | {reason} |')
    report+=['',f'{status["completed_directional_fits"]}/10 directional fits have completed their required updates. '
        'Each main arm requires 4,096 joint updates; uniform and mined arms share their first 2,048 updates per direction/seed. '
        'The shared prefix is counted once in compute and does not make independent experiments.','',
        'The new module keeps native evidence inside learned relative complete-action scores. It uses fixed-seed probability '
        'sampling, separate positive exposure, masked unknown alternatives, negative-only groups, complete lost-link utility, '
        'and one two-scale raw scene shared across candidate pairs. The 490,804-parameter encoder/head uses ordered attention.','',
        'Both inherited baselines retain all 4,108,943 observations. All graph hashes and baseline metric receipts were verified, '
        'with fresh ordinary/crowded replays in both embryos and independent official aggregation. '
        'P0 and C4_m6 remain unchanged.','',
        'This is exploratory source-only direct fitting on exposed P0 observations. Reused whole-clip source splits union exact '
        'frame overlaps, but missing global acquisition offsets prevent independent inner-validation certification. '
        'Neither a local score nor a new raw-scene encoder establishes clean OOF or hidden leaderboard performance.','',
        '[Training receipts](training_receipts.json), [fixed source curves](training_curves.csv), '
        '[complete source screens](source_scores.csv), [per-embryo scores](per_embryo_scores.csv), '
        '[sampling audit](sampling_audit.json), [validation](validation.json), [resources](resource.json).','',
        f'Measured exclusive GPU leases: {resource["exclusive_lease_seconds"]/3600:.3f} h; waits: {resource["wait_seconds"]:.1f} s. '
        'Full lease intervals include preprocessing. Kernel time, inference and checkpoint overhead remain separately recorded.','',
        'No production promotion, merge, Kaggle submission, weight publication or leaderboard claim has been made.']
    (RESULTS/'REPORT.md').write_text('\n'.join(report)+'\n')
    continuation=f'''Read REPORT.md and STATUS.json. Status: {status['status']}.

Run from `{REPO}` on the current branch. Inspect actual processes first; never resume an older study or launch duplicate workers.

```sh
ps -eo pid,ppid,etime,rss,args | rg division_generalization_v2
nvidia-smi
export PYTHONNOUSERSITE=1 PYTHONPATH=tools:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
STUDY_PY=/kaggle/envs/cell-tracking-annotation-selection-v1/bin/python
$STUDY_PY -m division_generalization_v2 queue
$STUDY_PY -m division_generalization_v2 report
```

The queue checkpoints every 128 joint updates and at milestones. Each matched arm loads its seed/source's identical 2,048-update model, optimizer, RNG and data-order state. Do not shrink 4,096 updates or treat interrupted fits as failed architectures.

Heavy artifacts are under `work/division-generalization-v2`, a symlink to `/kaggle/working/cell-tracking/division-generalization-v2`. Recreate that empty isolated root on another machine; inherited inputs resolve through input_manifest.json. Existing raw inputs and work/pipeline-error-training-20260915 are read-only references. No shared environment changes are needed.

Any stage failure is preserved with its exact command and log in queue/failure.json and logs/. Fix implementation defects, archive invalid attempts with reasons, and retain scientific settings. An unclosed GPU lease requires conservative accounting reconciliation before restart.

Outstanding gates are explicit in STATUS.json, validation.json, training_receipts.json and source_scores.csv. Target export requires source qualification; failed cheap control does not block main training. Both directional packages and both seeds must be frozen before target scoring. Full target scoring and fresh image proof, not partial clips, govern recommendation and the ≥0.95 claim.
'''
    (RESULTS/'CONTINUATION.md').write_text(continuation)
    return status


if __name__=='__main__':run()
