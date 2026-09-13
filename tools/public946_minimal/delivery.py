"""Locked robustness checks, offline notebook packaging, and sanitized results."""
from __future__ import annotations

import copy
import csv
import difflib
import os
import subprocess
import time
from pathlib import Path

from .common import ARMS, REPO, digest, load_arrays, now, read_json, sha, write_json
from .selection import ranking
from .study import execute, score


def candidates(args):
    if not (args.out/'scores/full/B0/summary.json').exists():
        return []
    baseline = read_json(args.out/'scores/full/B0/summary.json')
    singles = []
    transfers = []
    for arm in ARMS:
        if arm in ('B0', 'B1'):
            continue
        decision = args.out/'decisions'/(arm+'.json')
        if not decision.exists() or not read_json(decision)['eligible']:
            continue
        (transfers if arm.startswith('X') else singles).append(arm)
    def key(arm, parent):
        result = read_json(args.out/'scores/full'/arm/'summary.json')
        result['module_count'] = len(arm_modules(args, arm)) - int(arm.startswith('X'))
        return ranking(arm, result, parent,
                       read_json(args.out/'execution/full'/(arm+'.json'))['seconds'])
    selected = []
    if singles:
        selected.append(min(singles, key=lambda a: key(a, baseline)))
    if transfers:
        selected.append(min(transfers, key=lambda a: key(a, read_json(args.out/'scores/full/B1/summary.json'))))
    return selected


def arm_modules(args, arm):
    if arm.startswith('X'):
        return read_json(args.out/'finalist_lock.json')['recipes'][arm]+['no_motion']
    return ARMS[arm]['modules']


def pilot_rows(args):
    names = {p['dataset'] for p in read_json(args.out/'pilots.json')}
    return [r for r in read_json(args.out/'image_inventory.json') if r['dataset'] in names]


def robustness(args):
    from .diagnostics import prediction_disagreement
    rows = pilot_rows(args)
    arms = ['B0', 'B1', *candidates(args)]
    outcomes = {}
    for transform in ('reflect_x', 'reflect_y'):
        for arm in arms:
            _, failures = execute(args, arm, rows, transform, modules=arm_modules(args, arm), transform=transform)
            if failures:
                outcomes[arm+'_'+transform] = dict(status='failed', failures=failures)
                continue
            result = score(args, arm, rows, transform, transform=transform, baseline=None if arm=='B0' else ('B1' if arm.startswith('X') else 'B0'))
            comparisons = {r['dataset']: prediction_disagreement(args.out/'predictions/full'/arm/r['dataset']/'final.npz',
                           args.out/'predictions'/transform/arm/r['dataset']/'final.npz', r['image_shape'], transform) for r in rows}
            outcomes[arm+'_'+transform] = dict(status='measured', score=result, prediction_disagreement=comparisons)
    # Rename all stems and reverse outer dataset enumeration. The native image batch has size 1;
    # unet_batch_size is an unused public argument, so there is no independent inner batch order.
    renamed = []
    image_dir = args.out/'robustness_inputs/test'
    image_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        r = copy.deepcopy(row)
        alias = f'unknown_stem_{i:03d}'
        link = image_dir/(alias+'.zarr')
        if not link.exists():
            link.symlink_to(row['path'], target_is_directory=True)
        r.update(dataset=alias, original_dataset=row['dataset'], path=str(link), original_path=row['path'])
        renamed.append(r)
    for arm in arms:
        _, failures = execute(args, arm, renamed[::-1], 'renamed_reverse', modules=arm_modules(args, arm))
        matches = {}
        for row in renamed:
            path = args.out/'predictions/renamed_reverse'/arm/row['dataset']/'complete.json'
            if path.exists():
                a = read_json(args.out/'predictions/full'/arm/row['original_dataset']/'complete.json')
                b = read_json(path)
                matches[row['original_dataset']] = a['graph_hash'] == b['graph_hash']
        outcomes[arm+'_renamed_reverse'] = dict(status='passed' if len(matches)==4 and all(matches.values()) and not failures else 'failed',
            graph_identity=matches, fresh_process=True, inner_batch_order='Not applicable: native encode batch size is one; unet_batch_size is unused')
    # Required E04 one-voxel shift is a diagnostic even if E04 was scientifically rejected.
    if (args.out/'scores/full/E04/summary.json').exists():
        for arm in ('B0','E04'):
            _, failures = execute(args, arm, rows, 'shift_x1', modules=arm_modules(args, arm), transform='shift_x1')
            if not failures:
                comparisons = {r['dataset']: prediction_disagreement(args.out/'predictions/full'/arm/r['dataset']/'final.npz',
                    args.out/'predictions/shift_x1'/arm/r['dataset']/'final.npz',r['image_shape'],'shift_x1') for r in rows}
                outcomes[arm+'_shift_x1'] = dict(status='diagnostic_complete', prediction_disagreement=comparisons)
    noise = read_json(args.out/'numerical_noise.json')
    recommendation = {}
    for arm in candidates(args):
        parent = 'B1' if arm.startswith('X') else 'B0'
        checks = []
        for transform in ('reflect_x','reflect_y'):
            a, b = outcomes.get(arm+'_'+transform), outcomes.get(parent+'_'+transform)
            checks.append(bool(a and b and a['status']=='measured' and b['status']=='measured'
                and a['score']['pooled']['score']-b['score']['pooled']['score'] >= -noise['score']))
        checks.append(outcomes[arm+'_renamed_reverse']['status']=='passed')
        recommendation[arm] = dict(recommended=all(checks), checks=checks,
            reason='Both reflected pilot aggregates and rename/order identity must pass; no recipe revision after failure')
    write_json(args.out/'robustness.json',dict(outcomes=outcomes,recommendation=recommendation,
        caveat='Repeated nuisance tests on reused embryos; X reflection is part of E05 and is not independent confirmation'))


def package(args):
    from .portable import build
    from .evaluation import one, summarize
    rows = pilot_rows(args)
    robust = read_json(args.out/'robustness.json') if (args.out/'robustness.json').exists() else {'recommendation':{}}
    selected = [a for a in candidates(args) if robust['recommendation'].get(a,{}).get('recommended',False)]
    destination = args.out/'packages'
    records = []
    # Public controls are always delivered. Candidate slots with failed gates remain empty.
    for arm in ['B0','B1',*selected]:
        record = build(args, arm, arm_modules(args, arm), destination)
        image_dir = args.out/'package_pilots/test'
        image_dir.mkdir(parents=True,exist_ok=True)
        for row in rows:
            link = image_dir/(row['dataset']+'.zarr')
            if not link.exists():
                link.symlink_to(row['path'],target_is_directory=True)
        root = args.out/'package_validation'/arm
        env = os.environ.copy()
        env.update(PUBLIC946_OUTPUT=str(root), PUBLIC946_TEST_DIR=str(image_dir),
                   PUBLIC946_SUBMISSION=str(root/'submission.csv'),PUBLIC946_ARTIFACTS=str(args.artifacts),
                   PUBLIC946_GPU_GIB=str(read_json(args.out/'preflight.json')['limits']['gpu_allocated_gib_max']),
                   PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
        # Execute precisely the code cell in the actual packaged notebook through nbclient.
        # A dedicated kernel uses the same existing interpreter, without installing anything.
        job = destination/(arm+'_notebook_test.py')
        test_source = '''import json,os,sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
class RuntimeKernel(KernelSpecManager):
    def get_kernel_spec(self, name):
        from jupyter_client.kernelspec import KernelSpec
        return KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Pinned public946 runtime',language='python')
nb=nbformat.read(sys.argv[1],as_version=4)
client=NotebookClient(nb,timeout=43200,kernel_name='python3',resources={'metadata':{'path':str(Path(sys.argv[2]))}})
client.km=client.create_kernel_manager()
client.km.kernel_spec_manager=RuntimeKernel()
client.execute()
nbformat.write(nb,sys.argv[3])
'''
        job.write_text(test_source)
        root.mkdir(parents=True,exist_ok=True)
        prior_test=root/'package_test_receipt.json'
        if prior_test.exists() and read_json(prior_test)['notebook_sha256']==record['notebook_sha256'] and read_json(prior_test)['returncode']==0:
            test_record=read_json(prior_test)
        else:
            start=time.perf_counter()
            with (root/'notebook.log').open('a') as log:
                result=subprocess.run([str(args.runtime),str(job),record['notebook'],str(root),str(root/'executed.ipynb')],
                                      env=env,stdout=log,stderr=subprocess.STDOUT)
            test_record=dict(returncode=result.returncode,seconds=time.perf_counter()-start,
                             notebook_sha256=record['notebook_sha256'],completed_at=now())
            write_json(prior_test,test_record)
        record.update(notebook_returncode=test_record['returncode'],notebook_test_seconds=test_record['seconds'],
                      tested_pilots=[r['dataset'] for r in rows],full_cohort_notebook_executed=False,
                      full_cohort_scientific_worker_executed=(args.out/'scores/full'/arm/'summary.json').exists())
        if test_record['returncode']==0:
            comparisons=[]
            scoring=[]
            for row in rows:
                artifact=root/'predictions'/row['dataset']/'final.npz'
                a=read_json(root/'predictions'/row['dataset']/'complete.json')
                reference=args.out/'predictions/full'/arm/row['dataset']/'complete.json'
                if not reference.exists():
                    reference=args.out/'predictions/aa1'/arm/row['dataset']/'complete.json'
                b=read_json(reference)
                if sha(artifact)!=a['final_sha256']:
                    raise ValueError('Packaged prediction artifact drift')
                comparisons.append(a['graph_hash']==b['graph_hash'])
                scoring.append(one(row,artifact,args.out/'scores/package'/arm/(row['dataset']+'.json')))
            record.update(pilot_graph_identity=comparisons,pilot_score=summarize(scoring,[r['dataset'] for r in rows]),
                          ready_for_manual_test=all(comparisons),submission_sha256=sha(root/'submission.csv'))
        else:
            record.update(ready_for_manual_test=False,blocker='Actual packaged notebook failed; see retained notebook.log')
        records.append(record)
        write_json(destination/'manifest.json',records)
    (destination/'MANUAL_KAGGLE.md').write_text('''# Manual Kaggle execution

Create a notebook from the chosen .ipynb. Add the Biohub competition input and:

- pilkwang/biohub-tracking-support-pack-50ep-v1
- pilkwang/biohub-temporal-unet3d-seed314159-v1
- pilkwang/biohub-deepcenter-unet3d-center-prior-v1

Select a GPU, disable Internet, and run all cells. The exact public weights are hash checked.
Review manifest.json for actual validation scope, failures, runtime, source hashes and recipe.
The notebook writes /kaggle/working/submission.csv. Use Kaggle's manual save/run/submit controls
only if desired. This study performed no upload or submission.

Candidate order is the eligible B0-family notebook, then the eligible B1 transfer. Missing
slots remain empty. B0/B1 controls are separate references. Freeze notebook versions before
any new leaderboard feedback and do not revise knobs from those results. Record notebook
version, evaluator context, completion date and the actual returned leaderboard score.

The public v29 source is attributed to flexonafft; support code and models to pilkwang.
Retain original package LICENSE files and source attribution. Public availability alone is
not a new license grant. No clean embryo-disjoint validation was available in this study.

Offline local replay: use the existing notebook interpreter and run the .py export with
PUBLIC946_TEST_DIR, PUBLIC946_OUTPUT, PUBLIC946_SUBMISSION and PUBLIC946_ARTIFACTS set to
explicit image-only/public-artifact locations. No installation or model download is needed
in the audited local runtime. A fresh output directory produces a fresh pass.
''')
    # Human-reviewable scientific source beside compressed, self-contained notebook cells.
    from .neural import predictor_source
    native=(args.out/'public_source/tracking_repo/scripts/predict_unet_transformer.py').read_text()
    b0=predictor_source(native,[])
    (destination/'shared_runtime_changes.diff').write_text(''.join(difflib.unified_diff(native.splitlines(True),b0.splitlines(True),fromfile='materialized-v29',tofile='instrumented-B0')))
    for record in records:
        variant=predictor_source(native,record['modules'])
        diff=''.join(difflib.unified_diff(b0.splitlines(True),variant.splitlines(True),fromfile='B0',tofile=record['arm']))
        diff+='\nRegistered runtime mechanism switches: '+repr(record['modules'])+'\n'
        (destination/(record['arm']+'_scientific.diff')).write_text(diff)
    for name in ('modules.py','neural.py','worker.py','source.py'):
        (destination/name).write_text((REPO/'tools/public946_minimal'/name).read_text())
    report(args)


def report(args):
    destination=REPO/'results/public946-minimal-generalization-v1'
    destination.mkdir(parents=True,exist_ok=True)
    rows=read_json(args.out/'image_inventory.json')
    matrix=[]
    embryo_rows=[]
    for arm, registration in ARMS.items():
        summary_path=args.out/'scores/full'/arm/'summary.json'
        decision_path=args.out/'decisions'/(arm+'.json')
        execution_path=args.out/'execution/full'/(arm+'.json')
        summary=read_json(summary_path) if summary_path.exists() else None
        decision=read_json(decision_path) if decision_path.exists() else {}
        execution=read_json(execution_path) if execution_path.exists() else {}
        rec=dict(arm=arm,parent=registration['parent'],modules='+'.join(registration['modules']),
            status=decision.get('status','measured_control' if summary else 'not_yet_executed'),
            eligible=decision.get('eligible'),expected_clips=199,completed_clips=len(execution.get('completed',[])),
            failed_clips=len(execution.get('failed',[])),score=summary['pooled']['score'] if summary else None,
            original_export_score=summary['original_export']['score'] if summary else None,
            delta_B0=None,delta_B1=None,seconds=execution.get('seconds'),
            reason='; '.join(decision.get('reasons',[])) or decision.get('reason'),
            edge_tp=None,edge_fp=None,edge_fn=None,division_tp=None,division_fp=None,division_fn=None,division_jaccard=None)
        if summary:
            rec.update(summary['pooled']['counts'])
            rec['division_jaccard']=summary['pooled']['division_jaccard']
            for parent in ('B0','B1'):
                p=args.out/'scores/full'/parent/'summary.json'
                if p.exists():
                    rec['delta_'+parent]=rec['score']-read_json(p)['pooled']['score']
            for embryo,value in summary['per_embryo'].items():
                embryo_rows.append(dict(arm=arm,embryo=embryo,score=value['score'],division_jaccard=value['division_jaccard'],**value['counts']))
        matrix.append(rec)
    def csv_write(path,records,fields):
        with path.open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    csv_write(destination/'experiment_matrix.csv',matrix,list(matrix[0]))
    csv_write(destination/'per_embryo_scores.csv',embryo_rows,['arm','embryo','score','division_jaccard','edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes'])
    packages=read_json(args.out/'packages/manifest.json') if (args.out/'packages/manifest.json').exists() else []
    accounted=all(r['status']!='not_yet_executed' for r in matrix)
    scientifically_resolved=all(r['status'] in ('measured_control','measured','not_applicable','not_eligible','no_eligible_finalist') for r in matrix)
    novel_ready=any(r.get('ready_for_manual_test') and r['arm'] not in ('B0','B1') for r in packages)
    terminal=('ready_for_manual_lb_test' if novel_ready else 'completed_no_improvement') if scientifically_resolved and len(packages)>=2 and all(r.get('ready_for_manual_test') for r in packages) else 'partially_executed_with_named_blockers'
    status=dict(study='public946-minimal-generalization-v1',revision=2,status=terminal,
        all_registered_arms_accounted=accounted,measured_arms=[r['arm'] for r in matrix if r['score'] is not None],
        unrun_arms=[r['arm'] for r in matrix if r['status']=='not_yet_executed'],
        blockers=[dict(arm=r['arm'],status=r['status'],reason=r['reason']) for r in matrix if r['status'] in ('not_yet_executed','engineering_failures')],
        controls_only_is_complete=False,leaderboard_submission_performed=False,leaderboard_score=None,
        clean_validation_available=False,output_root=str(args.out),updated=now())
    write_json(destination/'status.json',status)
    historical={'B0':.911774,'B1':.934206}
    text=['# Expanded public946 study — measured execution\n',
          'Status: `'+terminal+'`. Full registered matrix: experiment_matrix.csv.\n',
          '## Known evidence versus this execution\n',
          'Public Harmonic Fusion v29 (script 347965685) was historically reported at 0.946 LB. This run made no submission and claims no new LB score. Historical all-199 controls: B0=0.911774, B1=0.934206. New numbers below are fresh official rematching of this run’s outputs.\n']
    for r in matrix:
        text.append('- '+r['arm']+': '+r['status']+(f"; score={r['score']:.12f}; delta B0={r['delta_B0']:+.12f}" if r['score'] is not None else '')+('; '+r['reason'] if r['reason'] else '')+'.\n')
    text += ['\n## Interpretation and limitations\n',
        'Both supplied embryos and overlapping clips have prior checkpoint/study exposure. These paired local measurements are exploratory and are not clean OOF or unseen-embryo validation. A local score above 0.95 is not a stopping rule or evidence of 0.95+ LB. No local delta is added to the historical public score.\n',
        'E03/E06 source-resolved no-op evidence, E01 proposal counts/possible B1 identity, failed methods, per-embryo regressions and conditional exclusions are retained in the JSON receipts. Missing scores are null, never zero.\n',
        'All inference uses the original public model/source and fixed settings. B0 explicitly retains motion. Mandatory bounds clipping is identical in every arm, applied after natural ties-to-even rounding. Original serialized outputs are scored separately; no old half-tie lookup corrections are used.\n',
        'The local CUDA/runtime/source/input/model fingerprints are frozen in execution_lock.json. Pilot score comparisons do not select recipes; full eligibility/ranking and the three combinations are fixed in the handover.\n',
        '\n## Reproduction and artifacts\n',
        'Raw outputs: `'+str(args.out)+'`. Predictions, dense/native evidence, original notebooks, models and submissions remain ignored and outside Git.\n',
        'Resume: `bash scripts/run_public946_minimal.sh run --resume --out '+str(args.out)+'`. Individual stages: `preflight`, `audit`, `pilot`, `controls`, `singles --arm E04`, `combinations`, `transfers`, `robustness`, `package`, `report`.\n',
        'Synthetic checks: `PYTHONNOUSERSITE=1 /kaggle/envs/cell-tracking-notebooks/bin/python -m unittest discover -s tests -p "test_public946_minimal*.py" -v`.\n',
        'Standalone offline notebooks, readable exports, exact public dependencies and manual Kaggle instructions: `'+str(args.out/'packages')+'`. Each manifest states actual fresh notebook test scope; worker full-cohort tests and actual notebook pilot tests are distinguished.\n']
    (destination/'REPORT.md').write_text('\n'.join(text))
    (destination/'START_HERE.md').write_text('# Public946 expanded revision 2\n\nRead [REPORT.md](REPORT.md), [status.json](status.json), [experiment_matrix.csv](experiment_matrix.csv), and [NEXT_AGENT.md](NEXT_AGENT.md).\n\nExecution status: `'+terminal+'`. No automatic Kaggle submission.\n')
    (destination/'NEXT_AGENT.md').write_text('# Continue exact registered execution\n\nRun `bash scripts/run_public946_minimal.sh run --resume --out '+str(args.out)+'`. Review the current status and per-arm failures first. Do not edit the handover registry or retune recipes; retain correctness failures and rerun affected stages after any documented engineering fix.\n\nOriginal scope includes all eight singles, conditional C01/C02/C03, and locked X01/X02. B0/B1 alone is incomplete. Earlier studies and preexisting workspace changes must remain untouched. Do not submit automatically.\n\nUnrun arms at report time: '+', '.join(status['unrun_arms'])+'.\n')
    for name in ('plan_freeze.json','execution_lock.json','applicability.json','source_audit.json','metric_identity.json','preflight.json',
                 'numerical_noise.json','finalist_lock.json','robustness.json'):
        p=args.out/name
        if p.exists():
            write_json(destination/name,read_json(p))
    write_json(destination/'packaging_receipts.json',packages)
    write_json(destination/'arm_decisions.json',{p.stem:read_json(p) for p in sorted((args.out/'decisions').glob('*.json'))})
    print('Wrote report:',destination,flush=True)
