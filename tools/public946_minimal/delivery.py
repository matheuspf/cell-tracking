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


def validate_csv(path,rows,prediction_root):
    """Validate the actual exported CSV against complete, validated graph files."""
    import itertools
    import numpy as np
    columns=['id','dataset','row_type','node_id','t','z','y','x','source_id','target_id']
    expected={r['dataset'] for r in rows}
    seen=set();index=0
    with Path(path).open(newline='') as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=columns:
            raise ValueError('Actual submission CSV schema differs from the public contract')
        for name,group in itertools.groupby(reader,key=lambda r:r['dataset']):
            if name not in expected or name in seen:
                raise ValueError('Unexpected or repeated CSV dataset group')
            seen.add(name);nodes=[];edges=[]
            for row in group:
                if None in row or int(row['id'])!=index:
                    raise ValueError('Malformed CSV row or nonunique global row ID')
                index+=1
                if row['row_type']=='node':
                    if (int(row['source_id']),int(row['target_id']))!=(-1,-1):
                        raise ValueError('Invalid node-row sentinels')
                    nodes.append([int(row[k]) for k in ('node_id','t','z','y','x')])
                elif row['row_type']=='edge':
                    if any(int(row[k])!=-1 for k in ('node_id','t','z','y','x')):
                        raise ValueError('Invalid edge-row sentinels')
                    edges.append([int(row[k]) for k in ('source_id','target_id')])
                else:
                    raise ValueError('Unknown CSV row type')
            graph=load_arrays(Path(prediction_root)/name/'final.npz')
            if not np.array_equal(np.asarray(nodes,np.int64).reshape(-1,5),graph['nodes']) or not np.array_equal(
                    np.asarray(edges,np.int64).reshape(-1,2),graph['edges']):
                raise ValueError('CSV content differs from the validated graph: '+name)
    if seen!=expected:
        raise ValueError('Missing expected full clip from actual CSV')
    return dict(valid=True,rows=index,datasets=sorted(seen),sha256=sha(path),
        validation='Exact schema, integer fields, sentinels, unique sequential row IDs, complete expected population, and exact node/edge identity to validated graph files')


def export_fresh_cohort(args,arm,record,rows):
    """Use the actual packaged export block after the all-model fresh workers."""
    root=args.out/'full_fresh_exports'/arm
    root.mkdir(parents=True,exist_ok=True)
    link=root/'predictions'
    prediction_root=args.out/'predictions/fresh_finalists'/arm
    if not link.exists():
        link.symlink_to(prediction_root,target_is_directory=True)
    driver=Path(record['python']).read_text()
    block=driver[driver.index("import numpy as np\ncolumns="):]
    first=read_json(prediction_root/rows[0]['dataset']/'job.json')
    target=root/'submission.csv'
    namespace=dict(csv=csv,rows=rows,ROOT=root,SUBMISSION=target,write_json=write_json,sha=sha,
        ARM=arm,MODULES=record['modules'],archive=Path(first['archive_source']),model_hashes=first['model_hashes'])
    exec(compile(block,'<actual-packaged-CSV-export>','exec'),namespace)
    result=validate_csv(target,rows,prediction_root)
    result.update(path=str(target),packaged_export_block_sha256=__import__('hashlib').sha256(block.encode()).hexdigest(),
        actual_full_notebook_invocation=False,
        scope='Full fresh image workers plus the actual packaged CSV export block; actual complete notebook invocation is separately tested on all four full pilots')
    write_json(root/'validation.json',result)
    return result


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
    fresh_receipts = {}
    full_rows = read_json(args.out/'image_inventory.json')
    for arm in selected:
        # The primary study permits legal neural replay for graph interventions.
        # A packaging finalist receives an additional full fresh pass with a new
        # DeepCenter cache, followed by the actual notebook's four-pilot test.
        for row in full_rows:
            cache = args.out/'fresh_finalist_cache'/arm/row['dataset']
            cache.mkdir(parents=True, exist_ok=True)
            link = cache/'input_hashes'
            if not link.exists():
                link.symlink_to(args.out/'input_hashes', target_is_directory=True)
        _, failures = execute(args, arm, full_rows, 'fresh_finalists', modules=arm_modules(args,arm))
        matches = {}
        for row in full_rows:
            reference = args.out/'predictions/full'/arm/row['dataset']/'complete.json'
            path = args.out/'predictions/fresh_finalists'/arm/row['dataset']/'complete.json'
            if path.exists():
                before, after = read_json(reference), read_json(path)
                matches[row['dataset']] = before['graph_hash']==after['graph_hash'] and after['neural_fresh']
        fresh_receipts[arm] = dict(expected_clips=len(full_rows), graph_identity=matches, failed=failures,
            passed=not failures and len(matches)==len(full_rows) and all(matches.values()),
            scope='Both tracking models and DeepCenter recomputed from public models and original images; isolated new heatmap cache')
    write_json(args.out/'fresh_finalist_receipts.json',fresh_receipts)
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
                      full_cohort_scientific_worker_executed=(args.out/'scores/full'/arm/'summary.json').exists(),
                      full_cohort_fresh_confirmation=fresh_receipts.get(arm))
        if test_record['returncode']==0:
            record['actual_csv_validation']=validate_csv(root/'submission.csv',rows,root/'predictions')
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
                          ready_for_manual_test=all(comparisons) and (arm in ('B0','B1') or fresh_receipts[arm]['passed']),
                          submission_sha256=sha(root/'submission.csv'))
        else:
            record.update(ready_for_manual_test=False,blocker='Actual packaged notebook failed; see retained notebook.log')
        if arm in selected and fresh_receipts[arm]['passed']:
            record['full_cohort_fresh_image_to_csv']=export_fresh_cohort(args,arm,record,full_rows)
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
    from .measurements import collect, append_report
    destination=REPO/'results/public946-minimal-generalization-v1'
    destination.mkdir(parents=True,exist_ok=True)
    rows=read_json(args.out/'image_inventory.json')
    measurements,cost=collect(args,destination)
    matrix=[]
    embryo_rows=[]
    for arm, registration in ARMS.items():
        summary_path=args.out/'scores/full'/arm/'summary.json'
        decision_path=args.out/'decisions'/(arm+'.json')
        execution_path=args.out/'execution/full'/(arm+'.json')
        summary=read_json(summary_path) if summary_path.exists() else None
        decision=read_json(decision_path) if decision_path.exists() else {}
        execution=read_json(execution_path) if execution_path.exists() else {}
        measured=measurements.get(arm,{})
        totals=measured.get('totals',{})
        modules=arm_modules(args,arm) if not arm.startswith('X') or (args.out/'finalist_lock.json').exists() and read_json(args.out/'finalist_lock.json')['recipes'][arm] is not None else []
        rec=dict(arm=arm,parent=registration['parent'],modules='+'.join(modules),
            status=decision.get('status','measured_control' if summary else 'not_yet_executed'),
            eligible=decision.get('eligible'),expected_clips=199,completed_clips=len(execution.get('completed',measured.get('completed',[]))),
            failed_clips=len(execution.get('failed',[])),score=summary['pooled']['score'] if summary else None,
            original_export_score=summary['original_export']['score'] if summary else None,
            delta_B0=None,delta_B1=None,min_embryo_delta_parent=None,
            seconds=execution.get('seconds',totals.get('worker_seconds')),
            reason='; '.join(decision.get('reasons',[])) or decision.get('reason'),
            edge_tp=None,edge_fp=None,edge_fn=None,division_tp=None,division_fp=None,division_fn=None,division_jaccard=None,
            adj_edge_jaccard=None,num_pred_nodes=None,original_tp_survived=None,original_tp_lost=None,recovered_tp=None,
            fresh_tracking_model_clips=totals.get('fresh_tracking_model_clips'),replayed_tracking_model_clips=totals.get('replayed_tracking_model_clips'),
            allocated_gpu_peak_bytes=totals.get('gpu_allocated_peak_bytes'),rss_peak_bytes=totals.get('rss_peak_bytes'),
            prediction_output_bytes=totals.get('prediction_output_bytes'),
            nodes_added_exact=totals.get('nodes_added_exact'),nodes_removed_exact=totals.get('nodes_removed_exact'),
            edges_added_exact=totals.get('edges_added_exact'),edges_removed_exact=totals.get('edges_removed_exact'))
        if summary:
            rec.update(summary['pooled']['counts'])
            rec['division_jaccard']=summary['pooled']['division_jaccard']
            rec['adj_edge_jaccard']=summary['pooled']['adj_edge_jaccard']
            for k in ('original_tp_survived','original_tp_lost','recovered_tp'):
                rec[k]=summary.get(k)
            for parent in ('B0','B1'):
                p=args.out/'scores/full'/parent/'summary.json'
                if p.exists():
                    rec['delta_'+parent]=rec['score']-read_json(p)['pooled']['score']
            deltas=[]
            for embryo,value in summary['per_embryo'].items():
                er=dict(arm=arm,embryo=embryo,score=value['score'],division_jaccard=value['division_jaccard'],
                        adj_edge_jaccard=value['adj_edge_jaccard'],original_export_score=summary.get('original_export_per_embryo',{}).get(embryo,{}).get('score'),**value['counts'])
                for parent in ('B0','B1'):
                    p=args.out/'scores/full'/parent/'summary.json'
                    er['delta_'+parent]=value['score']-read_json(p)['per_embryo'][embryo]['score'] if p.exists() else None
                own_parent='B1' if arm.startswith('X') else 'B0'
                if er['delta_'+own_parent] is not None:
                    deltas.append(er['delta_'+own_parent])
                embryo_rows.append(er)
            rec['min_embryo_delta_parent']=min(deltas) if deltas else None
        matrix.append(rec)
    def csv_write(path,records,fields):
        with path.open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    csv_write(destination/'experiment_matrix.csv',matrix,list(matrix[0]))
    csv_write(destination/'per_embryo_scores.csv',embryo_rows,['arm','embryo','score','original_export_score','delta_B0','delta_B1','division_jaccard','adj_edge_jaccard','edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes'])
    packages=read_json(args.out/'packages/manifest.json') if (args.out/'packages/manifest.json').exists() else []
    robust=read_json(args.out/'robustness.json') if (args.out/'robustness.json').exists() else {}
    outcomes=robust.get('outcomes',{})
    controls_portable=all(outcomes.get(arm+'_renamed_reverse',{}).get('status')=='passed' and
        all(outcomes.get(arm+'_'+transform,{}).get('status')=='measured' for transform in ('reflect_x','reflect_y'))
        for arm in ('B0','B1'))
    accounted=all(r['status']!='not_yet_executed' for r in matrix)
    scientifically_resolved=all(r['status'] in ('measured_control','measured','not_applicable','not_eligible','no_eligible_finalist') for r in matrix)
    novel_ready=any(r.get('ready_for_manual_test') and r['arm'] not in ('B0','B1') for r in packages)
    terminal=('ready_for_manual_lb_test' if novel_ready else 'completed_no_improvement') if scientifically_resolved and controls_portable and len(packages)>=2 and all(r.get('ready_for_manual_test') for r in packages) else 'partially_executed_with_named_blockers'
    status=dict(study='public946-minimal-generalization-v1',revision=2,status=terminal,
        all_registered_arms_accounted=accounted,measured_arms=[r['arm'] for r in matrix if r['score'] is not None],
        unrun_arms=[r['arm'] for r in matrix if r['status']=='not_yet_executed'],
        blockers=[dict(arm=r['arm'],status=r['status'],reason=r['reason']) for r in matrix if r['status'] in ('not_yet_executed','engineering_failures','measured_with_comparison_blocked')],
        control_portability_checks_complete=controls_portable,
        no_improvement_definition='No novel recipe passed every full-population, lineage, robustness and packaging gate; individual local gains remain reported.',
        controls_only_is_complete=False,leaderboard_submission_performed=False,leaderboard_score=None,
        clean_validation_available=False,output_root=str(args.out),updated=now())
    if not controls_portable:
        status['blockers'].append(dict(arm='B0/B1',status='portability_checks_pending_or_failed',reason='Both reflected pilot evaluations and exact renamed/reordered graphs must complete.'))
    status['blockers'] += [dict(arm=p['arm'],status='package_not_ready',reason=p.get('blocker','Notebook graph parity or full fresh finalist confirmation failed.')) for p in packages if not p.get('ready_for_manual_test')]
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
        'All inference uses the original public model/source and fixed settings. B0 explicitly retains motion. Mandatory bounds clipping is identical in every arm, applied after natural ties-to-even rounding. The original CSV lower clamp is reconstructed and scored separately from complete bounds sanitation; no old half-tie lookup corrections are used.\n',
        'The local CUDA/runtime/source/input/model fingerprints are frozen in execution_lock.json. Pilot score comparisons do not select recipes; full eligibility/ranking and the three combinations are fixed in the handover.\n',
        '\n## Reproduction and artifacts\n',
        'Raw outputs: `'+str(args.out)+'`. Predictions, dense/native evidence, original notebooks, models and submissions remain ignored and outside Git.\n',
        'Resume: `bash scripts/run_public946_minimal.sh run --resume --workers 4 --out '+str(args.out)+'`. Individual stages: `preflight`, `audit`, `pilot`, `controls`, `singles --arm E04`, `combinations`, `transfers`, `robustness`, `package`, `report`.\n',
        'Synthetic checks: `PYTHONNOUSERSITE=1 /kaggle/envs/cell-tracking-notebooks/bin/python -m unittest discover -s tests -p "test_public946_minimal*.py" -v`.\n',
        'Standalone offline notebooks, readable exports, exact public dependencies and manual Kaggle instructions: `'+str(args.out/'packages')+'`. Each manifest states actual fresh notebook test scope; worker full-cohort tests and actual notebook pilot tests are distinguished.\n']
    (destination/'REPORT.md').write_text('\n'.join(text))
    (destination/'START_HERE.md').write_text('# Public946 expanded revision 2\n\nRead [REPORT.md](REPORT.md), [status.json](status.json), [experiment_matrix.csv](experiment_matrix.csv), and [NEXT_AGENT.md](NEXT_AGENT.md).\n\nExecution status: `'+terminal+'`. No automatic Kaggle submission.\n')
    next_action=('The registered local study is complete. Review the measured report and package manifest; a new manual Kaggle test is the only way to obtain a new LB score. Do not reopen rejected recipes or fill empty finalist slots with unregistered changes.' if scientifically_resolved and terminal!='partially_executed_with_named_blockers' else 'Review named blockers and resume the exact registered matrix.')
    (destination/'NEXT_AGENT.md').write_text('# Exact study continuation and handoff\n\n'+next_action+'\n\nIdempotent verification/resume: `bash scripts/run_public946_minimal.sh run --resume --workers 4 --out '+str(args.out)+'`. Do not edit the handover registry or retune recipes; retain correctness failures and rerun affected stages after any documented engineering fix.\n\nOriginal scope includes all eight singles, conditional C01/C02/C03, and locked X01/X02. B0/B1 alone is incomplete. Earlier studies and preexisting workspace changes must remain untouched. Do not submit automatically.\n\nUnrun arms at report time: '+(', '.join(status['unrun_arms']) or 'none')+'.\n')
    for name in ('plan_freeze.json','execution_lock.json','applicability.json','source_audit.json','metric_identity.json','preflight.json',
                 'numerical_noise.json','finalist_lock.json','robustness.json','fresh_finalist_receipts.json',
                 'selection_code_lock.json','diagnostic_lock.json','scheduler_policy.json','scheduler_amendment_01.json',
                 'license_receipt.json','pilot_compute_projection.json','evaluation_code_lock.json','retention_policy.json'):
        p=args.out/name
        if p.exists():
            write_json(destination/name,read_json(p))
    write_json(destination/'packaging_receipts.json',packages)
    write_json(destination/'arm_decisions.json',{p.stem:read_json(p) for p in sorted((args.out/'decisions').glob('*.json'))})
    write_json(destination/'engineering_corrections.json',{
        str(p.parent.relative_to(args.out)):read_json(p) for p in (args.out/'engineering_failures').rglob('correction.json')})
    append_report(args,destination,measurements,cost)
    print('Wrote report:',destination,flush=True)
