"""Evaluation-side orchestration; each image-only child is guarded before imports.

The parent can read frozen graphs/GT only to compare completed child outputs. None
of these evaluation paths or arrays is passed to the package process.
"""
import csv,os,subprocess,sys,time
import pandas as pd
from .common import *

PRIMARY=['C0','J_native_frozen','H_general_J','H_probe_J','H_probe_native_J',
    'N_head_J','N_backbone_J','P_union_native_J','P_union_N_J','P_DC_native_J','P_DC_N_J','P_image_ablation']
NODE_DIAGNOSTICS=['P_union_native_J','P_union_N_J','P_DC_native_J','P_DC_N_J','P_image_ablation']


def freeze():
    path=OUT/'fresh_validation_plan.json'
    if path.exists():return read(path)
    density=read(OUT/'density_selection.json')['rows'];selected=[]
    for embryo in ['44b6','6bba']:
        rows=sorted([r for r in density if r['embryo']==embryo],key=lambda r:(r['density'],r['dataset']))
        for quantile in [.1,.5,.9]:
            rank=round((len(rows)-1)*quantile);row=rows[rank]
            selected.append(dict(**row,rank=rank,quantile=quantile,source='6bba' if embryo=='44b6' else '44b6',
                unfamiliar_name=f'volume_{len(selected)+1:02d}'))
    plan=dict(frozen=now(),clips=selected,variants=PRIMARY,
        selection='image-derived C0 density ranks 10%, 50%, 90% within each embryo; no GT or variant score selection',
        node_only_controls=NODE_DIAGNOSTICS,node_only_scope='six full clips only; not another 199-clip configuration',
        parity='exact graph preferred; otherwise fresh official integer counts and score equality within 1e-12 are required',
        decoder='frozen five-frame MILP; wall-clock limits can make graph identity depend on machine scheduling',
        package_guard='process-start Python audit hooks; no OS namespace claim')
    write(path,plan);return plan


def proposed_export(verified):
    from .report import csv_rows,REPLICAS,DETERMINISTIC
    scores=csv_rows('ablation_scores.csv');by={(r['variant'],r['embryo']):r for r in scores}
    def passes(v):
        return all((v,e) in by for e in BASE) and by[v,'pooled']['score']>BASE['pooled'] and all(by[v,e]['score']>=BASE[e]-1e-8 for e in ['44b6','6bba'])
    eligible=[v for v,r in REPLICAS.items() if passes(v) and passes(r) and v in verified]+[v for v in DETERMINISTIC if passes(v) and v in verified]
    return max(eligible,key=lambda v:by[v,'pooled']['score']) if eligible else 'C0'


def csv_graphs(path):
    rows=pd.read_csv(path);expected=['id','dataset','row_type','node_id','t','z','y','x','source_id','target_id']
    assert list(rows.columns)==expected
    assert np.array_equal(rows.id.to_numpy(),np.arange(len(rows))) and set(rows.row_type)<={'node','edge'}
    result={}
    for name in sorted(set(rows.dataset)):
        sub=rows[rows.dataset==name]
        result[name]=dict(nodes=sub[sub.row_type=='node'][['node_id','t','z','y','x']].to_numpy(np.int64),
            edges=sub[sub.row_type=='edge'][['source_id','target_id']].to_numpy(np.int64))
    return result


def compare(package,output,selection):
    from strong_tracker_v3.common import graph_hash,validate
    from annotation_selection.metric_adapter import evaluate_graph
    name=selection['dataset'];alias=selection['unfamiliar_name'];row=next(r for r in inventory() if r['dataset']==name)
    receipt=read(output/'inference_receipt.json');audits=[read(p) for p in (output/'read_audit').glob('*.json')]
    assert len(audits)>=2 and all(r['installed_before_numerical'] and not r['blocked_events'] for r in audits)
    assert any(r['categories'].get('image',0)>0 for r in audits)
    assert receipt['guard_installed_before_numerical'] and receipt['raw_images_read']
    assert all(receipt[k]==0 for k in ['cached_final_graph_reads','study_feature_cache_reads','source_labels_read'])
    manifest_hash=sha(package/'manifest.json');records=[];node_rows=[]
    gt=None
    for variant in receipt['variants']:
        actual=arrays(output/'predictions'/variant/f'{alias}.npz');expected=graph(name,variant)
        csvname='submission.csv' if variant==receipt['selected'] else f'submission_{variant}.csv'
        parsed=csv_graphs(output/csvname);assert set(parsed)=={alias}
        assert all(np.array_equal(actual[k],parsed[alias][k]) for k in ['nodes','edges'])
        validate(actual['nodes'],actual['edges'],row['image_shape'])
        exact=graph_hash(**actual)==graph_hash(**expected)
        if gt is None:gt=arrays(V1/'evaluation/gt'/f'{name}.npz')
        fresh,_,_=evaluate_graph(name,actual['nodes'],actual['edges'],gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
        reference=read(OUT/'evaluation'/variant/f'{name}.json')
        fields=['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes','matched_nodes']
        numeric=all(fresh[k]==reference[k] for k in fields)
        model=next(r for r in receipt['records'] if r['variant']==variant and r['dataset']==alias)
        input_parity={}
        if variant!='C0':
            assert model['model_executed'] and model['full_frame_proposals_executed']
            if variant in ['N_backbone_J','P_union_N_J','P_DC_N_J']:assert model['native_image_encoder_executed']
            if variant.startswith('H_'):assert model['hoct_backbone_executed']
            if variant.startswith('P_DC'):assert model['deepcenter_full_frames_executed']
            from .predict_batch import VARIANTS
            from .inference_fingerprints import inputs as input_fingerprints
            source=selection['source'];cfg=VARIANTS[variant];family=cfg['family'];seed=cfg['seed']
            c=arrays(OUT/'observations'/f'{name}.npz')
            bank=arrays(OUT/'banks'/source/('P0' if cfg['pop']=='P0' else 'P1')/f'{name}.npz')
            joint=read(OUT/'calibration'/f'{source}_J.json');cal=None;ms={}
            if family in ['N1','N2']:
                ms=arrays(OUT/'model_scores'/f'{family}_{seed}'/f'{name}.npz');cal=read(OUT/'calibration'/f'{source}_{family}_{seed}.json')['calibration']
            elif family.startswith('H'):
                ms=arrays(OUT/'model_scores/H'/f'{name}.npz');cal=read(OUT/'calibration'/f'{source}_H_{seed}.json')['models'][family]
            dc=arrays(OUT/'deepcenter'/f'{name}.npz')['confidence'] if cfg['pop']=='PDC' else None
            expected_inputs=input_fingerprints(c,bank,ms,cfg,joint,cal,dc)
            input_parity={k:model['input_fingerprints'].get(k)==h for k,h in expected_inputs.items()}
        records.append(dict(dataset=name,embryo=row['embryo'],unfamiliar_name=alias,variant=variant,source=selection['source'],
            full_shape=row['image_shape'],density=selection['density'],density_quantile=selection['quantile'],
            exact_graph_parity=exact,official_count_parity=numeric,passed=exact or numeric,csv_roundtrip=True,
            exact_predecoder_inputs=all(input_parity.values()) if input_parity else None,input_field_parity=input_parity,
            fresh_graph_hash=graph_hash(**actual),batch_graph_hash=graph_hash(**expected),
            fresh_counts={k:fresh[k] for k in fields},batch_counts={k:reference[k] for k in fields},
            package_manifest_sha256=manifest_hash,inference_receipt_sha256=sha(output/'inference_receipt.json'),
            inference_seconds=receipt['seconds'],clip_variant_cumulative_seconds=model['clip_elapsed_seconds'],
            max_parent_gpu_gib=receipt['peak_gpu_gib'],max_parent_rss_gib=receipt['rss_gib'],audit_processes=len(audits),
            guard_before_numerical=True,blocked_events=0,model_executed=model.get('model_executed',False)))
        if variant in NODE_DIAGNOSTICS:
            # Hold C0 edges fixed wherever both endpoints remain. Added nodes are
            # deliberately unlinked here; this isolates matching/count effects.
            base=graph(name);ids=set(map(int,actual['nodes'][:,0]));edges=np.asarray([e for e in base['edges'] if int(e[0]) in ids and int(e[1]) in ids],np.int64).reshape(-1,2)
            result,_,_=evaluate_graph(name,actual['nodes'],edges,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
            node_rows.append(dict(**result,variant=variant,embryo=row['embryo'],scope='six-clip fixed-C0-edge node-only diagnostic',
                inherited_edges_removed_for_absent_endpoint=len(base['edges'])-len(edges)))
    return records,node_rows


def guard_selftest(package):
    output=OUT/'inference_trials'/f'{package.name}_guard_test'
    if (output/'result.json').exists():return read(output/'result.json')
    output.mkdir(parents=True,exist_ok=True)
    target=V1/'evaluation/gt'/f"{inventory()[0]['dataset']}.npz"
    forbidden_cache=OUT/'observations'/f"{inventory()[0]['dataset']}.npz"
    script='''import json,os,socket,sys
from pathlib import Path
import sitecustomize
assert sitecustomize.INSTALLED_BEFORE_NUMERICAL
denied=[]
for path in sys.argv[1:3]:
    try:Path(path).read_bytes()
    except PermissionError:denied.append(path)
    else:raise AssertionError('Forbidden file was readable')
try:socket.getaddrinfo('example.com',443)
except PermissionError:denied.append('external DNS')
else:raise AssertionError('External DNS was permitted')
assert len(denied)==3
fresh=Path(os.environ['V5_FRESH_OUTPUT'])/'generated_prediction.geff'
fresh.mkdir(exist_ok=True);(fresh/'zarr.json').write_text('{}')
assert (fresh/'zarr.json').read_text()=='{}'
print(json.dumps(dict(passed=True,checks=4,guard_before_numerical=True,fresh_output_GEFF_roundtrip=True)))
'''
    env={**os.environ,'V5_PACKAGE_ROOT':str(package),'V5_FRESH_OUTPUT':str(output),'V5_AUDIT_DIR':str(output/'read_audit'),
        'V5_V1':str(V1),'V5_V2':str(V2),'PYTHONPATH':str(package/'tools'),'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'}
    result=subprocess.run([sys.executable,'-c',script,str(target),str(forbidden_cache)],env=env,cwd='/tmp',text=True,capture_output=True,check=True)
    r=json.loads(result.stdout);r.update(package_manifest_sha256=sha(package/'manifest.json'),
        scope='actual annotation, observation cache and DNS denial before numerical imports; Python audit hooks, not syscall sandbox')
    write(output/'result.json',r);return r


def image_child(package,item,output,options,label):
    view=OUT/'inference_trials/validation_images'/item['unfamiliar_name'];view.mkdir(parents=True,exist_ok=True)
    link=view/(item['unfamiliar_name']+'.zarr')
    if not link.exists():link.symlink_to(DATA/'train'/(item['dataset']+'.zarr'),target_is_directory=True)
    if (output/'inference_receipt.json').exists():return
    if output.exists():raise RuntimeError('Preserve partial fresh output; select a new attempt namespace: '+str(output))
    command=[str(package/'run.sh'),'--python',sys.executable,'--images',str(view),'--output',str(output),
        '--v1',str(V1),'--v2',str(V2),'--source-model',item['source'],*options]
    with gpu_aux(),(OUT/'logs'/f'{label}.log').open('w') as log:
        subprocess.run(command,cwd='/tmp',check=True,stdout=log,stderr=subprocess.STDOUT)


def payload_compatibility(tested,final):
    """Reuse image execution only when its entire runtime/primary payload is exact."""
    a,b=read(tested/'manifest.json'),read(final/'manifest.json')
    for bundle,m in [(tested,a),(final,b)]:
        for name,h in m['files'].items():assert sha(bundle/name)==h,(bundle,name)
    compared=[]
    for name,h in a['files'].items():
        if name=='winning_config.json':continue
        assert b['files'].get(name)==h,('Retest all six clips after a payload change',name)
        compared.append(name)
    extras=sorted(set(b['files'])-set(a['files']))
    for name in extras:
        assert (name.startswith('models/native/') or name.startswith('models/hoct/') or name.startswith('calibration/')) and '_314159.' in name,('Unexpected untested file',name)
    for key in ['base_dependencies','external_model_configs']:
        assert a[key]==b[key],('External inference dependency changed',key)
    result=dict(at=now(),passed=True,tested_manifest_sha256=sha(tested/'manifest.json'),
        final_manifest_sha256=sha(final/'manifest.json'),identical_files=len(compared),
        identical_file_hashes={name:a['files'][name] for name in compared},extra_replica_files=extras,
        changed_configuration='winning_config.json only; separately executed with no variant override',
        scope='byte-identical runtime, primary weights, source calibrations, preprocessing, solver and external dependency pins')
    write(OUT/'fresh_payload_compatibility.json',result);return result


def run(wait=False):
    plan=freeze();expected=set(read(OUT/'execution_protocol.json')['variants'])
    from .report import csv_rows
    def complete_variants():return {r['variant'] for r in csv_rows('ablation_scores.csv') if r['embryo']=='pooled'}
    def complete_fits():return [p for p in (OUT/'models/native').glob('*.json') if not read(p).get('tiny')]
    def primary_ready():
        paths=[OUT/root/f'{source}_{stage}_20260910.json' for root in ['models/native','calibration']
            for source in ['44b6','6bba'] for stage in ['N1','N2']]
        return all(p.exists() for p in paths) and (OUT/'hoct_queue_complete.json').exists()
    while not primary_ready():
        if not wait:raise RuntimeError('Primary native models and source calibrations are still pending')
        time.sleep(30)
    if wait:os.execv(sys.executable,[sys.executable,'-m','image_native_tracking_v5.fresh_validate'])
    from .package import build
    tested=OUT/'inference_package_validation'
    if not tested.exists():tested=build('C0',tested.name)
    guard=guard_selftest(tested);results=[];node_rows=[]
    schedule=OUT/'fresh_validation_schedule.json'
    if not schedule.exists():write(schedule,dict(at=now(),plan_sha256=sha(OUT/'fresh_validation_plan.json'),
        tested_manifest_sha256=sha(tested/'manifest.json'),
        change='Run the frozen six primary/control image executions during replica training, before waiting for scored comparisons.',
        unchanged='All clips, density ranks, 12 variants, model recipes and acceptance criteria.',
        final_requirement='Byte-identical tested payload plus actual final default and disable-switch image executions.'))
    # Do not wait for target scores between image executions. The guarded children
    # receive only images, the explicit source identifier, and the frozen bundle.
    for i,item in enumerate(plan['clips']):
        options=['--variant','C0']
        for variant in plan['variants']:
            if variant!='C0':options+=['--also-variant',variant]
        output=OUT/'inference_trials/validation'/item['unfamiliar_name']
        image_child(tested,item,output,options,f"fresh_validation_{item['unfamiliar_name']}")
        write(OUT/'fresh_validation_progress.json',dict(at=now(),image_clips_executed=i+1,compared_clips=0,
            status='image execution; official comparison pending',tested_manifest_sha256=sha(tested/'manifest.json')))
        print('Fresh image execution complete',item['dataset'],flush=True)
    while not set(PRIMARY)<=complete_variants():time.sleep(30)
    for i,item in enumerate(plan['clips']):
        output=OUT/'inference_trials/validation'/item['unfamiliar_name']
        with cpu_batch():r,n=compare(tested,output,item)
        results.extend(r);node_rows.extend(n)
        write(OUT/'fresh_validation_progress.json',dict(at=now(),image_clips_executed=6,compared_clips=i+1,rows=results))
        print('Fresh complete clip',item['dataset'],'passed',sum(v['passed'] for v in r),'/',len(r),flush=True)
    verified=[v for v in PRIMARY if len([r for r in results if r['variant']==v and r['passed']])==6]
    assert 'C0' in verified,'C0 must retain exact/count parity in every fresh clip'
    # A completed negative reproducibility test is retained, never relaxed into a
    # pass. Only pipelines meeting the original parity criterion can be exported.
    write(OUT/'fresh_primary_comparison.json',dict(at=now(),completed=True,all_passed=all(r['passed'] for r in results),
        verified_variants=verified,failed_parity_variants=[v for v in PRIMARY if v not in verified],rows=results,
        failed_parity_policy='retain failed measurements and exclude that pipeline from promotion; do not change acceptance tolerances'))
    while complete_variants()!=expected or len(complete_fits())!=8:time.sleep(30)
    selected=proposed_export(verified);package=OUT/'inference_package'
    if not package.exists():package=build(selected)
    assert read(package/'winning_config.json')['variant']==selected
    compatibility=payload_compatibility(tested,package);final_guard=guard_selftest(package)
    # No --variant override: this tests the actual selected default configuration.
    item=plan['clips'][0];default=OUT/'inference_trials/final_default'
    image_child(package,item,default,[],'fresh_final_default')
    with cpu_batch():default_rows,_=compare(package,default,item)
    assert len(default_rows)==1 and default_rows[0]['variant']==selected and default_rows[0]['passed']
    # Execute the real disable switch on the other input embryo/source direction.
    item=plan['clips'][3];fallback=OUT/'inference_trials/final_C0_disabled'
    image_child(package,item,fallback,['--disable-new-heads'],'fresh_final_disabled')
    with cpu_batch():disabled,_=compare(package,fallback,item)
    assert len(disabled)==1 and disabled[0]['variant']=='C0' and disabled[0]['exact_graph_parity']
    verified=[v for v in PRIMARY if len([r for r in results if r['variant']==v and r['passed']])==6]
    pd.DataFrame(node_rows).to_csv(OUT/'fresh_node_only_rows.csv',index=False)
    from annotation_selection.metric_adapter import aggregate
    summaries=[]
    for v in NODE_DIAGNOSTICS:
        for em in ['44b6','6bba','pooled']:
            sub=[r for r in node_rows if r['variant']==v and (em=='pooled' or r['embryo']==em)]
            a=aggregate(sub,[r['dataset'] for r in sub])
            c=[read(OUT/'evaluation/C0'/f"{r['dataset']}.json") for r in sub];b=aggregate(c,[r['dataset'] for r in c])
            summaries.append(dict(variant=v,embryo=em,samples=len(sub),score=a['score'],delta_subset_C0=a['score']-b['score'],
                **{**{k:val for k,val in a.items() if k not in ['score','counts']},**a['counts']},scope='six full clips; not a 199-clip arm'))
    pd.DataFrame(summaries).to_csv(OUT/'fresh_node_only_scores.csv',index=False)
    write(OUT/'fresh_validation.json',dict(at=now(),passed=all(r['passed'] for r in results) and guard['passed'],
        completed=True,export_passed=selected in verified and default_rows[0]['passed'] and disabled[0]['exact_graph_parity'] and guard['passed'] and final_guard['passed'] and compatibility['passed'],
        failed_parity_variants=[v for v in PRIMARY if v not in verified],
        clips=6,clip_variant_runs=len(results),verified_variants=verified,rows=results,guard_selftest=guard,final_guard_selftest=final_guard,
        plan_sha256=sha(OUT/'fresh_validation_plan.json'),package_manifest_sha256=sha(package/'manifest.json'),
        tested_package_manifest_sha256=sha(tested/'manifest.json'),payload_compatibility=compatibility,
        execution_scope='72 primary/control image executions of the byte-identical validation payload, plus two actual final-package executions',
        selected=selected,final_default=default_rows,C0_disablement=disabled,
        labels_visible_to_inference=False,cached_graphs_visible_to_inference=False,unknown_filename_test=True,
        node_only_scope='six-clips, explicit subset diagnostic',independent_biological_validation=False,kaggle_runtime_verified=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    print(freeze()) if a.freeze_only else run(a.wait)
