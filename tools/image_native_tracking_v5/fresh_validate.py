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


def proposed_export():
    from .report import csv_rows,REPLICAS,DETERMINISTIC
    scores=csv_rows('ablation_scores.csv');by={(r['variant'],r['embryo']):r for r in scores}
    def passes(v):
        return all((v,e) in by for e in BASE) and by[v,'pooled']['score']>BASE['pooled'] and all(by[v,e]['score']>=BASE[e]-1e-8 for e in ['44b6','6bba'])
    eligible=[v for v,r in REPLICAS.items() if passes(v) and passes(r)]+[v for v in DETERMINISTIC if passes(v)]
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
        if variant!='C0':
            assert model['model_executed'] and model['full_frame_proposals_executed']
            if variant in ['N_backbone_J','P_union_N_J','P_DC_N_J']:assert model['native_image_encoder_executed']
            if variant.startswith('H_'):assert model['hoct_backbone_executed']
            if variant.startswith('P_DC'):assert model['deepcenter_full_frames_executed']
        records.append(dict(dataset=name,embryo=row['embryo'],unfamiliar_name=alias,variant=variant,source=selection['source'],
            full_shape=row['image_shape'],density=selection['density'],density_quantile=selection['quantile'],
            exact_graph_parity=exact,official_count_parity=numeric,passed=exact or numeric,csv_roundtrip=True,
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


def run(wait=False):
    plan=freeze();expected=set(read(OUT/'execution_protocol.json')['variants'])
    from .report import csv_rows
    while True:
        scores=csv_rows('ablation_scores.csv');complete={r['variant'] for r in scores if r['embryo']=='pooled'}
        fits=list((OUT/'models/native').glob('*.json'));fits=[p for p in fits if not read(p).get('tiny')]
        ready=complete==expected and len(fits)==8 and (OUT/'hoct_queue_complete.json').exists()
        if ready:break
        if not wait:raise RuntimeError('Complete model and graph comparisons are still pending')
        time.sleep(30)
    if wait:os.execv(sys.executable,[sys.executable,'-m','image_native_tracking_v5.fresh_validate'])
    from .package import build
    selected=proposed_export();package=OUT/'inference_package'
    if not package.exists():package=build(selected)
    assert read(package/'winning_config.json')['variant']==selected
    guard=guard_selftest(package);results=[];node_rows=[]
    for item in plan['clips']:
        view=OUT/'inference_trials/final_images'/item['unfamiliar_name'];view.mkdir(parents=True,exist_ok=True)
        link=view/(item['unfamiliar_name']+'.zarr')
        if not link.exists():link.symlink_to(DATA/'train'/(item['dataset']+'.zarr'),target_is_directory=True)
        output=OUT/'inference_trials/final'/item['unfamiliar_name']
        if not (output/'inference_receipt.json').exists():
            if output.exists():raise RuntimeError('Preserve partial fresh output; select a new attempt namespace: '+str(output))
            command=[str(package/'run.sh'),'--python',sys.executable,'--images',str(view),'--output',str(output),
                '--v1',str(V1),'--v2',str(V2),'--source-model',item['source'],'--variant',selected]
            for variant in plan['variants']:
                if variant!=selected:command+=['--also-variant',variant]
            with gpu_aux(),(OUT/'logs'/f"fresh_final_{item['unfamiliar_name']}.log").open('w') as log:
                subprocess.run(command,cwd='/tmp',check=True,stdout=log,stderr=subprocess.STDOUT)
        with cpu_batch():r,n=compare(package,output,item)
        results.extend(r);node_rows.extend(n)
        write(OUT/'fresh_validation_progress.json',dict(at=now(),clips=len(results)//len(PRIMARY),rows=results))
        print('Fresh complete clip',item['dataset'],'passed',sum(v['passed'] for v in r),'/',len(r),flush=True)
    # Execute the actual disable switch in the final package as a separate child.
    item=plan['clips'][0];fallback=OUT/'inference_trials/final_C0_disabled'
    if not (fallback/'inference_receipt.json').exists():
        if fallback.exists():raise RuntimeError('Preserve partial final fallback output: '+str(fallback))
        view=OUT/'inference_trials/final_images'/item['unfamiliar_name']
        with gpu_aux(),(OUT/'logs/fresh_final_disabled.log').open('w') as log:
            subprocess.run([str(package/'run.sh'),'--python',sys.executable,'--images',str(view),'--output',str(fallback),
                '--v1',str(V1),'--v2',str(V2),'--source-model',item['source'],'--disable-new-heads'],
                cwd='/tmp',check=True,stdout=log,stderr=subprocess.STDOUT)
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
        clips=6,clip_variant_runs=len(results),verified_variants=verified,rows=results,guard_selftest=guard,
        plan_sha256=sha(OUT/'fresh_validation_plan.json'),package_manifest_sha256=sha(package/'manifest.json'),
        selected=selected,C0_disablement=disabled,
        labels_visible_to_inference=False,cached_graphs_visible_to_inference=False,unknown_filename_test=True,
        node_only_scope='six-clips, explicit subset diagnostic',independent_biological_validation=False,kaggle_runtime_verified=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    print(freeze()) if a.freeze_only else run(a.wait)
