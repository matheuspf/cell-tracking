"""Run the original independent graph-repair code in isolated clip shards."""
import csv
import ast
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .common import DATA,OUT,digest,graph_hash,now,read_json,sha,write_json


def adapter(root,names):
    full=OUT/'public_harmonic_full';root.mkdir(parents=True,exist_ok=True)
    original=(full/'harmonic_isolated.py').read_text()
    text=original.replace(str(full),str(root))
    images=root/'inputs/test';images.mkdir(parents=True,exist_ok=True)
    for name in names:
        p=images/(name+'.zarr')
        if not p.exists():p.symlink_to(DATA/'train'/(name+'.zarr'),target_is_directory=True)
    start='start_time = time.time()\navailable_gpu_count = _torch.cuda.device_count()'
    end='predict_seconds = time.time() - start_time'
    if text.count(start)!=1 or text.count(end)!=1:raise ValueError('Original neural execution boundary drift')
    a=text.index(start);b=text.index(end)+len(end)
    replacement=f'''_study_frozen_predictions = Path({str(full/'tracking_repo/predictions')!r})
_study_prediction_link = REPO_DIR / "predictions"
if _study_prediction_link.exists():
    raise RuntimeError("Fresh repair shard unexpectedly contains predictions")
_study_prediction_link.symlink_to(_study_frozen_predictions, target_is_directory=True)
_study_retention_lines = []
for _study_log in Path({str(full)!r}).glob("retention_guard_*.jsonl"):
    for _study_line in _study_log.read_text().splitlines():
        if _study_line.strip() and json.loads(_study_line)["dataset"] in test_stems:
            _study_retention_lines.append(_study_line)
(WORKING_DIR / "retention_guard_reused.jsonl").write_text("\\n".join(_study_retention_lines) + "\\n")
predict_seconds = 0.0  # Neural inference already completed; no repeated inference here.
print("Reusing frozen neural GEFF graphs; running the original repair functions unchanged.")
'''
    text=text[:a]+replacement+text[b:]
    anchor='    geffs = sorted((REPO_DIR / "predictions").glob(f"*/{METHOD}/split_0/*.geff"))'
    if text.count(anchor)!=1:raise ValueError('Original export collection boundary drift')
    text=text.replace(anchor,anchor+'\n    geffs = [p for p in geffs if p.stem in set(test_stems)]')
    original_functions={n.name:ast.get_source_segment(original,n) for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
    new_functions={n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    start=list(original_functions).index('graph_from_geff');repair_names=list(original_functions)[start:]
    repair_names.remove('write_test_submission')
    if any(original_functions[n]!=new_functions[n] for n in repair_names):raise ValueError('Original graph-repair function changed')
    path=root/'repair_isolated.py';compile(text,str(path),'exec');path.write_text(text)
    write_json(root/'repair_config.json',dict(created=now(),expected_samples=names,adapter_sha256=sha(path),
               original_adapter_sha256=sha(full/'harmonic_isolated.py'),
               unchanged_repair_function_sha256={n:digest(original_functions[n]) for n in repair_names},
               changes=['Output/input-view paths relocated','Completed neural inference replaced by read-only GEFF links','Export graph list restricted to shard clips','Original neural retention diagnostics subset reused'],
               graph_repair_functions_modified=False))
    return path


def execute(root,names):
    if (root/'complete.json').exists():
        receipt=read_json(root/'complete.json')
        if receipt['expected_samples']!=names or sha(root/'submission.csv')!=receipt['csv_sha256']:
            raise ValueError('Completed repair shard drift')
        return receipt
    path=adapter(root,names);start=time.perf_counter()
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',BIOHUB_ALLOW_PIP_INSTALL='0',
             BIOHUB_VALIDATOR_ENABLE='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',POLARS_MAX_THREADS='2',PYTHONUNBUFFERED='1')
    with (root/'run.log').open('w') as f:
        subprocess.run([sys.executable,str(path)],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=12*3600)
    receipt=dict(created=now(),expected_samples=names,seconds=time.perf_counter()-start,csv_sha256=sha(root/'submission.csv'))
    write_json(root/'complete.json',receipt)
    return receipt


def parity(old,new):
    columns=['node_id','t','z','y','x'];edges=['source_id','target_id'];rows=[]
    for name,g in old.groupby('dataset'):
        h=new[new.dataset==name]
        a=graph_hash(g[g.row_type=='node'][columns].to_numpy('int64'),g[g.row_type=='edge'][edges].to_numpy('int64'))
        b=graph_hash(h[h.row_type=='node'][columns].to_numpy('int64'),h[h.row_type=='edge'][edges].to_numpy('int64'))
        rows.append(dict(dataset=name,exact_graph_parity=a==b,original_hash=a,parallel_hash=b))
    if not all(r['exact_graph_parity'] for r in rows):raise ValueError('Parallel repair changed original pilot graphs')
    return rows


def pilot(args):
    names=read_json(OUT/'public_harmonic_pilot/adapter_manifest.json')['expected_samples']
    root=OUT/'public_repair_pilot';execute(root,names)
    rows=parity(pd.read_csv(OUT/'public_harmonic_pilot/submission.csv'),pd.read_csv(root/'submission.csv'))
    write_json(OUT/'public_repair_pilot_parity.json',dict(created=now(),passed=True,rows=rows))
    print('Original no-hook pilot and repair-only pilot graphs agree exactly.',flush=True)


def run(args):
    if not read_json(OUT/'public_repair_pilot_parity.json')['passed']:raise ValueError('Original pilot parity required before parallel repair')
    full=OUT/'public_harmonic_full';manifest=read_json(full/'adapter_manifest.json')
    if manifest['status']=='running':raise ValueError('Stop the owned serial repair process after pilot parity before starting shards')
    names=read_json(OUT/'fold_manifest.json')['expected_samples']
    geffs=list((full/'tracking_repo/predictions').glob('*/unet_transformer/split_0/*.geff'))
    if {p.stem for p in geffs}!=set(names):raise ValueError('Frozen neural graph sample set is incomplete')
    hashes={str(p.relative_to(full)):sha(p) for g in geffs for p in g.rglob('*') if p.is_file()}
    write_json(full/'serial_neural_and_partial_repair_manifest.json',manifest,immutable=True)
    progress={**manifest,'status':'graph_repair_running','repair_scheduling':'four isolated independent clip shards'}
    write_json(full/'adapter_manifest.json',progress)
    start=time.perf_counter();shards=[(OUT/f'public_repair_shard_{i}',names[i::4]) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts=list(pool.map(lambda item:execute(*item),shards))
    split=OUT/'public_repair_clip_csv';split.mkdir(exist_ok=True)
    columns=None;seen=set()
    for root,expected in shards:
        handle=None;current=None
        try:
            with (root/'submission.csv').open(newline='') as f:
                reader=csv.DictReader(f);columns=reader.fieldnames
                for row in reader:
                    name=row['dataset']
                    if name!=current:
                        if handle:handle.close()
                        if name in seen:raise ValueError('Duplicate repair dataset')
                        seen.add(name);current=name;handle=(split/(name+'.csv')).open('w',newline='');writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
                    writer.writerow(row)
        finally:
            if handle:handle.close()
    if seen!=set(names):raise ValueError('Repair shards skipped a dataset')
    target=full/'submission.parallel.tmp.csv';row_id=0
    with target.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for name in names:
            with (split/(name+'.csv')).open(newline='') as source:
                for row in csv.DictReader(source):
                    row['id']=row_id;writer.writerow(row);row_id+=1
    # Verify both original pilot graphs before replacing the owned partial CSV.
    pilot_frame=pd.read_csv(OUT/'public_harmonic_pilot/submission.csv')
    candidates=pd.concat([pd.read_csv(split/(n+'.csv')) for n in pilot_frame.dataset.unique()],ignore_index=True)
    pilot_parity=parity(pilot_frame,candidates)
    for p,h in hashes.items():
        if sha(full/p)!=h:raise ValueError('Read-only neural GEFF graph changed during repair')
    if (full/'submission.csv').exists():
        (full/'submission.csv').rename(full/'serial_partial_submission.csv')
    target.rename(full/'submission.csv')
    pd.concat([pd.read_csv(root/'run_stats.csv') for root,_ in shards],ignore_index=True).sort_values('dataset').to_csv(full/'run_stats.csv',index=False)
    write_json(OUT/'public_parallel_repair_receipt.json',dict(created=now(),complete=True,shards=receipts,samples=len(names),rows=row_id,
               seconds=time.perf_counter()-start,pilot_parity=pilot_parity,neural_graph_hashes=hashes,neural_graphs_unchanged=True,
               changed_original_functions=False,outer_annotation_outcomes_accessed=False))
    manifest.update(status='inference_complete',original_serial_returncode=manifest.get('returncode'),returncode=0,
                    completion='Original neural inference followed by four independent original graph-repair shards',
                    original_unchanged=sha(manifest['source_path'])==manifest['source_sha256'],
                    parallel_repair_receipt='../public_parallel_repair_receipt.json')
    manifest['deviations'].append('Graph repair/export scheduled in four isolated clip shards after exact original-pilot parity; original functions/settings unchanged')
    write_json(full/'adapter_manifest.json',manifest)
    refresh_runtime()
    print(f'Completed original public graph repair for all {len(names)} samples, exact pilot parity passed.',flush=True)


def refresh_runtime():
    """Separate interrupted serial duration from completed end-to-end wall time."""
    full=OUT/'public_harmonic_full';manifest=read_json(full/'adapter_manifest.json')
    if manifest['status']!='inference_complete':raise ValueError('Public graph completion is required')
    original=read_json(full/'serial_neural_and_partial_repair_manifest.json')
    repair=read_json(OUT/'public_parallel_repair_receipt.json')
    manifest.update(serial_seconds_before_reschedule=original['seconds'],parallel_repair_seconds=repair['seconds'],
                    seconds=(datetime.fromisoformat(repair['created'])-datetime.fromisoformat(original['created'])).total_seconds(),
                    runtime_note='End-to-end elapsed span, including the preserved serial repair attempt and parallel repair; shard run_stats prediction time is zero because neural GEFFs were reused')
    write_json(full/'adapter_manifest.json',manifest)
