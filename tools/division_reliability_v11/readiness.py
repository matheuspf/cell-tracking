"""Measured, source-only readiness and immutable scientific execution lock."""
from pathlib import Path
import json
import subprocess
import statistics
import importlib.metadata
from .common import REPO,ARCH,CONFIG,RESULTS,WORK,BRANCH,Blocked,read,write,sha,now

CORE=('common.py','guard.py','data.py','deterministic.py','upstream.py','graphs.py',
      'preprocess.py','resources.py','train_upstream.py','provenance.py')


def failures():
    reasons=[]
    for source in ('44b6','6bba'):
        path=RESULTS/f'pilot-{source}-20260918.json'
        if not path.exists() or read(path).get('status')!='real_optimizer_resume_passed':
            reasons.append(f'{source}: real batch-eight optimizer/resume proof missing')
        witnesses=list((WORK/'source_evaluation'/source/'20260918').glob('*/receipt.json'))
        if not witnesses or not any(read(p).get('positive_witness') and read(p).get('negative_witness') for p in witnesses):
            reasons.append(f'{source}: positive and false-fork complete-source witnesses missing')
        for sub in ('event_profile','mixed_profile','deployment_profile'):
            p=WORK/sub/source/'20260918/receipt.json'
            if not p.exists() or read(p).get('status')!='passed':reasons.append(f'{source}: {sub} missing')
    p=RESULTS/'production_resume.json'
    if not p.exists() or read(p).get('status')!='passed':reasons.append('Exact production sampler/optimizer resume proof missing')
    return reasons


def evidence():
    timing=[];event=[];deployment=[];crops=[];upstream_inference=[]
    for source in ('44b6','6bba'):
        for seed in (20260918,314159):
            folder=WORK/'resume_proof'/source/str(seed)/'continuous'
            rows=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
            if len(rows)!=20:raise Blocked('Incomplete upstream cost fixture')
            timing.append(dict(source=source,seed=seed,updates=len(rows),
                mean_gpu_lease_seconds=statistics.mean(r['gpu_lease_seconds'] for r in rows),
                maximum_gpu_lease_seconds=max(r['gpu_lease_seconds'] for r in rows),
                mean_input_seconds=statistics.mean(r['io_seconds'] for r in rows),
                history_sha256=sha(folder/'history.jsonl')))
        m=read(WORK/'mixed_profile'/source/'20260918/receipt.json')
        event.append(dict(source=source,rows=[{k:v for k,v in r.items() if k!='selection'} for r in m['rows']],
            receipt_sha256=sha(WORK/'mixed_profile'/source/'20260918/receipt.json'),
            encoder_recomputed_with_gradients=True,production_fit=False))
        c=read(WORK/'event_profile'/source/'20260918/receipt.json')
        if not all(r['pixels_exact'] and r['masks_exact'] for r in c['crop_parity']):raise Blocked('Raw crop parity failed')
        crops.append(dict(source=source,checks=c['crop_parity']))
        d=read(WORK/'deployment_profile'/source/'20260918/receipt.json')
        if d['profiled_complete_parent_groups']<4000:raise Blocked('Production head-batch cost profile missing')
        projected=d['all_node_embedding_gpu_seconds']+d['sampled_head_gpu_seconds']*d['total_deployed_parents']/d['profiled_complete_parent_groups']
        deployment.append(dict(source=source,all_nodes=d['all_nodes'],all_frames=d['all_frames'],
            sampled_complete_groups=d['profiled_complete_parent_groups'],projected_full_clip_gpu_seconds=projected,
            wall_seconds_all_embeddings_and_sampled_heads=d['wall_seconds'],
            measured_gpu_seconds=d['total_measured_gpu_seconds'],full_policy_wall_time_measured=False,
            embedding_parity=read(WORK/'deployment_profile'/source/'20260918/progress.json')['embedding_parity']))
        for p in (WORK/'source_inference-overfit'/source/'20260918').glob('*/receipt.json'):
            x=read(p);upstream_inference.append(dict(source=source,clip=x['clip'],frames=x['frames'],
                gpu_lease_seconds=x['gpu_lease_seconds'],wall_seconds=x['wall_seconds'],nodes=x['nodes']))
    write(RESULTS/'event_profile.json',dict(status='passed',mixed_batch_profiles=event,deployment=deployment))
    write(RESULTS/'crop_parity.json',dict(status='passed',sources=crops))
    return timing,event,deployment,upstream_inference


def lock():
    path=RESULTS/'execution_lock.json'
    if path.exists():require_production('lock');return read(path)
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!=BRANCH:raise Blocked('Incorrect execution branch')
    missing=failures()
    if missing:raise Blocked('; '.join(missing))
    subprocess.run(['python','handover/division-reliability-v11/check_plan.py'],cwd=REPO,check=True)
    config=read(CONFIG);timing,event,deployment,inference=evidence()
    up_rate=max(r['mean_gpu_lease_seconds'] for r in timing)
    event_rate=max(r['gpu_lease_seconds'] for e in event for r in e['rows'])
    policy_rate=max(r['projected_full_clip_gpu_seconds'] for r in deployment)
    baseline_rate=max(r['gpu_lease_seconds'] for r in inference)
    split=read(WORK/'source_partitions.json')
    fit_clips=sum(len(v['fit']) for v in split.values())*2
    calibration_clips=sum(len(v['calibration']) for v in split.values())*2
    target_clips=sum(config['expected_clip_counts'].values())*2
    cold_clips=2*4*3
    multiplier=config['budget']['projection_multiplier'];alloc=config['budget']['initial_allocations_hours']
    infer_h=multiplier*(baseline_rate*(fit_clips+calibration_clips+target_clips+cold_clips)+
                       policy_rate*(calibration_clips+target_clips+8))/3600
    candidates=[]
    for u in config['upstream_update_options']:
        for e in config['event_update_options']:
            uh=multiplier*4*u*up_rate/3600
            eh=multiplier*(4*e*event_rate+fit_clips*policy_rate)/3600
            candidates.append(dict(U=u,E=e,upstream_hours=uh,event_mining_hours=eh,inference_cold_hours=infer_h,
                feasible=uh<=alloc['upstream'] and eh<=alloc['event_mining'] and infer_h<=alloc['inference_cold']))
    selected=next((c for c in candidates if c['feasible']),None)
    projection=dict(status='passed' if selected else 'blocked',rates_seconds=dict(upstream_update=up_rate,event_update=event_rate,
        c00_clip=baseline_rate,compact_policy_clip=policy_rate),measurement=timing,source_inference=inference,
        population=dict(source_fit_clips=fit_clips,source_calibration_clips=calibration_clips,target_clips=target_clips,cold_clips=cold_clips),
        multiplier=multiplier,allocation_hours=alloc,candidates=candidates,selected=selected,
        caveat='Throughput projection from real source fixtures; model density can change. Runtime accounting enforces the ceilings.',
        allocation_rule='Keep initial allocations; no qualified upstream reuse releases capacity.')
    write(RESULTS/'allocation_projection.json',projection)
    if selected is None:raise Blocked('No registered minimum U/E pair fits measured costs and fixed reserve')
    from .features import NAMES,IDENTITY_NAMES
    from .actions import PROPOSAL
    from dataclasses import asdict
    from .provenance import digest
    root=Path(__file__).parent
    external={str(p.relative_to(REPO)):sha(p) for d in ('pipeline_error_training','strong_tracker_v3','annotation_selection')
              for p in (REPO/'tools'/d).glob('*.py')}
    docs={str(p.relative_to(REPO)):sha(p) for p in (REPO/'handover/division-reliability-v11').glob('*') if p.is_file()}
    for name in ('IMPLEMENTATION.md','VALIDATION.md'):
        p=REPO/'handover/clean-validation-division-v10'/name;docs[str(p.relative_to(REPO))]=sha(p)
    value=dict(status='locked',locked_utc=now(),branch=BRANCH,planning_commit='150aef2751b319ab3a4d1d877ae95cc543c2c56a',
        checkout_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),study_sha256=sha(CONFIG),
        upstream_updates=selected['U'],event_updates=selected['E'],allocation_hours=alloc,
        schedule=[dict(source=d['source'],target=d['target'],seed=s) for d in config['directions'] for s in config['seeds']],
        arms=config['arms'],upstream_reused_cells=[],all_components_random_initialization=True,
        source_partitions_sha256=sha(WORK/'source_partitions.json'),
        source_cache_manifest_sha256={s:sha(WORK/'preprocessed'/s/'manifest.json') for s in split},
        implementation_sha256={name:sha(root/name) for name in CORE},
        event_implementation_snapshot_sha256={p.name:sha(p) for p in root.glob('*.py') if p.name not in CORE},
        external_code_sha256=external,reference_file_sha256=docs,
        architecture_sha256={n:sha(ARCH/'src/biohub_tracking/models'/n) for n in ('temporal_unet.py','simple_node_transformer.py')},
        versions={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','zarr','polars','tracksdata','psutil')},
        metric_revision=config['metric_revision'],action_schema=list(NAMES),identity_schema=list(IDENTITY_NAMES),
        proposal_config=asdict(PROPOSAL),augmentation='same-image multiplicative intensity Uniform(0.9,1.1), clip [0,1], preserve zero padding; MSE against detached original encoder output',
        positive_event_unit='source GEFF persisted division parent ID, shared across timing anchors and overlapping clips',
        upstream_query_adapter='trilinear features at native coordinates; fixed four-axis sinusoidal coordinates; predicted supported queries use 7-um cardinality-first matching',
        limits=config['budget'],durable_write_buffer_gib=2,allocation_projection_sha256=sha(RESULTS/'allocation_projection.json'),
        exposure_class='source_isolated_reused_embryos',target_scores_opened=False,
        cold_selection='two clips per target chosen by median and highest image-only local-max density; lexical tie break; every arm and seed',
        implementation_repairs='Record named failures and recipe-preserving repairs; no target-selected change; immutable upstream numerical closure.')
    value['identity']=digest(value);write(path,value,immutable=True)
    write(WORK/'resources/allocation.json',dict(identity=value['identity'],hours=alloc),immutable=True)
    return value


def require_production(stage):
    path=RESULTS/'execution_lock.json'
    if not path.exists() or read(path).get('status')!='locked':raise Blocked(f'{stage}: no production lock')
    value=read(path)
    from .provenance import digest
    if value['identity']!=digest({k:v for k,v in value.items() if k!='identity'}):raise Blocked('Execution lock identity changed')
    for name,expected in value['implementation_sha256'].items():
        if sha(Path(__file__).parent/name)!=expected:raise Blocked('Locked upstream implementation changed: '+name)
    for name,expected in value['architecture_sha256'].items():
        if sha(ARCH/'src/biohub_tracking/models'/name)!=expected:raise Blocked('Architecture source changed')
    for name,expected in value['external_code_sha256'].items():
        if sha(REPO/name)!=expected:raise Blocked('Inherited implementation changed: '+name)
