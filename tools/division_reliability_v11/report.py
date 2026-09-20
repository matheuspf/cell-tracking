"""Sanitized completion accounting; missing matrix entries remain explicit nulls."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,os,shutil,sys
from .common import REPO,WORK,RESULTS,OLD,Blocked,read,write,sha,now
from .populations import clips

FIELDS=['arm','source','target','seed','dataset','status','missing_reason','clips_expected','clips_scored','frames',
        'edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes','estimated_total',
        'matched_nodes','edge_weight','adjusted_edge_contribution','edge_jaccard','adj_edge_jaccard','division_jaccard',
        'score','delta_C00','delta_C01','disabled_policy','accepted_actions','changed_edges','introduced_division_fp',
        'newly_recovered_division_tp','lost_division_tp','metric_revision']


def public(value):
    if isinstance(value,dict):return {k:public(v) for k,v in value.items()}
    if isinstance(value,list):return [public(v) for v in value]
    if isinstance(value,str):return value.replace(str(REPO),'<repo>').replace(str(Path.home()),'<home>')
    return value


def alive(pid):
    try:
        path=Path('/proc')/str(pid)
        return 'division_reliability_v11' in (path/'cmdline').read_bytes().decode() and (path/'stat').read_text().split()[2]!='Z'
    except (OSError,UnicodeError):return False


def aggregate_rows(rows,expected):
    """Restore undefined numerical values after JSON's explicit null encoding."""
    from annotation_selection.metric_adapter import aggregate
    fixed=[{**r,'gt_node_recall':float('nan') if r.get('gt_node_recall') is None else r['gt_node_recall']} for r in rows]
    return aggregate(fixed,expected)


def clip_summary(metrics,clip):
    from .evaluation import finite
    if sum(metrics[k] for k in ('edge_tp','edge_fp','edge_fn')):
        return finite(aggregate_rows([metrics],[clip]))
    from annotation_selection.metric_adapter import official
    import warnings
    raw=official.per_sample_metrics(official.EvaluationResult(*(int(metrics[k]) for k in official.COUNT_COLUMNS)),
        float(metrics['estimated_total']),float('nan') if metrics.get('gt_node_recall') is None else metrics['gt_node_recall'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');return finite(official.summarise([raw]))


def checkpoint(folder):
    path=folder/'resume.pt'
    if not path.exists():return None
    import torch,io,hashlib,subprocess
    # Atomic checkpoint replacement may occur during a live status refresh.
    # Read bytes, metadata and timing from one open inode so its hash and update
    # count always describe the same durable checkpoint.
    with path.open('rb') as stream:
        blob=stream.read();metadata=os.fstat(stream.fileno())
        born,stamp=subprocess.check_output(['stat','-L','--printf=%W|%w',f'/proc/self/fd/{stream.fileno()}'],
            text=True,pass_fds=(stream.fileno(),)).split('|',1)
    state=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
    result=dict(path=str(path.relative_to(REPO)),sha256=hashlib.sha256(blob).hexdigest(),durable_step=state['step'],source=state['source'],seed=state['seed'],
                lock_identity=state['lock_identity'],scheduler=state.get('scheduler'),optimizer_present='optimizer' in state,
                sampler_and_rng_present=all(k in state for k in ('sampler','cpu_rng','cuda_rng')))
    result['bytes']=metadata.st_size
    if born!='0' and '.' in stamp:
        birth_ns=int(born)*10**9+int(stamp.split('.',1)[1].split()[0].ljust(9,'0'))
        result['file_creation_to_last_write_seconds']=max(0.,(metadata.st_mtime_ns-birth_ns)/1e9)
        result['write_timing_scope']='Linux creation/mtime window of the latest atomic checkpoint; excludes state capture, close, directory sync and disk flush latency'
    del state
    return result


def training():
    cells=[]
    for source in ('44b6','6bba'):
        for seed in (20260918,314159):
            cell=dict(source=source,target='6bba' if source=='44b6' else '44b6',seed=seed)
            for kind in ('upstream','compact'):
                folder=WORK/'fits'/source/str(seed)/kind
                p=folder/'final.json';progress=read(folder/'progress.json') if (folder/'progress.json').exists() else dict(status='pending')
                result=read(p) if p.exists() else progress
                result={**result,'process_verified_alive':alive(progress.get('pid')),
                        'resume':checkpoint(folder),'history_summary':{}}
                hp=folder/'history.jsonl'
                if hp.exists():
                    count=0;sums=Counter();first=last=None;fallback=supervised=proposal=hard_fallback=0;visits=set();cache_peak=0;probabilities=[]
                    with hp.open() as f:
                        for line in f:
                            r=json.loads(line);count+=1;first=first or r;last=r
                            sums['gpu_lease_seconds']+=r.get('gpu_lease_seconds',0);sums['input_seconds']+=r.get('io_seconds',0)
                            for g in r.get('groups',[]):
                                fallback+=int(g['fallback']);supervised+=g['supervised'];proposal+=g['proposal']
                            for g in r.get('selection',[]):
                                visits.add((g['slot'],g['clip'],g['key']));hard_fallback+=int(g.get('hard_fallback',False))
                                probabilities.append(g['selection_probability'])
                            cache_peak=max(cache_peak,r.get('observation_cache_bytes',0))
                    keys=('step','loss','detection','association','losses','lr','denominators')
                    result['history_summary']=dict(recorded_updates=count,first={k:first[k] for k in keys if k in first},
                        last={k:last[k] for k in keys if k in last},totals=dict(sums),supervised_incoming_groups=supervised,
                        proposal_incoming_groups=proposal,query_fallback_batches=fallback,unique_group_visits=len(visits),
                        uniform_hard_slot_fallback_groups=hard_fallback,observation_cache_peak_bytes=cache_peak,
                        group_selection_probability_range=[min(probabilities),max(probabilities)] if probabilities else None,
                        history_sha256=sha(hp),full_history_committed=False)
                cell[kind]=result
            linear=WORK/'fits'/source/str(seed)/'linear/model.json'
            if linear.exists():
                r=read(linear);cell['linear']=dict(status=r['status'],census=r['census'],diagnostics=r['parameters']['diagnostics'],sha256=sha(linear))
            else:cell['linear']=dict(status='pending')
            cells.append(cell)
    value=dict(cells=cells,upstream_reused_cells=[],retained_final_only=True,all_components_random_initialization=True)
    write(RESULTS/'training_summary.json',public(value));return value


def resources():
    root=WORK/'resources';rows=[]
    if (root/'gpu-leases.jsonl').exists():
        with (root/'gpu-leases.jsonl').open() as f:rows=[json.loads(line) for line in f if line.strip()]
    prior=read(root/'prior_accounting.json');by_stage=Counter();status=Counter()
    for r in rows:by_stage[r['stage']]+=r['seconds'];status[r['status']]+=1
    upper=prior['conservative_gpu_lease_seconds'];total=sum(by_stage.values())+upper
    maxima={k:max((r.get(k,0) for r in rows),default=0) for k in ('total_device_peak_bytes','rss_peak_bytes','peak_reserved_bytes','peak_allocated_bytes')}
    minima={k:min((r[k] for r in rows if k in r),default=None) for k in ('device_free_min_bytes','host_available_min_bytes')}
    inherited=read(WORK/'preservation/inherited_compute.json')
    cpu_files=list((WORK/'controller/jobs').rglob('*.resources.json'))
    cpu=list({json.dumps(r,sort_keys=True):r for r in (read(p) for p in cpu_files)}.values())
    from statistics import median
    from math import ceil
    job_times=defaultdict(list);seen_jobs=set();job_status=defaultdict(Counter)
    for p in (WORK/'controller/jobs').rglob('*.json'):
        if p.name.endswith('.resources.json'):continue
        r=read(p)
        if 'wall_seconds' not in r:continue
        signature=tuple(r.get(k) for k in ('stage','source','seed','arm','clip','part','finished_utc','wall_seconds','status'))
        if signature in seen_jobs:continue
        seen_jobs.add(signature)
        label='/'.join(str(r[k]) for k in ('stage','arm','part') if r.get(k) is not None)
        job_times[label].append(r['wall_seconds']);job_status[label][r['status']]+=1
    job_summary={k:dict(executions=len(v),status_counts=dict(job_status[k]),sum_seconds=sum(v),
        minimum_seconds=min(v),median_seconds=median(v),p95_seconds=sorted(v)[ceil(.95*len(v))-1],maximum_seconds=max(v))
        for k,v in sorted(job_times.items())}
    bank_times=defaultdict(list)
    for p in (WORK/'banks').glob('*/*/*/*/receipt.json'):
        r=read(p);key='/'.join(p.relative_to(WORK/'banks').parts[:3])
        bank_times[key].append(r['wall_seconds'])
    bank_summary={k:dict(completed_clips=len(v),sum_seconds=sum(v),median_seconds=median(v),
        p95_seconds=sorted(v)[ceil(.95*len(v))-1],maximum_seconds=max(v)) for k,v in sorted(bank_times.items())}
    prediction_times=defaultdict(list)
    for p in (WORK/'predictions').glob('*/*/*/*/receipt.json'):
        r=read(p)
        if r.get('status')!='predicted_unscored':continue
        label='/'.join(p.relative_to(WORK/'predictions').parts[:3])
        prediction_times[label].append(r['wall_seconds'])
    prediction_summary={k:dict(completed_clips=len(v),sum_seconds=sum(v),median_seconds=median(v),
        p95_seconds=sorted(v)[ceil(.95*len(v))-1],maximum_seconds=max(v)) for k,v in sorted(prediction_times.items())}
    # Stable prediction receipts and the append-only lease journal survive a
    # controller's verified-cache reuse. Do not mistake the short reuse process
    # for the original optimizer or image-to-graph execution.
    from datetime import datetime
    locked=read(RESULTS/'execution_lock.json')['locked_utc'];optimizer_rows=defaultdict(list)
    for r in rows:
        if r['stage'] in ('upstream','event_mining') and not r.get('operation') and 'finished_utc' in r and r['started_utc']>=locked:
            optimizer_rows[(r['stage'],r['source'],r['seed'],r['pid'])].append(r)
    optimizer_windows=[]
    for (stage,source,seed,pid),group in sorted(optimizer_rows.items()):
        start=min(r['started_utc'] for r in group);end=max(r['finished_utc'] for r in group)
        optimizer_windows.append(dict(stage=stage,source=source,seed=seed,pid=pid,
            recorded_optimizer_leases=len(group),lease_status_counts=dict(Counter(r['status'] for r in group)),
            first_lease_started_utc=start,last_lease_finished_utc=end,
            observed_elapsed_seconds=(datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds(),
            gpu_lease_seconds=sum(r['seconds'] for r in group),process_verified_alive=alive(pid)))
    value=dict(new_study_gpu_lease_hours=total/3600,new_study_measured_journal_hours=sum(by_stage.values())/3600,
        pre_journal_conservative_charge_hours=upper/3600,pre_journal_accounting=prior,
        hours_by_stage={k:v/3600 for k,v in by_stage.items()},lease_count=len(rows),lease_status_counts=dict(status),
        maximum_lease_seconds=max((r['seconds'] for r in rows),default=0),sampled_maxima=maxima,sampled_minima=minima,
        current_durable_free_bytes=shutil.disk_usage(WORK).free,inherited_lifetime=inherited,
        cpu_job_resource_samples=len(cpu),cpu_job_resource_receipt_files=len(cpu_files),
        study_cpu_rss_peak_bytes=max((r.get('study_rss_peak_bytes',r['study_rss_bytes']) for r in cpu),default=None),
        completed_job_wall_time=job_summary,
        original_source_bank_wall_time=bank_summary,
        original_prediction_wall_time=prediction_summary,
        optimizer_execution_windows=optimizer_windows,
        optimizer_window_scope='First recorded optimizer lease through last closed optimizer lease in each production process. Includes inter-lease waiting; excludes startup, final serialization and time outside that interval. Active runs are partial. Lease counts include failures/replays and are not durable update counts.',
        prediction_timing_scope='One original retained prediction receipt per clip/arm, excluding verified-cache reuse. Source and target populations share each arm/source/seed key. Concurrent durations overlap; sums are not campaign duration.',
        controller_receipt_coverage_note='Some completed controller receipts were replaced by verified-cache reuse during the first pipeline restart. Their exact original process wall times are unavailable in the current job receipts. Original bank/prediction timings and all GPU lease charges remain available separately; optimizer windows are measured lower-bound intervals, not reconstructed process wall times.',
        source_bank_timing_scope='One original preparation per completed clip, excluding verified-cache reuse. Includes enumeration, crops, labels and full source scoring; sums overlap for concurrent clips.',
        job_wall_time_scope='Elapsed process time including input/output, CPU work and GPU waiting. Sums overlap for concurrent jobs and are not campaign duration. Distinct verified-cache reuse executions remain included; identical archived receipts are deduplicated. p95 uses nearest rank.',
        study_host_available_min_bytes=min((r.get('host_available_min_bytes',r['host_available_bytes']) for r in cpu),default=None),
        inherited_hours=inherited['charged_seconds']/3600,lifetime_v10_plus_v11_conservative_hours=(inherited['charged_seconds']+total)/3600,
        limits=read(RESULTS/'execution_lock.json')['limits'],reserve_hours=16,
        accounting_note='Earlier pilots use recorded wall-time upper bounds plus a 1200-second engineering allowance; subsequent leases include failures. Historical v10 bounds can include idle intervals.',
        timing_scope=dict(lease='Forward/backward/optimizer, parameter and optimizer-state transfers, and within-update proposal work while owning the GPU lease',
            input_io='Measured separately in optimizer histories and training_summary.json',
            checkpoint='Latest closed-file creation-to-last-write samples in training_summary.json; early overwritten checkpoints are not retrospectively timed',
            limitation='No separate CUDA-kernel-versus-transfer timing was collected for every production update; lease hours are not pure GPU kernel hours'),
        hardware=dict(gpu='NVIDIA RTX 4090',gpu_memory_gib=24,cpu='AMD Ryzen 9 7950X3D'),updated_utc=now())
    write(RESULTS/'resource.json',public(value));return value


def score_rows():
    from .evaluation import finite
    per_clip=[];directions=[];pooled=[];details={};calibrations=[]
    blockers=[read(p) for p in (WORK/'blocked_cells').glob('*.json')]
    revision=read(RESULTS/'execution_lock.json')['metric_revision']
    for source in ('44b6','6bba'):
        target='6bba' if source=='44b6' else '44b6'
        for seed in (20260918,314159):
            for arm in ('C00','C01','C11'):
                expected=clips(source,'target');rows=[];disabled=False if arm=='C00' else None
                causes=[b['stage']+': '+b['reason'] for b in blockers if b['source']==source and b['seed']==seed and arm in b['arms']]
                missing='; '.join(causes) if causes else 'complete frozen prediction and official score pending'
                calpath=WORK/'fits'/source/str(seed)/'calibration'/arm/'calibration.json'
                if calpath.exists():
                    cal=read(calpath);disabled=cal['disabled_policy'];calibrations.append({k:v for k,v in cal.items() if k not in ('parent_manifests',)})
                for clip in expected:
                    path=WORK/'evaluation'/source/str(seed)/arm/clip/'receipt.json'
                    result=dict(arm=arm,source=source,target=target,seed=seed,dataset=clip,status='blocked' if causes else 'pending',missing_reason=missing,
                                clips_expected=1,clips_scored=0,frames=None,disabled_policy=disabled,metric_revision=revision)
                    if path.exists():
                        r=read(path);details[source,seed,arm,clip]=r;metrics=r['metrics'];rows.append(metrics)
                        single=clip_summary(metrics,clip);w=sum(metrics[k] for k in ('edge_tp','edge_fp','edge_fn'))
                        result.update(metrics);result.update(status='scored',missing_reason=None,clips_scored=1,frames=r['frames'],
                            score=single['score'],division_jaccard=single['division_jaccard'],edge_weight=w,
                            adjusted_edge_contribution=w*metrics['adj_edge_jaccard'] if w else 0.)
                        if not w:result['missing_reason']='Official per-clip combined score is undefined: no evaluable edge denominator; counts remain in the full population'
                        prediction=read(WORK/'predictions'/arm/source/str(seed)/clip/'receipt.json')
                        result.update(accepted_actions=prediction.get('accepted_actions',0),changed_edges=prediction.get('changed_edges',0))
                    per_clip.append(result)
                base=dict(arm=arm,source=source,target=target,seed=seed,status='blocked' if causes else 'pending',missing_reason=missing,
                          clips_expected=len(expected),clips_scored=len(rows),frames=None,disabled_policy=disabled,metric_revision=revision)
                if len(rows)==len(expected):
                    summary=finite(aggregate_rows(rows,expected));base.update(summary);base.update(summary['counts']);base.pop('counts',None)
                    base.update(status='scored',missing_reason=None,frames=100*len(expected),estimated_total=sum(r['estimated_total'] for r in rows),
                                matched_nodes=sum(r['matched_nodes'] for r in rows))
                    w=sum(base[k] for k in ('edge_tp','edge_fp','edge_fn'));base['edge_weight']=w;base['adjusted_edge_contribution']=w*base['adj_edge_jaccard']
                directions.append(base)
    for seed in (20260918,314159):
        for arm in ('C00','C01','C11'):
            selected=[r for r in per_clip if r['arm']==arm and r['seed']==seed];known=[r for r in selected if r['status']=='scored']
            row=dict(arm=arm,source='pooled',target='pooled',seed=seed,status='pending',missing_reason='incomplete direction/seed matrix',
                clips_expected=len(selected),clips_scored=len(known),frames=None,
                disabled_policy=None if any(r['disabled_policy'] is None for r in selected) else any(r['disabled_policy'] for r in selected),metric_revision=revision)
            if any(r['status']=='blocked' for r in selected):
                row.update(status='blocked',missing_reason='; '.join(sorted({r['missing_reason'] for r in selected if r['status']=='blocked'})))
            if len(known)==len(selected):
                summary=finite(aggregate_rows(known,[r['dataset'] for r in selected]));row.update(summary);row.update(summary['counts']);row.pop('counts',None)
                row.update(status='scored',missing_reason=None,frames=100*len(known),estimated_total=sum(r['estimated_total'] for r in known),
                           matched_nodes=sum(r['matched_nodes'] for r in known))
                w=sum(row[k] for k in ('edge_tp','edge_fp','edge_fn'));row['edge_weight']=w;row['adjusted_edge_contribution']=w*row['adj_edge_jaccard']
            pooled.append(row)
    for group in (per_clip,directions,pooled):
        lookup={(r.get('dataset'),r['source'],r['seed'],r['arm']):r for r in group}
        for r in group:
            for arm in ('C00','C01'):
                baseline=lookup.get((r.get('dataset'),r['source'],r['seed'],arm))
                r['delta_'+arm]=r.get('score')-baseline['score'] if r.get('score') is not None and baseline and baseline.get('score') is not None else None
    for r in per_clip:
        key=r['source'],r['seed'],r['arm'],r['dataset'];bkey=r['source'],r['seed'],'C00',r['dataset']
        if key in details and bkey in details:
            a,b=details[key]['events'],details[bkey]['events']
            r.update(introduced_division_fp=len(set(a['fp_predicted_forks'])-set(b['fp_predicted_forks'])),
                newly_recovered_division_tp=len(set(a['recovered_gt_events'])-set(b['recovered_gt_events'])),
                lost_division_tp=len(set(b['recovered_gt_events'])-set(a['recovered_gt_events'])))
    for group in (directions,pooled):
        for r in group:
            ss=[x for x in per_clip if x['arm']==r['arm'] and x['seed']==r['seed'] and (r['source']=='pooled' or x['source']==r['source'])]
            if r['status']=='scored':
                for k in ('accepted_actions','changed_edges','introduced_division_fp','newly_recovered_division_tp','lost_division_tp'):
                    r[k]=sum(x.get(k,0) for x in ss)
    for name,rows in [('scores.csv',pooled),('per_embryo_scores.csv',directions),('per_clip_scores.csv',per_clip)]:
        with (RESULTS/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    write(RESULTS/'calibration.json',public(dict(status='complete' if len(calibrations)==8 else 'partial',cells=calibrations)))
    return pooled,directions,per_clip,details


def diagnostics(details):
    source=[];pilots=[];funnel=[]
    for s in ('44b6','6bba'):
        for seed in (20260918,314159):
            p=WORK/'source_diagnostics'/s/str(seed)/'summary.json'
            if p.exists():
                r=read(p);clean={k:v for k,v in r.items() if k!='per_clip'};clean['clips']={}
                for n,d in r['per_clip'].items():
                    clean['clips'][n]={k:v for k,v in d.items() if k!='funnel'}
                    if 'funnel' in d:clean['clips'][n]['funnel']={k:v for k,v in d['funnel'].items() if k!='events'}
                wp=p.parent/'witnesses/receipt.json'
                if wp.exists():
                    w=read(wp);clean['retained_source_witnesses']={k:v for k,v in w.items() if k!='parents'}
                source.append(clean)
        for p in (WORK/'source_evaluation'/s/'20260918').glob('*/receipt.json'):
            r=read(p);r={k:v for k,v in r.items() if k not in ('positive_witness','negative_witness')}
            raw=read(p)
            for k in ('positive_witness','negative_witness'):
                if raw.get(k):r[k]=dict(risk=raw[k]['risk'],official_score=raw[k]['official_score'],label_guided_not_model=True,
                    solver_and_global_edit_cap_applied=False,limitation='Local legal oracle only; tiny baseline graphs have zero allowed edits under the global 2% cap.')
            pilots.append(r)
    for (s,seed,arm,clip),r in details.items():
        d=r.get('diagnostics',{});counts=d.get('funnel',{}).get('stages',{})
        for stage,value in counts.items():funnel.append(dict(source=s,seed=seed,arm=arm,dataset=clip,stage=stage,count=value,status='measured'))
        for stage,value in d.get('stage_counts',{}).items():funnel.append(dict(source=s,seed=seed,arm=arm,dataset=clip,stage=stage,count=value,status='measured'))
        for stage,value in d.get('deployment_census',{}).get('census',{}).items():
            funnel.append(dict(source=s,seed=seed,arm=arm,dataset=clip,stage='deployment_'+stage,count=value,status='measured_lower_bound' if 'lower_bound' in stage else 'measured'))
        for stage,value in d.get('selected_action_local_risk',{}).items():
            funnel.append(dict(source=s,seed=seed,arm=arm,dataset=clip,stage='selected_local_'+stage,count=value,status='measured'))
        for stage,value in d.get('solver',{}).items():
            if isinstance(value,(int,float)) and not isinstance(value,bool) and not stage.startswith('max_'):
                funnel.append(dict(source=s,seed=seed,arm=arm,dataset=clip,stage='solver_'+stage,count=value,status='measured'))
        det=d.get('detection',{})
        for stage in ('gt_nodes','matched_nodes_7um','matched_nodes_3um','gt_edges_missing_endpoint','gt_edges_endpoints_present_link_missing','gt_edges_in_clean_candidate_union'):
            if stage in det:funnel.append(dict(source=s,seed=seed,arm=arm,dataset=clip,stage=stage,count=det[stage],status='measured'))
    if not funnel:funnel=[dict(status='pending',stage='target attribution requires complete frozen predictions and scoring',count=None)]
    with (RESULTS/'error_funnel.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['source','seed','arm','dataset','stage','count','status']);writer.writeheader();writer.writerows(funnel)
    tails=Counter()
    for key,r in details.items():
        for item in r.get('diagnostics',{}).get('raw_tail',[]):
            tails[f'{key[0]}/{key[1]}/{key[2]}/occurrence_target_{item["group_target"]}']+=1
            tails[f'{key[0]}/{key[1]}/{key[2]}/highest_action_risk_{item["highest_raw_gain_action_risk"]}']+=1
    masks=[read(p) for p in (WORK/'mask_audit').glob('*/receipt.json')]
    value=dict(status='complete' if len(source)==4 else 'partial',source_cells=source,engineering_pilots=pilots,
        source_mask_audits=[{k:v for k,v in m.items() if k!='per_clip'} for m in masks],
        raw_selected_target_tail_counts=dict(tails),target_tail_definition='Top 50 parent groups per full clip by uncalibrated complete-fork gain; unknown support stays unknown.',
        endpoint_action_groups_overlap=True,pilot_collapse_disclosed=True,biological_nondivision_identified=False,
        background_caveat='Low-intensity/low-variance background is a fixed heuristic. Sparse annotations cannot certify that unannotated background voxels contain no cells.')
    write(RESULTS/'source_diagnostics.json',public(value));return value


def interpretation(pooled,directional,details):
    """Describe complete populations without selecting or fitting another model."""
    evidence=[];by_key={(r['source'],r['seed'],r['arm']):r for r in directional}
    for r in directional:
        if r['status']!='scored':continue
        records=[v for (s,seed,arm,_),v in details.items() if (s,seed,arm)==(r['source'],r['seed'],r['arm'])]
        if len(records)!=r['clips_expected']:raise Blocked('Interpretation population differs from the scored population')
        det=Counter();stages=Counter();selected=Counter();solver=Counter()
        for record in records:
            d=record.get('diagnostics',{})
            for k,v in d.get('detection',{}).items():
                if isinstance(v,int) and not isinstance(v,bool):det[k]+=v
            stages.update(d.get('funnel',{}).get('stages',{}));stages.update(d.get('stage_counts',{}))
            selected.update(d.get('selected_action_local_risk',{}))
            for k,v in d.get('solver',{}).items():
                if isinstance(v,(int,float)) and not isinstance(v,bool) and not k.startswith('max_'):solver[k]+=v
        base=by_key[r['source'],r['seed'],'C00']
        keys=('score','delta_C00','delta_C01','edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn',
              'edge_jaccard','adj_edge_jaccard','num_pred_nodes','estimated_total','accepted_actions','changed_edges',
              'introduced_division_fp','newly_recovered_division_tp','lost_division_tp','disabled_policy')
        row=dict(source=r['source'],target=r['target'],seed=r['seed'],arm=r['arm'],clips=len(records),
            metrics={k:r.get(k) for k in keys},detection=dict(det),fork_stages=dict(stages),
            selected_local_risk=dict(selected),solver=dict(solver))
        if base['status']=='scored':
            row['edge_count_deltas_C00']={k:r[k]-base[k] for k in ('edge_tp','edge_fp','edge_fn')}
            row['fraction_C00_missed_divisions_newly_recovered']=r.get('newly_recovered_division_tp',0)/base['division_fn'] if base['division_fn'] else None
        evidence.append(row)
    value=dict(status='complete' if len(evidence)==12 else 'partial',cells=evidence,
        comparison_scope='Practical model-family value of C11 versus C01; not the isolated causal effect of factorization or one loss.',
        aggregation_scope='Each direction and seed is separate. Event counts refer to clip occurrences; seeds are not independent animals.',
        attribution_scope='Sparse matched endpoint/link counts and overlapping local division stages; these are not additive achievable score repairs.',
        unmatched_prediction_scope='Unmatched predictions are not automatically false positives. Node totals affect the official adjustment separately.')
    write(RESULTS/'interpretation.json',public(value));return value


def run():
    RESULTS.mkdir(parents=True,exist_ok=True)
    lock=read(RESULTS/'execution_lock.json');tr=training();res=resources();pooled,directional,per_clip,details=score_rows();diag=diagnostics(details)
    interpreted=interpretation(pooled,directional,details)
    packages={};clean=True
    from .provenance import unseal
    from .stage_provenance import validate_stages
    for s in ('44b6','6bba'):
        for seed in (20260918,314159):
            for arm in ('C00','C01','C11'):
                p=WORK/'packages'/s/str(seed)/arm
                if (p/'manifest.json').exists():
                    value=unseal(p);validate_stages(value['ancestry'],value['root_artifact'],s)
                    packages[f'{s}/{seed}/{arm}']=dict(path=str(p.relative_to(REPO)),identity=value['identity'],files=value['files'],ancestry=value['ancestry'])
    freeze=read(WORK/'freeze/public_summary.json') if (WORK/'freeze/public_summary.json').exists() else dict(status='pending',reason='Every retained model and complete target prediction must be frozen before target scoring')
    write(RESULTS/'prediction_manifest.json',freeze)
    cold=read(WORK/'cold/receipt.json') if (WORK/'cold/receipt.json').exists() else dict(status='pending')
    checks=read(WORK/'checks/runtime.json') if (WORK/'checks/runtime.json').exists() else dict(status='unrecorded',count=None)
    validation=dict(status='complete' if cold['status']=='passed' else 'partial',runtime_tests=checks,planning_contract_tests=37,
        tests_are_not_model_results=True,production_resume=read(RESULTS/'production_resume.json'),
        crop_parity=read(RESULTS/'crop_parity.json'),cold=cold,
        read_guard_limit='Python audit allowlist and fresh processes; tested direct/symlink/dirfd/subprocess denial, not an operating-system sandbox.')
    if (WORK/'cold_source_proof/receipt.json').exists():validation['source_cold_proof']=read(WORK/'cold_source_proof/receipt.json')
    if (WORK/'checks/empty_graph/receipt.json').exists():
        proof=read(WORK/'checks/empty_graph/receipt.json')
        validation['official_empty_graph_control']={k:v for k,v in proof.items() if k!='events'}
    if (WORK/'checks/mining_merge.json').exists():validation['mining_merge_contract']=read(WORK/'checks/mining_merge.json')
    if (WORK/'checks/source_prefetch.json').exists():validation['source_preparation_ownership']=read(WORK/'checks/source_prefetch.json')
    if (WORK/'checks/head_prefetch.json').exists():validation['head_preparation_ownership']=read(WORK/'checks/head_prefetch.json')
    if (WORK/'checks/resource_timing_reconciliation.json').exists():validation['resource_timing_reconciliation']=read(WORK/'checks/resource_timing_reconciliation.json')
    if (WORK/'checks/receipt_preservation.json').exists():validation['resource_receipt_preservation']=read(WORK/'checks/receipt_preservation.json')
    if (WORK/'checks/controller_archive_reuse.json').exists():validation['controller_archive_reuse']=read(WORK/'checks/controller_archive_reuse.json')
    if (RESULTS/'compact_precision_repair.json').exists():validation['compact_precision_repair']=read(RESULTS/'compact_precision_repair.json')
    if (WORK/'checks/source_attribution/receipt.json').exists():validation['source_attribution_execution']=read(WORK/'checks/source_attribution/receipt.json')
    if (WORK/'checks/target_gate.json').exists():validation['target_access_gate']=read(WORK/'checks/target_gate.json')
    if (WORK/'checks/retained_witness/receipt.json').exists():
        control=read(WORK/'checks/retained_witness/receipt.json')
        validation['retained_source_witness_control']={k:v for k,v in control.items() if k!='parents'}
    if (WORK/'checks/retained_positive_witness/receipt.json').exists():
        control=read(WORK/'checks/retained_positive_witness/receipt.json')
        validation['retained_source_positive_witness_control']={k:v for k,v in control.items() if k!='parents'}
    write(RESULTS/'validation.json',public(validation))
    write(RESULTS/'exposure_manifest.json',dict(exposure_class='source_isolated_reused_embryos',
        pristine_independent_generalization=False,seeds_are_not_new_animals=True,source_calibration_acquisition_independence_proven=False,
        historical_target_inspection=read(RESULTS/'reconciliation.json')['prior_target_exposure'],
        v11_target_metric_receipts=len(details),target_metrics_opened=bool(details),prediction_freeze_identity=freeze.get('identity'),
        qualified_packages=packages,unqualified_v10_weights_loaded=False,new_external_data_used=False))
    trained=all(c['upstream']['status']=='trained' and c['compact']['status']=='trained' and c['linear']['status']=='fitted' for c in tr['cells'])
    scored=all(r['status']=='scored' for r in pooled+directional)
    frozen=freeze['status']=='frozen';cold_pass=cold['status']=='passed';provenance=len(packages)==12
    complete=trained and scored and frozen and cold_pass and provenance
    active=[]
    for p in (WORK/'controller').glob('*.json'):
        r=read(p)
        for key in ('controller_pid','worker_pid'):
            if alive(r.get(key)):active.append(dict(role=p.stem+'/'+key,pid=r[key]))
    for p in (WORK/'controller/source-prefetch').glob('*/*/progress.json'):
        r=read(p)
        if alive(r.get('pid')):active.append(dict(role='source-prefetch/'+r['source']+'/'+str(r['seed']),pid=r['pid']))
    for p in (WORK/'controller/head-prefetch').glob('*/*/progress.json'):
        r=read(p)
        if alive(r.get('pid')):active.append(dict(role='head-prefetch/'+r['source']+'/'+str(r['seed']),pid=r['pid']))
    blocked=[]
    for p in (WORK/'controller/jobs').glob('*.json'):
        r=read(p)
        if r.get('status') in ('failed','blocked'):blocked.append({k:r.get(k) for k in ('stage','source','seed','arm','clip','status','exit_code','log_sha256')})
    for p in (WORK/'blocked_cells').glob('*.json'):blocked.append(read(p))
    command='PYTHONNOUSERSITE=1 PYTHONPATH=tools:. CUBLAS_WORKSPACE_CONFIG=:4096:8 /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m division_reliability_v11'
    milestone=complete and all(r['status']=='scored' and r['score']>=.95 for r in pooled if r['arm']=='C11')
    strong=complete and all(r['status']=='scored' and r['score']>=.95 for r in directional if r['arm']=='C11')
    base_by_cell={(r['source'],r['seed']):r for r in directional if r['arm']=='C00'}
    beneficial=any(r['status']=='scored' and base_by_cell[r['source'],r['seed']]['status']=='scored' and r.get('accepted_actions',0)>0 and
        (r.get('newly_recovered_division_tp',0)>0 or r['edge_tp']>base_by_cell[r['source'],r['seed']]['edge_tp'] or
         r['edge_fp']<base_by_cell[r['source'],r['seed']]['edge_fp']) for r in directional if r['arm']=='C11')
    promising=complete and all(r['delta_C01']>=.002 for r in pooled if r['arm']=='C11') and all(r['delta_C01']>=-.001 and r['delta_C00']>=-.001 for r in directional if r['arm']=='C11') and beneficial
    status=dict(status='complete' if complete else 'running' if active else 'blocked' if blocked else 'partial',updated_utc=now(),
        trained=trained,scored=scored,clean_provenance_passed=provenance,target_freeze_passed=frozen,all_four_cells_complete=complete,
        cold_inference_passed=cold_pass,upstream_reused_cells=[],blocked_stages=blocked,verified_active_processes=active,
        upstream_completed_cells=sum(c['upstream']['status']=='trained' for c in tr['cells']),compact_completed_cells=sum(c['compact']['status']=='trained' for c in tr['cells']),
        linear_completed_cells=sum(c['linear']['status']=='fitted' for c in tr['cells']),
        completed_score_rows=sum(r['status']=='scored' for r in per_clip),required_score_rows=len(per_clip),
        pooled_095_milestone_reached=milestone,strong_095_milestone_reached=strong,compact_promotion_gate_passed=promising,
        execution_lock_identity=lock['identity'],next_command=command+(' report' if active or complete else ' run --workers 3'),
        resume_command_when_no_existing_owner=command+' run --workers 3',
        resume_proof='Fresh-process 10+10 updates equal uninterrupted 20 updates, including optimizer/sampler/CPU+CUDA RNG and losses; compact mixed update replay also exact.',
        artifacts=dict(training='training_summary.json',predictions='prediction_manifest.json',resources='resource.json',exposure='exposure_manifest.json',interpretation='interpretation.json'),
        P0_modified=False,submitted_to_kaggle=False,weights_published=False)
    write(RESULTS/'STATUS.json',public(status))
    lines=[f'# Division reliability v11 — {status["status"]}',
        '',f'Actual status at {status["updated_utc"]}: {status["upstream_completed_cells"]}/4 C00 fits, {status["linear_completed_cells"]}/4 C01 fits and {status["compact_completed_cells"]}/4 C11 fits complete; {status["completed_score_rows"]}/{status["required_score_rows"]} required target clip/arm scores recorded.',
        '',f'The immutable schedule is U={lock["upstream_updates"]:,} and E={lock["event_updates"]:,} for both embryos and both seeds. Allocation stays 4/34/18/16 GPU lease-hours for pilots/upstream/event/inference. All six affordability candidates and the 25% margin are in allocation_projection.json. No target outcome selected the schedule.',
        '', 'The local v10 work was preserved. Neither completed nor partial v10 weights qualified for reuse because the matching trainer source and recursive ancestry were unavailable. Every retained v11 neural component starts randomly. P0 and the two pre-existing user-modified public946 reports remain unchanged.',
        '', 'Both embryos have historical research exposure. The claim is source_isolated_reused_embryos, not pristine independent generalization. Multiple seeds do not add embryos; calibration clips are not proven acquisition-independent. The original grouped split was retained. One persisted division event occurs in two source44 fit clips and receives one event unit across both.',
        '', 'C01/C11 independently edit their own C00 graph. Their comparison tests practical model-family value, not the isolated causal effect of factorization or one loss. Unknown legal forks remain in deployment denominators and do not become negative biological labels. Source safety and no-op outcomes are reported separately. CSV coordinates and IDs, full populations, official empty-division behavior and the pinned scorer are used; clip scores are never averaged.',
        '', 'The upstream training adapter uses annotation-matched proposal queries for supported incoming groups; complete inference uses dense detections. This leaves a training/inference attention-context difference. Low-intensity background masks are heuristics, not certification that unannotated voxels contain no cells. Full source mask audits and detector-collapse witnesses are retained.',
        '',f'New v11 GPU lease accounting: {res["new_study_gpu_lease_hours"]:.4f} hours, including measured failures and conservative early-pilot allowances. Historical v10 accounting is separate: {res["inherited_hours"]:.4f} hours, including an 8.4-hour unobserved-tail upper bound that may include idle time. Raw telemetry, private logs, checkpoints, arrays and complete predictions stay in work/division-reliability-v11/.',
        '', 'Resource reporting separates original prediction/bank timings, optimizer lease intervals and later cache-reuse process times. Some original controller wall-time receipts were overwritten during the first restart; their exact process durations are unavailable. All GPU lease charges remain accounted for. Optimizer intervals measured from the journal exclude startup and final serialization, so they are reported as observed intervals, not complete process wall times.',
        '', 'Source engineering proofs are actual executions, not retained model scores. They include batch-eight optimizer updates, exact resume, mixed 32-group compact gradients, native crop parity, complete source graphs, and official true/false-fork witnesses. The 250-update pilots produced excessive detections and almost no links; those failures are retained. Early pilot witnesses bypassed the global 2% cap. Later witnesses on retained C00 graphs use the registered solver and cap. Both kinds use labels to choose diagnostic edits and are not learned policies or achievable score bounds. Runtime tests and planning contracts are not evidence of trained accuracy.',
        '']
    if (RESULTS/'compact_precision_repair.json').exists():
        lines += ['A midpoint mining implementation failure exposed a TF32 singleton-reference discrepancy. C11 evaluation workers now set NVIDIA_TF32_OVERRIDE=0 before importing numerical libraries, symmetrically for mining, calibration, prediction and cold inference. Fitting settings, checkpoint parameters, bank definitions and the absolute 1e-5 parity tolerance are unchanged. The three tested 4,096-item embedding batches are bit identical; a complete source C00 image-to-CSV control under the override also matches every graph array and CSV byte. This source precision proof does not replace post-freeze target cold validation. Original failures and compute remain accounted for; compact_precision_repair.json records the correction.', '']
    for cal in read(RESULTS/'calibration.json')['cells']:
        label=f'{cal["source"]}/{cal["seed"]}/{cal["arm"]}'
        if cal['disabled_policy']:
            outcome='disabled: no registered margin passed source graph safety'
        else:
            trial=next(t for t in cal['trials'] if t['margin']==cal['margin'])
            outcome=f'margin {cal["margin"]}, {trial["accepted_actions"]} source calibration edits, score change {trial["combined_delta"]:+.6f}'
            if not trial['accepted_actions']:outcome+='; selected policy made no calibration edits'
        lines.append(f'- Source safety {label}: {outcome}. Occurrence support: {cal["distinct_positive_events"]} distinct positive events and {cal["negative_groups"]} negative groups; insufficient-support fallback {cal["insufficient_support"]}.')
    if read(RESULTS/'calibration.json')['cells']:lines.append('')
    available=[r for r in pooled if r['status']=='scored']
    if available:
        for r in available:
            deltas=', '.join(f'Δ{a} {r["delta_"+a]:+.6f}' for a in ('C00','C01') if r.get('delta_'+a) is not None)
            lines.append(f'- Seed {r["seed"]}, {r["arm"]}: pooled score {r["score"]:.6f}; {deltas}.')
    for r in directional:
        if r['status']=='scored':
            deltas=', '.join(f'Δ{a} {r["delta_"+a]:+.6f}' for a in ('C00','C01') if r.get('delta_'+a) is not None)
            lines.append(f'- Source {r["source"]} → target {r["target"]}, seed {r["seed"]}, {r["arm"]}: {r["score"]:.6f}; {deltas}; {r["clips_scored"]} complete clips.')
    if scored:
        lines += ['',f'The pooled 0.95 milestone is {"reached" if milestone else "not reached"}; the per-embryo/both-seed milestone is {"reached" if strong else "not reached"}. The compact promotion gate is {"passed" if promising else "not passed"}. These are local measurements, not a public/private leaderboard guarantee.']
    else:lines += ['Comparisons for unfinished arms/populations remain unmeasured. Available complete-population scores are retained; missing comparisons are blank with a reason in the three score CSVs. No 0.95 milestone is claimed from an incomplete matrix.']
    for row in pooled:
        if row['arm']!='C01' or row['status']!='scored':continue
        base=next(r for r in pooled if r['arm']=='C00' and r['seed']==row['seed'])
        if base['status']!='scored':continue
        lines += ['',f'C01 fork recovery, seed {row["seed"]}: {row["newly_recovered_division_tp"]} newly recovered out of {base["division_fn"]} C00-missed annotated division occurrences, {row["lost_division_tp"]} lost recoveries and {row["introduced_division_fp"]} introduced division FP. Its pooled score change is {row["delta_C00"]:+.6f}. This quantifies how much the linear comparator solves without interpreting a disabled policy as learned improvement.']
    comparisons=[r for r in directional if r['arm']=='C11' and r['status']=='scored' and r.get('delta_C01') is not None]
    if comparisons:
        better=sum(r['delta_C01']>0 for r in comparisons);worse=sum(r['delta_C01']<0 for r in comparisons)
        lines += ['',f'C11 versus C01: {len(comparisons)}/4 direction/seed comparisons are measured; {better} improve, {worse} regress and {len(comparisons)-better-worse} tie. The registered practical promotion gate is {"passed" if promising else "not passed"}. Seeds remain optimization replications on the same two embryos.']
    for cell in interpreted['cells']:
        label=f'source {cell["source"]} → target {cell["target"]}, seed {cell["seed"]}'
        if cell['arm']=='C00':
            d=cell['detection'];m=cell['metrics']
            if d:
                lines += ['',f'C00 error attribution ({label}): {d.get("matched_nodes_7um",0)}/{d.get("gt_nodes",0)} annotated nodes matched at 7 µm and {d.get("matched_nodes_3um",0)} at 3 µm. Among annotated edges, {d.get("gt_edges_missing_endpoint",0)} lack a matched endpoint and {d.get("gt_edges_endpoints_present_link_missing",0)} have matched endpoints but no recovered link. Raw/adjusted edge Jaccard is {m["edge_jaccard"]:.6f}/{m["adj_edge_jaccard"]:.6f}; predicted/estimated node totals are {m["num_pred_nodes"]}/{m["estimated_total"]:g}. Unmatched predictions are not automatically false positives.']
        elif cell['arm']=='C11':
            f=cell['fork_stages'];m=cell['metrics']
            lines += ['',f'C11 division funnel ({label}): {m["division_tp"]+m["division_fn"]} annotated event occurrences; {f.get("endpoint_window_present",0)} have endpoint windows, {f.get("anchor_present",0)} anchors, {f.get("daughter_paths_present",0)} daughter paths and {f.get("legal_compatible_action",0)} legal compatible actions. Compatible actions rank first in {f.get("conditional_top1",0)} events; {f.get("positive_calibrated_margin_gain",0)} have positive calibrated gain and {f.get("actually_recovered",0)} are recovered in the final graph. These diagnostic groups overlap and cannot be added as independent repairs.']
    lines += ['',f'Global target freeze: {frozen}. Complete cold validation: {cold_pass}. Clean provenance for all twelve packages: {provenance}.',
        '', 'Resume from the repository root after verifying no existing controller owns the study:', '', '```sh',command+' run --workers 3','```',
        '', 'While a controller is active, use the same command with `report` in place of `run --workers 3` to refresh STATUS. Cell owner locks prevent duplicate training. Durable checkpoint paths, SHA-256 hashes, exact saved update counts, and optimizer/RNG availability are in training_summary.json. Each stage uses atomic output receipts and can be invoked individually with `--source`, `--seed`, and the relevant arm/clip/partition.',
        '', 'The resume proof is a fresh-process 10+10 versus uninterrupted 20-update comparison across source cells, including the transition into proposal queries. Compact mixed-objective replay is also bit exact. A CUDA lazy-initialization RNG reset found during testing was repaired before the lock; failed attempts remain private. Max-pool backward carries a PyTorch determinism warning, so claims of exactness are limited to the tested executions and cold results.',
        '', 'Current next decision: '+('Review measured C01/C11 deltas, cold validation and error attribution before proposing any new experiment. P0 stays unchanged.' if complete else 'Finish the locked matrix using the existing owners/checkpoints; do not select a new model from partial source or target outcomes.'),
        '', 'No merge, Kaggle submission, weight publication, all-data refit or unrelated handover study was launched.']
    if blocked:lines += ['',f'{len(blocked)} unresolved stage failure/blocker receipt(s) are listed in STATUS.json; missing metrics have not been replaced by C00 or P0 values.']
    (RESULTS/'REPORT_BACK.md').write_text('\n'.join(lines)+'\n')
    return status
