"""Sanitized measured report and standalone offline dashboard."""
from collections import Counter
from pathlib import Path
import html
import shutil
import numpy as np
import pandas as pd
from .common import *

def choose(points):
    pooled=points[points.embryo=='pooled'].sort_values('score',ascending=False)
    eligible=[]
    for r in pooled.to_dict('records'):
        v=r['variant'];rr=points[points.variant==v]
        if v.startswith('C_') or v in ['historical_v1','R_image_local']:continue
        if r['delta_v2']>1e-10 and (rr.delta_v2>=-1e-10).all():eligible.append(v)
    return eligible[0] if eligible else 'incumbent',eligible

def family_outcomes(points):
    """All family comparisons use the preserved v2 population baselines."""
    result={}
    for family,prefixes in [('association',('A_',)),('division',('D_',)),
                            ('rescue',('R_',)),('combinations',('AD_', 'ADR_'))]:
        variants=sorted(v for v in points.variant.unique() if v.startswith(prefixes))
        records=[]
        for variant in variants:
            groups={r['embryo']:r for r in points[points.variant==variant].to_dict('records')}
            pooled=groups['pooled']
            valid=variant!='R_image_local'
            records.append(dict(variant=variant,score=pooled['score'],delta_v2=pooled['delta_v2'],
                embryo_deltas_v2={e:groups[e]['delta_v2'] for e in ['44b6','6bba']},
                promotion_gate_passed=valid and pooled['delta_v2']>1e-10 and
                    all(r['delta_v2']>=-1e-10 for r in groups.values()),
                invalid_control=not valid,
                counts={k:int(pooled[k]) for k in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn']}))
        result[family]=sorted(records,key=lambda r:r['score'],reverse=True)
    return result

def compact_event_diagnostics(ctx):
    coverage=read_json(ctx.out/'event_candidate_coverage.json')
    lock=read_json(ctx.out/'event_model_lock.json')
    fields=['source_embryo','target_embryo','pool','seed','actual_steps','seconds','model_sha256',
        'training_rows','positive_rows','positive_rows_seen','all_positive_alternatives_seen',
        'unique_positive_groups','unique_negative_groups','source_weighted_bce','actual_iterations','objective']
    models={name:{k:r[k] for k in fields if k in r} for name,r in lock['models'].items()}
    return dict(coverage=coverage,models=models,total_training_elapsed_seconds=lock['total_elapsed_seconds'],
        training_optimizer_steps=sum(r.get('actual_steps',0) for r in models.values()),
        scores_are_calibrated_posteriors=False)

def resource_summary(ctx):
    p=ctx.out/'resource_samples.csv'
    if not p.exists():return {}
    d=pd.read_csv(p)
    return dict(peak_task_rss_gib=float(d.rss_bytes.max()/1024**3),peak_gpu_gib=float(d.gpu_used_mib.max()/1024),
        min_free_disk_gib=float(d.free_disk_bytes.min()/1024**3),observed_elapsed_seconds=float(d.elapsed_seconds.max()),
        observations=len(d))

def audit_changes(ctx,selected):
    """Final whole-graph matches give pair and recovered-truth regret separately."""
    records=[]
    for s in ctx.samples():
        name=s['dataset'];p0=ctx.out/'evaluation/matches/incumbent'/f'{name}.npz';p1=ctx.out/'evaluation/matches'/selected/f'{name}.npz'
        b=load_graph(p0);c=load_graph(p1)
        row=dict(dataset=name,embryo=s['embryo'],variant=selected)
        for key,prefix in [('tp_pairs','predicted_tp_pairs'),('fp_pairs','fp_pairs'),('recovered_gt_edges','recovered_gt_edges')]:
            old=set(map(tuple,b[key]));new=set(map(tuple,c[key]))
            row[prefix+'_lost']=len(old-new);row[prefix+'_new']=len(new-old);row[prefix+'_retained']=len(old&new)
        records.append(row)
    pd.DataFrame(records).to_csv(ctx.out/'selected_regret_by_clip.csv',index=False)
    result={k:int(sum(r[k] for r in records)) for k in records[0] if k not in ['dataset','embryo','variant']}
    write_json(ctx.out/'selected_regret_summary.json',result);return result

def worst_clips(ctx,rows,selected):
    fields=['dataset','embryo','adj_edge_jaccard','edge_tp','edge_fp','edge_fn',
        'division_tp','division_fp','division_fn','num_pred_nodes']
    b=rows[rows.variant=='incumbent'][fields]
    c=rows[rows.variant==selected][fields]
    pairs=b.merge(c,on=['dataset','embryo'],suffixes=('_v2','_selected'),validate='one_to_one')
    for field in fields[2:]:pairs['delta_'+field]=pairs[field+'_selected']-pairs[field+'_v2']
    pairs=pairs.sort_values(['delta_adj_edge_jaccard','dataset'])
    pairs.to_csv(ctx.out/'selected_clip_changes.csv',index=False)
    return pairs.head(5)[['dataset','embryo',*['delta_'+f for f in fields[2:]]]].to_dict('records')

def oracle_comparison(ctx,points):
    existing=read_json(ctx.out/'source_event_oracle_summary.json')['summary']
    baseline={r['embryo']:r['score'] for r in points.to_dict('records') if r['variant']=='incumbent'}
    result={}
    for arm,groups in existing.items():
        result[arm]={embryo:dict(**row,delta_v2=row['score']-baseline[embryo])
                     for embryo,row in groups.items()}
    result['source_association_oracle']=read_json(ctx.out/'source_association_oracle_score_summary.json')['results']
    write_json(ctx.out/'source_oracle_comparison.json',dict(results=result,
        inference_eligible=False,scope='Source-only heuristic feasibility, not a global bound or a promotion candidate'))
    return result

def dashboard(ctx,summary,points,score_rows):
    template=Path(__file__).with_name('dashboard_template.html').read_text()
    columns=['dataset','embryo','variant','adj_edge_jaccard','edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes']
    data=dict(summary=summary,points=points.to_dict('records'),scores=score_rows[columns].to_dict('records'))
    encoded=__import__('json').dumps(clean(data),allow_nan=False).replace('</','<\\/')
    (ctx.out/'dashboard.html').write_text(template.replace('__V3_DATA__',encoded))

def run(ctx,args=None):
    points=pd.read_csv(ctx.out/'operating_points.csv');rows=pd.read_csv(ctx.out/'score_rows.csv')
    names={s['dataset'] for s in ctx.samples()}
    if len(names)!=199 or rows.variant.nunique()!=32 or len(rows)!=32*199:
        raise ValueError('Completed report requires exactly 32 configurations × 199 expected clips')
    for variant,group in rows.groupby('variant'):
        if len(group)!=199 or set(group.dataset)!=names:
            raise ValueError(f'Incomplete or duplicate score coverage: {variant}')
    if (len(points)!=32*3 or points.duplicated(['variant','embryo']).any()
            or set(points.variant)!=set(rows.variant)
            or any(set(g.embryo)!={'44b6','6bba','pooled'} for _,g in points.groupby('variant'))):
        raise ValueError('Aggregate population coverage is incomplete')
    selected,eligible=choose(points)
    config=read_json(ctx.out/'winning_config.json')
    if config['variant']!=selected:raise RuntimeError('Selected inference package does not match measured promotion gate')
    export=read_json(ctx.out/'selected_prediction_lock.json')
    if export['variant']!=selected or export['samples']!=199 or set(export['hashes'])!=names:
        raise ValueError('Selected export variant or coverage differs from the measured selection')
    scored_hashes=rows[rows.variant==selected].set_index('dataset')['graph_file_sha256'].to_dict()
    for name,expected_hash in export['hashes'].items():
        source=ctx.incumbent(name) if selected=='incumbent' else ctx.out/'candidate_graphs'/selected/f'{name}.npz'
        if (sha(ctx.out/'selected_predictions'/f'{name}.npz')!=expected_hash
                or sha(source)!=expected_hash or scored_hashes[name]!=expected_hash):
            raise ValueError('Selected graph changed after scoring or export')
    base=points[(points.variant=='incumbent')&(points.embryo=='pooled')].iloc[0].to_dict()
    best=points[(points.variant==selected)&(points.embryo=='pooled')].iloc[0].to_dict()
    historical=points[(points.variant=='historical_v1')&(points.embryo=='pooled')].iloc[0].to_dict()
    embryos={e:points[(points.variant==selected)&(points.embryo==e)].iloc[0].to_dict() for e in ['44b6','6bba']}
    fresh=read_json(ctx.out/'fresh_delivery_receipt.json')
    if (fresh.get('variant')!=selected or fresh.get('samples')!=199
            or fresh.get('status')!='full_fresh_selected_complete'
            or fresh.get('annotations_unavailable_from_start') is not True
            or fresh.get('cached_graph_reads_for_prediction')!=0
            or fresh.get('fresh_selected_graphs_equal_to_scored')!=199):
        raise ValueError('Complete annotation-unavailable inference for the selected variant is not proven')
    for sample in ctx.samples():
        record=read_json(ctx.out/'fresh_evidence'/f"{sample['dataset']}.json")
        if record.get('annotation_reads')!=0 or record.get('cached_graph_reads_for_prediction')!=0:
            raise ValueError('Fresh evidence accessed forbidden prediction inputs')
    smoke=read_json(ctx.out/'package_smoke_receipt.json')
    if (smoke.get('variant')!=selected or smoke.get('pilot_names')!=['44b6_87bba6c4','6bba_789f8168']
            or smoke.get('selected_graph_parity')!=2 or smoke.get('cached_graph_reads_for_prediction')!=0
            or any(smoke.get(k) is not True for k in ['passed','actual_copied_package_execution',
                'annotations_unavailable_from_start','csv_validation_passed','source_modules_within_copied_package'])):
        raise ValueError('The copied selected package must pass both real image pilot executions')
    teacher_parity=read_json(ctx.out/'fresh_teacher_native_parity_summary.json')
    if (teacher_parity.get('samples')!=199 or teacher_parity.get('unexplained_coordinate_changes')!=0
            or any(teacher_parity.get(k) is not True for k in ['all_E_hgb_edges_exact',
                'all_E_native_edges_exact','all_current_native_feature_fields_exact',
                'all_current_native_numeric_arrays_finite','all_old_teacher_edges_exact'])):
        raise ValueError('Full fresh teacher/native evidence audit is incomplete or unexplained')
    regret=audit_changes(ctx,selected);resources=resource_summary(ctx)
    worst=worst_clips(ctx,rows,selected)
    oracles=oracle_comparison(ctx,points)
    events=compact_event_diagnostics(ctx)
    if events['training_optimizer_steps']!=40000 or sum('actual_steps' in r for r in events['models'].values())!=4:
        raise ValueError('Four complete 10,000-step source/seed image fits are required')
    fresh_scores=read_json(ctx.out/'fresh_selected_score_summary.json')
    fresh_lock=read_json(ctx.out/'fresh_selected_score_lock.json')
    if (not fresh_scores['all_counts_and_scores_match'] or fresh_scores['variant']!=selected
            or fresh_scores['samples']!=199 or fresh_lock['variant']!=selected
            or set(fresh_lock['hashes'])!=names
            or fresh_scores['fingerprint']!=sha(ctx.out/'fresh_selected_score_lock.json')):
        raise ValueError('Fresh selected official scores differ or belong to another selection')
    for name,expected_hash in fresh_lock['hashes'].items():
        path=ctx.out/'fresh_candidates'/selected/f'{name}.npz'
        if sha(path)!=expected_hash:raise ValueError('Fresh selected graph changed after independent scoring')
        a=load_graph(path);b=load_graph(ctx.out/'selected_predictions'/f'{name}.npz')
        if graph_hash(a['nodes'],a['edges'])!=graph_hash(b['nodes'],b['edges']):
            raise ValueError('Fresh graph differs from the exported selected graph')
    families=family_outcomes(points)
    write_json(ctx.out/'family_outcomes.json',families)
    ledger=read_json(ctx.out/'ledger_delivery_summary.json')
    delta=best['score']-base['score']
    decision='retain_v2' if selected=='incumbent' else 'stretch_local_gain' if delta>=.02 else 'useful_local_gain' if delta>=.002 else 'small_local_gain'
    summary=dict(study_id='strong-tracker-v3',status='measured_complete',created=now(),selected_variant=selected,
        decision=decision,invalid_for_promotion={'R_image_local':'Synthetic test found flat-background maxima can pass temporal persistence; retained as failed control, corrected in R_image_persistent'},baseline_v2=base,selected=best,embryos=embryos,delta_v2=delta,eligible_variants=eligible,
        historical_v1_score=historical['score'],delta_v1_historical_only=best['score']-historical['score'],
        variants=int(rows.variant.nunique()),expected_clips=199,score_rows=len(rows),
        source_directions=['44b6->6bba','6bba->44b6'],metric_revision=ctx.metric_revision,
        validation='Operational exploratory. Both embryos repeatedly reused; public upstream contamination. Unknown crop overlaps; no independent inner folds or confidence intervals.',
        fresh_inference=fresh,fresh_official_evaluation=fresh_scores,package_smoke=smoke,
        fresh_teacher_parity=teacher_parity,resources=resources,regret=regret,
        events=events,source_oracles=oracles,worst_adjusted_edge_changes=worst,
        family_outcomes=families,ledger_delivery=ledger,
        census=read_json(ctx.out/'census_summary.json'),
        limitations=['Promotion requires pooled gain and nonnegative delta in both reused embryos; it is not hidden-test evidence.',
            'Teacher votes are correlated. Source event observations may overlap; augmentation does not create independent events.',
            'GT-derived matches, source labels and oracles are excluded from inference inputs.',
            'RTX 4090 runtime does not establish Kaggle accelerator performance.'])
    write_json(ctx.out/'summary.json',summary)
    history=points[(points.variant=='historical_v1')&(points.embryo=='pooled')].iloc[0]
    lines=['# Strong tracker v3 — measured results','',
        f"Decision: **{decision}**. Selected **{selected}**, score **{best['score']:.15f}**, delta against v2 **{delta:+.15f}** on all **199 clips**.",
        '',f"The preserved v2 incumbent independently rescored to **{base['score']:.16f}**. V1's {history.score:.16f} is historical context only; the selected delta versus v1 is {best['score']-history.score:+.15f}, and was not used for promotion.",
        '',*[f"- {e}: {r['score']:.15f}, delta versus v2 {r['delta_v2']:+.15f}." for e,r in embryos.items()],
        '',summary['validation'],'',
        '## Measurements and decision', '',
        f"Selected edge TP/FP/FN: {int(best['edge_tp'])} / {int(best['edge_fp'])} / {int(best['edge_fn'])}. Division TP/FP/FN: {int(best['division_tp'])} / {int(best['division_fp'])} / {int(best['division_fn'])}. Predicted nodes: {int(best['num_pred_nodes'])}.",
        '',f"{summary['variants']} complete scored graph configurations, {len(rows)} rows. Every scored graph gets fresh official node matching and division assignment. Aggregation checks the exact expected sample set, fixed per-clip estimates and official denominator weights.",
        '', '## V300–V310: incumbent, residual census and conditional controls','',
        'All 199 incumbent file hashes, the original notebook hash and the selected v2 lock were verified before experiments. The metric source matches current upstream byte-for-byte; the exact pinned revision remains unchanged.',
        '', '[Pinned official metric source](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f5a52d11077f9dc2b074644618f26939e2/src/tracking_cellmot) and [metric documentation](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md) define the evaluation. The per-file local/upstream SHA checks are in metric_verification.json.',
        '', 'The preserved v2 incumbent has 98 missed divisions with both daughter lineages matched but no surviving fork, 18 with missing daughter candidates, four with competing assignment and two missing parent-side evidence. The census covers all 151 GT division observations, 92 division FP, 5,860 edge FN and 4,930 edge FP. It does not reuse the old census.',
        '', 'The small safe-division/smoothing grid and no-pruning/no-gap controls all keep motion relinking off. Original and canonical teacher coordinates are accounted separately, and inserted donor IDs are excluded when correspondence is unproven.',
        '', '## V320–V350: local arbitration, events and image rescue','',
        'Association features were rebuilt at incumbent coordinates and adjacency. Opposite-embryo native-residual logistic and seven-leaf tree models use supported positive and contradictory edges; unknown sparse contexts stay masked. Local edits include displaced owners, fixed boundaries, no-op, an edit cap and bounded MILP conflicts. Frozen v2 model transfer is a named control.',
        '', 'Division supervision targets supported events and their pair/timing/path alternatives. The expanded pool allows alternatives excluding the current continuation, image-derived temporal anchors and candidate daughter persistence. The individual-owner guard was tested against the free-daughter/owned-daughter counterexample. Models and both source directions were frozen before comparative scoring.',
        '', 'The image-rescue lane tests native reappearance, small image-centroid refinements and secondary maxima. A synthetic test exposed flat-background false persistence in the initial R_image_local arm; it is retained as an invalid-for-promotion control. R_image_persistent adds actual foreground-contrast evidence for every temporal peak. The DeepCenter-confirmed arm uses cached heatmaps and local checkpoint queries for missing frames. Continuous centroid shifts are capped at 0.75 um before integer serialization; realized shifts in the initial arm reach 0.908403 um, which is recorded explicitly.',
        '', '## V360: combinations and regret','',
        'Combination inputs are regenerated after association edits. Component scores are not added together. The selected regret audit distinguishes changed predicted pairs from recovered GT-edge identities using each full graph’s fresh matches. The local edit_ledger.parquet preserves prediction-time actions; evaluation/edit_effects.parquet and stage/variant regret tables remain separate. Per-action edge attribution requires unchanged nodes and matching plus exact reconstruction of the scored stage graph. Division credit and combined-score changes are reported for whole graphs, not allocated additively to individual actions.',
        '',f"Recovered GT-edge identity changes for the selected graph: {regret['recovered_gt_edges_new']} new, {regret['recovered_gt_edges_lost']} lost. Predicted TP-pair changes: {regret['predicted_tp_pairs_new']} new, {regret['predicted_tp_pairs_lost']} lost.",
        '', 'The five lowest per-clip adjusted-edge changes are listed below. This ordering is diagnostic: division Jaccard is pooled, so these changes are not additive per-clip combined-score contributions or independent biological replicates.',
        '',*[f"- {r['dataset']}: adjusted-edge delta {r['delta_adj_edge_jaccard']:+.9f}; edge TP {int(r['delta_edge_tp']):+d}, FP {int(r['delta_edge_fp']):+d}; division TP {int(r['delta_division_tp']):+d}, FP {int(r['delta_division_fp']):+d}." for r in worst],
        '', '## V370: inference, runtime and portability','',
        'The actual notebook neural and repair code was copied into an isolated v3 entry. The first two full 100-frame pilots were selected by median image contrast, one per embryo, before their scores were inspected. Both reproduced raw and repaired cached graphs exactly with annotations unavailable from process start.',
        '', 'All 199 selected fresh-image graphs received an additional independent official evaluation. Pooled and both embryo counts/scores agree with the selected cached experiment. This reproduction does not add a new inference policy to the 32-configuration experiment budget.',
        '', 'The final copied inference package also ran both preselected image pilots through its own run.sh from outside this checkout. Both selected graphs and CSV round trips matched, with annotations unavailable from process startup and zero cached graph inputs for prediction. The package smoke receipt pins its code, policy and dependencies.',
        '', f"The full fresh teacher audit found {teacher_parity['changed_old_teacher_nodes']} old-teacher coordinate differences: {teacher_parity['bounded_nodes']} pre-existing bounds corrections and {teacher_parity['documented_half_integer_serialization_nodes']} documented half-integer serialization cases. There are zero unexplained changes. All old/E teacher edges and every current native feature array match exactly across 199 clips; the final selected graphs also match. The original old-teacher node arrays are therefore not claimed to be byte-identical.",
        '',f"The full fresh neural/baseline pass took {fresh['full_neural_wall_seconds']/60:.2f} minutes of measured wall time. Per-clip neural/I/O time was median {fresh['neural_io_seconds']['median']:.2f} seconds (p90 {fresh['neural_io_seconds']['p90']:.2f}); graph ILP/export was median {fresh['graph_ilp_export_seconds']['median']:.2f} seconds (p90 {fresh['graph_ilp_export_seconds']['p90']:.2f}); teacher/native-feature reconstruction and decoding was median {fresh['teacher_native_feature_decode_seconds']['median']:.2f} seconds (p90 {fresh['teacher_native_feature_decode_seconds']['p90']:.2f}). These stages overlapped across processes, so their sums are not end-to-end wall time.",
        '',f"The two copied-package pilots took {smoke['wall_seconds']:.2f} seconds concurrently. The separate warm-filesystem startup probe took {read_json(ctx.out/'fresh_startup_benchmark.json')['total_seconds']:.2f} seconds; this is not a cold-cache measurement. See [fresh_delivery_receipt.json](fresh_delivery_receipt.json), [package_smoke_receipt.json](package_smoke_receipt.json) and [fresh_startup_benchmark.json](fresh_startup_benchmark.json) for exact timing boundaries and dependency hashes.", '',
        f"Observed peak study RSS {resources.get('peak_task_rss_gib',0):.2f} GiB; GPU memory {resources.get('peak_gpu_gib',0):.2f} GiB. Full timing, source hashes, graph hashes and artifact dependencies are included in the local manifests.",
        '', 'See [dashboard.html](dashboard.html), [score_rows.csv](score_rows.csv), [operating_points.csv](operating_points.csv), [event diagnostic](event_diagnostic.md), [winning_config.json](winning_config.json), [NEXT_AGENT.md](NEXT_AGENT.md) and [reproduction instructions](../../docs/strong-tracker-v3.md).',
        '', 'Raw microscopy, detailed annotation evidence, crop tensors, graph predictions and weights stay outside Git. The complete local v3 root plus v1/v2 stores, raw competition inputs, official scorer, notebook code and local model weights are needed on another host. No external artifact backup or persistence is implied.',
        '']
    event_lines=['## Event coverage and actual training','']
    for pool in ['base','fallback']:
        if pool not in events['coverage']:continue
        for embryo,r in events['coverage'][pool].items():
            event_lines.append(f"- {pool}, source {embryo}: {r['covered_events']}/{r['total_events']} supported events ({r['coverage']:.2%}); {r['proposals']:,} alternatives, {r['positive_alternatives']:,} supported positive alternatives and {r['unknown']:,} masked unknowns.")
    event_lines+=['',f"The four image fits used **{events['training_optimizer_steps']:,} actual optimizer steps** in total, 10,000 per source/seed. Seeds are 20260909 and 314159. The compact nine-frame parent-centered image encoder was trained from scratch on event bags. Daughter geometry is permutation invariant; cached image embedding inference was checked against direct inference. Both source directions and trained model hashes were frozen before comparative event scoring.",
        '', 'The image objective samples equal positive/negative event-group risk. Logistic fits normalize positive bag weight and balance classes. Their scores are not calibrated event-prevalence posteriors. All supported positive alternatives remain in training; incomplete sparse contexts are masked. The 25/50/100% source positive-group diagnostic fits are source resubstitution evidence, not independent validation. Secondary seeds were trained and compared at the probability level; they do not add a retrospectively selected deployment configuration.', '']
    event_lines+=['The source-label audit found that the frozen negative-sampling inclusion field records a parent-group sampling fraction, not the exact row inclusion probability when hard and ordinary strata coexist. It was not used by training. The exported table names that field parent_group_sampling_fraction; it must not be used as an inverse-probability weight or calibration propensity. The audit preserved the original label/model fingerprints.', '']
    for name,r in events['models'].items():
        if 'actual_steps' in r:
            event_lines.append(f"- {name}: {r['actual_steps']:,} steps, {r['seconds']:.1f} seconds, {r['unique_positive_groups']} positive groups, {r['positive_rows_seen']:,}/{r['positive_rows']:,} positive alternatives observed by the optimizer.")
    event_lines+=['', 'These are observed event bags, not certified distinct biological divisions across overlapping crops. Candidate counts measure search cost; they do not increase the number of independent positive events.', '']
    cov=events['coverage']
    if 'fallback' in cov:
        sizes={p:sum(r['proposals'] for r in cov[p].values()) for p in ['base','fallback']}
        covered={p:sum(r['covered_events'] for r in cov[p].values()) for p in ['base','fallback']}
        unknown=sum(r['unknown'] for r in cov['fallback'].values())
        event_lines += [f"The wider pool costs {sizes['fallback']/sizes['base']:.2f}× as many alternatives for {covered['fallback']-covered['base']} additional covered event observations ({covered['base']} → {covered['fallback']}). Of its {sizes['fallback']:,} alternatives, {unknown:,} ({unknown/sizes['fallback']:.2%}) have unknown sparse labels and remain masked. This is candidate-label availability, not biological division prevalence.", '']
    cut=lines.index('## V360: combinations and regret')
    lines[cut:cut]=event_lines
    try:
        from .association_report import paragraphs
        extra=paragraphs(ctx)
    except ImportError:extra=[]
    if extra:
        cut=lines.index('## V360: combinations and regret')
        lines[cut:cut]=extra
    oracle_lines=['## Source-only oracle feasibility','',
        'These probes read source annotations and cannot run as inference. They are legal heuristic constructions, not global bounds. Both non-oracle prediction directions were frozen before the retrospective event-oracle diagnostics. Their deltas still use the v2 incumbent.', '']
    for arm,groups in oracles.items():
        for embryo,r in groups.items():
            oracle_lines.append(f"- {arm}, {embryo}: {r['score']:.12f} ({r['delta_v2']:+.12f} versus v2); division TP/FP/FN {r['division_tp']}/{r['division_fp']}/{r['division_fn']}.")
    oracle_lines+=['']
    cut=lines.index('## V360: combinations and regret');lines[cut:cut]=oracle_lines
    measured=['## Measured event, rescue and combination outcomes','']
    if all(not r['promotion_gate_passed'] for r in families['division']):
        r=families['division'][0];c=r['counts']
        measured += [f"Every event arm failed the both-embryo promotion gate. The best pooled event arm, {r['variant']}, changed division TP by {c['division_tp']-int(base['division_tp']):+d} and FP by {c['division_fp']-int(base['division_fp']):+d}; its delta against v2 is {r['delta_v2']:+.12f}. The oracle contrast establishes useful candidate feasibility on the observed labels, while the learned policies fail to select those events reliably. Class-balanced uncalibrated scores and largely unlabelled candidate contexts are plausible contributors; these results do not isolate their causal contributions.", '']
    for family in ['division','rescue','combinations']:
        measured += [f"### {family.capitalize()}", '']
        for r in families[family]:
            gate='invalid control' if r['invalid_control'] else 'passes both-embryo gate' if r['promotion_gate_passed'] else 'does not pass promotion gate'
            d=r['embryo_deltas_v2'];c=r['counts']
            measured.append(f"- {r['variant']}: {r['score']:.12f}, delta versus v2 {r['delta_v2']:+.12f}; 44b6 {d['44b6']:+.12f}, 6bba {d['6bba']:+.12f}; division TP/FP/FN {c['division_tp']}/{c['division_fp']}/{c['division_fn']}; {gate}.")
        measured += ['']
    cut=lines.index('## V360: combinations and regret');lines[cut:cut]=measured
    (ctx.out/'final_report.md').write_text('\n'.join(lines))
    dashboard(ctx,summary,points,rows)
    public=ctx.repo/'results/strong-tracker-v3';public.mkdir(parents=True,exist_ok=True)
    names=['summary.json','final_report.md','dashboard.html','score_rows.csv','operating_points.csv','winning_config.json',
        'selected_prediction_lock.json','baseline_verification.json','metric_verification.json','census_summary.json',
        'score_completeness.json','census_endpoint_summary.json','heatmap_rescue_summary.json','preservation_check.json','selected_regret_summary.json','selected_regret_by_clip.csv',
        'rescue_summary.json','rescue_quantization_diagnostic.json','fresh_delivery_receipt.json',
        'fresh_selected_score_summary.json','fresh_selected_score_rows.csv','selected_clip_changes.csv',
        'source_oracle_comparison.json','family_outcomes.json']
    for name in names:
        p=ctx.out/name
        if p.exists():shutil.copyfile(p,public/name)
    write_json(public/'runtime_summary.json',resources)
    print(f'Report generated: {selected} score {best["score"]:.16f} delta {delta:+.16f}',flush=True)
