"""Measured offline report and dashboard; no hypothetical gains are promoted."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from .common import OUT,REPO,clean,now,read_json,stage,write_json
from .evaluate import collect


def family(v):
    if v=='identity':return 'Baseline'
    if 'bypass' in v:return 'Pipeline ablations'
    if v.startswith(('DE','EF')):return 'Combinations'
    if v.startswith('D_'):return 'Divisions'
    if v.startswith('E_'):return 'Associations'
    if 'risk' in v:return 'Learned risk'
    if 'temporal' in v:return 'Temporal image'
    return 'Native selection'


def run(args):
    collect()
    completeness=read_json(OUT/'score_completeness.json')
    assert len(completeness['complete_variants'])==104 and completeness['score_rows']==104*199,completeness
    f=pd.read_csv(OUT/'operating_points.csv');f['family']=f.variant.map(family)
    base=f[f.variant=='identity'].set_index('embryo');pool=f[f.embryo=='pooled']
    eligible=[]
    for v in pool.variant:
        r=f[f.variant==v]
        if (r[r.embryo!='pooled'].delta>=-1e-12).all():eligible.append(v)
    best=pool[pool.variant.isin(eligible)].sort_values(['score','variant'],ascending=[False,True]).iloc[0]
    # Prefer the separately scored, strict-bounds export at equal score.
    if abs(float(pool.set_index('variant').loc['bypass_motion_bounds','score'])-best.score)<1e-12:
        best=pool[pool.variant=='bypass_motion_bounds'].iloc[0]
    selected=f[f.variant==best.variant].set_index('embryo')
    decision='significant_local_gain' if best.delta>=.02 else 'smaller_valid_gain' if best.delta>0 else 'no_gain'
    assert best.variant=='bypass_motion_bounds','Selected export must be updated for a different measured winner'
    census=pd.read_csv(OUT/'division_failure_census.csv');edges=pd.read_csv(OUT/'edge_failure_census.csv')
    temporal=read_json(OUT/'temporal_model_lock.json');models=read_json(OUT/'model_lock.json');risk=read_json(OUT/'risk_model_lock.json')
    resources=pd.read_csv(OUT/'resource_samples.csv')
    stage_summary=read_json(OUT/'stage_summary.json');oracles=read_json(OUT/'oracle_diagnostics.json')
    fullfits=[r for r in temporal['full_fits'] if r['fraction']==1.]
    details=[d for p in (OUT/'evaluation/risk').glob('*.json') for d in read_json(p)['actual_deletions']]
    summary=dict(study_id='strong-tracker-v2',created=now(),status='measured_complete',decision=decision,
        selected_variant=best.variant,selection='Exploratory stage-ablation winner chosen after inspection; not an untouched prespecified primary comparison',
        provenance='Contaminated public upstream checkpoints; learned repairs and selectors use source-only labels in both directions',
        validation='Two reused embryos; partial image registration; no independent inner folds or bootstrap confidence intervals',
        samples=199,variants=104,per_sample_score_rows=20696,
        baseline={e:base.loc[e].to_dict() for e in ['44b6','6bba','pooled']},
        selected={e:selected.loc[e].to_dict() for e in ['44b6','6bba','pooled']},
        division_failure_reasons=Counter(census[census.kind=='gt_division'].primary_reason),
        division_fp_reasons=Counter(census[census.kind=='predicted_division_fp'].primary_reason),
        edge_failure_reasons=Counter(edges.kind+':'+edges.primary_reason),
        baseline_identity=read_json(OUT/'baseline_verification.json'),replay=read_json(OUT/'replay_identity_receipt.json'),
        registration=read_json(OUT/'registration_validation.json'),source_training=models['sources'],
        temporal_training=temporal['full_fits'],temporal_gpu_hours=temporal['gpu_synchronized_hours'],
        risk_training=risk['training'],actual_group_deletions=len(details),
        deletions_with_rematching_changed_tp_loss=sum(d['actual_tp_loss']!=d['survivor_tp_loss'] for d in details),
        peak_aggregate_rss_gib=float(resources.rss_bytes.max()/1024**3),peak_gpu_used_gib=float(resources.gpu_used_mib.max()/1024),
        new_output_gib=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())/1024**3,
        historical_v1_clean_gain=.002790679234791149,historical_public_transfer_score=float(pool.set_index('variant').loc['F_v1_transfer_control','score']),
        focus='Not run; no FOCUS checkpoint was present, no gated terms were accepted, and no images were uploaded. Frozen-node graph headroom was dominant.')
    # Detailed registrations and event IDs remain only in their local artifact.
    summary['registration']={k:v for k,v in summary['registration'].items() if k not in ['pairs','inconsistent_cycles']}
    write_json(OUT/'summary.json',summary)
    def val(v,e='pooled',key='score'):
        return float(f[(f.variant==v)&(f.embryo==e)].iloc[0][key])
    native_best=pool[pool.family=='Native selection'].sort_values('score',ascending=False).iloc[0]
    image_best=pool[pool.family=='Temporal image'].sort_values('score',ascending=False).iloc[0]
    primary=val('DEF_risk_transfer')
    text=f'''# Strong tracker v2 — measured results

Decision: **{decision}**. The selected exploratory pipeline scores **{base.loc['pooled','score']:.6f} → {best.score:.6f} (+{best.delta:.6f})** on all 199 clips. It bypasses the original motion relinker, retains the learned neural associations, and runs the remaining Harmonic Fusion repairs unchanged. The exported winner enforces image bounds; the separately rescored bounds correction changes the score by zero.

Both embryo directions improve: **44b6 {base.loc['44b6','score']:.6f} → {selected.loc['44b6','score']:.6f} ({selected.loc['44b6','delta']:+.6f})**; **6bba {base.loc['6bba','score']:.6f} → {selected.loc['6bba','score']:.6f} ({selected.loc['6bba','delta']:+.6f})**. This exceeds the proposed +0.02 pooled local target. Selection followed the failure census and phase ablations, so the winner is exploratory and selected after inspecting results.

The public checkpoints were trained or selected using supplied embryos, and both embryos had already been examined. Source-only repair/selector training does not remove upstream contamination. These are local operational results, not clean OOF estimates or a hidden-test forecast. The historical 0.946 public leaderboard score is a different evaluation population; no local delta is added to it.

## V200–V210: exact baseline, real stages, failure census

The fresh official baseline reproduces 122,201 edge TP, 6,885 FP, 6,682 FN, and 23 division TP / 97 FP / 128 FN. All 199 final input hashes and the original two replay pilots were verified. Instrumentation follows six actual coarse phases, rather than treating 27 helper functions as 27 transformations.

All 199 repaired graphs reproduce the sealed baseline after a documented serializer compatibility adjustment at two floating-point half-integer ties. Native replay floats and the two failed parity receipts are retained. All edges and IDs already agreed. Native rounding was independently scored and gives the same combined baseline score. The six original out-of-bounds points are retained for baseline identity; the separate bounds-only experiment also leaves the score unchanged.

The raw neural graph scores **0.914903**. Motion relinking lowers this to **0.892978**; gap closing gives **0.894511**, safe divisions **0.903685**, pruning **0.903789**, and smoothing **0.911774**. Cumulative changes are descriptive, not proof that a stage should be removed. Complete bypass experiments establish that skipping safe divisions scores **0.902672**, skipping smoothing **0.903789**, and skipping motion relinking **{best.score:.6f}**.

The census covers all **151 GT division observations, 97 division FP, 6,682 edge FN and 6,885 edge FP**, using fresh official local division assignments at every phase. Initially, 99 missed divisions had both daughter lineages locally matched but no surviving fork; 21 lacked daughter candidates, four involved assignment competition, three lacked parent evidence, and one had incorrect local topology. The final census adds explicit stage attribution and flags for correct forks lost or relocalized by later repair. There are **3,327** FN edges in the available alternative pool; **1,437** have exported native pre-ILP scores. These counts are diagnostics, not guaranteed recoverable gains.

Stable raw node IDs map exactly to pre-ILP coordinates, one-to-one. Only **37,822** final nodes are inserted relative to the raw graph. The old nearest-center confidence mapping marked 383,339 nodes missing, including many relocated original detections; v2 carries original confidence and displacement explicitly instead of treating missing confidence as false-cell evidence.

Three impossible-inference interventions were actually rescored: fixed-node GT-guided legal rewiring **{oracles['summaries']['oracle_fixed_nodes_links']['pooled']['score']:.6f}**, GT-centered candidate injection plus rewiring **{oracles['summaries']['oracle_gt_center_injection']['pooled']['score']:.6f}**, and GT-selected pruning **{oracles['summaries']['oracle_gt_pruning']['pooled']['score']:.6f}**. Their search is heuristic and incomplete, not a rigorous global upper bound. Scores above one are possible under the official count adjustment. Oracle code, graphs and labels are isolated under evaluation paths, and inference blocks those paths.

## V220–V230: constrained divisions and associations

The bounded fork pool covers **102 of 151** annotated division observations and supplies **204** positive hypotheses. Source 44b6 contributes 39 hypotheses from 18 covered event observations; source 6bba contributes 165 from 84. All available positive hypotheses are used. Uncovered events remain explicit coverage failures, not silent negative labels. Contradictory negatives are sampled by predicted tracklet, with inverse inclusion weights; incomplete or time-shifted division neighborhoods are masked.

The native edge learner uses **125,528** positive candidate transitions. Fork features include native probabilities/margins, physical distances, barycenter motion, image intensity, daughter separation across neighboring frames, and boundary masks. Continuation/fork choices are solved jointly over competing source/target components using a bounded binary program; association assignments keep confident external links and existing forks fixed. Every output is checked for merges, duplicate edges, degree limits, integer coordinates and consecutive frames.

The fixed temporal division arm at p=0.05 recovers 35 divisions but raises FP to 220 and scores **{val('D_temporal_p0.05'):.6f}**. This negative result is retained. The strongest measured association setting scores **{val('E_hgb_m0.5'):.6f} (+{val('E_hgb_m0.5',key='delta'):.6f})** with gains in both directions. It recovers 605 previously missed TP edges and loses 31 baseline TP edges. Its FP count falls from 6,885 to 5,941. The more conservative prespecified E setting scores **{val('E_hgb_m1.5'):.6f}**; prespecified D+E scores **{val('DE_primary'):.6f}**.

The selected complete motion-bypass pipeline has **{int(best.edge_tp):,} edge TP, {int(best.edge_fp):,} FP, {int(best.edge_fn):,} FN**, and **{int(best.division_tp)} division TP / {int(best.division_fp)} FP / {int(best.division_fn)} FN**. Its main improvement is association accuracy. Division recovery remains a substantial unresolved opportunity.

## V240: native selectors and observed deletion risk

Native logistic and seven-leaf boosted models use every source matched-positive node: **19,980** in 44b6 and **110,979** in 6bba. Geometry-free quality/temporal features are primary; normalized geometry is an ablation. Negative sampling is stratified within tracklets by origin, depth and density, with recorded inclusion weights. Metrics are evaluated on the full natural node population. Labels mean matched sparse annotation, not biological cell truth.

Node, whole-tracklet, five-frame segment and predicted-fork-protected actions are tested at requested retention 1, .995, .99, .98, .95, .9, .8, .7 and .5. Realized budgets, TP survival, new TP, counts and divisions are recomputed exactly. The best native selection setting in hindsight is **{native_best.variant}**, scoring **{native_best.score:.6f} ({native_best.delta:+.6f})**. The historical DoG transfer control remains **{val('F_v1_transfer_control'):.6f}**, consistent with the preserved v1 failure.

The learned risk model uses all **164,765** source action groups, including **6,729** groups incident to matched TP edges, plus **{len(details):,} actual group-deletion rescoring experiments**. Actual run-level loss includes rematching, division changes, FP changes and count weights. Rematching changes the naive TP-loss estimate in **{summary['deletions_with_rematching_changed_tp_loss']}** rescored actions. The primary risk threshold can abstain and caps removal at 10%; it does not force deletion in every clip. Its score is **{val('F_risk_threshold'):.6f} ({val('F_risk_threshold',key='delta'):+.6f})**.

Quality profiles and explicit low-response/missing-confidence summaries are retained. There is no dense biological true/false-cell audit, so neither native selection nor the risk target establishes latent annotator preference independently of ordinary detection quality.

## V250: full temporal image study

Six **1,421,057-parameter** five-frame triplanar fits were completed on the RTX 4090: 10%, 30% and 100% of positive tracklet groups in each source embryo, **5,000 optimization steps per fit (30,000 total)**. Both full fits used every positive observation and every positive source tracklet group: 19,980 / 981 groups and 110,979 / 6,344 groups. The encoder combines 13 µm local and 26 µm context views; panel geometry and temporal boundary masks are explicit. Shared intensity gain is the only augmentation; no inconsistent panel flips or rolled borders are used.

Learning curves record actual steps, unique positives/groups, weighted BCE, memory and synchronized time. All six final checkpoints were frozen before comparative image scoring. Source-prior correction is used for probabilities, with no target-label calibration. Classification results for all three training fractions use the full opposite-embryo population. The best full-image selector in hindsight is **{image_best.variant}**, scoring **{image_best.score:.6f} ({image_best.delta:+.6f})**. It does not beat the selected graph pipeline.

The six fits consumed **{temporal['gpu_synchronized_hours']:.6f} GPU-synchronized training hours**; actual steps and positive coverage establish the fit size, not elapsed time alone. FOCUS was not downloaded or run. No authorized local FOCUS checkpoint was present, and the census showed greater immediate headroom from existing nodes and links. This image arm tests selection risk; it cannot restore absent daughter candidates.

## V260: combinations and validation limits

There are **104 complete variants and 20,696 per-sample score rows**. Identity, D-only, E-only, F-only, D+E and D+E+F are all evaluated. The prespecified D+E+learned-risk combination scores **{primary:.6f} ({primary-base.loc['pooled','score']:+.6f})**. Additional E+F combinations are explicitly exploratory. Filters calibrated on the original graph are labelled as transfer arms when applied after repair; their new action groups and fork protection are recomputed from predicted structure.

Every graph variant uses fresh official matching and division evaluation. Every aggregation requires the full expected sample set and is checked against the pinned official run-level weighting. FP changes alter weights, so neither average clip scores nor count-only approximations are used as results. The selected 199 graph files also reproduce byte-for-byte in annotation-unavailable export, with strict image bounds and graph validity.

All **219** fresh image-patch checks for the 73 known crop translations pass, with no inconsistent cycles. Transforms cover 65 clips of 44b6 and none of 6bba. They identify 149 provisional event groups among 151 observations, but unknown overlaps remain. Independent purged source blocks cannot be certified, so fixed limited settings replace inner tuning; no bootstrap confidence intervals are fabricated. A clean upstream source-trained replication remains unavailable. The completed v1 clean classical gain (+0.002791) and failed public transfer remain historical evidence, unchanged.

## V270: resources, artifacts and reproduction

The original GPU study interpreter, pinned metric revision `075fc5f5a52d11077f9dc2b074644618f26939e2`, and documented PyTorch metadata exception are preserved. No package stack or driver was upgraded. CUDA library-path initialization follows `scripts/root_remote_env.sh`. Jobs run in tmux with per-stage logs/checkpoints, at most about 40 active CPU threads, and GPU 0.

Recorded peak aggregate process RSS is **{summary['peak_aggregate_rss_gib']:.2f} GiB** and peak GPU use is **{summary['peak_gpu_used_gib']:.2f} GiB**. New v2 disk use at report generation is **{summary['new_output_gib']:.2f} GiB**, within the 60 GiB cap. Resource logs remain in `resource_samples.csv`; training checkpoints and exact source/config/model hashes are retained.

The complete local artifact root is `/kaggle/working/cell-tracking/strong-tracker-v2/`. Start with `dashboard.html`, `score_rows.csv`, `operating_points.csv`, `stage_scores.csv`, both failure censuses, `oracle_diagnostics.json`, `native_classifier_metrics.csv`, `learning_curves.csv`, `winning_config.json`, `selected_prediction_lock.json`, `validation_receipt.json` and `artifact_manifest.json`. Detailed event coordinates, predictions, image stores, weights and matching files remain local. Reproduction commands are in `docs/strong-tracker-v2.md` and `reproduce.sh`.

Before this instance is destroyed, copy out the **entire v2 output root**, `work/strong-tracker-v2/`, and the new Git commit. Reproduction also needs the preserved v1 store, pinned evaluator and original competition inputs already mirrored on this host. The instance disk is not a persistent attached volume. No Kaggle submission, forum post, paid API, external model download or remote publication was performed.
'''
    (OUT/'v2_report.md').write_text(text)
    data=dict(summary=summary,points=f.to_dict('records'),stages=stage_summary,
        oracles=oracles['summaries'],curves=pd.read_csv(OUT/'temporal_learning_curves.csv').to_dict('records'),
        classifier=pd.read_csv(OUT/'native_classifier_metrics.csv').to_dict('records'))
    template=(REPO/'tools/strong_tracker_v2/dashboard_template.html').read_text()
    (OUT/'dashboard.html').write_text(template.replace('__DATA__',json.dumps(clean(data),allow_nan=False).replace('</','<\\/')))
    stage('V270','report_rendered_pending_browser_and_seal',decision=decision,score=float(best.score),delta=float(best.delta))
    print(decision,best.variant,float(best.score),float(best.delta),flush=True)
