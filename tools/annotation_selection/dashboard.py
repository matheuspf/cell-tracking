"""Self-contained HTML, exportable measured plots, and the final Markdown report."""
from __future__ import annotations

import json
import base64
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import OUT,REPO,clean,now,read_json,sha,write_json
from .features import FEATURES


def fmt(value,digits=4):
    return 'unavailable' if value is None or isinstance(value,float) and not np.isfinite(value) else f'{value:,.{digits}f}'


def table(df,columns):
    text='| '+' | '.join(columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'
    for _,r in df.iterrows():
        values=[]
        for c in columns:
            v=r[c];values.append(fmt(v) if isinstance(v,(float,np.floating)) else str(v))
        text+='| '+' | '.join(values)+' |\n'
    return text


def plots(sweep,coverage):
    root=OUT/'plots';root.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    cv=coverage[coverage.embryo!='pooled']
    fig,ax=plt.subplots(figsize=(6.5,4))
    x=np.arange(len(cv));ax.bar(x-.17,cv.reference_coverage_ratio_of_sums*100,.34,label='GT / supplied total estimate')
    ax.bar(x+.17,cv.candidate_membership_prevalence*100,.34,label='Matched / candidate observations')
    ax.set_xticks(x,cv.embryo);ax.set_ylabel('Percent of the stated denominator');ax.legend(fontsize=8)
    ax.set_title('Measured sparse coverage; estimated all-cell denominator')
    fig.tight_layout();fig.savefig(root/'coverage.png');fig.savefig(root/'coverage.svg');plt.close(fig)
    mids=['hgb_all_leaf7','quality','image_only_seed20260908','image_plus_tabular_seed20260908']
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for embryo,ax in zip(cv.embryo,axs[0]):
        for mid in mids:
            policy='quality_nodes' if mid=='quality' else 'membership_nodes'
            s=sweep[(sweep.embryo==embryo)&(sweep.model_id==mid)&(sweep.policy==policy)].sort_values('realized_keep')
            ax.plot([*s.realized_keep,1.],[*s.annotation_recall,1.],'o-',label=mid.replace('_seed20260908',''))
        ax.plot([0,1],[0,1],':',color='#888',label='random expectation')
        ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Realized node keep fraction',ylabel='Baseline annotation-match recall',title=embryo)
    for embryo,ax in zip(cv.embryo,axs[1]):
        for mid in mids:
            policy='quality_tracklets' if mid=='quality' else 'membership_tracklets'
            s=sweep[(sweep.embryo==embryo)&(sweep.model_id==mid)&(sweep.policy==policy)].sort_values('realized_keep')
            ax.plot([*s.realized_keep,1.],[*s.delta,0.],'o-',label=mid.replace('_seed20260908',''))
        ax.axhline(0,color='#888',linestyle=':');ax.set(xlabel='Realized tracklet node retention',ylabel='Official local score delta',title=embryo)
    axs[0,0].legend(fontsize=7);axs[1,0].legend(fontsize=7)
    fig.suptitle('Frozen cross-embryo selectors — every plotted point was measured')
    fig.tight_layout();fig.savefig(root/'retention_and_score.png');fig.savefig(root/'retention_and_score.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for embryo,ax in zip(cv.embryo,axs):
        s=sweep[(sweep.embryo==embryo)&(sweep.model_id=='hgb_all_leaf7')&(sweep.policy=='membership_tracklets')].sort_values('requested_keep')
        ax.plot(s.realized_keep,s.deleted_annotation_rate,'o-',label='Deleted annotation rate')
        ax.axhline(float(s.prevalence.iloc[0]),color='#888',linestyle=':',label='Natural candidate prevalence')
        ax.set(xlabel='Realized node retention',ylabel='Annotation rate among deleted nodes',title=embryo);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'deleted_group.png');fig.savefig(root/'deleted_group.svg');plt.close(fig)
    sweep.to_csv(root/'all_plot_data.csv',index=False)
    profiles=pd.read_csv(OUT/'feature_profiles.csv')
    fig,axs=plt.subplots(4,4,figsize=(13,10))
    for feature,ax in zip(FEATURES,axs.flat):
        sub=profiles[profiles.feature==feature]
        for embryo,g in sub.groupby('embryo'):
            ax.plot(g.mean_feature,g.annotation_rate*100,'o-',markersize=3,label=embryo)
        if sub.empty:ax.text(.5,.5,'Constant feature; no bins',ha='center',va='center',transform=ax.transAxes)
        ax.set_title(feature,fontsize=9);ax.tick_params(labelsize=7);ax.set_ylabel('Matched GT (%)',fontsize=7)
    axs[0,0].legend(fontsize=7)
    fig.suptitle('Post-lock feature maps on the clean candidate population — associations, not causal effects')
    fig.tight_layout();fig.savefig(root/'feature_profiles.png');fig.savefig(root/'feature_profiles.svg');plt.close(fig)
    calibration=pd.read_csv(OUT/'calibration.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for embryo,ax in zip(cv.embryo,axs):
        maximum=.01
        for mid in ['hgb_all_leaf7','hgb_appearance_quality_leaf7','image_only_seed20260908','image_plus_tabular_seed20260908']:
            s=calibration[(calibration.embryo==embryo)&(calibration.model_id==mid)]
            ax.plot(s.mean_score,s.annotation_rate,'o-',label=mid.replace('_seed20260908',''))
            maximum=max(maximum,s.mean_score.max(),s.annotation_rate.max())
        ax.plot([0,maximum],[0,maximum],':',color='#888');ax.set(title=embryo,xlabel='Mean predicted probability in bin',ylabel='Measured annotation fraction')
    axs[0].legend(fontsize=7);fig.suptitle('Natural-prevalence calibration · fixed probability bins · no independent-group intervals')
    fig.tight_layout();fig.savefig(root/'calibration.png');fig.savefig(root/'calibration.svg');plt.close(fig)


def build(summary,sweep,coverage,classifiers):
    plots(sweep,coverage)
    primary=pd.DataFrame(summary['primary']);inv=read_json(OUT/'inventory_summary.json');env=read_json(OUT/'environment.json')
    folds=read_json(OUT/'fold_manifest.json');lock=read_json(OUT/'prediction_lock.json')
    cls=classifiers[(classifiers.model_id=='hgb_all_leaf7')&(classifiers.subset=='all')]
    baseline=sweep[(sweep.model_id=='identity')&(sweep.policy=='identity')]
    oracle=sweep[sweep.model_id=='oracle'].sort_values('score',ascending=False).groupby('embryo',sort=False).head(1)
    primary_md=table(primary,['embryo','n','realized_keep','annotation_recall','phi','baseline_score','score','delta'])
    cover_md=table(coverage,['embryo','clips','annotated_nodes','annotated_edges','gt_divisions','estimated_total','reference_coverage_ratio_of_sums','candidate_nodes','matched_nodes','gt_detection_recall'])
    classifier_md=table(cls,['embryo','n','prevalence','auroc','average_precision','ap_over_prevalence','brier'])
    attribution_md=table(primary,['embryo','delta','count_effect','graph_effect','division_effect','alpha0_delta'])
    exact_md=table(pd.DataFrame(summary['matched_budget_primary']),['embryo','primary_score','control_score','primary_minus_control','primary_annotation_recall','control_annotation_recall','realized_nodes'])
    required_md=table(primary,['embryo','break_even_weighted_tp_ratio','measured_weighted_tp_ratio'])
    oracle_md=table(oracle,['embryo','requested_keep','realized_keep','score','delta'])
    image_fits=[]
    for e in folds['embryos']:
        for f in read_json(OUT/'models'/e/'image_complete.json')['fits']:
            image_fits.append(dict(source=e,model=f['id'],candidates=f['fit_candidates'],gpu_seconds=f['gpu_seconds'],
                                   peak_vram_mib=f['peak_vram_bytes']/2**20,observations_per_second=f['observations_per_second'],loader_seconds=f['loader_seconds']))
    image_md=table(pd.DataFrame(image_fits),['source','model','candidates','gpu_seconds','peak_vram_mib','observations_per_second','loader_seconds'])
    gpu_seconds=sum(f['gpu_seconds'] for f in image_fits)
    baseline_md=table(baseline,['embryo','num_pred_nodes','baseline_count_ratio_of_sums','edge_tp','edge_fp','edge_fn','edge_jaccard','division_jaccard','adj_edge_jaccard','score'])
    graph_md=table(primary,['embryo','edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','old_tp_endpoint_survival','old_tp_rematched_survival','newly_recovered_tp','rematched_gt_node_recall'])
    comparator_ids=['hgb_all_leaf7','hgb_appearance_quality_leaf7','quality','image_only_seed20260908','image_plus_tabular_seed20260908','image_only_seed314159','image_plus_tabular_seed314159']
    comparator_md=table(classifiers[(classifiers.subset=='all')&classifiers.model_id.isin(comparator_ids)],['embryo','model_id','auroc','average_precision','ap_over_prevalence'])
    sensitivity_md=table(classifiers[(classifiers.model_id=='hgb_all_leaf7')&(classifiers.subset!='independent_randomized_outer_target')],
                         ['embryo','subset','n','prevalence','auroc','average_precision','ap_over_prevalence'])
    randomized=classifiers[classifiers.subset=='independent_randomized_outer_target']
    randomized_md=table(randomized,['embryo','model_id','prevalence','auroc','average_precision'])
    negative_md=table(sweep[sweep.model_id.str.contains('shuffled|synthetic')&(sweep.policy=='membership_tracklets')&(sweep.requested_keep==.9)],
                     ['embryo','model_id','annotation_recall','phi','delta'])
    seed_md=table(pd.read_csv(OUT/'image_seed_results.csv'),['embryo','model_id','annotation_recall','score','delta'])
    random_md=table(pd.read_csv(OUT/'random_exact_seed_results.csv'),['embryo','seeds','mean_control_score','min_control_score','max_control_score','mean_primary_minus_control'])
    diagnostics=pd.read_csv(OUT/'matching_diagnostics.csv');diagnostics['embryo']=diagnostics.dataset.str.split('_').str[0]
    match_summary=diagnostics.groupby('embryo')[['candidate_nodes','matched_nodes','ambiguous_candidates','missed_gt']].sum().reset_index()
    match_summary['ambiguous_candidate_fraction']=match_summary.ambiguous_candidates/match_summary.candidate_nodes
    match_summary.to_csv(OUT/'matching_summary.csv',index=False)
    matching_md=table(match_summary,['embryo','candidate_nodes','matched_nodes','ambiguous_candidates','ambiguous_candidate_fraction','missed_gt'])
    public_sweep=pd.read_csv(OUT/'public_retention.csv') if (OUT/'public_retention.csv').exists() else pd.DataFrame()
    public_md=table(pd.DataFrame(summary['public']['primary']),['embryo','baseline_count_ratio_of_sums','realized_keep','annotation_recall','baseline_score','score','delta']) if summary.get('public') else 'Public scoring unavailable.'
    confidence_note=(f"Public detector confidence is assigned from the nearest same-frame pre-ILP center within 2 um. "
                     f"{summary['public']['confidence_missing_nodes']:,} final nodes ({summary['public']['confidence_missing_fraction']:.2%}) have no such center; "
                     "their confidence is zero with a persisted missing flag. Final graph nodes without an associated pre-ILP detection limit the interpretation of this diagnostic confidence control. "
                     "Source: [public_summary.json](public_summary.json), per-clip receipts under `baseline/public/`.") if summary.get('public') else ''
    public_pooled=next((r for r in (summary.get('public') or {}).get('primary',[]) if r['embryo']=='pooled'),None)
    public_executive=(f"Transferring the frozen primary rule to Harmonic Fusion changes its pooled diagnostic score from "
                      f"{public_pooled['baseline_score']:.6f} to {public_pooled['score']:.6f} ({public_pooled['delta']:+.6f}). "
                      "This primary transfer harms the stronger tracker. The public lane is contaminated by checkpoint training/selection on the supplied embryos, "
                      "so it is an operational diagnostic and cannot establish clean unseen-embryo transfer. Source: [public_retention.csv](public_retention.csv).") if public_pooled else ''
    resource=pd.read_csv(OUT/'resource_samples.csv')
    resource_note=f'Measured late-run monitoring maxima: {resource.summed_rss_gib.max():.2f} GiB summed task RSS (shared pages can be counted more than once), {resource.device_used_mib.max()/1024:.2f} GiB total device memory including desktop/other processes. Sampling began after training; image-fit memory is measured separately below. The full input archive verification checked 24,886 files and 87,609,892,618 bytes by CRC32. Detailed samples are in [resource_samples.csv](resource_samples.csv).'
    public_note='The public checkpoint lane is reported separately in [public_summary.json](public_summary.json). It is contaminated: the temporal checkpoint trained on both embryos, and other checkpoint selection also used supplied embryos. Frozen clean selectors are transferred to its candidate population without refitting. It is not clean OOF evidence.' if (OUT/'public_summary.json').exists() else 'Public-model inference/evaluation status is recorded in the local stage ledger; no unmeasured public score is asserted here.'
    pooled=primary[primary.embryo=='pooled'].iloc[0]
    quality_pooled=next(r for r in summary['matched_budget_primary'] if r['embryo']=='pooled')
    answers=f'''- **H1, observable membership:** primary AUROC is {cls.auroc.min():.3f}–{cls.auroc.max():.3f}, with AP {cls.ap_over_prevalence.min():.2f}–{cls.ap_over_prevalence.max():.2f} times natural candidate prevalence. At the primary budget, pooled baseline annotation-match recall is {pooled.annotation_recall:.2%} and phi/MCC is {pooled.phi:.4f}. These are sparse-match targets, not adjudicated true-cell or annotator-selection probabilities.
- **H2, official local score:** pooled score changes from {pooled.baseline_score:.6f} to {pooled.score:.6f} ({pooled.delta:+.6f}) at {pooled.realized_keep:.2%} realized node retention. The gain relative to ordinary confidence filtering at exactly matched per-video node counts is {quality_pooled['primary_minus_control']:+.6f}. Direction-specific results remain visible below.
- **H3, source of the change:** count adjustment contributes {pooled.count_effect:+.6f}, graph changes {pooled.graph_effect:+.6f}, and divisions {pooled.division_effect:+.6f}. The alpha=0 pooled delta is {pooled.alpha0_delta:+.6f}. These contributions use the full measured aggregator. Sources: [retention.csv](retention.csv), [classifier_metrics.csv](classifier_metrics.csv), [matched_budget_summary.csv](matched_budget_summary.csv).'''
    text=f'''# Annotation-selection study: measured local results

Status: **{summary['status']}**. Generated {summary['created']}.

## Executive answer

{answers}

The exact-budget comparison demonstrates an advantage over raw DoG response confidence on these candidates. It does not isolate annotator preference: appearance-only learned selectors also predict sparse membership, and unmatched candidates mix real unannotated cells with detection, localization and duplicate errors. The missing independent cell-quality/census labels leave annotation-specific selection among real cells unresolved.

{public_executive}

The complete clean study evaluates **199 training clips in both embryo directions**, using image-only classical candidates and source-embryo annotation membership targets. Both directions' models and all 199 prediction files were frozen and reproduced byte-for-byte with annotation reads blocked before official outer evaluation. The primary rule was fixed before outer outcomes: boosted trees with 7 leaves, all allowed feature groups, and coherent tracklet selection at requested keep 0.9.

{primary_md}

These are local scores from the pinned official implementation, not hidden-test or leaderboard gains. Inner tuning was unavailable: released crop/time origins are absent, and image fingerprints demonstrate overlaps. One conservative overlap supergroup per embryo prevents claiming independent clip-level validation or useful bootstrap confidence intervals. The protocol's fixed-setting fallback was therefore used; this is **not a source-selected optimum**.

Exact sparse annotation counts and detector-conditional match membership are measured. The all-cell denominator is a supplied estimate; exact true-cell prevalence and latent annotator selection probability remain unidentified. Candidate confidence, localization, duplicate competition, and annotator membership can all contribute to predictability. No independent manual cell-quality labels were available.

The frozen primary outcome is **{summary['primary_outcome']}**. The study decision also considers the full descriptive sweep and the separately marked public diagnostic lane; it does not promote their best outer point as a newly validated configuration.

## Data and labeling

{cover_md}

Source: [coverage.csv](coverage.csv), [sample_inventory.csv](sample_inventory.csv). Counts are cell observations across dataset clips, not unique biological cells. Overlapping crops may count the same observation repeatedly. Both ratio-of-sums and mean-of-sample-ratios are preserved in the coverage table. Per-frame annotation counts are available in `evaluation/annotation_frames.csv`; no video-wide total estimate was distributed over frames as if measured.

The GEFF reader checked integer representability, unique node IDs, coordinate bounds, edge endpoints, duplicate edges, forward time, divisions and physical-scale agreement. All {len(folds['expected_samples'])} expected samples were present. Actual image metadata gives `(T,Z,Y,X)=(100,64,256,256)`, uint16 and spatial scale `(1.625,0.40625,0.40625)` micrometers. Detailed immutable source fingerprints are in [data_fingerprints.json](data_fingerprints.json).

The clean detector uses fixed multiscale difference-of-Gaussian peaks (1.2 and 1.8 um), threshold 0.025, physical NMS at 3.25 um, and a deterministic 7 um one-to-one temporal linker. The separate 0.05 response threshold is scored afresh. This baseline has no predicted forks, so it cannot recover divisions. That limits extrapolation to an advanced tracking pipeline.

Candidate ambiguity and missed supplied GT:

{matching_md}

Source: [matching_summary.csv](matching_summary.csv), [matching_diagnostics.csv](matching_diagnostics.csv). The ambiguity stratum contains unmatched candidates within 7 um of a supplied GT center; this is a localization/competition diagnostic, not an adjudicated cell-quality label. Per-clip match-distance medians and 95th percentiles remain available without mislabeling a median of clip medians as a pooled distance quantile.

The [blinded census pack](blinded_census/README.md) contains 48 independently sampled image ROIs, with known inclusion probabilities and nonoverlapping grid cores within each clip/frame. Raw crops have image halos and contain no GT markers or keep decisions. Manual labels remain blank. Global cross-clip nonoverlap is unresolved; no all-cell prevalence estimate is fabricated.

An additional [48-candidate precision audit](blinded_candidate_audit/README.md) samples within embryo, image-response and predicted-persistence strata. Its design uses no GT or keep decisions and was added after prediction lock solely for future independent manual assessment. Both the [census viewer](blinded_census/viewer.html) and [candidate viewer](blinded_candidate_audit/viewer.html) work offline, expose the z stack, and export entered judgments. No such judgments are included in training or claimed as measurements here.

## Provenance and validation

There are {folds['positive_overlap_pairs']} image-confirmed overlap pairs, recorded with consistent crop/time translations in [overlap_registration.json](overlap_registration.json). Failure to match sampled image patches was not treated as proof of independence. The resulting split is documented in [fold_manifest.json](fold_manifest.json), and fixed settings in [preregistration.json](preregistration.json).

- Metric revision: `{env['metric_revision']}`; the current official repository revision agreed with the handover pin. The [pinned metric source](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/{env['metric_revision']}/src/tracking_cellmot/metrics.py), local Kaggle evaluation page and organizer patch announcements agree on the published count adjustment and local division contract. Private scorer internals were not queried.
- Data hash: `{summary['data_hash']}`.
- Split hash: `{summary['split_hash']}`.
- Prediction lock: `{lock['created']}`; [model_lock.json](model_lock.json), [prediction_lock.json](prediction_lock.json).
- Existing notebook environments and originals were preserved. The study uses an additive venv inheriting the notebook runtime; exact packages and hardware are in [environment.json](environment.json).
- The pinned scorer was imported from its source checkout. Its package metadata requests PyTorch >=2.9, while the preserved notebook stack supplies 2.8.0+cu128. The exercised scoring paths passed the upstream fixtures in this recorded environment; no broad package install or claim of a fully resolved upstream package environment is made.
- The original 29 arithmetic tests, repository checks, 102 upstream metric/division fixtures, and added integration tests were run. Final receipts and commands are linked from the artifact ledger; arithmetic helpers were not used as a substitute for official graph matching.

{public_note}

Full public diagnostic transfer, all 199 clips:

{public_md}

Source: [public_retention.csv](public_retention.csv), [public_score_rows.csv](public_score_rows.csv). The original no-export pilot and full export-hook graphs agreed exactly on both pilot clips; [public_graph_parity.json](public_graph_parity.json) records the comparison. The public candidate graphs, pre-ILP scores and transferred selector predictions are frozen separately. This local score is not a reproduction of the notebook's quoted leaderboard score on hidden videos.

{confidence_note}

After neural inference, the serial public graph-repair stage was rescheduled across four isolated clip shards. All 27 repair functions remained byte-for-byte unchanged, and a repair-only pilot matched both original pilot graphs before the owned serial process was stopped. Its partial CSV and log are preserved. The shards reuse the original neural GEFFs and their frame-retention diagnostics, then merge CSV rows in sample order. [public_parallel_repair_receipt.json](public_parallel_repair_receipt.json) verifies the complete sample set, unchanged neural graphs and repeated final parity. An initial repair-pilot export check correctly rejected missing neural diagnostic logs; those original diagnostics were then carried through, with the failed attempt retained. No public annotation outcomes informed this scheduling change.

The 48 fully exported graphs in the preserved original serial CSV also match the parallel repair exactly. The terminal partially written dataset is explicitly excluded from this additional full-graph comparison, while all 199 completed parallel outputs remain in evaluation. See [public_serial_prefix_parity.json](public_serial_prefix_parity.json).

The original public export contains six centers at z=64 across five clips, one voxel beyond the image. The strict spatial check caught this before public outcomes were evaluated. Those nodes and all graph coordinates were retained unchanged for diagnostic scoring; the pinned metric accepts spatial coordinates without an image-shape bound. Image-feature sampling uses the existing half-sample reflection convention, while geometry and distances use the original centers. This explicit spatial-integrity exception further limits the public lane; it does not affect the clean lane. See [public_coordinate_audit.json](public_coordinate_audit.json). The failed preparation log is preserved.

## Learnability and operating points

Primary classifier, natural candidate prevalence:

{classifier_md}

Fixed feature/quality comparators and both image seeds:

{comparator_md}

Source: [classifier_metrics.csv](classifier_metrics.csv). The complete constant/logistic/boosted-tree grids, both image families and seeds, high-confidence subset, ambiguous-candidate sensitivity fit, shuffled-target control, and independent random-tracklet target are all retained. [calibration.csv](calibration.csv) contains measured probability calibration; [feature_profiles.csv](feature_profiles.csv) contains post-lock feature maps. GT-derived match IDs/distances, estimates, annotation geometry, embryo/file identity, and matched-graph attributes were excluded from inference features.

Primary-model sensitivity populations:

{sensitivity_md}

The high-confidence population uses only response >=0.05 and predicted tracklet length >=3. It is a label-blind quality proxy, not a human-verified real-cell population. Excluding ambiguous near-GT unmatched candidates is an evaluation diagnostic. Neither establishes selection among all real cells. Source: [classifier_metrics.csv](classifier_metrics.csv).

![Post-lock feature associations](plots/feature_profiles.png)

![Natural-prevalence calibration](plots/calibration.png)

Transferred probabilities are poorly calibrated across the embryos' different candidate prevalences. AUROC and retention assess rankings; the raw probabilities should not be read as calibrated annotator-selection probabilities. No outer-label calibration was fitted. The Brier scores and fixed-bin counts remain available for checking this limitation.

Every budget is present in [retention.csv](retention.csv): 1.0 (identity), 0.9, 0.8, 0.7, 0.5, 0.3, and 0.1. The table records realized retention, annotation recall, precision, specificity, MCC/phi, and annotation rate in deleted candidates. Coherent units use their 0.9 score quantile and include the unit crossing the budget; actual overshoot remains visible. Baseline TP endpoint survival, survival after rematching, and newly recovered TPs are reported separately; no squared-recall approximation substitutes for graph evaluation.

The identity measurement supplies the common r=1 curve endpoint; it is not counted as a separate independent run for each selector. Endpoint keep correlations by clip, model and budget are in [endpoint_correlations.csv](endpoint_correlations.csv).

![Measured retention and score](plots/retention_and_score.png)

## Actual graph score

Unfiltered clean baseline counts and components:

{baseline_md}

Filtered fixed-primary counts and measured old/new TP behavior:

{graph_md}

The primary combined scores and deltas are shown in the executive table. Count ratios above one remain visible; no supplied estimate is clamped.

Full source: [score_rows.csv](score_rows.csv), [retention.csv](retention.csv). Each distinct listed graph variant was reconstructed and evaluated using official `evaluate`, `per_sample_metrics`, and `summarise`. Required sample sets are locked; missing samples, invalid estimates, GT/count drift, and nonfinite required counts fail evaluation. Node matching is one-to-one, time-aware, anisotropic, and limited to 7 um; edgeless populations use the actual official node matcher. Division evaluation is rerun on fresh graph copies for filtered variants.

Run-level adjusted edges use current per-sample TP+FP+FN weights; divisions are micro-aggregated. No-division samples are explicit, and undefined sample division Jaccards remain undefined. Identity preserves the original graph; isolated-node cleanup is a separately named ablation. Filtering removes incident edges and creates no bridges or GT-guided associations.

## Controls, attribution and uncertainty

Two-factor Shapley decomposition using the exact aggregator:

{attribution_md}

`count_effect + graph_effect + division_effect = delta` is checked numerically. Alpha=0 is a diagnostic score, not the competition score. Node-count/edge-count mixtures used for attribution are arithmetic counterfactuals, not realizable graph predictions.

The complete random-node and random-tracklet controls include 20 fixed seeds at every deletion budget. Ordinary response confidence uses the same node/coherent/fork-context policies. Requested and realized budgets are both reported; coherent rounding means equal requested budgets need not be exactly equal realized budgets. Do not interpret those small differences as perfectly matched-budget evidence. Shuffled and synthetic target controls use predicted tracklets as the randomization units; they do not shuffle individual time-adjacent nodes independently.

Independently randomized outer targets produce AUROCs from {randomized.auroc.min():.3f} to {randomized.auroc.max():.3f}, with AP close to each randomized target's measured prevalence:

{randomized_md}

The shuffled control permutes whole label sequences between predicted tracklets, resampling sequence indices when lengths differ; candidate-weighted prevalence can therefore change. The synthetic control draws one Bernoulli label per complete tracklet. Source fits and outer diagnostic targets use separate random seeds. These are leakage diagnostics, with no independent-group confidence intervals. Source: [classifier_metrics.csv](classifier_metrics.csv).

Their actual sparse-GT outcomes at the fixed coherent 0.9 budget are:

{negative_md}

Both randomized-target selectors lose official score in both embryo directions at this budget. Source: [retention.csv](retention.csv).

Additional confidence controls match the primary selector's **exact realized node count within every video**, retaining complete tracklets. A suffix subset-sum feasibility calculation chooses the lexicographically highest-ranked feasible quality subset. The primary prefix is already the highest-ranked feasible subset for its own realized cost. This removes coherent-unit rounding as an explanation of a primary-versus-confidence difference:

{exact_md}

Source: [matched_budget_summary.csv](matched_budget_summary.csv), [matched_budget_controls.csv](matched_budget_controls.csv). Confidence controls cover all six deletion budgets; 20 exact-cost random coherent controls cover the primary 0.9 budget. No GT labels enter the budget or subset construction.

Exact-cost random-seed variability at the primary budget:

{random_md}

Source: [random_exact_seed_results.csv](random_exact_seed_results.csv). These ranges describe random-filter seeds, not uncertainty about new embryos.

Both image-probe seeds at the same fixed 0.9 coherent budget:

{seed_md}

Source: [image_seed_results.csv](image_seed_results.csv). Seed 20260908 remains primary; seed 314159 is a robustness diagnostic.

Each primary/control uncertainty calculation ran 2,000 paired **complete-supergroup** resamples per embryo and recomputed the full score. With one conservative supergroup, those resamples degenerate to the same contribution. [uncertainty.csv](uncertainty.csv) therefore reports **unavailable confidence intervals**, not falsely precise node/clip-based intervals. Two embryo directions are two replications; neither these counts nor a bootstrap establishes generalization to a population of new embryos.

Hindsight oracle headroom (descriptive, never eligible for rule selection):

{oracle_md}

These membership oracles are heuristics after rematching, not proofs of a global score bound. High-confidence membership and ambiguous-neighbor sensitivity remain distinct from an independently audited real-cell population.

## Decision and next action

The study decision is **{summary['status']}**, and the fixed primary outcome is **{summary['primary_outcome']}**. H1's observable membership-prediction question is assessed by the fixed feature/quality comparators and exact-budget confidence controls. Inferring latent human selection among all real cells remains unresolved without independent quality/census labels. H2 is assessed by the fresh graph deltas in both directions. H3 is assessed by count attribution and the alpha=0 diagnostic; a count contribution larger than the net gain can occur when graph damage offsets it. The full sweep shows alternative operating points but does not authorize choosing a threshold after outer revelation and relabeling it confirmatory.

The following break-even requirements are **derived from the measured count multipliers, remaining FP denominators, and division outcomes**. The weighted TP ratio is `sum(TP1_i * multiplier1_i) / sum(TP0_i * multiplier1_i)`. Its required value is `(baseline_score - filtered_division_term) * sum(G_i + FP1_i) / sum(TP0_i * multiplier1_i)`. This is exact arithmetic for those measured quantities, not an independent-node recall assumption:

{required_md}

No Kaggle submission, notebook publication, forum post, push, or hidden-test claim was made. A follow-up needs a new experiment version and must acknowledge reuse of these embryos. Source-independent crop/time origins and a new embryo would enable stronger validation; manual census labels would address the separate real-cell denominator question.

## Runtime and reproduction

{resource_note}

The bounded image probe used 30,000 uniformly sampled source candidates per fit, six fixed epochs, 3x32x32 triplanar patches centered at predicted positions, and train-only image/tabular normalization. Patch pixels represent 0.8125 um (about a 26 um field). All classes receive the same flips and at-most-one-pixel roll augmentation. The roll wraps panel borders; image extraction reflects at volume borders. Fixed epoch choice replaces unavailable group-disjoint early stopping. Total measured probe training time was {gpu_seconds:.2f} GPU-synchronized wall seconds.

{image_md}

`loader_seconds` is the shared source patch preload repeated on each fit's receipt, not four separately incurred loads. Candidate image-read/detection/patch times and graph sizes are in [candidate_inventory.csv](candidate_inventory.csv); isolated notebook wall time is in [public_harmonic_full/adapter_manifest.json](public_harmonic_full/adapter_manifest.json). Resource and throughput figures describe this local machine and execution.

Actual command history: [commands.jsonl](commands.jsonl). From the repository root, using the existing study environment:

```sh
bash scripts/run_annotation_selection.sh environment inventory candidates splits audit labels train image freeze infer
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/tools" POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 \\
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m annotation_selection infer \\
  --output /kaggle/working/cell-tracking/annotation-selection-v1/predictions_gt_unavailable
bash scripts/run_annotation_selection.sh lock
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/tools" POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 \\
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python -m annotation_selection evaluate --workers 8
bash scripts/run_annotation_selection.sh public public-prepare public-evaluate export matched-controls audit-viewer report
```

See the tracked [reproduction instructions]({str(REPO/'docs/annotation-selection-v1.md')}) for environment/source setup, safe resume boundaries, integrity checks and final test commands. Existing raw inputs and notebook originals remain untouched. Predictions, patches, model files and detailed matching tables remain in ignored local storage.

All machine-readable artifacts and source/command hashes are indexed in [artifact_manifest.json](artifact_manifest.json); stage receipts are in [status.json](status.json). The [offline dashboard](dashboard.html) embeds its plot data and requires no network connection.
'''
    (OUT/'final_report.md').write_text(text)
    payload=dict(summary=summary,coverage=coverage.to_dict('records'),sweep=sweep.to_dict('records'),
                 public_sweep=public_sweep.to_dict('records'),classifiers=classifiers.to_dict('records'))
    data=json.dumps(clean(payload),separators=(',',':'),allow_nan=False).replace('</','<\\/')
    page=HTML.replace('__DATA__',data).replace('__STATUS__',summary['status'].replace('_',' '))
    for marker,name in [('__FEATURE_PLOT__','feature_profiles.png'),('__CALIBRATION_PLOT__','calibration.png')]:
        page=page.replace(marker,'data:image/png;base64,'+base64.b64encode((OUT/'plots'/name).read_bytes()).decode())
    (OUT/'dashboard.html').write_text(page)


HTML='''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotation selection · measured local study</title>
<style>
:root{color-scheme:dark;--bg:#101820;--panel:#1a2631;--ink:#e6eef5;--muted:#a8bac8;--accent:#58d4bb;--warm:#f6be6a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}main{max-width:1380px;margin:auto;padding:32px}h1{font-size:34px;letter-spacing:-1px;margin:8px 0}h2{font-size:20px;margin:0 0 14px}.eyebrow{color:var(--accent);letter-spacing:2px;font-size:12px;text-transform:uppercase}.muted{color:var(--muted)}a{color:var(--accent)}.pill{display:inline-block;padding:5px 12px;border:1px solid #67806e;border-radius:20px;color:var(--warm)}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}.card,.panel{background:var(--panel);border:1px solid #30414e;border-radius:12px;padding:20px}.value{font-size:30px;font-variant-numeric:tabular-nums}.label{color:var(--muted);font-size:13px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}.notice{padding:16px 20px;border-left:3px solid var(--warm);background:#262b2d;margin:20px 0}.controls{display:flex;flex-wrap:wrap;gap:18px;align-items:end}label{display:flex;flex-direction:column;gap:5px;color:var(--muted);font-size:13px}select{background:#111c25;color:var(--ink);border:1px solid #526675;border-radius:7px;padding:9px;max-width:370px}svg{display:block;width:100%;height:auto}.legend{display:flex;gap:15px;font-size:12px;color:var(--muted)}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:right;padding:9px;border-bottom:1px solid #33434f;font-variant-numeric:tabular-nums}td:first-child,th:first-child{text-align:left}th{color:var(--muted);font-weight:500}.scroll{overflow:auto}details{margin-top:18px}code{font-size:12px;overflow-wrap:anywhere}footer{color:var(--muted);padding:28px 0}button{background:var(--accent);color:#122720;border:0;border-radius:6px;padding:8px 13px;cursor:pointer}@media(max-width:850px){.cards,.grid{grid-template-columns:1fr 1fr}main{padding:18px}.grid{grid-template-columns:1fr}}@media(max-width:500px){.cards{grid-template-columns:1fr}h1{font-size:27px}}
</style><style>.panel{min-width:0}.controls label{min-width:0;max-width:100%}select{max-width:min(370px,100%)}.scroll{max-width:100%}</style><main>
<div class="eyebrow">Biohub · annotation selection v1 · local experiment</div>
<h1>Can we preserve annotations while keeping fewer nodes?</h1>
<p class="muted">Actual measurements across 199 clips and two embryos. All primary predictions frozen before outer outcomes.</p>
<span class="pill">__STATUS__</span>
<p id="measuredoutcome"></p>
<div class="cards" id="cards"></div>
<div class="notice"><strong>Interpretation boundary.</strong> Sparse-GT membership is observable; being a real cell and being selected by an annotator are different targets. Supplied total counts are estimates. One conservative overlap group per embryo makes independent-clip confidence intervals unavailable.</div>
<div class="panel"><div class="controls">
<label>Evidence lane<select id="lane"><option value="clean">Clean classical candidates</option><option value="public">Public checkpoint · contaminated diagnostic</option></select></label>
<label>Outer embryo<select id="embryo"></select></label>
<label>Frozen selector<select id="model"></select></label>
<label>Graph policy<select id="policy"><option value="tracklets">Coherent tracklets</option><option value="nodes">Individual nodes</option><option value="tracklets_fork_protected">Tracklets + fork context</option></select></label>
<label>Budget view<select id="focus"><option value="all">All measured budgets</option><option value="high">Keep 70–100%</option></select></label>
<button id="download">Download plotted data</button></div><p class="muted" id="laneinfo"></p></div>
<div class="grid"><section class="panel"><h2>Baseline annotation-match recall</h2><div id="recall"></div><div class="legend"><span><i class="dot" style="background:#58d4bb"></i>Selected model</span><span><i class="dot" style="background:#f6be6a"></i>Confidence control</span><span><i class="dot" style="background:#a8bac8"></i>Random mean (20 seeds)</span></div><p class="muted">Fraction of the baseline candidate-to-GT matches retained before fresh matching. This denominator excludes GT nodes missed by the baseline detector.</p></section>
<section class="panel"><h2>Actual official local score change</h2><div id="score"></div><p class="muted">Fresh node matching and division evaluation for every filtered graph; current sample weights in aggregation.</p></section>
<section class="panel"><h2>Annotation rate among deleted candidates</h2><div id="deleted"></div><p class="muted">Low deleted-group annotation rate can also reflect detection quality or localization errors.</p></section>
<section class="panel"><h2>Where the score change comes from</h2><div id="attribution"></div><p class="muted">Exact count / graph / division Shapley attribution. Counterfactual count mixtures are arithmetic diagnostics.</p></section></div>
<div class="grid"><section class="panel"><h2>Edge preservation after rematching</h2><div id="edge"></div></section><section class="panel"><h2>Division tradeoff</h2><div id="division"></div><p class="muted">The clean classical linker has no predicted forks. The public diagnostic lane includes its actual division outcomes.</p></section></div>
<section class="panel"><h2>Clean fixed primary rule — both directions</h2><div class="scroll" id="primary"></div><p class="muted">7-leaf boosted tree · all allowed features · 0.9-quantile tracklet ranking · requested keep 90%. Fixed-setting fallback because independent source inner folds could not be established. No post-hoc optimum is promoted.</p><div id="intervals"></div></section>
<section class="panel" style="margin-top:20px"><h2>Public checkpoint transfer · contaminated diagnostic</h2><div class="scroll" id="publicprimary"></div><p class="muted">All 199 clips, original notebook settings. Checkpoint training or selection used supplied embryos. Six original centers lie one voxel outside the image; graphs are unchanged and image features use reflected sampling. These outcomes cannot establish clean unseen-embryo transfer.</p></section>
<div class="grid"><section class="panel"><h2>Counts and denominators</h2><div class="scroll" id="coverage"></div><p class="muted">Observations across clips; overlapping views may repeat the same biological observation. Exact all-cell prevalence: unavailable.</p></section>
<section class="panel"><h2>Classifier diagnostics</h2><div class="scroll" id="classifier"></div><p class="muted">Natural candidate prevalence; calibration and sensitivity populations are in the downloadable tables.</p></section></div>
<div class="grid"><section class="panel"><h2>Sparse coverage by embryo</h2><div id="coverageplot"></div><p class="muted">Two explicitly different denominators: the supplied all-cell estimate and the frozen clean candidate population.</p></section><section class="panel"><h2>Measured count ratios · active lane</h2><div id="countplot"></div><p class="muted">Ratio of summed predicted nodes to summed supplied estimates; baseline versus the fixed primary filter. Actual score adjustment remains per sample.</p></section></div>
<section class="panel"><h2>Current plotted operating points</h2><div class="scroll" id="points"></div></section>
<details class="panel"><summary>Clean feature profiles and calibration · generated after prediction lock</summary><p class="muted">Both embryos, natural candidate prevalence. These observational profiles cannot identify causal annotator preferences or true-cell status. Constant features have no fabricated bins.</p><img src="__FEATURE_PLOT__" alt="Measured membership versus allowed geometry, appearance and predicted temporal features" style="width:100%;height:auto"><img src="__CALIBRATION_PLOT__" alt="Measured classifier probability calibration by embryo" style="width:100%;height:auto"></details>
<details class="panel"><summary>Provenance, limitations, and artifacts</summary><div id="provenance"></div><p>102 upstream metric/division fixtures plus arithmetic and repository integration checks. Both directions' 199 inference outputs reproduced with annotation-file reads blocked. Public checkpoints are diagnostic only because training/selection used supplied embryos.</p><p>The blinded census pack contains 48 image ROIs with known inclusion probabilities. No human census labels were supplied. The primary classical linker has no forks, so its division recall is zero. Two thousand complete-supergroup bootstrap draws degenerate with one group per embryo; useful intervals are unavailable.</p><p><a href="final_report.md">Full report</a> · <a href="retention.csv">All operating points</a> · <a href="score_rows.csv">Official per-sample counts</a> · <a href="classifier_metrics.csv">Classifier metrics</a> · <a href="artifact_manifest.json">Artifact manifest</a> · <a href="status.json">Stage ledger</a></p></details>
<footer>Offline artifact · all chart data embedded · no network dependencies · no hidden-test gain claimed</footer>
</main><script type="application/json" id="data">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id), fmt=(v,n=4)=>v==null?'unavailable':Number(v).toLocaleString(undefined,{maximumFractionDigits:n}), pct=v=>v==null?'—':fmt(100*v,2)+'%';
const pool=D.coverage.find(x=>x.embryo==='pooled');
$('cards').innerHTML=[['Annotated observations',fmt(pool.annotated_nodes,0)],['Candidate observations',fmt(pool.candidate_nodes,0)],['GT / supplied total estimate',pct(pool.reference_coverage_ratio_of_sums)],['GT-free reproduced clips','199 / 199']].map(([a,b])=>`<div class="card"><div class="label">${a}</div><div class="value">${b}</div></div>`).join('');
['44b6','6bba','pooled'].forEach(e=>$('embryo').add(new Option(e,e)));$('embryo').value='pooled';
function active(){return $('lane').value==='public'?D.public_sweep:D.sweep}
function models(){const old=$('model').value,mids=[...new Set(active().filter(x=>x.policy.startsWith('membership')&&x.model_id!=='oracle').map(x=>x.model_id))];$('model').innerHTML='';mids.forEach(m=>$('model').add(new Option(m,m)));$('model').value=mids.includes(old)?old:'hgb_all_leaf7'}models();
function tbl(rows,cols){return '<table><thead><tr>'+cols.map(([k,l])=>`<th>${l}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(([k,l,f])=>`<td>${f?f(r[k]):typeof r[k]==='number'?fmt(r[k]):r[k]??'unavailable'}</td>`).join('')+'</tr>').join('')+'</tbody></table>'}
function chart(id,series,ykey,{zero=false}={}){
 const xlo=$('focus').value==='high'?.7:0,valid=r=>r[ykey]!=null&&r.realized_keep>=xlo;
 let pts=series.flatMap(s=>s.rows).filter(valid);if(!pts.length){$(id).textContent='No measured points';return}
 let values=pts.map(r=>r[ykey]),lo=Math.min(...values,...(xlo&&ykey==='annotation_recall'?[]:[0])),hi=Math.max(...values);
 if(hi===lo)hi=lo+1;const pad=(hi-lo)*.12;lo-=pad;hi+=pad;
 if(['annotation_recall','deleted_annotation_rate','edge_jaccard','division_jaccard'].includes(ykey)){lo=Math.max(0,lo);hi=Math.min(1,hi)}
 const W=570,H=300,L=65,R=15,T=15,B=48,x=v=>L+(v-xlo)/(1-xlo)*(W-L-R),y=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${ykey} versus measured retention">`;
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;s+=`<path d="M${L} ${y(v)}H${W-R}" stroke="#344654"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end" fill="#a8bac8" font-size="11">${fmt(v,3)}</text>`}
 for(let i=0;i<=5;i++){const v=xlo+(1-xlo)*i/5;s+=`<text x="${x(v)}" y="${H-B+23}" text-anchor="middle" fill="#a8bac8" font-size="11">${pct(v)}</text>`}
 if(zero)s+=`<path d="M${L} ${y(0)}H${W-R}" stroke="#b2bcc3" stroke-dasharray="4 4"/>`;
 for(const ser of series){const rows=ser.rows.filter(valid).sort((a,b)=>a.realized_keep-b.realized_keep);s+=`<path d="${rows.map((r,i)=>(i?'L':'M')+x(r.realized_keep)+' '+y(r[ykey])).join(' ')}" fill="none" stroke="${ser.color}" stroke-width="2.4"/>`;for(const r of rows)s+=`<circle cx="${x(r.realized_keep)}" cy="${y(r[ykey])}" r="4" fill="${ser.color}"><title>${ser.name}: keep ${pct(r.realized_keep)}; ${ykey} ${fmt(r[ykey],6)}</title></circle>`}
 s+=`<text x="${W/2}" y="${H-5}" text-anchor="middle" fill="#a8bac8" font-size="12">Realized node retention</text></svg>`;$(id).innerHTML=s
}
function bars(id,rows,cols){const W=570,H=260,L=170,max=Math.max(...rows.flatMap(r=>cols.map(c=>r[c[0]])),.01);let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${id}">`;rows.forEach((r,i)=>{s+=`<text x="12" y="${60+i*90}" fill="#e6eef5" font-size="14">${r.embryo}</text>`;cols.forEach(([k,label,color],j)=>{const y=35+i*90+j*28,v=r[k];s+=`<rect x="${L}" y="${y}" width="${v/max*300}" height="18" fill="${color}"/><text x="${L-8}" y="${y+13}" text-anchor="end" fill="#a8bac8" font-size="10">${label}</text><text x="${L+v/max*300+6}" y="${y+13}" fill="#e6eef5" font-size="11">${fmt(v,3)}</text>`})});$(id).innerHTML=s+'</svg>'}
let plotted={};
function render(){
 const e=$('embryo').value,m=$('model').value,p=$('policy').value,data=active(),base=data.find(r=>r.embryo===e&&r.policy==='identity');
 const endpoint=base?{...base,shared_identity_endpoint:true}:null,withIdentity=rows=>endpoint?[...rows,endpoint]:rows;
 const rows=withIdentity(data.filter(r=>r.embryo===e&&r.model_id===m&&r.policy==='membership_'+p)),quality=withIdentity(data.filter(r=>r.embryo===e&&r.model_id==='quality'&&r.policy==='quality_'+p));
 const random=data.filter(r=>r.embryo===e&&r.model_id==='random'&&r.policy==='random_'+(p==='nodes'?'nodes':'tracklets'));
 const means=withIdentity([.9,.8,.7,.5,.3,.1].map(f=>{const a=random.filter(r=>r.requested_keep===f),o={requested_keep:f};for(const k of ['realized_keep','annotation_recall','delta','deleted_annotation_rate','edge_jaccard','division_jaccard']){const v=a.map(r=>r[k]).filter(x=>x!=null);o[k]=v.length?v.reduce((s,x)=>s+x,0)/v.length:null}return o}));
 const series=[{name:m,rows,color:'#58d4bb'},{name:'confidence',rows:quality,color:'#f6be6a'},{name:'random mean',rows:means,color:'#a8bac8'}];
 chart('recall',series,'annotation_recall');chart('score',series,'delta',{zero:true});chart('deleted',series,'deleted_annotation_rate');chart('edge',series,'edge_jaccard');chart('division',series,'division_jaccard');
 chart('attribution',[{name:'count',rows,color:'#58d4bb'},{name:'graph',rows:rows.map(r=>({...r,count_effect:r.graph_effect})),color:'#f6be6a'},{name:'division',rows:rows.map(r=>({...r,count_effect:r.division_effect})),color:'#a8bac8'}],'count_effect',{zero:true});
 $('points').innerHTML=tbl(rows,[['requested_keep','Requested',pct],['realized_keep','Realized',pct],['annotation_recall','Recall',pct],['phi','MCC / phi'],['deleted_annotation_rate','Deleted label rate',pct],['old_tp_endpoint_survival','Old TP endpoints kept',pct],['old_tp_rematched_survival','Old TPs after rematching',pct],['newly_recovered_tp','New TPs'],['edge_jaccard','Edge J'],['division_jaccard','Division J'],['score','Score'],['delta','Delta']]);
 const cs=D.classifiers.filter(r=>r.model_id===m&&r.subset==='all'&&(e==='pooled'||r.embryo===e));$('classifier').innerHTML=$('lane').value==='public'?'Classifiers were fitted on opposite-embryo clean candidates and transferred without refitting. Public membership classification metrics were not independently evaluated; use its measured retention and graph outcomes above.':tbl(cs,[['embryo','Embryo'],['prevalence','Prevalence',pct],['auroc','AUROC'],['average_precision','AP'],['ap_over_prevalence','AP / prevalence'],['brier','Brier']]);
 $('laneinfo').textContent=($('lane').value==='public'?'Known checkpoint contamination: diagnostic transfer only. ':'Clean source-only selectors on fixed image-only candidates. ')+ 'The r=1 endpoint is the shared measured identity graph. Confidence curves use the same policy; random fork-context comparisons retain the unprotected random tracklet control. Exact-cost controls are reported separately.';
 bars('countplot',data.filter(r=>r.source_rule_role==='fixed_primary'&&r.embryo!=='pooled'),[['baseline_count_ratio_of_sums','Baseline','#f6be6a'],['filtered_count_ratio_of_sums','Fixed primary','#58d4bb']]);plotted={lane:$('lane').value,embryo:e,selector:m,policy:p,series}
}
$('primary').innerHTML=tbl(D.summary.primary,[['embryo','Embryo'],['n','Clips'],['realized_keep','Realized keep',pct],['annotation_recall','Baseline-match recall',pct],['baseline_score','Baseline'],['score','Filtered'],['delta','Delta']]);$('coverage').innerHTML=tbl(D.coverage,[['embryo','Embryo'],['annotated_nodes','GT observations'],['estimated_total','Supplied estimate'],['candidate_nodes','Candidates'],['gt_detection_recall','GT recall',pct]]);$('provenance').innerHTML=`<p>Metric: <code>${D.summary.metric_revision}</code><br>Data: <code>${D.summary.data_hash}</code><br>Split: <code>${D.summary.split_hash}</code></p>`;
$('publicprimary').innerHTML=tbl(D.summary.public?.primary||[],[['embryo','Embryo'],['n','Clips'],['realized_keep','Realized keep',pct],['baseline_score','Baseline'],['score','Filtered'],['delta','Delta']]);
const cp=D.summary.primary.find(r=>r.embryo==='pooled'),pp=D.summary.public?.primary.find(r=>r.embryo==='pooled');$('measuredoutcome').textContent=`Fixed primary, pooled: clean score ${fmt(cp.baseline_score,6)} → ${fmt(cp.score,6)}. `+(pp?`Public diagnostic transfer: ${fmt(pp.baseline_score,6)} → ${fmt(pp.score,6)}. The primary filter harms the stronger public tracker.`:'');
$('intervals').innerHTML='<p class="muted">Paired uncertainty: 2,000 complete-supergroup draws per embryo; one independent supergroup each. Confidence intervals are unavailable. No zero-width error bars are presented as evidence of precision.</p>';
bars('coverageplot',D.coverage.filter(r=>r.embryo!=='pooled'),[['reference_coverage_ratio_of_sums','GT / estimate','#58d4bb'],['candidate_membership_prevalence','Matched / candidates','#f6be6a']]);
['embryo','model','policy','focus'].forEach(id=>$(id).addEventListener('change',render));$('lane').onchange=()=>{models();render()};$('download').onclick=()=>{const b=new Blob([JSON.stringify(plotted,null,2)],{type:'application/json'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='measured-operating-points.json';a.click();URL.revokeObjectURL(u)};render();
</script></html>'''
