"""Measured tables, exact attribution, conservative uncertainty, and offline report."""
from __future__ import annotations

import html
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score

from .common import OUT,REPO,SEED,clean,digest,load_graph,now,read_json,revision,sha,stage,write_json
from .features import FEATURES
from .metric_adapter import aggregate

PRIMARY='hgb_all_leaf7__membership_tracklets__s20260908__r0.9'
BASE='identity__identity__s20260908__r1'


def score_formula(rows,counts_from=None,nodes_from=None,alpha=.1,weights=None):
    c=rows if counts_from is None else counts_from
    p=rows if nodes_from is None else nodes_from
    tp=c.edge_tp.to_numpy(float);fp=c.edge_fp.to_numpy(float);fn=c.edge_fn.to_numpy(float)
    mul=1+alpha-alpha*p.num_pred_nodes.to_numpy(float)/p.estimated_total.to_numpy(float)
    freq=np.ones(len(c)) if weights is None else np.asarray(weights,float)
    denom=np.sum(freq*(tp+fp+fn))
    edge=np.sum(freq*tp*np.maximum(mul,0))/denom if denom else np.nan
    dv=c[['division_tp','division_fp','division_fn']].to_numpy(float)
    totals=(dv*freq[:,None]).sum(axis=0)
    division=totals[0]/totals.sum() if totals.sum() else np.nan
    return dict(edge=edge,division=division,score=edge+(.1*division if np.isfinite(division) else 0.))


def membership_aggregate(group):
    tp,fp,fn,tn=[int(group[k].sum()) for k in ['membership_tp','membership_fp','membership_fn','membership_tn']]
    n=tp+fp+fn+tn;k=tp+fp;a=tp+fn
    denom=float(k)*a*(n-a)*(n-k)
    rate=fn/(n-k) if n>k else None;p=a/n if n else None
    return dict(realized_keep=k/n if n else None,prevalence=p,annotation_recall=tp/a if a else None,
                specificity=tn/(tn+fp) if tn+fp else None,precision=tp/k if k else None,
                phi=(tp*tn-fp*fn)/np.sqrt(denom) if denom>0 else None,deleted_annotation_rate=rate,
                deleted_rate_relative_to_population=rate/p if rate is not None and p else None,
                candidate_nodes=n,matched_nodes=a,kept_matched_nodes=tp,
                old_tp_endpoint_survival=group.baseline_tp_endpoints_kept.sum()/group.baseline_tp.sum() if group.baseline_tp.sum() else None,
                old_tp_rematched_survival=group.baseline_tp_rematched_surviving.sum()/group.baseline_tp.sum() if group.baseline_tp.sum() else None,
                newly_recovered_tp=int(group.newly_recovered_tp.sum()))


def sweep_summary(df,output_prefix=''):
    manifest=read_json(OUT/'fold_manifest.json');records=[];attribution=[]
    gt_sizes={r['dataset']:r['annotated_nodes'] for r in read_json(OUT/'inventory.json')}
    for embryo in [*manifest['embryos'],'pooled']:
        sub=df if embryo=='pooled' else df[df.embryo==embryo]
        expected=manifest['expected_samples'] if embryo=='pooled' else [n for n in manifest['expected_samples'] if n.split('_')[0]==embryo]
        base=sub[sub.variant_id==BASE].sort_values('dataset')
        if set(base.dataset)!=set(expected):raise ValueError('Missing baseline samples')
        baseline=aggregate(base.to_dict('records'),expected)
        for variant,g in sub.groupby('variant_id',sort=False):
            g=g.sort_values('dataset')
            if list(g.dataset)!=list(base.dataset):raise ValueError('Missing/duplicate scored sample')
            if not np.array_equal(g.estimated_total.values,base.estimated_total.values):raise ValueError('Count estimate drift')
            result=aggregate(g.to_dict('records'),expected)
            counts=result.pop('counts');result.update(counts)
            r=g.iloc[0]
            record=dict(embryo=embryo,variant_id=variant,model_id=r.model_id,policy=r.policy,seed=int(r.seed),requested_keep=r.requested_keep,
                        source_rule_role='fixed_primary' if variant==PRIMARY else 'baseline' if variant==BASE else 'hindsight_oracle' if r.model_id=='oracle' else 'preregistered_descriptive',
                        baseline_score=baseline['score'],delta=result['score']-baseline['score'],**result,**membership_aggregate(g))
            record.update(baseline_edge_jaccard=baseline['edge_jaccard'],baseline_division_jaccard=baseline['division_jaccard'],
                          baseline_adj_edge_jaccard=baseline['adj_edge_jaccard'],baseline_edge_tp=int(base.edge_tp.sum()),
                          baseline_edge_fp=int(base.edge_fp.sum()),baseline_edge_fn=int(base.edge_fn.sum()),
                          baseline_count_ratio_of_sums=float(base.num_pred_nodes.sum()/base.estimated_total.sum()),
                          filtered_count_ratio_of_sums=float(g.num_pred_nodes.sum()/g.estimated_total.sum()),
                          baseline_count_ratio_mean_of_ratios=float(base.baseline_count_ratio.mean()),
                          filtered_count_ratio_mean_of_ratios=float(g.count_ratio.mean()),
                          rematched_nodes=int(g.matched_nodes.sum()),
                          rematched_gt_node_recall=float(g.matched_nodes.sum()/sum(gt_sizes[n] for n in expected)),
                          gt_node_recall_mean_of_ratios=float(g.gt_node_recall.mean()),
                          fp_retained_ratio=float(g.edge_fp.sum()/base.edge_fp.sum()) if base.edge_fp.sum() else None)
            record['alpha0_score']=score_formula(g,alpha=0)['score']
            record['alpha0_delta']=record['alpha0_score']-score_formula(base,alpha=0)['score']
            e00=score_formula(base)['edge'];e11=score_formula(g)['edge']
            e01=score_formula(base,nodes_from=g)['edge'];e10=score_formula(g,nodes_from=base)['edge']
            count_effect=.5*(e01-e00+e11-e10);graph_effect=.5*(e10-e00+e11-e01)
            division_effect=(result['score']-e11)-(baseline['score']-e00)
            if not np.isclose(count_effect+graph_effect+division_effect,record['delta'],atol=1e-12):
                raise ValueError('Shapley attribution fails exact score decomposition')
            record.update(count_effect=count_effect,graph_effect=graph_effect,division_effect=division_effect)
            record['count_share_of_positive_delta']=count_effect/record['delta'] if record['delta']>0 else None
            multiplier=np.maximum(1.1-.1*g.num_pred_nodes.to_numpy(float)/g.estimated_total.to_numpy(float),0.)
            weighted_base_tp=float(np.sum(base.edge_tp.to_numpy(float)*multiplier))
            remaining_denominator=float((g.edge_tp+g.edge_fp+g.edge_fn).sum())
            div_term=result['score']-result['adj_edge_jaccard']
            record['break_even_weighted_tp_ratio']=((baseline['score']-div_term)*remaining_denominator/weighted_base_tp) if weighted_base_tp else None
            record['measured_weighted_tp_ratio']=float(np.sum(g.edge_tp.to_numpy(float)*multiplier))/weighted_base_tp if weighted_base_tp else None
            records.append(record)
    summary=pd.DataFrame(records);summary.to_csv(OUT/(output_prefix+'retention.csv'),index=False)
    summary.to_parquet(OUT/(output_prefix+'retention.parquet'),index=False)
    return summary


def classifier_tables():
    cache=OUT/'classifier_report_receipt.json'
    inputs=dict(prediction_lock=sha(OUT/'prediction_lock.json'),schema_version=1,
                labels={n:sha(OUT/'evaluation/membership'/f'{n}.npz') for n in read_json(OUT/'fold_manifest.json')['expected_samples']})
    if cache.exists():
        receipt=read_json(cache)
        if receipt['inputs']!=inputs:raise ValueError('Classifier report input drift')
        for p,h in receipt['outputs'].items():
            if sha(OUT/p)!=h:raise ValueError('Classifier report output drift')
        return pd.read_csv(OUT/'classifier_metrics.csv')
    manifest=read_json(OUT/'fold_manifest.json');metrics=[];calibration=[];profiles=[];candidates=[]
    from .train import control_targets
    for d in manifest['directions']:
        ys=[];ambs=[];features=[];pred=[];units=[];offset=0;keys=[]
        for name in d['outer_samples']:
            b=load_graph(OUT/'baseline/clean'/f'{name}.npz')
            with np.load(OUT/'evaluation/membership'/f'{name}.npz') as f:
                ys.append(f['annotation_label']);ambs.append(f['ambiguous'])
            with np.load(OUT/'predictions'/d['source']/f'{name}.npz') as f:
                model_ids=list(f.files);pred.append(np.column_stack([f[k] for k in model_ids]))
            features.append(b['features']);units.append(b['tracklet']+offset)
            offset+=int(b['tracklet'].max(initial=-1))+1
            keys.append(pd.DataFrame(dict(dataset=name,candidate_id=b['nodes'][:,0])))
        y=np.concatenate(ys);amb=np.concatenate(ambs);x=np.concatenate(features);ps=np.concatenate(pred);u=np.concatenate(units)
        high=(x[:,5]>=.05)&(x[:,10]>=3)
        for j,model in enumerate(model_ids):
            for subset,mask in [('all',np.ones(len(y),bool)),('high_confidence',high),('exclude_ambiguous_diagnostic',~amb)]:
                yy=y[mask];p=ps[mask,j]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    auc=roc_auc_score(yy,p) if len(np.unique(yy))==2 else None
                    ap=average_precision_score(yy,p) if yy.sum() else None
                prob=np.clip(p,0,1)
                metrics.append(dict(embryo=d['outer'],direction=d['source']+'->'+d['outer'],model_id=model,subset=subset,
                                    n=len(yy),prevalence=float(yy.mean()),auroc=auc,average_precision=ap,
                                    ap_over_prevalence=ap/yy.mean() if yy.mean() and ap is not None else None,
                                    brier=brier_score_loss(yy,prob) if model!='quality' else None,
                                    interpretation='Matched sparse-GT membership, not true-cell status'))
            if model!='quality':
                bins=np.minimum(np.floor(np.clip(ps[:,j],0,1)*10).astype(int),9)
                for k in np.unique(bins):
                    mask=bins==k
                    calibration.append(dict(embryo=d['outer'],model_id=model,bin=int(k),n=int(mask.sum()),
                                            mean_score=float(ps[mask,j].mean()),annotation_rate=float(y[mask].mean())))
        for kind in ['shuffled','synthetic']:
            target=control_targets(y,u,SEED+1,kind)
            model=f'hgb_all_leaf7_{kind}';p=ps[:,model_ids.index(model)]
            metrics.append(dict(embryo=d['outer'],direction=d['source']+'->'+d['outer'],model_id=model,
                                subset='independent_randomized_outer_target',n=len(y),prevalence=float(target.mean()),
                                auroc=roc_auc_score(target,p),average_precision=average_precision_score(target,p),
                                brier=brier_score_loss(target,p),randomization_unit='maximal predicted nonbranching tracklet'))
        for j,feature in enumerate(FEATURES):
            cuts=np.unique(np.quantile(x[:,j],np.linspace(0,1,11)))
            if len(cuts)<2:continue
            bins=np.searchsorted(cuts[1:-1],x[:,j],side='right')
            for k in np.unique(bins):
                mask=bins==k
                profiles.append(dict(embryo=d['outer'],feature=feature,bin=int(k),n=int(mask.sum()),
                                     mean_feature=float(x[mask,j].mean()),annotation_rate=float(y[mask].mean()),
                                     ambiguous_rate=float(amb[mask].mean())))
        selected=pd.concat(keys,ignore_index=True)
        selected['annotation_label']=y;selected['score']=ps[:,model_ids.index('hgb_all_leaf7')]
        selected['fold']=d['source']+'->'+d['outer'];selected['model_id']='hgb_all_leaf7';selected['provenance_lane']='clean'
        selected.to_parquet(OUT/'evaluation'/f'candidate_scores_{d["outer"]}.parquet',index=False)
        candidates.append(selected)
    pd.concat(candidates,ignore_index=True).to_csv(OUT/'candidate_scores.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'classifier_metrics.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'calibration.csv',index=False)
    pd.DataFrame(profiles).to_csv(OUT/'feature_profiles.csv',index=False)
    outputs=['candidate_scores.csv','classifier_metrics.csv','calibration.csv','feature_profiles.csv']
    outputs.extend(f'evaluation/candidate_scores_{e}.parquet' for e in manifest['embryos'])
    write_json(cache,dict(created=now(),inputs=inputs,outputs={p:sha(OUT/p) for p in outputs},
                interpretation='Outer diagnostics after both directional prediction locks; no model/configuration selection'))
    return pd.DataFrame(metrics)


def classifiers(args):
    if not (OUT/'prediction_lock.json').exists():raise ValueError('Outer diagnostics require both direction locks')
    print(classifier_tables().query("model_id == 'hgb_all_leaf7' and subset == 'all'").to_string(index=False),flush=True)


def coverage_table(df):
    inv=pd.read_csv(OUT/'sample_inventory.csv');base=df[df.variant_id==BASE]
    merged=inv.merge(base[['dataset','num_pred_nodes','matched_nodes','baseline_count_ratio']],on='dataset',validate='one_to_one')
    rows=[]
    for embryo in [*sorted(inv.embryo.unique()),'pooled']:
        sub=merged if embryo=='pooled' else merged[merged.embryo==embryo]
        a=int(sub.annotated_nodes.sum());p=int(sub.num_pred_nodes.sum());m=int(sub.matched_nodes.sum());estimate=float(sub.estimated_total.sum())
        rows.append(dict(embryo=embryo,clips=len(sub),conservative_overlap_groups=sub.overlap_group.nunique(),annotated_nodes=a,
                         annotated_edges=int(sub.annotated_edges.sum()),gt_divisions=int(sub.gt_divisions.sum()),estimated_total=estimate,
                         reference_coverage_ratio_of_sums=a/estimate,reference_coverage_mean_of_ratios=float(sub.reference_coverage.mean()),
                         candidate_nodes=p,matched_nodes=m,candidate_membership_prevalence=m/p,gt_detection_recall=m/a,
                         count_ratio_of_sums=p/estimate,count_ratio_mean_of_ratios=float(sub.baseline_count_ratio.mean()),
                         true_cell_prevalence=None))
    out=pd.DataFrame(rows);out.to_csv(OUT/'coverage.csv',index=False)
    return out


def uncertainty(df):
    manifest=read_json(OUT/'fold_manifest.json');rng=np.random.default_rng(SEED);records=[];draw_rows=[]
    variants=[PRIMARY,'quality__quality_tracklets__s20260908__r0.9',
              'image_only_seed20260908__membership_tracklets__s20260908__r0.9',
              'image_only_seed314159__membership_tracklets__s20260908__r0.9']
    for embryo in manifest['embryos']:
        sub=df[df.embryo==embryo];base=sub[sub.variant_id==BASE].sort_values('dataset')
        group_ids=base.overlap_group.to_numpy();groups=np.unique(group_ids)
        for variant in variants:
            other=sub[sub.variant_id==variant].sort_values('dataset');draws=[]
            for i in range(2000):
                sampled=rng.choice(groups,len(groups),replace=True)
                freq=np.sum(group_ids[:,None]==sampled[None,:],axis=1)
                draws.append(score_formula(other,weights=freq)['score']-score_formula(base,weights=freq)['score'])
            valid=len(groups)>1
            records.append(dict(embryo=embryo,variant_id=variant,groups=len(groups),draws=2000,
                                delta=score_formula(other)['score']-score_formula(base)['score'],
                                ci_low=float(np.quantile(draws,.025)) if valid else None,
                                ci_high=float(np.quantile(draws,.975)) if valid else None,
                                bootstrap_min=min(draws),bootstrap_max=max(draws),interval_estimable=valid,
                                limitation='One conservative supergroup: resampling degenerates; independent-clip intervals would be unjustified' if not valid else 'Conditional within-embryo interval; not population generalization'))
            draw_rows.extend(dict(embryo=embryo,variant_id=variant,draw=i,delta=v) for i,v in enumerate(draws))
    pd.DataFrame(records).to_csv(OUT/'uncertainty.csv',index=False)
    pd.DataFrame(draw_rows).to_csv(OUT/'bootstrap_draws.csv',index=False)
    return records


def exact_budget_summary(df):
    controls=pd.read_parquet(OUT/'matched_budget_controls.parquet');out=[]
    manifest=read_json(OUT/'fold_manifest.json')
    for embryo in [*manifest['embryos'],'pooled']:
        expected=manifest['expected_samples'] if embryo=='pooled' else [n for n in manifest['expected_samples'] if n.split('_')[0]==embryo]
        sub=controls if embryo=='pooled' else controls[controls.embryo==embryo]
        for (model,seed,fraction),g in sub.groupby(['model_id','seed','requested_keep']):
            selected=df[(df.model_id=='hgb_all_leaf7')&(df.policy=='membership_tracklets')&(df.requested_keep==fraction)]
            if embryo!='pooled':selected=selected[selected.embryo==embryo]
            a=selected.sort_values('dataset');b=g.sort_values('dataset')
            if not np.array_equal(a.num_pred_nodes.values,b.num_pred_nodes.values):raise ValueError('Realized budgets are not matched per video')
            score=aggregate(g.to_dict('records'),expected)['score'];primary=aggregate(selected.to_dict('records'),expected)['score']
            out.append(dict(embryo=embryo,control=model,seed=int(seed),requested_keep=float(fraction),
                            primary_score=primary,control_score=score,primary_minus_control=primary-score,
                            primary_annotation_recall=float(a.kept_matched_nodes.sum()/a.baseline_matched_nodes.sum()),
                            control_annotation_recall=float(b.kept_matched_nodes.sum()/b.baseline_matched_nodes.sum()),
                            realized_nodes=int(g.num_pred_nodes.sum()),samples=len(g),exact_per_video_node_counts=True))
    result=pd.DataFrame(out);result.to_csv(OUT/'matched_budget_summary.csv',index=False)
    return result


def verify_scoring_provenance():
    manifest=read_json(OUT/'fold_manifest.json');prediction=read_json(OUT/'prediction_lock.json')
    for name in manifest['expected_samples']:
        labels=read_json(OUT/'evaluation/membership'/f'{name}.json')
        sample=read_json(OUT/'evaluation/score_samples'/f'{name}.json')
        if sha(OUT/'evaluation/gt'/f'{name}.npz')!=labels['gt_sha256'] or sample['inputs']['gt_sha']!=labels['gt_sha256']:
            raise ValueError('GT cache changed since source labels were generated')
        if sha(OUT/'baseline/clean'/f'{name}.npz')!=labels['baseline_sha256']:
            raise ValueError('Frozen baseline changed')
    from .common import DATA
    for hashes in read_json(OUT/'data_fingerprints.json').values():
        for path,h in hashes.items():
            if sha(DATA/path)!=h:raise ValueError('Input metadata/GEFF bytes changed')
    for path,h in prediction['prediction_hashes'].items():
        if sha(OUT/'predictions'/path)!=h:raise ValueError('Frozen prediction changed')
    for path,h in read_json(OUT/'model_lock.json')['models'].items():
        if sha(OUT/path)!=h:raise ValueError('Frozen model or source provenance changed')
    from .common import OFFICIAL
    for path,h in read_json(OUT/'environment.json')['metric_files'].items():
        if sha(OFFICIAL/path)!=h:raise ValueError('Pinned metric source changed')
    for pack in read_json(OUT/'public_provenance.json'):
        for weight in pack['weights']:
            if sha(weight['path'])!=weight['sha256']:raise ValueError('Existing public checkpoint changed')
        for split in pack['splits']:
            if sha(split['path'])!=split['hash']:raise ValueError('Existing public split manifest changed')
    public=read_json(OUT/'public_harmonic_full/adapter_manifest.json')
    if sha(public['source_path'])!=public['source_sha256']:raise ValueError('Original public notebook changed')
    write_json(OUT/'final_integrity_receipt.json',dict(created=now(),samples=len(manifest['expected_samples']),
                gt_edge_and_estimate_fingerprints_unchanged=True,baseline_and_prediction_hashes_unchanged=True,
                source_models_unchanged=True,public_weights_and_notebook_unchanged=True,pinned_metric_sources_unchanged=True))


def seed_tables(sweep,exact):
    image=sweep[sweep.model_id.str.startswith('image_')&(sweep.policy=='membership_tracklets')&(sweep.requested_keep==.9)].copy()
    image['family']=image.model_id.str.replace(r'_seed\d+$','',regex=True)
    image.to_csv(OUT/'image_seed_results.csv',index=False)
    random=exact[exact.control=='random_exact'].groupby('embryo').agg(seeds=('seed','nunique'),
           mean_control_score=('control_score','mean'),min_control_score=('control_score','min'),max_control_score=('control_score','max'),
           std_control_score=('control_score','std'),mean_primary_minus_control=('primary_minus_control','mean'),
           min_primary_minus_control=('primary_minus_control','min'),max_primary_minus_control=('primary_minus_control','max')).reset_index()
    random.to_csv(OUT/'random_exact_seed_results.csv',index=False)
    return image,random


def controls(args):
    df=pd.read_parquet(OUT/'score_rows.parquet')
    uncertainty(df)
    stage('S100','complete_with_identifiability_limits',bootstrap_draws=2000,independent_groups_per_embryo=1)


def run(args):
    if not read_json(OUT/'score_manifest.json')['complete']:raise ValueError('Full sample evaluation must finish before final reporting')
    stage('S110','running')
    verify_scoring_provenance()
    df=pd.read_parquet(OUT/'score_rows.parquet')
    early=OUT/'evaluation/primary_early_rows.parquet'
    if early.exists():
        columns=['dataset','variant_id','graph_hash','edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes']
        a=pd.read_parquet(early)[columns].sort_values(['dataset','variant_id']).reset_index(drop=True)
        b=df[df.variant_id.isin([BASE,PRIMARY])][columns].sort_values(['dataset','variant_id']).reset_index(drop=True)
        if not a.equals(b):
            # Persisted CSV-derived integer/float dtypes may differ, so compare values too.
            if a.shape!=b.shape or not (a.to_numpy()==b.to_numpy()).all():raise ValueError('Full sweep disagrees with early primary measurement')
        write_json(OUT/'early_primary_verification.json',dict(created=now(),rows=len(a),exact_graph_and_official_count_parity=True))
    if not np.array_equal(df.baseline_tp_rematched_surviving+df.newly_recovered_tp,df.edge_tp):
        raise ValueError('Predicted TP edge accounting disagrees with official counts')
    sweep=sweep_summary(df);coverage=coverage_table(df);classifiers=classifier_tables();intervals=uncertainty(df)
    exact=exact_budget_summary(df)
    seed_tables(sweep,exact)
    public_summary=None
    if (OUT/'public_score_rows.parquet').exists():
        from .public_analysis import verify_prediction_lock
        verify_prediction_lock()
        if not read_json(OUT/'public_score_manifest.json')['complete']:raise ValueError('Public score table is incomplete')
        public_df=pd.read_parquet(OUT/'public_score_rows.parquet')
        if not np.array_equal(public_df.baseline_tp_rematched_surviving+public_df.newly_recovered_tp,public_df.edge_tp):
            raise ValueError('Public TP edge accounting disagrees with official counts')
        public_sweep=sweep_summary(public_df,output_prefix='public_')
        missing_confidence=sum(read_json(OUT/'baseline/public'/f'{n}.json')['confidence_missing']
                               for n in read_json(OUT/'fold_manifest.json')['expected_samples'])
        public_nodes=int(public_sweep[(public_sweep.embryo=='pooled')&(public_sweep.variant_id==BASE)].num_pred_nodes.iloc[0])
        public_summary=dict(lane='diagnostic_contaminated_transfer',primary=public_sweep[public_sweep.variant_id==PRIMARY].to_dict('records'),
                            baseline=public_sweep[public_sweep.variant_id==BASE].to_dict('records'),
                            rows=len(public_df),sample_count=public_df.dataset.nunique(),graph_parity=read_json(OUT/'public_graph_parity.json'),
                            confidence_missing_nodes=missing_confidence,confidence_missing_fraction=missing_confidence/public_nodes,
                            confidence_rule='Nearest same-frame pre-ILP center within 2 um; missing scores set to zero with explicit flag',
                            spatial_integrity=read_json(OUT/'public_coordinate_audit.json'))
        write_json(OUT/'public_summary.json',public_summary)
    primary=sweep[sweep.variant_id==PRIMARY]
    directions=primary[primary.embryo!='pooled']
    descriptive=sweep[(~sweep.model_id.isin(['oracle','random','identity']))&(~sweep.model_id.str.contains('shuffled|synthetic'))]
    transferable_descriptive=descriptive[descriptive.embryo!='pooled'].groupby('variant_id').delta.min().max()
    public_gain=any(r['delta']>0 for r in public_summary['primary']) if public_summary else False
    decision='promising_but_uncertain' if (directions.delta>0).any() or transferable_descriptive>0 or public_gain else 'no_transferable_signal'
    primary_outcome=('gain_in_both_directions' if (directions.delta>0).all() else
                     'mixed_directions' if (directions.delta>0).any() else 'no_gain_in_either_direction')
    summary=dict(created=now(),status=decision,primary_rule='Fixed hgb_all_leaf7, upper-0.9-quantile tracklet selection, requested keep 0.9',
                 source_inner_tuning_available=False,primary=primary.to_dict('records'),coverage=coverage.to_dict('records'),
                 primary_outcome=primary_outcome,descriptive_best_minimum_direction_delta=float(transferable_descriptive),
                 diagnostic_public_primary_gain=public_gain,
                 intervals=intervals,gt_unavailable_reproduction_passed=True,metric_revision=read_json(OUT/'environment.json')['metric_revision'],
                 data_hash=read_json(OUT/'inventory_summary.json')['data_hash'],split_hash=sha(OUT/'fold_manifest.json'),
                 true_cell_prevalence=None,hidden_test_gain_claimed=False,
                 matched_budget_primary=exact[(exact.requested_keep==.9)&(exact.control=='quality_exact')].to_dict('records'),
                 public=public_summary,
                 limitation='Only two embryos; unresolved clip overlaps force fixed settings and make within-embryo bootstrap intervals non-identifiable')
    write_json(OUT/'summary.json',summary)
    from .dashboard import build
    build(summary,sweep,coverage,classifiers)
    stage('S100','complete_with_identifiability_limits',bootstrap_draws=2000,independent_groups_per_embryo=1)
    stage('S110','report_generated_validation_pending',decision=decision,report=str(OUT/'final_report.md'),dashboard=str(OUT/'dashboard.html'))
    print(json.dumps(clean(summary['primary']),indent=2),flush=True)
