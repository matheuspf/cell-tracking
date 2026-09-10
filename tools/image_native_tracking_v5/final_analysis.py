"""Measured family interpretation, manifests, and post-freeze score attrition."""
import importlib.metadata,platform,subprocess,sys,time
import pandas as pd
from .common import *


def provenance():
    models=[];uses=[];event_coverage=[]
    for source in ['44b6','6bba']:
        catalog=read(OUT/'training_labels'/f'{source}_native_catalog.json')
        native_forks=0;H_forks=0;H_positive=0;H_missing=0
        for row in catalog['rows']:
            _,counts=np.unique(np.asarray(row['positive'])[:,0],return_counts=True)
            native_forks+=int((counts==2).sum())
        for row in inventory():
            if row['embryo']!=source:continue
            name=row['dataset'];data=arrays(OUT/'hoct_training_features'/f'{name}.npz')
            pairs=arrays(OUT/'banks'/source/'P0'/f'{name}.npz')['pairs'][data['pair_indices']]
            positive=pairs[data['labels']==1];_,counts=np.unique(positive[:,0],return_counts=True)
            H_forks+=int((counts==2).sum());H_positive+=len(positive)
            H_missing+=read(OUT/'hoct_training_features'/f'{name}.json')['missing_positive_features']
        event_coverage.append(dict(source=source,clips=sum(r['embryo']==source for r in inventory()),
            native_supported_windows=len(catalog['rows']),native_supported_positive_edges=catalog['positive_edges'],
            native_censored_edges=catalog['censored_edges'],native_supported_two_daughter_parents=native_forks,
            HOCT_supported_positive_edges=H_positive,HOCT_missing_positive_feature_rows=H_missing,
            HOCT_supported_two_daughter_parents=H_forks,
            interpretation='exact incoming-parent training targets with represented C0 endpoints; distinct from official local timing-window division recall'))
    write(OUT/'source_event_coverage.json',event_coverage)
    for path in sorted((OUT/'models').glob('*/*.json')):
        r=read(path)
        if 'steps' not in r or r.get('tiny'):continue
        checkpoint=path.with_suffix('.pt');assert sha(checkpoint)==r['sha256']
        family=r.get('stage','H1');source=r['source']
        keys=['source','seed','steps','seconds','sha256','trainable_modules','trainable_parameters','source_windows',
            'seen_windows','source_positive_edges','seen_positives','missed_windows','encoder_before','encoder_after',
            'first_encoder_tensor_l2_change','first_step_gradients','supported_positives','supported_negatives',
            'seen_rows','total_rows','head_parameter_change_l2','consistency_control','native_image_optimizer_updates',
            'teacher_stability_weight','teacher_confidence_min','precision','code_sha256']
        models.append(dict(path=str(checkpoint.relative_to(OUT)),bytes=checkpoint.stat().st_size,family=family,
            training_receipt_sha256=sha(path),**{k:r[k] for k in keys if k in r}))
        assert r.get('missed_windows',0)==0
        if family in ['N1','N2']:assert r['seen_positives']==r['source_positive_edges']
        else:assert r['seen_rows']==r['total_rows']
        if family=='N2':assert r['encoder_before']!=r['encoder_after'] and r['first_encoder_tensor_l2_change']>0
        if family=='N1':assert r['encoder_before']==r['encoder_after']
        uses.append(dict(family=family,source=source,target='6bba' if source=='44b6' else '44b6',seed=r['seed'],
            source_clips=sum(x['embryo']==source for x in inventory()),optimizer_updates=r['steps'],
            eligible_positive_transitions=r.get('source_positive_edges',r.get('supported_positives')),
            observed_positive_transitions=r.get('seen_positives',r.get('supported_positives')),
            source_dev='resubstitution only',direct_labels='own training embryo only',
            inherited_exposure='public checkpoints, repeated embryos and C0 teachers; not independently clean'))
    write(OUT/'model_manifest.json',dict(at=now(),models=models,pretrained=[dict(name=n,sha256=sha(OUT/'models/hoct'/f'{n}.pt')) for n in ['general_v1','ctc_v0']],
        source_calibrations={p.name:sha(p) for p in sorted((OUT/'calibration').glob('*.json'))},
        all_supported_source_windows_seen=True,all_supported_positive_transitions_seen=True))
    pd.DataFrame(uses).to_csv(OUT/'dataset_use.csv',index=False)
    versions={}
    names=['torch','numpy','scipy','pandas','polars','zarr','geff','tracksdata','scikit-learn','scikit-image','ilpy','joblib','psutil','matplotlib']
    for name in names:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not found'
    for distribution in importlib.metadata.distributions(path=[str(WORK/'python')]):
        versions[distribution.metadata['Name']]=distribution.version
    runtime=dict(at=now(),python=sys.version,interpreter=sys.executable,platform=platform.platform(),packages=versions,
        GPU=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        Python_user_site_disabled=os.environ.get('PYTHONNOUSERSITE')=='1',Kaggle_runtime_verified=False)
    write(OUT/'runtime_versions.json',runtime)
    base=read(V4/'inference_package/base/manifest.json');package=OUT/'inference_package'
    dependencies=dict(at=now(),package_manifest_sha256=sha(package/'manifest.json'),base_dependencies=base,
        external_model_configs=read(package/'manifest.json')['external_model_configs'],
        bundled_model_sha256={str(p.relative_to(package)):sha(p) for p in (package/'models').rglob('*.pt')},
        runtime_versions_sha256=sha(OUT/'runtime_versions.json'),
        inference_inputs=['raw image Zarr','new bundled model weights and source calibrations','pinned inherited C0 models and source files'],
        prohibited_inputs=['GEFF annotations','estimated node counts','source training arrays','cached selected graphs','study observation or score caches'],
        offline_guard='Python audit hooks before numerical imports and in inherited subprocesses; loopback IPC allowed; not syscall isolation')
    write(OUT/'inference_dependency_manifest.json',dependencies)
    config=read(REPO/'handover/image-native-tracking-v5/config.json')
    hoct_pin=subprocess.check_output(['git','-C',str(WORK/'hoct'),'rev-parse','HEAD'],text=True).strip();assert hoct_pin==config['hoct']['commit']
    write(OUT/'source_manifest.json',dict(at=now(),HOCT=dict(repository='https://github.com/royerlab/hoct',commit=hoct_pin,
        license='MIT',license_sha256=sha(WORK/'hoct/LICENSE'),weights=config['hoct'],
        interface='supported feature-bearing IndexedRXGraph; 19 actual morphology/position inputs, 288-dimensional true edge embeddings',
        biological_exposure='not independently certified'),native=dict(source='preserved installed patched U-Net/transformer',
        architecture_sha256=sha(OUT/'native_architecture.json'),C0_base_manifest_sha256=sha(V4/'inference_package/base/manifest.json')),
        metric=dict(repository='https://github.com/royerlab/kaggle-cell-tracking-competition',commit=config['metric_commit']),
        competition_inputs='199 supplied clips; 71 from 44b6 and 128 from 6bba',synthetic_training_used=False,
        FOCUS_used=False,external_microscopy_uploads=False,new_terms_accepted=False))
    repairs=[
        dict(component='official HOCT graph adapter',failure='bulk_add_edges returns no edge-ID list in the installed tracksdata API',
            correction='read actual edge IDs through the supported edge_attrs table and retain explicit ID mapping',model_recipe_changed=False),
        dict(component='source proposal pilot',failure='source6 expanded candidate bank had not yet been materialized',
            correction='build that source-specific bank with the existing frozen label-free producer before decoding',model_recipe_changed=False),
        dict(component='early package read guard',failure='inherited secondary config and v2 hash-only teacher lock were undeclared dependencies',
            correction='pin each exact file in the manifest and verify its SHA before invoking the inherited predictor',model_recipe_changed=False),
        dict(component='early package read guard',failure='newly generated prediction GEFF output was treated as input annotation',
            correction='allow GEFF inside the new fresh-output directory only; resolved annotation/cache paths remain blocked',model_recipe_changed=False),
        dict(component='additional actual-network pixel fixture',failure='fixture added an extra channel dimension to model.encode input',
            correction='use the installed B,T,Z,Y,X encode contract; actual production training was already using that contract',model_recipe_changed=False),
        dict(component='original HOCT source6 decoder diagnostic',failure='fixed central source tile contained insufficient valid objects',
            correction='one bounded repeat on the densest image-derived grid tile; original failed receipt retained',model_recipe_changed=False),
        dict(component='evaluation-only heuristic oracle runtime',failure='one edge at a time triggered avoidable repeated full-graph work',
            correction='apply the same legal supported-edge set in one batch; truth remains evaluation-only',model_recipe_changed=False)]
    names=['package_early_fallback_retry1.log','package_early_fallback_retry2.log','package_early_fallback_retry3.log',
        'package_early_paths.log','native_integrity.log','source_proposals_final.log','HOCT_sparse_ROI_retry.log']
    write(OUT/'execution_repairs.json',dict(at=now(),repairs=repairs,
        preserved_logs={n:sha(OUT/'logs'/n) for n in names if (OUT/'logs'/n).exists()},
        post_target_hyperparameter_repairs=0,failed_attempts_retained=True,
        model_optimizer_recipe_changed_after_opposite_source_scores=False))


def attrition():
    from .headroom import force_legal
    from .temporal_decode import protected_context
    from .calibrate import apply
    variants={'J_native_frozen':'N0','H_general_J':'H0','H_probe_J':'H1','H_probe_native_J':'H2','N_head_J':'N1','N_backbone_J':'N2'}
    rows=[]
    for row in inventory():
        name=row['dataset'];source='6bba' if row['embryo']=='44b6' else '44b6';base=graph(name)
        events=read(OUT/'oracle_events'/f'{name}.json')['fixed'];bank=arrays(OUT/'banks'/source/'P0'/f'{name}.npz')
        hh=arrays(OUT/'model_scores/H'/f'{name}.npz');j=read(OUT/'calibration'/f'{source}_J.json');cfg=j['config']
        hoct=read(OUT/'calibration'/f'{source}_H_20260910.json')['models'];old=set(map(tuple,base['edges']))
        for variant,family in variants.items():
            if family=='N0':scores=j['native_scale']*bank['native_logits']+j['native_offset']
            elif family=='H0':scores=apply(hoct[family],hh['H0'])
            elif family=='H1':scores=apply(hoct[family],hh['H1_20260910'])
            elif family=='H2':scores=apply(hoct[family],hh['H1_20260910'],bank['native_logits'])
            else:
                values=arrays(OUT/'model_scores'/f'{family}_20260910'/f'{name}.npz')['P0']
                scores=apply(read(OUT/'calibration'/f'{source}_{family}_20260910.json')['calibration'],values)
            lookup=dict(zip(map(tuple,bank['pairs']),map(float,scores)));finite={p for p,s in lookup.items() if np.isfinite(s)}
            enabled={p for p,s in lookup.items() if np.isfinite(s) and s>-(cfg['birth_cost']+cfg['termination_cost']+cfg['incumbent_edge_bonus']+4)}|old
            protected,_=protected_context(base['edges']);protected.update(old-finite)
            counts=dict(C0_missed_events=0,with_local_candidate_option=0,with_finite_model_option=0,after_score_pruning=0,after_local_protection=0)
            for event in events:
                if event['official_recovered']:continue
                counts['C0_missed_events']+=1;options=[set(map(tuple,op)) for op in event['options']]
                counts['with_local_candidate_option']+=bool(options)
                finite_options=[op for op in options if op<=finite];counts['with_finite_model_option']+=bool(finite_options)
                active=[op for op in finite_options if op<=enabled];counts['after_score_pruning']+=bool(active)
                counts['after_local_protection']+=any(force_legal(protected,op,protected)[1] for op in active)
            receipt=read(OUT/'prediction_receipts'/variant/f'{name}.json')['decode']
            rows.append(dict(dataset=name,embryo=row['embryo'],variant=variant,**counts,
                candidate_edges=len(scores),finite_model_edges=len(finite),decoder_enabled_edges=len(enabled),
                missing_feature_C0_edges_preserved=receipt['missing_feature_C0_edges_preserved'],
                scope='bounded local timing options; support/protection feasibility is not global MILP selection probability'))
    pd.DataFrame(rows).to_csv(OUT/'score_attrition_rows.csv',index=False)
    numeric=[k for k in rows[0] if k not in ['dataset','embryo','variant','scope']];summary=[]
    for variant in variants:
        for em in ['44b6','6bba','pooled']:
            sub=[r for r in rows if r['variant']==variant and (em=='pooled' or r['embryo']==em)]
            summary.append(dict(variant=variant,embryo=em,samples=len(sub),**{k:sum(r[k] for r in sub) for k in numeric}))
    pd.DataFrame(summary).to_csv(OUT/'score_attrition.csv',index=False)


def analysis():
    scores=pd.read_csv(OUT/'ablation_scores.csv');index={(r['variant'],r['embryo']):r for r in scores.to_dict('records')}
    comparisons=[]
    pairs=[('H_probe_J','H_general_J'),('H_probe_native_J','H_general_J'),('N_head_J','J_native_frozen'),
        ('N_backbone_J','N_head_J'),('N_backbone_J','J_native_frozen'),('P_union_native_J','J_native_frozen'),
        ('P_union_N_J','N_backbone_J'),('P_DC_native_J','J_native_frozen'),('P_DC_N_J','N_backbone_J'),
        ('P_union_N_J','P_union_native_J'),('P_DC_N_J','P_DC_native_J'),('P_image_ablation','P_union_native_J')]
    for candidate,control in pairs:
        for em in ['44b6','6bba','pooled']:
            a=index[candidate,em];b=index[control,em]
            comparisons.append(dict(candidate=candidate,control=control,embryo=em,delta_score=a['score']-b['score'],
                delta_edge_tp=a['edge_tp']-b['edge_tp'],delta_edge_fp=a['edge_fp']-b['edge_fp'],
                delta_division_tp=a['division_tp']-b['division_tp'],delta_division_fp=a['division_fp']-b['division_fp'],
                delta_nodes=a['num_pred_nodes']-b['num_pred_nodes'],interpretation='difference of two complete measured graph results; changed matching is included'))
    pd.DataFrame(comparisons).to_csv(OUT/'matched_controls.csv',index=False)
    families={'H':['H_general_J','H_probe_J','H_probe_native_J'],'N':['N_head_J','N_backbone_J'],
        'P':['P_union_native_J','P_union_N_J','P_DC_native_J','P_DC_N_J','P_image_ablation']}
    result=[]
    regrets=read(OUT/'regret_summary.json')
    for family,variants in families.items():
        best=max(variants,key=lambda v:index[v,'pooled']['score'])
        result.append(dict(family=family,best_primary_exploratory=best,
            scores={e:index[best,e]['score'] for e in BASE},delta_C0={e:index[best,e]['delta_C0'] for e in BASE},
            regret=[r for r in regrets if r['variant']==best],
            status='valid measured representation/observation experiment',hindsight_selection=True))
    decoders=[]
    for folder in sorted((OUT/'prediction_receipts').iterdir()):
        rr=[read(p) for p in folder.glob('*.json')]
        if not rr:continue
        windows=[w for r in rr for w in r['decode']['windows']]
        decoders.append(dict(variant=folder.name,samples=len(rr),windows=len(windows),
            fallback_windows=sum(w.get('fallback',False) for w in windows),optimal_windows=sum(w.get('optimal',False) for w in windows),
            time_limit_windows=sum(w.get('status')==1 for w in windows),summed_decode_seconds=sum(r['seconds'] for r in rr),
            maximum_window_variables=max(w.get('variables',0) for w in windows),
            missing_model_edges=sum(r['decode']['model_feature_missing'] for r in rr)))
    pd.DataFrame(decoders).to_csv(OUT/'decoder_runtime.csv',index=False)
    source=[read(p) for p in sorted((OUT/'source_pilots').glob('*_final_objective.json'))]
    source_summary=[{**{k:v for k,v in r.items() if k not in ['decode','score','C0']},
        'counts':{k:r['score'][k] for k in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes','matched_nodes']},
        'C0_counts':{k:r['C0'][k] for k in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes','matched_nodes']}} for r in source]
    write(OUT/'family_analysis.json',dict(at=now(),families=result,source_proposal_pilots=source_summary,
        attribution='Supported edge regret, node/matching changes, division identity and count arithmetic are measured separately. Sparse unlabeled cells do not establish dense detector precision.',
        causal_limits='Same reused embryos and inherited models; no independent hidden-test forecast or cell/clip bootstrap interval',
        node_only_control='six full density-selected clips; full-199 count decomposition is arithmetic, explicitly not an additional measured graph arm',
        P2=read(OUT/'P2_decision.json'),original_HOCT_diagnostics=[read(p) for p in sorted((OUT/'source_pilots').glob('*HOCT_original_decoder*.json'))],
        CTC_source_diagnostics=[{k:v for k,v in read(p).items() if k!='decode'} for p in sorted((OUT/'source_pilots').glob('*HOCT_ctc_J.json'))]))


def run(wait=False):
    required=['regret_summary.json','fresh_validation.json','optical_review_receipt.json','P2_decision.json']
    while any(not (OUT/p).exists() for p in required):
        if not wait:raise RuntimeError('Final diagnostic dependencies remain pending')
        time.sleep(30)
    if wait:os.execv(sys.executable,[sys.executable,'-m','image_native_tracking_v5.final_analysis'])
    provenance();attrition();analysis()
    write(OUT/'final_analysis_complete.json',dict(at=now(),complete=True))
    print('Final measured analysis and dependency manifests complete',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args();run(a.wait)
