"""Build measured progress/final artifacts from complete official results."""
import csv,html
import pandas as pd
from .common import *

REPLICAS={'N_backbone_J':'N_final_seed2','H_probe_native_J':'H_final_seed2',
          'P_union_N_J':'P_final_seed2','P_DC_N_J':'P_DC_final_seed2'}
DETERMINISTIC=['J_native_frozen','H_general_J','P_union_native_J','P_DC_native_J']


def csv_rows(name):
    import fcntl
    path=OUT/name
    with (OUT/'aggregation.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_SH)
        return pd.read_csv(path).replace({np.nan:None}).to_dict('records') if path.exists() else []


def decision(scores):
    by={(r['variant'],r['embryo']):r for r in scores}
    def passes(v):
        return all((v,em) in by for em in BASE) and by[v,'pooled']['score']>BASE['pooled'] and all(by[v,em]['score']>=BASE[em]-1e-8 for em in ['44b6','6bba'])
    gates=[]
    for primary,replica in REPLICAS.items():
        complete=all((v,em) in by for v in [primary,replica] for em in BASE)
        gates.append(dict(primary=primary,replica=replica,complete=complete,primary_pass=passes(primary),
            replica_pass=passes(replica),eligible=complete and passes(primary) and passes(replica)))
    fresh=read(OUT/'fresh_validation.json') if (OUT/'fresh_validation.json').exists() else {}
    verified=fresh.get('verified_variants',[])
    for primary in DETERMINISTIC:
        complete=all((primary,em) in by for em in BASE)
        gates.append(dict(primary=primary,replica='independent fresh execution',complete=complete,
            primary_pass=passes(primary),replica_pass=primary in verified,eligible=complete and passes(primary) and primary in verified))
    eligible=[r['primary'] for r in gates if r['eligible']]
    selected=max(eligible,key=lambda v:by[v,'pooled']['score']) if eligible else 'C0'
    score=by.get((selected,'pooled'),{'score':BASE['pooled']})['score']
    exploratory=[r for r in scores if r['embryo']=='pooled' and not r['variant'].startswith('Oracle_')]
    best=max(exploratory,key=lambda r:r['score']) if exploratory else None
    return dict(selected=selected,score=score,delta_C0=score-BASE['pooled'],target_met=score>=.95,
        gates=gates,best_exploratory=best,selection_pending=not (OUT/'execution_complete.json').exists(),
        interpretation='Repeated operational data and inherited checkpoint exposure; no independent biological generalization claim.')


def training():
    summaries=[];curves=[];seen=set()
    for p in sorted((OUT/'models').glob('*/*.json')):
        r=read(p)
        if 'steps' not in r:continue
        key=r.get('key',f"{r['source']}_H1_{r['seed']}");seen.add(key)
        stage=r.get('stage','H1');history=r.get('history',[])
        summaries.append(dict(key=key,source=r['source'],family=stage,seed=r['seed'],steps=r['steps'],complete=True,
            tiny=r.get('tiny',False),seconds=r['seconds'],sha256=r['sha256'],
            trainable_parameters=r.get('trainable_parameters',289),
            native_image_updates=r.get('native_image_optimizer_updates',0),
            encoder_changed=r.get('encoder_before')!=r.get('encoder_after') if 'encoder_before' in r else False,
            encoder_tensor_delta=r.get('first_encoder_tensor_l2_change',0),
            source_positive_edges=r.get('source_positive_edges',r.get('supported_positives')),
            source_windows=r.get('source_windows'),seen_windows=r.get('seen_windows'),missed_windows=r.get('missed_windows'),
            final_recorded_loss=history[-1]['loss'] if history else None))
        curves.extend(dict(key=key,source=r['source'],family=stage,seed=r['seed'],tiny=r.get('tiny',False),**h) for h in history)
    for p in sorted((OUT/'training').glob('*_progress.json')):
        r=read(p)
        if r['key'] in seen:continue
        history=r['history'];tiny='_tiny' in r['key']
        summaries.append(dict(key=r['key'],source=r['source'],family=r['stage'],seed=r['seed'],steps=r['steps'],
            target_steps=r['target_steps'],complete=False,tiny=tiny,seconds=history[-1]['seconds'],
            final_recorded_loss=history[-1]['loss'],native_image_updates=r['steps'] if r['stage']=='N2' else 0))
        curves.extend(dict(key=r['key'],source=r['source'],family=r['stage'],seed=r['seed'],tiny=tiny,**h) for h in history)
    return summaries,curves


def coverage():
    rows=[read(p) for p in sorted((OUT/'headroom').glob('*.json'))];summary=[]
    numeric=['gt_nodes','C0_nodes','C0_matched','gt_edges','endpoints_present','candidate_edges','candidate_gt_edges',
        'crowded_endpoint_edges','missed_candidate_edges_crowded','mitotic_endpoint_edges','missed_candidate_edges_mitotic',
        'valid_HOCT_regions','novel_peaks','union_matched','new_peak_matched_nodes','new_peak_new_GT_recovery',
        'union_C0_GT_matches_lost','official_local_candidate_feasible','exact_id_candidate_divisions']
    for em in ['44b6','6bba','pooled']:
        sub=[r for r in rows if em=='pooled' or r['embryo']==em]
        if not sub:continue
        buckets={b:sum(r['division_buckets'].get(b,0) for r in sub) for b in sorted({b for r in sub for b in r['division_buckets']})}
        summary.append(dict(embryo=em,samples=len(sub),complete=len(sub)==(199 if em=='pooled' else 71 if em=='44b6' else 128),
            **{k:sum(r[k] for r in sub) for k in numeric},division_buckets=buckets,
            cap_coverage={k:{q:sum(r['cap_coverage'][k][q] for r in sub) for q in ['edges','gt_transitions_covered','gt_transitions_lost_given_endpoints']} for k in ['4','8','16']}))
    return summary


def build(final=False):
    scores=csv_rows('ablation_scores.csv');train,curves=training();cov=coverage();selection=decision(scores)
    protocol=read(OUT/'execution_protocol.json');complete_variants={r['variant'] for r in scores if r['embryo']=='pooled'}
    status=dict(study_id='image-native-tracking-v5',updated=now(),status='complete' if final else 'executing',
        target_score=.95,incumbent_score=BASE['pooled'],selected=selection['selected'],selected_score=selection['score'],target_met=selection['target_met'],
        complete_configurations=len(complete_variants),registered_configurations=len(protocol['variants']),
        complete_new_operational_variants=len([v for v in complete_variants if v!='C0' and not v.startswith('Oracle_')]),
        observations=len(list((OUT/'observations').glob('*.json'))),HOCT_feature_shards=len(list((OUT/'hoct_training_features').glob('*.json'))),
        DeepCenter_clips=len(list((OUT/'deepcenter').glob('*.json'))),expected_clips=199,
        complete_native_fits=sum(r['complete'] and not r['tiny'] and r['family'] in ['N1','N2'] for r in train),
        planned_native_fits=8,source_only=True,prior_studies_preserved=True)
    if final:
        assert complete_variants==set(protocol['variants']),sorted(set(protocol['variants'])-complete_variants)
        assert status['complete_native_fits']==8
        assert (OUT/'fresh_validation.json').exists() and read(OUT/'fresh_validation.json')['passed']
    exposures=[
        dict(family='C0',direct_fit='legacy source/opposite-embryo repair pipeline',inherited='public primary/secondary native, DeepCenter and v2 E_hgb; embryos repeatedly reused',independent=False),
        dict(family='H0',direct_fit='source-only logit scale/offset and C0 structural prior',inherited='public HOCT general_v1 training biology not independently certified; C0 centers/regions inherit native exposure',independent=False),
        dict(family='H1/H2',direct_fit='supported transitions in own source embryo; unknown edges masked; no hard ILP consistency',inherited='HOCT public backbone plus inherited C0 and native evidence; reused source resubstitution calibration',independent=False),
        dict(family='N1/N2',direct_fit='own source images and supported incoming-parent targets; source-only teacher KL on confident unknown targets',inherited='installed native primary checkpoint and C0 teachers may have seen either embryo',independent=False),
        dict(family='P1/PDC',direct_fit='proposal extraction fixed before target scores; N2 association direct fitting source-only',inherited='native and optional DeepCenter public detectors; repeated source pilots',independent=False),
        dict(family='Oracles',direct_fit='explicit target truth-assisted evaluation-only interventions',inherited='not deployable, not promotion candidates, not upper bounds',independent=False)]
    budget=dict(C0=BASE['pooled'],target=.95,required_gain=.95-BASE['pooled'],
        C0_adjusted_edge=.9228682178819851,C0_division_jaccard=29/243,
        required_division_jaccard_with_edges_fixed=(.95-.9228682178819851)/.1,
        required_adjusted_edge_with_divisions_fixed=.95-.1*29/243,
        note='Exact additive arithmetic; counts and matching must be re-evaluated for each changed graph.')
    resource=read(OUT/'resource_current.json') if (OUT/'resource_current.json').exists() else None
    payload=dict(status=status,scores=scores,selection=selection,training=train,curves=curves,coverage=cov,exposure=exposures,budget=budget,
        resources=resource,protocol=protocol,supervisor=read(OUT/'supervisor_state.json') if (OUT/'supervisor_state.json').exists() else None)
    write(OUT/'status.json',status);write(OUT/'selection.json',selection);write(OUT/'exposure.json',exposures);write(OUT/'score_budget.json',budget)
    pd.DataFrame(train).to_csv(OUT/'training_summary.csv',index=False);pd.DataFrame(curves).to_csv(OUT/'learning_curves.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in cov]).to_csv(OUT/'coverage.csv',index=False)
    pd.DataFrame(selection['gates']).to_csv(OUT/'family_outcomes.csv',index=False)
    template=(Path(__file__).parent/'dashboard_template.html').read_text()
    encoded=json.dumps(payload,default=lambda v:v.item() if isinstance(v,np.generic) else v,allow_nan=False).replace('<','\\u003c')
    (OUT/'dashboard.html').write_text(template.replace('__STUDY_DATA__',encoded))
    write(OUT/'dashboard_data.json',payload)
    body=f'''# Image-native tracking v5 — {'measured report' if final else 'execution progress'}

Updated {status['updated']}. {len(complete_variants)} of {len(protocol['variants'])} registered complete configurations are scored.
C0 is freshly reproduced at {BASE['pooled']:.15f}; the target is 0.95 (+{budget['required_gain']:.15f}).
Current eligible export: {selection['selected']} at {selection['score']:.15f}. {'Final selection.' if final else 'Selection remains provisional while execution continues.'}

The offline [dashboard](dashboard.html) contains the official complete-clip comparisons, source learning curves, candidate coverage and replication gate.
Only complete 199-clip variants enter the comparison. Oracles are separately identified as truth-assisted feasibility diagnostics.

The native tracker is the installed 2,076,706-parameter U-Net/transformer, with a 1,496,320-parameter temporal 3D encoder. N1 fits its association module for 8,000 updates per source; N2 warm-starts N1 and updates the actual encoder for 12,000 more, at a tenfold smaller learning rate. Both recipes repeat with seed 314159. The detector output layer remains frozen; N2 is an association-representation experiment at fixed coordinates, followed by the proposal factorial.

HOCT runs the pinned official 6,252,593-parameter general_v1 JIT with 19 genuine region/position features and 288-dimensional edge embeddings. H1 fits its linear edge probe on supported source transitions. H2 fits one source-only residual calibration with the unchanged native evidence. Hard ILP consistency is disabled so missing C0 links do not become negatives. The ctc_v0 model is a source-only diagnostic within the 18-complete-configuration cap.

P1 discovers full-field native peaks with real image contrast, assigns new IDs, derives image-supported watershed morphology, and rebuilds candidate features at the new coordinates. PDC uses continuous frozen DeepCenter confirmation. The image ablation removes proposal confidence. The rolling five-frame MILP compares births, continuation, bifurcation and incumbent explanations under ownership and one-cell/two-cell exclusion constraints. Timing aliases on predicted paths share a maximum complete-explanation choice. C0 fork predecessor/daughter/grandchild edges are protected in operational inference. Oracles relax protection explicitly.

Native comparison tensors use the primary checkpoint without the incumbent's secondary model/eight-view harmonic ensemble. Their full-frame two-frame inputs use the exact installed downsampling, quantiles and positional/indexing conventions. This intentional change is isolated by N0. The complete inherited C0 path and pre-ILP arrays were independently reproduced on two full density-selected clips.

Both direct adaptation directions use only their own source labels and source calibration. All source transitions with represented endpoints are eligible; missing parents and possible unannotated second daughters remain censored. Public checkpoints, C0 teachers and repeated embryo use prevent an independent biological-generalization claim. Seed replication measures training sensitivity, not embryo independence.

The server reboot interrupted execution after the native log reached update 2,941. Hash checks recovered 122 image shards, 120 HOCT shards, all C0 results and optimizer/RNG state at update 2,000. The last 941 updates were repeated. Prior critical hashes remained unchanged. The crash cause is unavailable from container kernel logs. Resumption uses tmux, two bounded GPU lanes and shared CPU-pool locking; native resume snapshots are now every 250 updates.

Detailed GT identities, optical-review images, raw data, checkpoints and submissions remain local. No Kaggle submission or notebook publication is performed.
'''
    (OUT/('final_report.md' if final else 'progress_report.md')).write_text(body)
    print(json.dumps(status),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();build(a.final)
