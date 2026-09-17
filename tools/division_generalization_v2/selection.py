"""Predeclared source-only duration, checkpoint, application and family rules."""
import json
from .common import WORK,RESULTS,read_json,write_json,sha


def select(source,arm,seed,steps=(3072,4096)):
    candidates=[]
    for step in steps:
        path=WORK/'screens'/arm/source/str(seed)/str(step)/'summary.json'
        if not path.exists():continue
        r=read_json(path)
        for app,score in r['applications'].items():
            if score['delta']>=-1e-12:
                candidates.append(dict(step=step,application=app,score=score['score'],delta=score['delta'],
                    lost_supported_edges=score['lost_supported_edges'],summary=str(path),
                    summary_sha256=sha(path),checkpoint_sha256=r['checkpoint_sha256'],
                    calibration={k:r['calibration'][k] for k in ('status','temperature','intercept')}))
    if not candidates:
        return dict(source=source,arm=arm,seed=seed,status='source_failed',fallback='P0',eligible=[])
    # Application is a predeclared last tie break, favoring protected support.
    candidates.sort(key=lambda r:(-r['score'],r['lost_supported_edges'],r['step'],r['application']!='protected'))
    return dict(source=source,arm=arm,seed=seed,status='qualified',selected=candidates[0],eligible=candidates)


def extension(source,seed):
    decisions={}
    for arm in ('J_uniform','J_mined'):
        path=WORK/'training'/arm/source/str(seed)/'diagnostics.jsonl'
        diagnostics={r['step']:r for r in [json.loads(line) for line in path.read_text().splitlines()]}
        before=diagnostics[3072]['calibration']['event'];after=diagnostics[4096]['calibration']['event']
        screens=[read_json(WORK/'screens'/arm/source/str(seed)/str(step)/'summary.json') for step in (3072,4096)]
        scores=[max(x['score'] for x in s['applications'].values()) for s in screens]
        decisions[arm]=dict(loss_3072=before,loss_4096=after,relative_improvement=(before-after)/max(abs(before),1e-12),
                           score_3072=scores[0],score_4096=scores[1],
                           eligible=after<=.99*before and scores[1]>=scores[0]-1e-12)
    # The same direction-specific rule must hold for both matched branches;
    # never extend one arm in isolation or choose duration using target results.
    result=dict(source=source,seed=seed,branches=decisions,extend=all(x['eligible'] for x in decisions.values()),
                same_extended_budget_for_both=True,target_used=False)
    write_json(RESULTS/f'extension-{source}-{seed}.json',result,immutable=True)
    return result


def freeze(primary,replication):
    selected={}
    for arm in ('G30','J_uniform','J_mined'):
        for seed in ((20260916,) if arm=='G30' else (20260916,314159)):
            for source in ('44b6','6bba'):
                steps=(4096,) if arm=='G30' else (3072,4096,8192)
                selected[f'{arm}/{source}/{seed}']=select(source,arm,seed,steps)
    family=[]
    for arm in ('J_uniform','J_mined'):
        directional=[selected[f'{arm}/{s}/20260916'] for s in ('44b6','6bba')]
        if all(x['status']=='qualified' for x in directional):
            family.append((min(x['selected']['delta'] for x in directional),
                           sum(x['selected']['delta'] for x in directional),arm))
    nominee=max(family,key=lambda x:(x[0],x[1],x[2]=='J_uniform'))[2] if family else None
    result=dict(status='frozen_before_target_predictions',nominee=nominee,selected=selected,
        no_favorable_embryo_selection=True,target_results_read=False,
        shared_prefix_not_independent=True,primary_seed=20260916,replication_seed=314159,
        qualified_exports=[arm for arm in ('G30','J_uniform','J_mined') if all(
            selected[f'{arm}/{s}/{seed}']['status']=='qualified'
            for s in ('44b6','6bba') for seed in ((20260916,) if arm=='G30' else (20260916,314159)))])
    write_json(RESULTS/'target_freeze.json',result,immutable=True)
    return result
