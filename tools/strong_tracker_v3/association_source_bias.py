"""Frozen source resubstitution bias audit with fresh official whole-graph scores."""
from __future__ import annotations

from .common import load_graph,now,read_json,run_pool,sha,write_json
from .association import VARIANTS,cached,decode,score_features
from .context import RunContext


def one(task):
    from annotation_selection.metric_adapter import evaluate_graph
    from .association_diagnostics import edge_evidence
    ctx,s,row=task;name=s['dataset'];b=load_graph(ctx.incumbent(name));c=cached(ctx,s,b['nodes'],b['edges'])
    gt=load_graph(ctx.v1/'evaluation/gt'/f'{name}.npz')
    baseline,bm,bt=evaluate_graph(name,b['nodes'],b['edges'],gt['nodes'],gt['edges'],s['physical_scale'],row['estimated_total'])
    _,bf,br=edge_evidence(b['edges'],bm,gt['edges']);records=[]
    for variant,cfg in VARIANTS.items():
        score=score_features(ctx,s['embryo'],c,cfg['model']);e,ledger=decode(b['nodes'],b['edges'],c,score,cfg['margin'])
        result,m,tp=evaluate_graph(name,b['nodes'],e,gt['nodes'],gt['edges'],s['physical_scale'],row['estimated_total'])
        _,fp,recovered=edge_evidence(e,m,gt['edges'])
        record=dict(dataset=name,source=s['embryo'],variant=variant,
            count_deltas={k:result[k]-baseline[k] for k in ['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn']},
            predicted_tp_gained=len(tp-bt),predicted_tp_lost=len(bt-tp),gt_edges_gained=len(recovered-br),gt_edges_lost=len(br-recovered),
            fp_removed=len(bf-fp),fp_added=len(fp-bf),accepted_actions=ledger['accepted_actions'],changed_edges=ledger['changed_edges'])
        records.append(record)
    write_json(ctx.out/'evaluation/association_source_bias'/f'{name}.json',dict(records=records,
        selection='first five lexicographic image dataset identifiers per source; frozen before source outcomes',
        scope='same-source training resubstitution bias diagnostic, no independent validation claim',
        model_lock_sha256=sha(ctx.out/'association_model_lock.json')))
    return records


def run(ctx=None,workers=2):
    ctx=ctx or RunContext.default();samples=ctx.samples();rows={r['dataset']:r for r in ctx.eval_rows()}
    selected=[s for embryo in ['44b6','6bba'] for s in sorted([s for s in samples if s['embryo']==embryo],key=lambda s:s['dataset'])[:5]]
    lock=dict(samples=[s['dataset'] for s in selected],variants=VARIANTS,model_lock_sha256=sha(ctx.out/'association_model_lock.json'),
        selection='first five lexicographic names per source',used_for_policy_tuning=False)
    lock_path=ctx.out/'association_source_bias_lock.json'
    if lock_path.exists():
        if read_json(lock_path)!=lock:raise ValueError('Source bias audit lock drift')
    else:write_json(lock_path,lock)
    records=[r for chunk in run_pool(one,[(ctx,s,rows[s['dataset']]) for s in selected],workers) for r in chunk]
    summary=[]
    for source in ['44b6','6bba']:
        for variant in VARIANTS:
            rr=[r for r in records if r['source']==source and r['variant']==variant]
            summary.append(dict(source=source,variant=variant,samples=len(rr),
                count_deltas={k:sum(r['count_deltas'][k] for r in rr) for k in rr[0]['count_deltas']},
                **{k:sum(r[k] for r in rr) for k in ['predicted_tp_gained','predicted_tp_lost','gt_edges_gained','gt_edges_lost','fp_removed','fp_added','accepted_actions','changed_edges']}))
    write_json(ctx.out/'association_source_bias_summary.json',dict(created=now(),summary=summary,
        scope='source resubstitution bias only; all variants remain frozen regardless of this diagnostic',
        lock_sha256=sha(lock_path)))


if __name__=='__main__':run()
