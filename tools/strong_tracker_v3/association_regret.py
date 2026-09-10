"""Evaluation-only fresh-match edge accounting and division-path regret audit."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .common import adjacency,graph_hash,load_graph,now,read_json,sha,write_json
from .association import VARIANTS
from .context import RunContext


def sets(m):
    return {key:set(map(tuple,m[key])) for key in ['tp_pairs','fp_pairs','recovered_gt_edges']}


def run(ctx=None):
    import pandas as pd
    ctx=ctx or RunContext.default();rows=[];edit_rows=[]
    for sample in ctx.samples():
        name=sample['dataset'];base=load_graph(ctx.out/'evaluation/matches/incumbent'/f'{name}.npz');bs=sets(base)
        bm=dict(map(tuple,base['matched_ids']));br=read_json(ctx.out/'evaluation/scores/incumbent'/f'{name}.json')['result']
        for variant in VARIANTS:
            mp=ctx.out/'evaluation/matches'/variant/f'{name}.npz';m=load_graph(mp);ss=sets(m)
            mm=dict(map(tuple,m['matched_ids']));vr=read_json(ctx.out/'evaluation/scores'/variant/f'{name}.json')['result']
            changed_matches=bm!=mm
            record=dict(dataset=name,embryo=sample['embryo'],variant=variant,
                fresh_matching_changed=changed_matches,
                lost_predicted_tp=len(bs['tp_pairs']-ss['tp_pairs']),gained_predicted_tp=len(ss['tp_pairs']-bs['tp_pairs']),
                lost_gt_edges=len(bs['recovered_gt_edges']-ss['recovered_gt_edges']),gained_gt_edges=len(ss['recovered_gt_edges']-bs['recovered_gt_edges']),
                removed_fp=len(bs['fp_pairs']-ss['fp_pairs']),introduced_fp=len(ss['fp_pairs']-bs['fp_pairs']),
                official_edge_tp_delta=vr['edge_tp']-br['edge_tp'],official_edge_fp_delta=vr['edge_fp']-br['edge_fp'],
                official_division_tp_delta=vr['division_tp']-br['division_tp'],
                official_division_fp_delta=vr['division_fp']-br['division_fp'],official_division_fn_delta=vr['division_fn']-br['division_fn'])
            ledger=read_json(ctx.out/'association_ledgers'/variant/f'{name}.json')
            counts=Counter()
            for index,edit in enumerate(ledger['edits']):
                removed=set(map(tuple,edit['removed']));added=set(map(tuple,edit['added']))
                local=dict(dataset=name,variant=variant,edit_index=index,
                    removed_tp=len(removed&bs['tp_pairs']),added_tp=len(added&ss['tp_pairs']),
                    removed_fp=len(removed&bs['fp_pairs']),added_fp=len(added&ss['fp_pairs']),
                    removed_unknown=len(removed-bs['tp_pairs']-bs['fp_pairs']),added_unknown=len(added-ss['tp_pairs']-ss['fp_pairs']),
                    termination=edit['termination'],predicted_value=edit['value'],fresh_matching_changed=changed_matches,
                    scope='Per-edge costs under fresh whole-graph matches; division costs remain whole-clip assignments')
                counts['tp']+=local['added_tp']-local['removed_tp'];counts['fp']+=local['added_fp']-local['removed_fp'];edit_rows.append(local)
            if not changed_matches:
                assert counts['tp']==record['official_edge_tp_delta'] and counts['fp']==record['official_edge_fp_delta']
            rows.append(record)
    out=ctx.out/'evaluation/association_regret';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_parquet(out/'clip_regret.parquet',index=False)
    pd.DataFrame(edit_rows).to_parquet(out/'edit_regret.parquet',index=False)
    aggregates=[]
    for variant in VARIANTS:
        for embryo in ['44b6','6bba','pooled']:
            rr=[r for r in rows if r['variant']==variant and (embryo=='pooled' or r['embryo']==embryo)]
            keys=[k for k in rr[0] if k not in ['dataset','embryo','variant']]
            aggregates.append(dict(variant=variant,embryo=embryo,samples=len(rr),**{k:sum(r[k] for r in rr) for k in keys}))
    write_json(ctx.out/'association_regret_summary.json',dict(created=now(),aggregate=aggregates,edit_rows=len(edit_rows),
        full_official_assignment_per_graph=True,edge_edit_ledger_reconciles_exactly_when_matches_unchanged=True,
        scope='Post-hoc evaluation only; never packaged as inference evidence'))
    oracle_division_audit(ctx)


def oracle_division_audit(ctx):
    from strong_tracker_v2.census import division_census
    cases=[];eval_rows={r['dataset']:r for r in ctx.eval_rows()}
    for path in sorted((ctx.out/'evaluation/oracle_scores/source_association_oracle').glob('*.json')):
        name=path.stem;orr=read_json(path)['result'];baseline=read_json(ctx.out/'evaluation/scores/incumbent'/path.name)['result']
        if all(orr[k]==baseline[k] for k in ['division_tp','division_fp','division_fn']):continue
        b=load_graph(ctx.incumbent(name));g=load_graph(ctx.out/'evaluation/source_association_oracle'/f'{name}.npz')
        gt=load_graph(ctx.v1/'evaluation/gt'/f'{name}.npz');scale=eval_rows[name]['physical_scale']
        before=division_census(name,b['nodes'],b['edges'],gt['nodes'],gt['edges'],scale)
        after=division_census(name,g['nodes'],g['edges'],gt['nodes'],gt['edges'],scale)
        old_events={r['event_id']:r for r in before if r['kind']=='gt_division'}
        changes=[dict(event_id=r['event_id'],before=old_events[r['event_id']],after=r) for r in after
            if r['kind']=='gt_division' and r['recovered']!=old_events[r['event_id']]['recovered']]
        _,_,succ=adjacency(b['nodes'],b['edges']);forks={int(b['nodes'][i,0]) for i,s in enumerate(succ) if len(s)==2}
        frozen={(int(a),int(d)) for a,d in b['edges'] if int(a) in forks}
        preserved=frozen<=set(map(tuple,g['edges']))
        ledger=read_json(ctx.out/'evaluation/source_association_oracle'/f'{name}.json')['edits']
        record=dict(dataset=name,before_graph_hash=graph_hash(b['nodes'],b['edges']),after_graph_hash=graph_hash(g['nodes'],g['edges']),
            frozen_fork_outgoing_edges_preserved=preserved,
            count_deltas={k:orr[k]-baseline[k] for k in ['edge_tp','edge_fp','division_tp','division_fp','division_fn']},
            changed_divisions=changes,edits=ledger,
            interpretation='Fixed immediate fork edges do not protect all timing-tolerant daughter paths; an edge globally counted FP can support a valid early division')
        write_json(ctx.out/'evaluation/association_regret/oracle_division'/f'{name}.json',record)
        cases.append({k:record[k] for k in ['dataset','frozen_fork_outgoing_edges_preserved','count_deltas','interpretation']})
    summary=dict(created=now(),cases=cases,
        fork_freeze_scope='Only each incumbent fork parent and its two immediate outgoing edges are fixed; downstream daughter trajectories can change',
        no_new_variants_or_refitting=True)
    write_json(ctx.out/'association_oracle_division_regret_summary.json',summary)
    write_json(ctx.out/'source_oracle_division_regret_summary.json',summary)


if __name__=='__main__':run()
