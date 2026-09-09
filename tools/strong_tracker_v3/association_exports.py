"""Stream label-free disagreement/edit tables and sanitized V320 aggregates."""
from __future__ import annotations

import json

import numpy as np

from .common import load_graph,now,read_json,sha,write_json
from .context import RunContext
from .features import EDGE_FEATURES
from .association import VARIANTS


def run(ctx=None):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    ctx=ctx or RunContext.default();samples=ctx.samples();writer=None;totals={};rows=0
    disagreement_path=ctx.out/'disagreements.parquet'
    try:
        for s in samples:
            name=s['dataset'];b=load_graph(ctx.incumbent(name));c=load_graph(ctx.out/'features'/f'{name}.npz')
            f=c['edge_features'];pairs=c['pairs'];ex=f[:,2].astype(bool);votes=f[:,25:29].astype(bool)
            disagree=np.any(votes!=ex[:,None],axis=1);x=f[disagree];p=pairs[disagree]
            data={'dataset':np.repeat(name,len(x)),'source_id':b['nodes'][p[:,0],0],
                'target_id':b['nodes'][p[:,1],0],**{k:x[:,i] for i,k in enumerate(EDGE_FEATURES)}}
            table=pa.Table.from_pydict(data)
            if writer is None:writer=pq.ParquetWriter(disagreement_path,table.schema,compression='zstd')
            writer.write_table(table);rows+=len(x)
            meta=read_json(ctx.out/'features'/f'{name}.json')
            for k in ['teacher_native_union','total_candidates','image_neighbor_added','inserted_nodes','relocated_native_nodes','nodes']:
                totals[k]=totals.get(k,0)+meta[k]
    finally:
        if writer is not None:writer.close()
    edits=[];variant_counts=[]
    for variant in VARIANTS:
        for embryo in ['44b6','6bba','pooled']:
            group=[s for s in samples if embryo=='pooled' or s['embryo']==embryo]
            counts=dict(variant=variant,embryo=embryo,samples=len(group),proposals=0,components=0,abstained=0,accepted_actions=0,changed_edges=0,seconds=0.)
            for s in group:
                rec=read_json(ctx.out/'association_ledgers'/variant/f"{s['dataset']}.json")
                for k in ['proposals','components','abstained','accepted_actions','changed_edges','seconds']:counts[k]+=rec[k]
                if embryo!='pooled':
                    for idx,e in enumerate(rec['edits']):
                        edits.append(dict(dataset=s['dataset'],variant=variant,edit_id=idx,source=rec['source'],
                            value=e['value'],termination=e['termination'],solver_status=e['solver_status'],
                            removed_edges=json.dumps(e['removed']),added_edges=json.dumps(e['added']),
                            canonical_nodes=json.dumps(e['canonical_nodes']),affected_sources=json.dumps(e['affected_sources']),
                            native_support=json.dumps(e['native_support']),read_window=json.dumps(e['read_window'])))
            variant_counts.append(counts)
    pd.DataFrame(edits).to_parquet(ctx.out/'association_edit_ledger.parquet',index=False)
    write_json(ctx.out/'association_summary.json',dict(created=now(),samples=len(samples),feature_totals=totals,
        disagreements_rows=rows,disagreements_sha256=sha(disagreement_path),edit_ledger_rows=len(edits),
        variants=variant_counts,model_lock=read_json(ctx.out/'association_model_lock.json'),
        scope='Inference and source-fit counts only. Root official scorer supplies every gain relative to v2.',
        features_recomputed_at_incumbent_centers_and_adjacency=True,
        teacher_mapping='Only proven native IDs transfer. Different centers measured, inserted teacher nodes excluded.',
        no_target_gt_in_disagreement_or_edit_tables=True))


if __name__=='__main__':run()
