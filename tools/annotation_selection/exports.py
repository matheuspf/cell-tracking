"""Streaming contract tables and input-preservation receipts."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .common import DATA,OUT,REPO,METRIC_REV,load_graph,now,read_json,sha,write_json
from .features import FEATURES,VERSION
from .filter_graph import PreparedFilter
from .metric_adapter import evaluate_graph


def duplicate_audit():
    archive=read_json(REPO/'work/data-manifest.json')
    by_dataset=defaultdict(list)
    for r in archive['files']:
        parts=r['path'].split('/')
        if len(parts)>4 and parts[1].endswith('.zarr') and '/0/c/' in r['path']:
            by_dataset[(parts[0],parts[1][:-5])].append(('/'.join(parts[2:]),r['bytes'],r['crc32']))
    lookup=defaultdict(list)
    for key,entries in by_dataset.items():lookup[tuple(sorted(entries))].append(key)
    duplicate_groups=[]
    for datasets in lookup.values():
        if len(datasets)<2:continue
        checks=[]
        for split,name in datasets:
            checks.append([sha(DATA/split/f'{name}.zarr/0/c/{t}/0/0/0') for t in [0,49,99]])
        equal=all(c==checks[0] for c in checks)
        if not equal:raise ValueError('Archive duplicate fingerprint content mismatch')
        duplicate_groups.append(dict(datasets=datasets,archive_chunks_equal=True,checked_chunk_sha256=checks[0],content_equal=True))
        train=[n for s,n in datasets if s=='train']
        if len({n.split('_')[0] for n in train})>1:raise ValueError('Duplicate training images cross outer embryos')
    write_json(OUT/'duplicate_audit.json',dict(created=now(),groups=duplicate_groups,training_images=sum(k[0]=='train' for k in by_dataset),
               visible_test_used_for_independent_validation=False,cross_embryo_training_duplicates=False))


def run(args):
    if not (OUT/'prediction_lock.json').exists():raise ValueError('Detailed report exports remain embargoed before prediction lock')
    duplicate_audit()
    rows=read_json(OUT/'inventory.json');writers={};missed=[];stats=[];endpoints=[]
    manifest=read_json(OUT/'fold_manifest.json')
    try:
        for count,row in enumerate(rows,1):
            name=row['dataset'];b=load_graph(OUT/'baseline/clean'/f'{name}.npz');gt=load_graph(OUT/'evaluation/gt'/f'{name}.npz')
            n,e=b['nodes'],b['edges']
            meta=read_json(OUT/'baseline/clean'/f'{name}.json')
            f=pd.DataFrame(b['features'],columns=FEATURES)
            for j,key in enumerate(['candidate_id','t','z','y','x']):f[key]=n[:,j]
            f['dataset']=name;f['generator_id']='classical_dog_v1';f['prediction_graph_hash']=meta['graph_hash']
            f['feature_version']=VERSION;f['predicted_tracklet_id']=b['tracklet']
            label=load_graph(OUT/'evaluation/membership'/f'{name}.npz')
            labels=pd.DataFrame(dict(dataset=name,candidate_id=n[:,0],matched_gt_id=label['matched_gt_id'],
                      annotation_label=label['annotation_label'],ambiguity_stratum=np.where(label['ambiguous'],'unmatched_within_7um','matched_or_far_unmatched'),matching_revision=METRIC_REV))
            for key,df in [('candidate_features',f),('membership_labels',labels)]:
                path=OUT/(key+'.parquet') if key=='candidate_features' else OUT/'evaluation'/(key+'.parquet')
                table=pa.Table.from_pandas(df,preserve_index=False)
                if key not in writers:writers[key]=pq.ParquetWriter(path,table.schema,compression='zstd')
                writers[key].write_table(table)
            missing=gt['nodes'][~np.isin(gt['nodes'][:,0],label['matched_gt_id'][label['matched_gt_id']>=0])]
            missed.extend(dict(dataset=name,gt_node_id=int(node[0])) for node in missing)
            er,matches,tp=evaluate_graph(name,n,e,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
            if er['matched_nodes']!=int(label['annotation_label'].sum()):raise ValueError('Fresh baseline matching drift')
            tp_array=np.array(sorted(tp),np.int64).reshape(-1,2)
            np.savez_compressed(OUT/'evaluation/membership'/f'{name}_baseline_tp.npz',edges=tp_array)
            gt_index={int(r[0]):r[2:] for r in gt['nodes']};node_index={int(r[0]):i for i,r in enumerate(n)}
            distance=np.array([np.linalg.norm((n[node_index[i],2:]-gt_index[j])*row['physical_scale']) for i,j in matches.items()])
            stats.append(dict(dataset=name,matched_nodes=len(matches),ambiguous_candidates=int(label['ambiguous'].sum()),
                              candidate_nodes=len(n),missed_gt=len(missing),matched_distance_median=float(np.median(distance)) if len(distance) else None,
                              matched_distance_p95=float(np.quantile(distance,.95)) if len(distance) else None))
            source=next(d['source'] for d in manifest['directions'] if d['outer']==row['embryo'])
            with np.load(OUT/'predictions'/source/f'{name}.npz') as scores:
                prepare=PreparedFilter(n,e)
                for mid in ['quality','hgb_all_leaf7','image_only_seed20260908']:
                    for policy in ['nodes','tracklets']:
                        for r in [.9,.8,.7,.5,.3,.1]:
                            keep=prepare.mask(scores[mid],r,('quality_' if mid=='quality' else 'membership_')+policy,key=mid)
                            a=np.array([keep[node_index[int(i)]] for i in tp_array[:,0]])
                            c=np.array([keep[node_index[int(i)]] for i in tp_array[:,1]])
                            ta=float(a.mean()) if len(a) else np.nan;tc=float(c.mean()) if len(c) else np.nan
                            both=float((a&c).mean()) if len(a) else np.nan;den=ta*(1-ta)*tc*(1-tc)
                            endpoints.append(dict(dataset=name,embryo=row['embryo'],model_id=mid,policy=policy,requested_keep=r,baseline_tp=len(tp),
                                      source_kept=int(a.sum()),target_kept=int(c.sum()),both_kept=int((a&c).sum()),
                                      source_keep_rate=ta,target_keep_rate=tc,tp_keep_rate=both,
                                      endpoint_keep_correlation=(both-ta*tc)/np.sqrt(den) if den>0 else None))
            print(f'Contract tables {count}/{len(rows)} {name}',flush=True)
    finally:
        for writer in writers.values():writer.close()
    pd.DataFrame(missed).to_csv(OUT/'evaluation/missed_gt_nodes.csv',index=False)
    pd.DataFrame(stats).to_csv(OUT/'matching_diagnostics.csv',index=False)
    pd.DataFrame(endpoints).to_csv(OUT/'endpoint_correlations.csv',index=False)
    write_json(OUT/'contract_table_receipt.json',dict(created=now(),samples=len(rows),baseline_tp_tables=True,
               gt_unavailable_feature_columns=FEATURES,metadata=dict(schema_version=1,run_id='annotation-selection-v1',producer='python -m annotation_selection export')))
