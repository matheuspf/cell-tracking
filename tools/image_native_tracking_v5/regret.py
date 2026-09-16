"""Evaluation-only edge/node regret and explicitly arithmetic score decomposition."""
import time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from .common import *


def one(task):
    variant,row=task;name=row['dataset'];dest=OUT/'regret'/variant/f'{name}.json'
    if dest.exists():return read(dest)
    base=graph(name);new=graph(name,variant)
    b=arrays(OUT/'evaluation_matches/C0'/f'{name}.npz');n=arrays(OUT/'evaluation_matches'/variant/f'{name}.npz')
    bm=dict(map(tuple,b['matches']));nm=dict(map(tuple,n['matches']))
    be=set(map(tuple,base['edges']));ne=set(map(tuple,new['edges']))
    bg=set(map(tuple,b['recovered']));ng=set(map(tuple,n['recovered']))
    bt=set(map(tuple,b['tp']));nt=set(map(tuple,n['tp']))
    bi=set(base['nodes'][:,0].astype(int));ni=set(new['nodes'][:,0].astype(int));added=ni-bi;removed=bi-ni
    touched=set(int(i) for edge in new['edges'] for i in edge);linked_new=added&touched
    br=read(OUT/'evaluation/C0'/f'{name}.json');nr=read(OUT/'evaluation'/variant/f'{name}.json')
    assert len(bg)==br['edge_tp'] and len(ng)==nr['edge_tp']
    base_coords={int(v[0]):tuple(v[1:]) for v in base['nodes']};new_coords={int(v[0]):tuple(v[1:]) for v in new['nodes']}
    moved=sum(base_coords[i]!=new_coords[i] for i in bi&ni)
    config=read(OUT/'execution_protocol.json')['variants'][variant]
    r=dict(dataset=name,embryo=row['embryo'],variant=variant,seed=config.get('seed'),family=config['family'],population=config.get('pop'),
        added_edges=len(ne-be),removed_edges=len(be-ne),changed_edges=len(be^ne),
        predicted_TP_edges_gained=len(nt-bt),predicted_TP_edges_lost=len(bt-nt),
        GT_edges_gained=len(ng-bg),GT_edges_lost=len(bg-ng),GT_edge_net=len(ng)-len(bg),
        GT_nodes_gained=len(set(nm.values())-set(bm.values())),GT_nodes_lost=len(set(bm.values())-set(nm.values())),
        added_nodes=len(added),removed_nodes=len(removed),moved_existing_nodes=moved,
        new_nodes_in_selected_edges=len(linked_new),selected_new_nodes_isolated=len(added-linked_new),
        selected_edges_incident_to_new_nodes=sum(int(a) in added or int(b) in added for a,b in ne),
        matched_new_nodes=sum(i in nm for i in added),removed_matched_C0_nodes=sum(i in bm for i in removed),
        retained_predicted_ID_match_changes=sum(i in nm and bm[i]!=nm[i] for i in bm if i in ni),
        delta_edge_tp=nr['edge_tp']-br['edge_tp'],delta_edge_fp=nr['edge_fp']-br['edge_fp'],
        delta_division_tp=nr['division_tp']-br['division_tp'],delta_division_fp=nr['division_fp']-br['division_fp'],
        graph_hash=nr['graph_hash'])
    assert r['GT_edges_gained']-r['GT_edges_lost']==r['delta_edge_tp']
    if config.get('pop')=='P0':assert not added and not removed and moved==0
    save(OUT/'regret_examples'/variant/f'{name}.npz',GT_edges_gained=np.asarray(sorted(ng-bg),np.int64).reshape(-1,2),
        GT_edges_lost=np.asarray(sorted(bg-ng),np.int64).reshape(-1,2),new_nodes=np.asarray(sorted(added),np.int64))
    write(dest,r);return r


def decomposition(scores):
    from annotation_selection.metric_adapter import aggregate
    samples=inventory();base={r['dataset']:read(OUT/'evaluation/C0'/f"{r['dataset']}.json") for r in samples}
    index={(r['variant'],r['embryo']):r for r in scores};rows=[]
    for variant in sorted({r['variant'] for r in scores}):
        if variant=='C0':continue
        current={r['dataset']:read(OUT/'evaluation'/variant/f"{r['dataset']}.json") for r in samples}
        for em in ['44b6','6bba','pooled']:
            names=[r['dataset'] for r in samples if em=='pooled' or r['embryo']==em]
            # Counts/matches remain those of each measured graph. Only the count-penalty
            # arithmetic changes here; this is not a new node-only graph experiment.
            c0_counts_new_size=[{**base[n],'num_pred_nodes':current[n]['num_pred_nodes']} for n in names]
            new_counts_c0_size=[{**current[n],'num_pred_nodes':base[n]['num_pred_nodes']} for n in names]
            a=aggregate(c0_counts_new_size,names)['score'];b=aggregate(new_counts_c0_size,names)['score']
            real=index[variant,em];inc=index['C0',em]
            div=.1*(real['division_jaccard']-inc['division_jaccard'])
            node=real['score']-b;association=b-inc['score']-div
            assert abs(node+association+div-real['delta_C0'])<1e-12
            rows.append(dict(variant=variant,embryo=em,samples=len(names),measured_delta=real['delta_C0'],
                adjusted_edge_delta=real['adj_edge_jaccard']-inc['adj_edge_jaccard'],division_contribution_delta=div,
                arithmetic_node_effect_at_C0_edges=a-inc['score'],arithmetic_node_effect_at_variant_edges=node,
                association_and_matching_at_C0_count=association,count_edge_interaction=node-(a-inc['score']),
                interpretation='arithmetic counterfactual; not a graph experiment or causal attribution'))
    return rows


def run(wait=False):
    expected=set(read(OUT/'execution_protocol.json')['variants'])
    while True:
        complete={p.name for p in (OUT/'evaluation').iterdir() if len(list(p.glob('*.json')))==199}
        if not wait or complete==expected:break
        time.sleep(30)
    variants=sorted(complete-{'C0'});tasks=[(v,r) for v in variants for r in inventory()]
    with cpu_batch(),ProcessPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(one,tasks,chunksize=2))
    pd.DataFrame(result).to_csv(OUT/'regret_rows.csv',index=False)
    scores=pd.read_csv(OUT/'ablation_scores.csv').to_dict('records')
    pd.DataFrame(decomposition(scores)).to_csv(OUT/'score_decomposition.csv',index=False)
    numeric=[k for k in result[0] if k not in ['dataset','embryo','variant','seed','family','population','graph_hash']] if result else []
    summary=[]
    for v in variants:
        for em in ['44b6','6bba','pooled']:
            sub=[r for r in result if r['variant']==v and (em=='pooled' or r['embryo']==em)]
            summary.append(dict(variant=v,embryo=em,samples=len(sub),**{k:sum(r[k] for r in sub) for k in numeric}))
    write(OUT/'regret_summary.json',summary)
    print('Complete edge/node regret',len(variants),'variants',len(result),'clip-variant rows',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args();run(a.wait)
