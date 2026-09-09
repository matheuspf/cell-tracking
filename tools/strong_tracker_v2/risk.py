"""Source-only observed-metric risk targets, including actual group rescoring."""
from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from annotation_selection.filter_graph import filtered
from annotation_selection.metric_adapter import aggregate,evaluate_graph

from .common import (OUT,V1,SEED,inventory,load_graph,now,read_json,run_pool,save_arrays,sha,write_json)
from .evaluate import baseline_fp_set
from .policy import group_actions,group_features

CONFIG=dict(positive_groups_per_clip_rescored=3,negative_groups_per_clip_rescored=5,
    surrogate='All source groups; unique incident matched TP edges and FP edges per removed node',
    correction='Actual official group-deletion run-level score loss per node, including rematching',
    tree=dict(max_leaf_nodes=7,max_iter=100,min_samples_leaf=20,l2_regularization=10.,learning_rate=.05),
    seed=SEED,primary_threshold=-.005,primary_max_removed_fraction=.1,
    primary_action='whole tracklets, predicted fork context protected',
    keep_fractions=[.995,.99,.98,.95,.9,.8,.7,.5])


def prepare_one(row):
    name=row['dataset'];path=OUT/'evaluation/risk'/f'{name}.npz'
    if path.exists() and path.with_suffix('.json').exists():return name
    b=load_graph(V1/'baseline/public'/f'{name}.npz');n,e=b['nodes'],b['edges']
    c=load_graph(OUT/'native'/f'{name}.npz');m=load_graph(OUT/'evaluation/membership'/f'{name}.npz')
    gt=load_graph(V1/'evaluation/gt'/f'{name}.npz')
    matches={int(i):int(j) for i,j in zip(n[:,0],m['matched_gt_id']) if j>=0}
    tp=set(map(tuple,m['tp_edges']));fp=baseline_fp_set(n,e,gt,matches)
    groups,protected=group_actions(n,e,'fork_protected');x=group_features(b,c,groups)
    unit=np.empty(len(n),int)
    for k,g in enumerate(groups):unit[g]=k
    ix={int(r[0]):i for i,r in enumerate(n)};lost=np.zeros((len(groups),2))
    for kind,es in enumerate([tp,fp]):
        for a,d in es:
            for j in {unit[ix[int(a)]],unit[ix[int(d)]]}:lost[j,kind]+=1
    cost=np.array([len(g) for g in groups]);target=lost/cost[:,None]
    eligible=np.array([not protected[g].any() for g in groups])
    positive=np.flatnonzero((lost[:,0]>0)&eligible);negative=np.flatnonzero((lost[:,0]==0)&eligible)
    rng=np.random.default_rng(SEED+sum(name.encode()))
    selected=[];weights=[]
    for members,kmax in [(positive,3),(negative,5)]:
        k=min(len(members),kmax)
        if not k:continue
        selected.extend(rng.choice(members,k,replace=False));weights.extend([len(members)/k]*k)
    baselines=[read_json(OUT/'evaluation/census'/f"{r['dataset']}.json")['base'] for r in inventory() if r['embryo']==row['embryo']]
    expected=[r['dataset'] for r in baselines];source_score=aggregate(baselines,expected)['score']
    actual=[];details=[]
    for j in selected:
        keep=np.ones(len(n),bool);keep[groups[j]]=False
        nn,ee=filtered(n,e,keep);tic=time.perf_counter()
        new,_,newtp=evaluate_graph(name,nn,ee,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
        changed=[new if r['dataset']==name else r for r in baselines]
        score=aggregate(changed,expected)['score']
        loss=(source_score-score)*1e6/cost[j]
        actual.append(loss)
        details.append(dict(group=int(j),removed_nodes=int(cost[j]),run_score_before=source_score,run_score_after=score,
            actual_loss_per_node_micropoints=loss,survivor_tp_loss=int(lost[j,0]),
            actual_tp_loss=len(tp)-len(newtp),newly_matched_tp=len(newtp-tp),
            division_tp=new['division_tp'],division_fp=new['division_fp'],division_fn=new['division_fn'],
            seconds=time.perf_counter()-tic))
    save_arrays(path,features=x,surrogate_target=target,actual_index=np.array(selected,int),
        actual_loss=np.array(actual),actual_sampling_weight=np.array(weights),cost=cost,eligible=eligible)
    write_json(path.with_suffix('.json'),dict(dataset=name,embryo=row['embryo'],all_groups=len(groups),
        positive_groups=int((lost[:,0]>0).sum()),positive_eligible_groups=len(positive),negative_eligible_groups=len(negative),
        actual_deletions=details,independence='Predicted groups are action units, not certified biological replicates'))
    return name


def fit():
    locks={};stats={}
    for source in ['44b6','6bba']:
        data=[load_graph(OUT/'evaluation/risk'/f"{r['dataset']}.npz") for r in inventory() if r['embryo']==source]
        x=np.concatenate([d['features'] for d in data]);sur=np.concatenate([d['surrogate_target'] for d in data])
        heads=[]
        for k in [0,1]:
            h=HistGradientBoostingRegressor(**CONFIG['tree'],random_state=SEED,early_stopping=False)
            h.fit(x,sur[:,k]);heads.append(h)
        xx=np.concatenate([d['features'][d['actual_index']] for d in data])
        yy=np.concatenate([d['actual_loss'] for d in data]);w=np.concatenate([d['actual_sampling_weight'] for d in data]);w/=w.mean()
        predicted=np.column_stack([h.predict(xx) for h in heads])
        head=HistGradientBoostingRegressor(**CONFIG['tree'],random_state=SEED,early_stopping=False)
        head.fit(np.column_stack([xx,predicted]),yy,sample_weight=w)
        dest=OUT/'risk_models'/f'{source}.joblib';dest.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(dict(source=source,surrogate_heads=heads,actual_head=head,config=CONFIG,feature_count=x.shape[1]),dest)
        locks[source]=sha(dest);stats[source]=dict(all_action_groups=len(x),positive_action_groups=int((sur[:,0]>0).sum()),
            actual_rescored_deletions=len(yy),source_fit_weighted_mae=float(np.average(np.abs(head.predict(np.column_stack([xx,predicted]))-yy),weights=w)))
    write_json(OUT/'risk_model_lock.json',dict(created=now(),models=locks,config=CONFIG,training=stats,
        both_sources_frozen_before_scoring=True,independent_source_validation=False),immutable=True)


def run(args):
    write_json(OUT/'risk_config.json',CONFIG,immutable=True)
    if (OUT/'risk_model_lock.json').exists():
        lock=read_json(OUT/'risk_model_lock.json')
        assert lock['config']==CONFIG,'Locked risk fit configuration changed'
        assert all(sha(OUT/'risk_models'/f'{s}.joblib')==h for s,h in lock['models'].items()),'Locked risk model changed'
        print('Risk fits already complete; model/config hashes verified',flush=True)
        return
    rows=inventory();rows=rows[:args.limit] if args.limit else rows
    list(run_pool(prepare_one,rows,args.workers))
    if not args.limit:fit()
