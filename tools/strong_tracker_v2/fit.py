"""Fit small regularized models in both source-only directions before scoring."""
from __future__ import annotations

import time
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from annotation_selection.features import FEATURES

from .common import (OUT,V1,SEED,inventory,load_graph,now,read_json,sha,stage,write_json)
from .hypotheses import FORK_FEATURES
from .native import EDGE_FEATURES,NODE_EXTRA

NODE_FEATURES=FEATURES+NODE_EXTRA
CONFIG=dict(seed=SEED,node_negative_per_tracklet_stratum=12,
    logistic_C=.1,logistic_iterations=400,
    node_tree=dict(max_leaf_nodes=7,max_iter=160,min_samples_leaf=100,l2_regularization=10.,learning_rate=.06),
    edge_tree=dict(max_leaf_nodes=7,max_iter=120,min_samples_leaf=80,l2_regularization=10.,learning_rate=.06),
    fork_tree=dict(max_leaf_nodes=7,max_iter=100,min_samples_leaf=16,l2_regularization=20.,learning_rate=.05),
    selection='Fixed limited settings; no valid independent source inner folds; no outer tuning',
    division_probability_thresholds=[.02,.05,.1],edge_replacement_logodds_margins=[.5,1.5],
    keep_fractions=[1.,.995,.99,.98,.95,.9,.8,.7,.5],
    primary=dict(D='D_temporal_p0.05',E='E_hgb_m1.5',F='F_risk_threshold'),
    provenance='Public upstream checkpoints contaminated; selectors/repairs use only the opposite embryo labels; exploratory')


def source_data(source):
    rows=[r for r in inventory() if r['embryo']==source]
    node_x=[];node_y=[];node_w=[];edge_x=[];edge_y=[];edge_w=[];fork_x=[];fork_y=[];fork_w=[]
    stats=defaultdict(int);positive_groups=set();event_groups=set()
    for r in rows:
        name=r['dataset'];b=load_graph(V1/'baseline/public'/f'{name}.npz')
        c=load_graph(OUT/'native'/f'{name}.npz');h=load_graph(OUT/'hypotheses'/f'{name}.npz')
        t=load_graph(OUT/'evaluation/training'/f'{name}.npz')
        m=load_graph(OUT/'evaluation/membership'/f'{name}.npz');y=m['annotation_label']
        x=np.column_stack([b['features'],c['node_extra']]).astype(np.float32)
        positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
        rng=np.random.default_rng(SEED+sum(name.encode()));groups=defaultdict(list)
        # Origin, depth and density strata within tracklets cover repaired nodes
        # and crowded/dim candidates without treating missing scores as falsehood.
        for i in negative:
            key=(int(b['tracklet'][i]),int(c['native_index'][i]<0),int(b['features'][i,1]*3),int(b['features'][i,8]>=8))
            groups[key].append(i)
        chosen=[];weights=[]
        for members in groups.values():
            k=min(CONFIG['node_negative_per_tracklet_stratum'],len(members))
            chosen.extend(rng.choice(members,k,replace=False));weights.extend([len(members)/k]*k)
        idx=np.r_[positive,np.array(chosen,int)]
        node_x.append(x[idx]);node_y.append(y[idx]);node_w.append(np.r_[np.ones(len(positive)),weights])
        positive_groups.update((name,int(u)) for u in b['tracklet'][positive])
        stats['node_positive_observations']+=len(positive);stats['node_negative_population']+=len(negative)
        stats['node_negative_sampled']+=len(chosen);stats['node_negative_tracklet_strata']+=len(groups)
        edge_x.append(c['edge_features'][t['edge_index']]);edge_y.append(t['edge_y']);edge_w.append(1/t['edge_probability'])
        fork_x.append(h['fork_features'][t['fork_index']]);fork_y.append(t['fork_y']);fork_w.append(1/t['fork_probability'])
        event_groups.update((name,int(i)) for i in t['fork_event'] if i>=0)
    stats.update(samples=len(rows),positive_tracklet_groups=len(positive_groups),division_event_observations=len(event_groups),
                 independent_embryo_clusters=1,biological_event_deduplication_complete=False)
    result={}
    for name,x,y,w in [('node',node_x,node_y,node_w),('edge',edge_x,edge_y,edge_w),('fork',fork_x,fork_y,fork_w)]:
        x=np.concatenate(x);y=np.concatenate(y);w=np.concatenate(w).astype(np.float64)
        assert np.isfinite(x).all() and set(np.unique(y))=={0,1}
        w/=w.mean()
        result[name]=(x,y,w)
        stats[name+'_fit_rows']=len(y);stats[name+'_fit_positives']=int(y.sum())
        stats[name+'_weighted_prevalence']=float(np.average(y,weights=w))
    return result,dict(stats)


def make_model(kind,model_type):
    if model_type=='logistic':
        return make_pipeline(StandardScaler(),LogisticRegression(C=CONFIG['logistic_C'],max_iter=CONFIG['logistic_iterations'],random_state=SEED))
    return HistGradientBoostingClassifier(**CONFIG[kind+'_tree'],random_state=SEED,early_stopping=False)


def fit_model(model,x,y,w):
    if hasattr(model,'steps'):
        model.fit(x,y,standardscaler__sample_weight=w,logisticregression__sample_weight=w)
    else:model.fit(x,y,sample_weight=w)


def run(args):
    if (OUT/'fit_config.json').exists():
        saved=read_json(OUT/'fit_config.json')
        assert {k:v for k,v in saved.items() if k!='created'}==CONFIG,'Locked native fit configuration changed'
    if (OUT/'model_lock.json').exists():
        lock=read_json(OUT/'model_lock.json')
        assert sha(OUT/'fit_config.json')==lock['config_sha256']
        assert all(sha(OUT/p)==h for p,h in lock['models'].items()),'Locked native model changed'
        print('Native fits already complete; model/config hashes verified',flush=True)
        return
    stage('V220','fitting');stage('V230','fitting');stage('V240','fitting')
    write_json(OUT/'fit_config.json',dict(created=now(),**CONFIG),immutable=True) if not (OUT/'fit_config.json').exists() else None
    metrics=[];curves=[];stats={}
    quality=list(range(5,len(NODE_FEATURES)))
    graph=[i for i,c in enumerate(FORK_FEATURES) if c not in ['daughter_separation_change_1','daughter_separation_change_2','daughter_separation_change_3','future_missing_count','motion_barycenter_residual_um']]
    specs=[('F_logistic_quality','node','logistic',quality),('F_hgb_quality','node','hgb',quality),
           ('F_hgb_geometry','node','hgb',list(range(len(NODE_FEATURES)))),
           ('E_logistic','edge','logistic',list(range(len(EDGE_FEATURES)))),
           ('E_hgb','edge','hgb',list(range(len(EDGE_FEATURES)))),
           ('D_graph','fork','logistic',graph),('D_temporal','fork','logistic',list(range(len(FORK_FEATURES)))),
           ('D_hgb','fork','hgb',list(range(len(FORK_FEATURES))))]
    for source in ['44b6','6bba']:
        path=OUT/'models'/source;path.mkdir(parents=True,exist_ok=True)
        data,stats[source]=source_data(source)
        write_json(path/'training_counts.json',stats[source])
        print(source,stats[source],flush=True)
        for mid,kind,mt,cols in specs:
            out=path/f'{mid}.joblib'
            x,y,w=data[kind];xx=x[:,cols]
            if out.exists():model=joblib.load(out)['model'];seconds=0.
            else:
                model=make_model(kind,mt);start=time.perf_counter();fit_model(model,xx,y,w);seconds=time.perf_counter()-start
                names={'node':NODE_FEATURES,'edge':EDGE_FEATURES,'fork':FORK_FEATURES}[kind]
                joblib.dump(dict(model=model,columns=cols,feature_names=[names[i] for i in cols],kind=kind,source=source,
                    config=CONFIG,source_samples=[r['dataset'] for r in inventory() if r['embryo']==source]),out)
            pred=model.predict_proba(xx)[:,1]
            metrics.append(dict(source=source,model=mid,scope='source_fit_inverse_probability_weighted',rows=len(y),positives=int(y.sum()),
                average_precision=average_precision_score(y,pred,sample_weight=w),auroc=roc_auc_score(y,pred,sample_weight=w),
                log_loss=log_loss(y,pred,sample_weight=w),seconds=seconds))
            curves.append(dict(source=source,model=mid,positive_fraction=1.,steps=int(getattr(model,'n_iter_',CONFIG['logistic_iterations']) if not hasattr(model,'steps') else model[-1].n_iter_[0]),
                positives=int(y.sum()),loss=metrics[-1]['log_loss'],scope='source_fit',seconds=seconds))
            print(source,mid,metrics[-1],flush=True)
    pd.DataFrame(metrics).to_csv(OUT/'native_classifier_metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'learning_curves.csv',index=False)
    write_json(OUT/'model_lock.json',dict(created=now(),models={str(p.relative_to(OUT)):sha(p) for p in (OUT/'models').rglob('*.joblib')},
        config_sha256=sha(OUT/'fit_config.json'),sources=stats,both_directions_frozen_before_comparative_scoring=True),immutable=True)
    stage('V220','models_frozen');stage('V230','models_frozen');stage('V240','native_models_frozen_risk_pending')
