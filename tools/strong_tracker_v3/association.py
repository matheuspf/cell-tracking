"""Source-only residual reranking and bounded compatible disagreement edits."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.sparse import coo_matrix
from scipy.special import expit

from .common import (adjacency,digest,graph_hash,load_graph,now,read_json,run_pool,
                     save_arrays,save_graph,sha,validate,write_json)
from .context import RunContext
from .features import EDGE_FEATURES,build,cached,prepare_one

CONFIG=dict(seed=20260909,window_frames=5,max_component_alternatives=256,
 margin_logodds=[1.5,3.0],max_changed_edge_fraction=.02,termination_cost=5.,
 fixed_agreement='all four teachers, or three teachers with native>=.95 and distance<=5um',
 source_fit='all supported positives and endpoint-contradictory negatives; unknown masked',
 logistic=dict(native_offset_coefficient=1.,l2=1.,max_iter=250),
 tree=dict(max_leaf_nodes=7,max_iter=120,min_samples_leaf=80,l2_regularization=10.,learning_rate=.06),
 primary='A_residual_m1.5',policy='same fixed policy both directions; opposite-embryo fit; reused exploratory data')
VARIANTS={f'A_{kind}_m{m}':dict(model=kind,margin=m) for kind in ['residual','hgb7'] for m in [1.5,3.0]}
VARIANTS['A_v2_transfer_m3.0']=dict(model='v2_transfer',margin=3.)


class NativeResidual:
    """Regularized logistic correction with native log odds as a fixed offset."""
    def fit(self,x,y):
        self.columns=np.array([i for i,k in enumerate(EDGE_FEATURES) if k not in ['native_probability','native_logit']],int)
        v=x[:,self.columns].astype(np.float64);self.mean=v.mean(axis=0);self.scale=np.maximum(v.std(axis=0),1e-4)
        z=np.column_stack([np.ones(len(v)),(v-self.mean)/self.scale]);offset=x[:,23].astype(np.float64)
        yy=y.astype(np.float64);penalty=np.r_[0.,np.ones(z.shape[1]-1)]*CONFIG['logistic']['l2']
        def objective(beta):
            a=offset+z@beta;p=expit(a)
            return (np.logaddexp(0,a).sum()-yy@a+.5*np.dot(penalty*beta,beta))/len(y), (z.T@(p-yy)+penalty*beta)/len(y)
        result=minimize(objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',
            options=dict(maxiter=CONFIG['logistic']['max_iter'],ftol=1e-10,gtol=1e-6))
        self.beta=result.x;self.optimizer=dict(success=bool(result.success),message=str(result.message),iterations=result.nit)
        return self

    def decision_function(self,x):
        return x[:,23]+self.beta[0]+((x[:,self.columns]-self.mean)/self.scale)@self.beta[1:]

    def predict_proba(self,x):
        p=expit(self.decision_function(x));return np.column_stack([1-p,p])


def supported_labels(pairs,matched_gt_id,gt_edges):
    """One supported endpoint suffices to refute; unlabeled pairs stay unknown."""
    truth=set(map(tuple,gt_edges));out=set(gt_edges[:,0]);inc=set(gt_edges[:,1])
    return np.array([1 if (int(matched_gt_id[a]),int(matched_gt_id[b])) in truth else
        0 if matched_gt_id[a] in out or matched_gt_id[b] in inc else -1 for a,b in pairs],np.int8)


def training_one(task):
    ctx,sample=task;name=sample['dataset'];p=ctx.out/'evaluation/association_training'/f'{name}.npz'
    b=load_graph(ctx.incumbent(name));c=cached(ctx,sample,b['nodes'],b['edges'])
    gt_path=ctx.v1/'evaluation/gt'/f'{name}.npz'
    inputs=dict(graph_hash=graph_hash(b['nodes'],b['edges']),features_sha256=sha(ctx.out/'features'/f'{name}.npz'),
        gt_sha256=sha(gt_path),metric_revision=ctx.metric_revision,code_sha256=sha(Path(__file__)))
    if p.with_suffix('.json').exists():
        saved=read_json(p.with_suffix('.json'))
        if saved['inputs']!=inputs or saved['sha256']!=sha(p):raise ValueError('Source-label fingerprint drift')
        return name
    gt=load_graph(gt_path);mp=ctx.out/'evaluation/matches/incumbent'/f'{name}.npz'
    if mp.exists():
        mm=load_graph(mp)
        mapping=dict(map(tuple,mm['matched_ids']))
        m=np.array([mapping.get(int(i),-1) for i in b['nodes'][:,0]],np.int64)
    else:
        from annotation_selection.metric_adapter import match_nodes
        mm=match_nodes(b['nodes'],b['edges'],gt['nodes'],gt['edges'],sample['physical_scale'])
        m=np.array([mm.get(int(i),-1) for i in b['nodes'][:,0]],np.int64)
    y=supported_labels(c['pairs'],m,gt['edges']);idx=np.flatnonzero(y>=0)
    save_arrays(p,index=idx,label=y[idx],matched_gt_id=m)
    write_json(p.with_suffix('.json'),dict(inputs=inputs,positive=int(np.sum(y==1)),negative=int(np.sum(y==0)),
        unknown=int(np.sum(y<0)),sha256=sha(p),no_unknown_negatives=True))
    return name


def fit(ctx):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss,average_precision_score
    config_path=ctx.out/'association_config.json';lock_path=ctx.out/'association_model_lock.json'
    configuration=dict(**CONFIG,variants=VARIANTS,feature_names=EDGE_FEATURES)
    if config_path.exists():
        if read_json(config_path)!=configuration:raise ValueError('Association frozen config drift')
    else:write_json(config_path,configuration)
    if lock_path.exists():
        lock=read_json(lock_path)
        if sha(config_path)!=lock['config_sha256'] or not all(sha(ctx.out/p)==h for p,h in lock['models'].items()):
            raise ValueError('Association model lock drift')
        return lock
    counts={};metrics=[]
    for source in ['44b6','6bba']:
        names=[s['dataset'] for s in ctx.samples() if s['embryo']==source];xs=[];ys=[]
        for name in names:
            c=load_graph(ctx.out/'features'/f'{name}.npz');t=load_graph(ctx.out/'evaluation/association_training'/f'{name}.npz')
            xs.append(c['edge_features'][t['index']]);ys.append(t['label'])
        x=np.concatenate(xs);y=np.concatenate(ys);counts[source]=dict(samples=len(names),rows=len(y),positive=int(y.sum()),negative=int(np.sum(y==0)))
        print('Association source',source,counts[source],flush=True)
        for kind in ['residual','hgb7']:
            tic=time.perf_counter()
            model=NativeResidual() if kind=='residual' else HistGradientBoostingClassifier(**CONFIG['tree'],random_state=CONFIG['seed'],early_stopping=False)
            model.fit(x,y);p=model.predict_proba(x)[:,1]
            dest=ctx.out/'association_models'/source/f'{kind}.joblib';dest.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(dict(model=model,source=source,source_samples=names,feature_names=EDGE_FEATURES,
                config_sha256=sha(config_path),feature_hashes={n:sha(ctx.out/'features'/f'{n}.npz') for n in names}),dest)
            metrics.append(dict(source=source,model=kind,source_fit_log_loss=log_loss(y,p),source_fit_ap=average_precision_score(y,p),
                seconds=time.perf_counter()-tic,optimizer=getattr(model,'optimizer',None),**counts[source]))
            print('Association fitted',metrics[-1],flush=True)
    lock=dict(created=now(),config_sha256=sha(config_path),models={str(p.relative_to(ctx.out)):sha(p) for p in (ctx.out/'association_models').rglob('*.joblib')},
        source_counts=counts,source_fit_metrics=metrics,both_directions_frozen_before_scoring=True,
        feature_schema_sha256=digest(EDGE_FEATURES),code_sha256=sha(Path(__file__)),
        transfer_control_model_lock_sha256=sha(ctx.v2/'model_lock.json'),
        transfer_control_models={source:sha(ctx.v2/'models'/source/'E_hgb.joblib') for source in ['44b6','6bba']})
    write_json(lock_path,lock);return lock


def score_features(ctx,source,c,kind):
    if kind=='v2_transfer':
        p=ctx.v2/'models'/source/'E_hgb.joblib';lock=read_json(ctx.v2/'model_lock.json')
        if sha(p)!=lock['models'][str(p.relative_to(ctx.v2))]:raise ValueError('Frozen transfer model drift')
        spec=joblib.load(p);prob=spec['model'].predict_proba(c['edge_features'][:,:23][:,spec['columns']])[:,1]
        return np.log(np.clip(prob,1e-5,1-1e-5)/(1-np.clip(prob,1e-5,1-1e-5)))
    p=ctx.out/'association_models'/source/f'{kind}.joblib';lock=read_json(ctx.out/'association_model_lock.json')
    if sha(p)!=lock['models'][str(p.relative_to(ctx.out))]:raise ValueError('Frozen association model drift')
    model=joblib.load(p)['model']
    if kind=='residual':return model.decision_function(c['edge_features'])
    prob=model.predict_proba(c['edge_features'])[:,1]
    return np.log(np.clip(prob,1e-5,1-1e-5)/(1-np.clip(prob,1e-5,1-1e-5)))


def decode(nodes,edges,c,scores,margin=1.5,max_fraction=.02):
    """Enumerate complete source/owner alternatives, then solve local conflicts.

    Current forks and agreeing supported continuations form fixed boundaries.
    Donor reassignment or explicit costly termination closes every stolen target.
    Five-frame windows contain at most four edge transitions. No-op has zero value.
    """
    ix,pred,succ=adjacency(nodes,edges);pairs=c['pairs'];f=c['edge_features'];old={(ix[int(a)],ix[int(b)]) for a,b in edges}
    score={tuple(p):float(s) for p,s in zip(pairs,scores)};row={tuple(p):k for k,p in enumerate(pairs)}
    fixed={p for p in old if len(succ[p[0]])==2}
    for p in old:
        k=row[p]
        votes=f[k,25:29].sum()
        if votes==4 or (votes>=3 and f[k,0]>=.95 and f[k,3]<=5.):fixed.add(p)
    fs={a for a,b in fixed};ft={b for a,b in fixed};by_source=defaultdict(list)
    for a,b in pairs:
        a,b=int(a),int(b)
        if a not in fs and b not in ft:by_source[a].append(b)
    for a in by_source:by_source[a]=sorted(by_source[a],key=lambda b:(-score[a,b],b))[:4]
    actions=[];seen=set();rejected=defaultdict(int)
    def add_action(sources,new_edges,termination):
        removed={p for a in sources for p in [(a,b) for b in succ[a]]};new=set(new_edges)
        rem=removed-new;adds=new-removed
        if not rem and not adds:return
        if len({b for a,b in new})!=len(new):return
        if any(pred[b] and pred[b][0] not in sources for a,b in new):return
        key=(tuple(sorted(rem)),tuple(sorted(adds)))
        if key in seen:return
        seen.add(key);changed=len(rem)+len(adds)
        value=sum(score.get(p,-5.) for p in adds)-sum(score.get(p,-5.) for p in rem)-margin*len(adds)-CONFIG['termination_cost']*termination
        if value<=1e-9:return
        touched={i for p in rem|adds for i in p};t=int(min(nodes[a,1] for a in sources))
        actions.append(dict(sources=set(sources),removed=rem,added=adds,touched=touched,
            value=float(value),changed=changed,window=t//4,termination=termination))
    for a,targets in by_source.items():
        if len(succ[a])>1:continue
        for b in targets:
            if (a,b) in old:continue
            k=row[a,b]
            # Work only where a teacher supplies an alternative or image/native evidence is strong.
            if f[k,25:29].sum()==0 and f[k,0]<.5 and f[k,3]>5.:continue
            if pred[b]:
                donor=pred[b][0]
                if donor in fs or len(succ[donor])!=1:rejected['fixed_donor']+=1;continue
                options=[j for j in by_source.get(donor,[]) if j!=b and (not pred[j] or pred[j][0] in [a,donor])]
                for j in options:add_action({a,donor},{(a,b),(donor,j)},0)
                add_action({a,donor},{(a,b)},1)
            else:add_action({a},{(a,b)},0)
    # Conflicts include all changed sources, targets and shared path nodes, within each five-frame window.
    parents=list(range(len(actions)))
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    seen_node={}
    for k,a in enumerate(actions):
        for n in a['touched']:
            key=(a['window'],n)
            if key in seen_node:parents[root(k)]=root(seen_node[key])
            else:seen_node[key]=k
    comps=defaultdict(list)
    for k in range(len(actions)):comps[root(k)].append(k)
    chosen=[];cap=int(np.floor(max_fraction*len(edges)));spent=0;abstained=0
    for ks in sorted(comps.values(),key=lambda ks:-max(actions[k]['value'] for k in ks)):
        if len(ks)>CONFIG['max_component_alternatives']:abstained+=1;continue
        keys=sorted({n for k in ks for n in actions[k]['touched']});ri={n:i for i,n in enumerate(keys)}
        rr=[];cc=[];data=[]
        for j,k in enumerate(ks):
            for n in actions[k]['touched']:rr.append(ri[n]);cc.append(j);data.append(1.)
            rr.append(len(keys));cc.append(j);data.append(actions[k]['changed'])
        matrix=coo_matrix((data,(rr,cc)),shape=(len(keys)+1,len(ks))).tocsc()
        upper=np.r_[np.ones(len(keys)),max(cap-spent,0)]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Unrecognized options detected.*')
            result=milp(c=-np.array([actions[k]['value'] for k in ks]),integrality=np.ones(len(ks)),
                bounds=Bounds(0.,1.),constraints=LinearConstraint(matrix,np.zeros(len(upper)),upper),
                options=dict(time_limit=2.,mip_rel_gap=0.,presolve=True,threads=1))
        if not result.success or result.x is None:abstained+=1;continue
        take=[k for j,k in enumerate(ks) if result.x[j]>.5];chosen.extend(take);spent+=sum(actions[k]['changed'] for k in take)
    selected=set(old);ledger=[]
    def ids(es):return sorted([int(nodes[a,0]),int(nodes[b,0])] for a,b in es)
    for k in chosen:
        a=actions[k]
        # A fresh compatibility check guards overlap at shared window boundaries.
        if not a['removed']<=selected:abstained+=1;continue
        trial=(selected-a['removed'])|a['added'];targets=[b for _,b in trial]
        if len(targets)!=len(set(targets)):abstained+=1;continue
        selected=trial;t=min(int(nodes[i,1]) for i in a['touched'])
        ledger.append(dict(removed=ids(a['removed']),added=ids(a['added']),
            affected_sources=[int(nodes[i,0]) for i in sorted(a['sources'])],
            canonical_nodes=nodes[sorted(a['touched'])].tolist(),value=a['value'],termination=a['termination'],
            read_window=[max(0,t-2),t+2],solver_status='optimal',forks_frozen=True,
            native_support=[dict(edge=ids({p})[0],probability=float(f[row[p],0]),missing=bool(f[row[p],1]),
                votes=f[row[p],25:29].tolist()) for p in sorted(a['added'])]))
    if selected==old:out=edges.copy()
    else:out=np.asarray(sorted((int(nodes[a,0]),int(nodes[b,0])) for a,b in selected),np.int64).reshape(-1,2)
    return out,dict(proposals=len(actions),components=len(comps),abstained=abstained,
        accepted_actions=len(ledger),changed_edges=len(selected^old),edit_cap=cap,fixed_edges=len(fixed),
        rejected=dict(rejected),edits=ledger)


def apply(ctx,sample,base_nodes,base_edges,model_family='residual',margin=1.5):
    name=sample['dataset'];b=load_graph(ctx.incumbent(name))
    if np.array_equal(base_nodes,b['nodes']) and np.array_equal(base_edges,b['edges']):c=cached(ctx,sample,base_nodes,base_edges)
    else:
        old=cached(ctx,sample,b['nodes'],b['edges'])
        same_nodes=np.array_equal(base_nodes,b['nodes'])
        c,_=build(ctx,sample,base_nodes,base_edges,node_features=old['node_features'] if same_nodes else None)
    source='6bba' if sample['embryo']=='44b6' else '44b6';scores=score_features(ctx,source,c,model_family)
    ee,ledger=decode(base_nodes,base_edges,c,scores,margin)
    validate(base_nodes,ee,sample['image_shape'])
    return base_nodes.copy(),ee,ledger


_ANNOTATIONS_DENIED=False


def deny_annotations():
    """Install once in a fresh inference worker, before model/feature loading."""
    import os
    import sys
    global _ANNOTATIONS_DENIED
    if _ANNOTATIONS_DENIED:return
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.path.abspath(os.fsdecode(args[0]))
            if '.geff' in p or '/evaluation/' in p or '/oracle' in p or 'census' in p or p.endswith('/inventory.json'):
                raise PermissionError(f'Annotations unavailable: {p}')
    sys.addaudithook(audit);_ANNOTATIONS_DENIED=True


def inference_one(task):
    deny_annotations()
    ctx,sample=task;name=sample['dataset'];b=load_graph(ctx.incumbent(name));c=cached(ctx,sample,b['nodes'],b['edges'])
    source='6bba' if sample['embryo']=='44b6' else '44b6';scores={k:score_features(ctx,source,c,k) for k in ['residual','hgb7','v2_transfer']}
    records={}
    for variant,cfg in VARIANTS.items():
        p=ctx.out/'candidate_graphs'/variant/f'{name}.npz';receipt=ctx.out/'association_ledgers'/variant/f'{name}.json'
        inputs=dict(incumbent_sha256=sha(ctx.incumbent(name)),features_sha256=sha(ctx.out/'features'/f'{name}.npz'),
            model_lock_sha256=sha(ctx.out/'association_model_lock.json'),config_sha256=digest(cfg),code_sha256=sha(Path(__file__)))
        if receipt.exists():
            record=read_json(receipt)
            if record['inputs']!=inputs or record['sha256']!=sha(p):raise ValueError('Association prediction drift')
            records[variant]=record;continue
        tic=time.perf_counter();e,ledger=decode(b['nodes'],b['edges'],c,scores[cfg['model']],cfg['margin'])
        validate(b['nodes'],e,sample['image_shape']);save_graph(p,b['nodes'],e)
        record=dict(inputs=inputs,sha256=sha(p),graph_hash=graph_hash(b['nodes'],e),source=source,seconds=time.perf_counter()-tic,**ledger)
        write_json(receipt,record);records[variant]=record
    return dict(dataset=name,changed={v:r['changed_edges'] for v,r in records.items()})


def source_oracle_one(task):
    """Retrospective source feasibility only; never read by inference."""
    ctx,sample=task;name=sample['dataset'];b=load_graph(ctx.incumbent(name));c=cached(ctx,sample,b['nodes'],b['edges'])
    gt=load_graph(ctx.v1/'evaluation/gt'/f'{name}.npz');t=load_graph(ctx.out/'evaluation/association_training'/f'{name}.npz')
    y=supported_labels(c['pairs'],t['matched_gt_id'],gt['edges'])
    scores=np.where(y==1,12.,np.where(y==0,-12.,0.))
    e,ledger=decode(b['nodes'],b['edges'],c,scores,margin=1.5)
    validate(b['nodes'],e,sample['image_shape'])
    save_graph(ctx.out/'evaluation/source_association_oracle'/f'{name}.npz',b['nodes'],e)
    write_json(ctx.out/'evaluation/source_association_oracle'/f'{name}.json',dict(dataset=name,
        source=sample['embryo'],scope='same source training diagnostic; heuristic legal rewire feasibility, not global bound',
        original_graph_hash=graph_hash(b['nodes'],b['edges']),graph_hash=graph_hash(b['nodes'],e),**ledger))
    return dict(dataset=name,oracle_changed_edges=ledger['changed_edges'])


def run(ctx=None,args=None,workers=4):
    if args is not None:workers=args.workers
    ctx=ctx or RunContext.default();ctx.check_outputs();samples=ctx.samples();tic=time.perf_counter()
    list(run_pool(prepare_one,[(ctx,s) for s in samples],workers))
    feature_lock=ctx.out/'association_feature_lock.json'
    if not feature_lock.exists():write_json(feature_lock,dict(created=now(),feature_schema=EDGE_FEATURES,
        files={s['dataset']:sha(ctx.out/'features'/f"{s['dataset']}.npz") for s in samples},
        current_incumbent_features=True,config=CONFIG))
    list(run_pool(training_one,[(ctx,s) for s in samples],workers));fit(ctx)
    lock_path=ctx.out/'association_round_lock.json'
    if not lock_path.exists():write_json(lock_path,dict(created=now(),variants=VARIANTS,
        model_lock_sha256=sha(ctx.out/'association_model_lock.json'),both_directions_frozen=True,
        outer_score_exposure=False,initial_round=True,feature_lock_sha256=sha(ctx.out/'association_feature_lock.json')))
    list(run_pool(inference_one,[(ctx,s) for s in samples],workers))
    result=dict(created=now(),variants=VARIANTS,samples=len(samples),seconds=time.perf_counter()-tic,
        model_lock_sha256=sha(ctx.out/'association_model_lock.json'),round_lock_sha256=sha(lock_path),
        predictions={v:{s['dataset']:sha(ctx.out/'candidate_graphs'/v/f"{s['dataset']}.npz") for s in samples} for v in VARIANTS})
    write_json(ctx.out/'association_inference_receipt.json',result)
    # Oracle labels are exposed only after both directions' non-oracle predictions freeze.
    list(run_pool(source_oracle_one,[(ctx,s) for s in samples],workers))
    return result


if __name__=='__main__':
    import argparse
    # Persist model classes under their importable module, never __main__.
    from strong_tracker_v3.association import run as run_study
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();run_study(workers=args.workers)
