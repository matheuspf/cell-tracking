"""Matched point control with complete native evidence and fresh official scoring.

No segmenter, classical or learned, is substituted when masks are unavailable.
The unchanged v3 association decoder freezes existing forks. The mask-division
extension is a separate, explicitly unexecuted gate until real masks exist.
"""
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from strong_tracker_v3.association import decode
from strong_tracker_v3.common import validate, save_arrays
from strong_tracker_v3.features import EDGE_FEATURES
from .common import *

CONFIG = dict(seed=20260910, l2=1., max_iter=250, margin=3., max_changed_edge_fraction=.02,
    scorer='L2 logistic residual with fixed full-native logit offset',
    decoder='unchanged strong_tracker_v3.association.decode; existing forks frozen',
    features=EDGE_FEATURES, sparse_targets='positive GT links; contradictory incoming parents only; other links unknown',
    source_fit='all supported transitions of source embryo, opposite-source application',
    max_complete_configurations=6, replication_slots=2, gpu_hours_cap=24,
    mask_recipes_um=[6.,9.], recipe_selection='not selected: no learned backend',
    target_results_select_recipes=False)

def evidence(name):
    graph = c0(name)
    path = V3/'fresh_evidence'/f'{name}.npz'
    saved = load_graph(path)
    assert np.array_equal(saved['incumbent_nodes'], graph['nodes'])
    # Preserve the exact incumbent preselector evidence and candidate bank. This
    # includes full primary/secondary/eight-view native logits and teacher votes.
    x = saved['edge_features'].copy()
    pairs = saved['pairs']
    assert x.shape == (len(pairs), len(EDGE_FEATURES)) and np.isfinite(x).all()
    return graph, pairs, x

def supported_labels(pairs, nodes, matches, gt_edges):
    reverse = {int(g): int(p) for p,g in matches}
    positives = {(reverse[int(a)], reverse[int(b)]) for a,b in gt_edges
                 if int(a) in reverse and int(b) in reverse}
    # A recorded daughter refutes competing incoming parents for that daughter;
    # it does NOT refute an unrecorded second daughter of the same parent.
    targets = {b for _,b in positives}
    ids = nodes[pairs, 0]
    return np.asarray([1 if tuple(p) in positives else 0 if int(p[1]) in targets else -1
                       for p in ids], np.int8)

def fit_residual(x, y, offset, names):
    mean = x.mean(0, dtype=np.float64)
    scale = np.maximum(x.std(0, dtype=np.float64), 1e-4)
    z = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    penalty = np.r_[0., np.ones(x.shape[1])] * CONFIG['l2']
    def objective(beta):
        logits = offset + z @ beta
        return ((np.logaddexp(0,logits).sum()-y@logits + .5*np.dot(penalty*beta,beta))/len(y),
                (z.T@(expit(logits)-y)+penalty*beta)/len(y))
    result = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B',
                      options=dict(maxiter=CONFIG['max_iter'],ftol=1e-10,gtol=1e-6))
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError('Nonfinite source fit')
    return dict(mean=mean, scale=scale, beta=result.x, feature_names=names,
                iterations=result.nit, success=bool(result.success), message=str(result.message),
                objective=float(result.fun), rows=len(y), positives=int(y.sum()))

def fit():
    rows = inventory()
    fit_columns = [i for i,k in enumerate(EDGE_FEATURES) if k not in ['native_probability','native_logit']]
    records = []
    for source in ['44b6','6bba']:
        destination = OUT/'models'/f'P0_{source}.json'
        if destination.exists():
            spec = read_json(destination)
            assert spec['configuration_sha256'] == digest(CONFIG)
            for name,expected in spec['training_hashes'].items():
                assert sha(V3/'fresh_evidence'/f'{name}.npz')==expected['evidence']
                assert sha(V1/'evaluation/gt'/f'{name}.npz')==expected['gt']
                assert sha(V5/'evaluation_matches/C0'/f'{name}.npz')==expected['matching']
            records.append(spec); continue
        start = time.monotonic(); xs=[]; ys=[]; offsets=[]; hashes={}; unknown=0
        for row in rows:
            if row['embryo'] != source: continue
            name=row['dataset']; g,pairs,x=evidence(name)
            matches=load_graph(V5/'evaluation_matches/C0'/f'{name}.npz')['matches']
            gt=load_graph(V1/'evaluation/gt'/f'{name}.npz')
            y=supported_labels(pairs,g['nodes'],matches,gt['edges']); known=y>=0
            xs.append(x[known][:,fit_columns]); ys.append(y[known]); offsets.append(x[known,23])
            unknown += int((~known).sum())
            hashes[name] = dict(evidence=sha(V3/'fresh_evidence'/f'{name}.npz'),
                gt=sha(V1/'evaluation/gt'/f'{name}.npz'), matching=sha(V5/'evaluation_matches/C0'/f'{name}.npz'))
        model=fit_residual(np.concatenate(xs),np.concatenate(ys).astype(float),np.concatenate(offsets),
                           [EDGE_FEATURES[i] for i in fit_columns])
        spec=dict(source=source, columns=fit_columns, model=model, training_hashes=hashes,
                  unknown_rows=unknown, configuration_sha256=digest(CONFIG), seconds=time.monotonic()-start)
        write(destination,spec); records.append(spec)
        print('source fit',source,model['rows'],model['iterations'],model['success'],flush=True)
    write(OUT/'model_lock.json',dict(frozen=now(),both_directions_frozen_before_target_scores=True,
        configuration_sha256=digest(CONFIG), models={s:sha(OUT/'models'/f'P0_{s}.json') for s in ['44b6','6bba']}))
    return records

def predict_scores(x, source):
    spec=read_json(OUT/'models'/f'P0_{source}.json'); m=spec['model']
    return x[:,23]+m['beta'][0]+((x[:,spec['columns']]-m['mean'])/m['scale'])@np.asarray(m['beta'][1:])

def one(row):
    from annotation_selection.metric_adapter import evaluate_graph
    name=row['dataset']; base,pairs,x=evidence(name)
    gt=load_graph(V1/'evaluation/gt'/f'{name}.npz'); source='6bba' if row['embryo']=='44b6' else '44b6'
    scores=predict_scores(x,source); begin=time.monotonic()
    edges,ledger=decode(base['nodes'],base['edges'],dict(pairs=pairs,edge_features=x),scores,
                         margin=CONFIG['margin'],max_fraction=CONFIG['max_changed_edge_fraction'])
    prediction_seconds=time.monotonic()-begin
    reserve(SCRATCH)
    save_arrays(SCRATCH/'predictions/P0'/f'{name}.npz',nodes=base['nodes'],edges=edges)
    results=[]
    for arm,ee in [('C0',base['edges']),('P0',edges)]:
        dest=OUT/'evaluation'/arm/f'{name}.json'
        expected=graph_hash(base['nodes'],ee)
        if dest.exists():
            result=read_json(dest); assert result['graph_hash']==expected
            assert result['gt_sha256']==sha(V1/'evaluation/gt'/f'{name}.npz')
            results.append(result); continue
        validate(base['nodes'],ee,row['image_shape'])
        start=time.monotonic()
        result,matching,tp=evaluate_graph(name,base['nodes'],ee,gt['nodes'],gt['edges'],
                                          row['physical_scale'],row['estimated_total'])
        result.update(variant=arm,embryo=row['embryo'],seconds=time.monotonic()-start,
                      prediction_seconds=0 if arm=='C0' else prediction_seconds,
                      fresh_official_matching=True,gt_sha256=sha(V1/'evaluation/gt'/f'{name}.npz'),
                      matched_nodes=len(matching),changed_edges=0 if arm=='C0' else ledger['changed_edges'])
        write(dest,result);results.append(result)
    return name

def run():
    if not (OUT/'preflight.json').exists(): raise Blocked('Run preflight first')
    config_path=OUT/'configuration.json'
    if config_path.exists(): assert read_json(config_path)==CONFIG
    else: write(config_path,CONFIG)
    start=time.monotonic(); fit()
    reserve(SCRATCH); SCRATCH.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,name in enumerate(pool.map(one,inventory()),1):
            if i%10==0 or i==199: print('official C0/P0',i,'/199',flush=True)
            write(OUT/'control_progress.json',dict(completed_clips=i,expected_clips=199,at=now()))
    write(OUT/'control_receipt.json',dict(complete=True,clips=199,seconds=time.monotonic()-start,
        official_evaluator='pinned annotation_selection.metric_adapter.evaluate_graph',
        solver='unchanged v3 association decoder; no new division decisions in P0',
        scratch=str(SCRATCH),gpu_training_seconds=0))
