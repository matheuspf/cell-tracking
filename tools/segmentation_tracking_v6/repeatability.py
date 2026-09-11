"""Repeat deterministic point fits and all 199 graph decisions, without new tuning."""
import time
import numpy as np
from .common import *
from .controls import CONFIG, evidence, supported_labels, fit_residual, predict_scores
from strong_tracker_v3.features import EDGE_FEATURES
from strong_tracker_v3.association import decode

def run():
    start=time.monotonic();models=[]
    for source in ['44b6','6bba']:
        saved=read_json(OUT/'models'/f'P0_{source}.json');columns=saved['columns']
        xs=[];ys=[];offsets=[]
        for row in inventory():
            if row['embryo']!=source: continue
            name=row['dataset'];g,pairs,x=evidence(name)
            matches=load_graph(V5/'evaluation_matches/C0'/f'{name}.npz')['matches']
            gt=load_graph(V1/'evaluation/gt'/f'{name}.npz')
            y=supported_labels(pairs,g['nodes'],matches,gt['edges']);known=y>=0
            xs.append(x[known][:,columns]);ys.append(y[known]);offsets.append(x[known,23])
        repeated=fit_residual(np.concatenate(xs),np.concatenate(ys).astype(float),np.concatenate(offsets),
                              [EDGE_FEATURES[i] for i in columns])
        checks={k:bool(np.array_equal(repeated[k],saved['model'][k])) for k in ['mean','scale','beta']}
        if not all(checks.values()): raise ValueError('Deterministic source fit differs')
        models.append(dict(source=source,exact=checks))
    hashes=[]
    for i,row in enumerate(inventory(),1):
        name=row['dataset'];g,pairs,x=evidence(name);source='6bba' if row['embryo']=='44b6' else '44b6'
        e,_=decode(g['nodes'],g['edges'],dict(pairs=pairs,edge_features=x),predict_scores(x,source),
                   margin=CONFIG['margin'],max_fraction=CONFIG['max_changed_edge_fraction'])
        expected=read_json(OUT/'evaluation/P0'/f'{name}.json')['graph_hash']
        assert graph_hash(g['nodes'],e)==expected,name
        hashes.append(expected)
        if i%50==0: print('deterministic repeat',i,'/199',flush=True)
    write(OUT/'repeatability.json',dict(passed=True,source_fits=models,exact_graphs=len(hashes),
        graph_hashes_sha256=digest(hashes),seconds=time.monotonic()-start,
        seed_interpretation='L-BFGS and fixed decoder have no RNG; repeated full fits and graph decisions, no pretend second-seed training',
        extra_complete_score_runs=0))

if __name__=='__main__': run()
