import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('tracksdata',reason='Optional study runtime; see docs/annotation-selection-v1.md')
pytest.importorskip('pandas')

from annotation_selection.common import REPO,OFFICIAL,graph_hash,write_json
if not (OFFICIAL/'src/tracking_cellmot/metrics.py').exists():
    pytest.skip('Pinned metric checkout required for study integration tests',allow_module_level=True)
from annotation_selection.features import assert_features
from annotation_selection.filter_graph import filtered,keep_mask
from annotation_selection.inventory import validate_graph
from annotation_selection.metric_adapter import aggregate,evaluate_graph,match_nodes


def fixture():
    nodes=np.array([[0,0,10,20,20],[1,1,10,20,20],[2,2,10,18,20],[3,2,10,22,20],[4,3,10,16,20],[5,3,10,24,20]])
    return nodes,np.array([[0,1],[1,2],[1,3],[2,4],[3,5]])


def test_golden_fork_and_identity():
    n,e=fixture();h=graph_hash(n,e)
    r,m,tp=evaluate_graph('x',n,e,n,e,(1.625,.40625,.40625),100.)
    assert (r['edge_tp'],r['edge_fp'],r['edge_fn'])==(5,0,0)
    assert (r['division_tp'],r['division_fp'],r['division_fn'])==(1,0,0)
    assert len(m)==6 and len(tp)==5 and graph_hash(n,e)==h
    for policy in ['identity','random_nodes','random_tracklets','membership_tracklets_fork_protected']:
        nn,ee=filtered(n,e,keep_mask(n,e,np.ones(len(n)),1,policy))
        assert graph_hash(nn,ee)==h


def test_empty_and_singleton_matching():
    n,e=fixture()
    r,_,_=evaluate_graph('x',n[:0],e[:0],n,e,(1.625,.40625,.40625),100.)
    assert (r['edge_tp'],r['edge_fp'],r['edge_fn'])==(0,0,5)
    assert r['division_fn']==1
    matches=match_nodes(n[:1],e[:0],n,e,(1.625,.40625,.40625))
    assert matches=={0:0}


def test_anisotropy_time_and_duplicate_competition():
    n,e=fixture();p=np.repeat(n[:1],2,axis=0);p[:,0]=[10,11]
    assert len(match_nodes(p,e[:0],n,e,(1.625,.40625,.40625)))==1
    p=p[:1].copy();p[0,2]+=5
    assert not match_nodes(p,e[:0],n[:1],e[:0],(1.625,.40625,.40625))
    p=n[:1].copy();p[0,4]+=5
    assert len(match_nodes(p,e[:0],n[:1],e[:0],(1.625,.40625,.40625)))==1
    p[0,1]=1
    assert not match_nodes(p,e[:0],n[:1],e[:0],(1.625,.40625,.40625))


def test_filters_and_no_leakage():
    n,e=fixture()
    for policy in ['random_nodes','random_tracklets','membership_tracklets','membership_tracklets_fork_protected']:
        keep=keep_mask(n,e,np.arange(len(n)),.3,policy)
        nn,ee=filtered(n,e,keep)
        validate_graph(nn,ee,(4,64,256,256),prediction=True)
        assert np.array_equal(keep,keep_mask(n,e,np.arange(len(n)),.3,policy))
    assert not keep_mask(n,e,np.arange(len(n)),0,'random_nodes').any()
    with pytest.raises(ValueError):assert_features(['nearest_gt_um'])
    with pytest.raises(ValueError):validate_graph(n,np.vstack([e,e[:1]]),(4,64,256,256),prediction=True)
    with pytest.raises(ValueError):validate_graph(n,np.array([[0,4]]),(4,64,256,256),prediction=True)


def test_aggregation_parity_and_strictness():
    rng=np.random.default_rng(20260908)
    for _ in range(100):
        rows=[]
        for name in ['a','b','c']:
            counts=rng.integers(0,100,size=7)
            rows.append(dict(dataset=name,estimated_total=100.,**dict(zip(['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes'],counts))))
        assert np.isfinite(aggregate(rows,['a','b','c'])['score'])
    with pytest.raises(ValueError):aggregate(rows,['a','b'])
    for value in [0,float('nan'),float('inf'),-1,True]:
        rows[0]['estimated_total']=value
        with pytest.raises(ValueError):aggregate(rows,['a','b','c'])


def test_locked_json(tmp_path):
    p=tmp_path/'lock.json';write_json(p,{'a':1},immutable=True)
    write_json(p,{'a':1},immutable=True)
    with pytest.raises(RuntimeError):write_json(p,{'a':2},immutable=True)
