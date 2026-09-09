import importlib.util
from pathlib import Path
import numpy as np
import pytest
from strong_tracker_v3.context import RunContext
from strong_tracker_v3 import common

def fixture():
    n=np.array([[0,0,4,5,5],[1,1,4,5,5],[2,2,4,3,5],[3,2,4,7,5],[4,3,4,2,5],[5,3,4,8,5]])
    e=np.array([[0,1],[1,2],[1,3],[2,4],[3,5]])
    return n,e

def test_pure_graph_primitives_legacy_parity(tmp_path):
    from annotation_selection.common import graph_hash,save_graph,load_graph
    n,e=fixture();a=tmp_path/'a.npz';b=tmp_path/'b.npz'
    save_graph(a,n,e);common.save_graph(b,n,e)
    assert common.sha(a)==common.sha(b)
    assert common.graph_hash(n,e)==graph_hash(n,e)
    assert np.array_equal(common.load_graph(a)['edges'],load_graph(b)['edges'])
    common.validate(n,e,(4,10,10,10))
    assert common.graph_hash(n,e)==graph_hash(n,e)

def test_context_rejects_overlap_and_is_frozen(tmp_path):
    from dataclasses import FrozenInstanceError,replace
    ctx=RunContext.default(out=tmp_path/'out',work=tmp_path/'work')
    ctx.check_outputs()
    with pytest.raises(FrozenInstanceError):ctx.out=tmp_path
    with pytest.raises(ValueError):replace(ctx,out=ctx.v2/'new').check_outputs()
    alias=tmp_path/'alias';alias.symlink_to(ctx.v2,target_is_directory=True)
    with pytest.raises(ValueError):replace(ctx,out=alias/'new').check_outputs()

def test_metric_and_helper_unequal_weight_parity():
    from annotation_selection.metric_adapter import evaluate_graph,aggregate
    ctx=RunContext.default();spec=importlib.util.spec_from_file_location('v3_counts',ctx.repo/'handover/strong-tracker-v3/scorecard.py')
    counts=importlib.util.module_from_spec(spec);spec.loader.exec_module(counts)
    n,e=fixture();original=common.graph_hash(n,e)
    a,_,_=evaluate_graph('a',n,e,n,e,(1,1,1),10)
    b,_,_=evaluate_graph('b',n,e[:2],n,e,(1,1,1),15)
    for x in [a,b]:x.update(variant='test',embryo='x')
    measured=aggregate([a,b],['a','b']);independent=counts.summary([a,b],['a','b'])
    assert measured['score']==pytest.approx(independent['score'],abs=1e-12)
    assert original==common.graph_hash(n,e)
    for bad in [0,-1,np.nan,True]:
        with pytest.raises(ValueError):evaluate_graph('a',n,e,n,e,(1,1,1),bad)
    with pytest.raises(ValueError):counts.summary([a],['a','b'])
    with pytest.raises(ValueError):counts.summary([a,a],['a','b'])

def test_metric_empty_graph_and_no_divisions():
    from annotation_selection.metric_adapter import evaluate_graph,aggregate
    n,e=fixture();empty=np.empty((0,2),np.int64)
    row,match,_=evaluate_graph('empty',n,empty,n,e,(1,1,1),10)
    assert row['edge_tp']==0 and row['edge_fn']==len(e) and len(match)==len(n)
    row,_,_=evaluate_graph('linear',n[:2],e[:1],n[:2],e[:1],(1,1,1),10)
    assert aggregate([row],['linear'])['score']==row['adj_edge_jaccard']

def test_graph_failures(tmp_path):
    n,e=fixture()
    for ee in [np.vstack([e,e[0]]),np.vstack([e,[0,5]]),np.vstack([e,[2,5]])]:
        with pytest.raises((ValueError,KeyError)):common.validate(n,ee,(4,10,10,10))
    with pytest.raises(FileNotFoundError):common.load_graph(tmp_path/'missing.npz')
    p=tmp_path/'bad.npz';p.write_text('not a graph')
    with pytest.raises(ValueError):common.load_graph(p)
