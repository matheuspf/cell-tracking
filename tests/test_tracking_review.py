"""Scoring boundaries and cross-frame evidence for the full tracking review."""
import json
import sqlite3

import numpy as np
import pytest

from center_comparison.tracking_index import SCHEMA, edge_flags, scenarios, aggregate_counts
from center_comparison.tracking_server import list_cases


def test_edge_partition_distinguishes_missing_observations_and_wrong_identity():
    # GT 10→20→30→40. Center 20 is lost after selection. 50 competes with
    # the correctly matched prediction for 30; an entirely unknown edge is ignored.
    gt = np.array([[10,0,0,0,0],[20,1,0,0,0],[30,2,0,0,0],[40,3,0,0,0]])
    nodes = np.array([[1,0,0,0,0],[3,2,0,0,0],[4,3,0,0,0],[50,2,0,0,1],[90,8,0,0,0],[91,9,0,0,0]])
    common = dict(gt=gt,gt_edges=np.array([[10,20],[20,30],[30,40]]))
    final = dict(nodes=nodes,matches=np.array([[1,10,0],[3,30,0],[4,40,0]]),
                 edge_status=np.array([[50,4,-1],[90,91,0]]))
    centers = {g:dict(kind='matched',raw_count=1,final_count=1) for g in [10,20,30,40]}
    centers[20] = dict(kind='proposal_only',raw_count=1,final_count=0)
    flags = edge_flags(common,final,centers,np.ones(3))
    assert [f['category'] for f in flags].count('edge_endpoint') == 2
    assert [f['category'] for f in flags].count('edge_link') == 1
    assert sum(f['category']=='edge_fp' for f in flags) == 1
    assert all(f['stage']=='selection' for f in flags if f['category']=='edge_endpoint')
    wrong = flags[-1]
    assert wrong['stage']=='competing'
    assert wrong['evidence']['competing']==[dict(prediction=50,expected_gt=30,assigned_prediction=3,distance_um=1.)]
    # The same coordinates from a different timepoint cannot establish competition.
    nodes[3,1] = 1
    assert edge_flags(common,final,centers,np.ones(3))[-1]['stage']=='unmatched'


def test_no_proposal_and_assignment_conflict_have_different_attribution():
    common=dict(gt=np.array([[11,0,0,0,0],[22,1,0,0,0]]),gt_edges=np.array([[11,22]]))
    final=dict(nodes=np.array([[7,0,0,0,0]]),matches=np.array([[7,11,0]]),edge_status=np.empty((0,3),int))
    centers={11:dict(kind='matched',raw_count=1,final_count=1),22:dict(kind='conflict',raw_count=2,final_count=1)}
    assert edge_flags(common,final,centers,np.ones(3))[0]['stage']=='assignment'
    centers[22].update(kind='missing',raw_count=0,final_count=0)
    assert edge_flags(common,final,centers,np.ones(3))[0]['stage']=='proposal_gap'


def test_pagination_filters_and_deep_links_reach_the_last_flag():
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row;db.executescript(SCHEMA)
    for i in range(103):
        db.execute('INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?,?)',(f'flag{i:03}', 'pooled','sample','44b6','division_fn','division_no_fork',i,i,None,'{}'))
    first=list_cases(db,dict(filter='division_fn',embryo='44b6'))
    assert first['total']==103 and len(first['rows'])==40
    last=list_cases(db,dict(key='flag102'))
    assert last['offset']==80 and last['rows'][-1]['key']=='flag102'
    assert list_cases(db,dict(offset='99999'))==last
    assert list_cases(db,dict(embryo='6bba'))['rows']==[]
    assert list_cases(db,dict(stage='division_no_fork',t='50'))['rows'][0]['key']=='flag050'
    with pytest.raises(ValueError):list_cases(db,dict(filter="' OR 1=1"))


def test_accounting_scenarios_reweight_clips_and_do_not_modify_receipts():
    pytest.importorskip('tracksdata')
    rows=[dict(dataset='a',edge_tp=90,edge_fn=10,edge_fp=20,division_tp=1,division_fp=1,division_fn=2,num_pred_nodes=100,estimated_total=200,
               categories=dict(edge_endpoint=6,edge_link=4)),
          dict(dataset='b',edge_tp=9,edge_fn=1,edge_fp=0,division_tp=0,division_fp=0,division_fn=0,num_pred_nodes=200,estimated_total=100,
               categories=dict(edge_endpoint=0,edge_link=1))]
    frozen=json.dumps(rows,sort_keys=True);base=aggregate_counts(rows)
    results={x['key']:x for x in scenarios(rows,base)}
    # Remove FP: weight becomes 100+10, retaining distinct 1.05 and 0.9 multipliers.
    expected=(90*1.05+9*.9)/110+.1*(1/4)
    assert results['edge_fp']['score']==pytest.approx(expected)
    assert results['division_all']['delta']==pytest.approx(.075)
    assert results['division_all']['delta'] != pytest.approx(results['division_fn']['delta']+results['division_fp']['delta'])
    assert json.dumps(rows,sort_keys=True)==frozen


def test_division_classification_uses_official_temporal_window():
    pytest.importorskip('tracksdata')
    from center_comparison.tracking_index import division_flags
    from annotation_selection.metric_adapter import evaluate_graph
    # All observations exist, but a single branch is selected at the split.
    gt=np.array([[101,0,0,0,0],[102,1,0,0,0],[103,2,0,0,0],[104,2,0,0,10],
                 [105,3,0,0,0],[106,3,0,0,10]])
    ge=np.array([[101,102],[102,103],[102,104],[103,105],[104,106]])
    nodes=gt.copy();nodes[:,0]=np.arange(1,7)
    edges=np.array([[1,2],[2,3],[3,5],[4,6]])
    row,_,_=evaluate_graph('example',nodes,edges,gt,ge,np.ones(3),6)
    flags=division_flags(dict(gt=gt,gt_edges=ge),dict(nodes=nodes,edges=edges),np.ones(3),row)
    assert row['division_fn']==1 and len(flags)==1
    assert flags[0]['stage']=='division_no_fork'
    assert flags[0]['node']==102
    assert set(g for _,g in flags[0]['evidence']['local_matches'])==set(gt[:,0])
    # Adding the second child recovers the division; no fabricated FN remains.
    edges=np.vstack([edges,[2,4]])
    row,_,_=evaluate_graph('example',nodes,edges,gt,ge,np.ones(3),6)
    assert row['division_tp']==1
    assert division_flags(dict(gt=gt,gt_edges=ge),dict(nodes=nodes,edges=edges),np.ones(3),row)==[]
