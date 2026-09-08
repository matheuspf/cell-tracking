"""Strict adapter to the pinned, unmodified official metric implementation."""
from __future__ import annotations

import sys
import warnings

import numpy as np
import polars as pl
import tracksdata as td
from tracksdata.metrics import DistanceMatching
from tracksdata.options import set_options

from .common import OFFICIAL,METRIC_REV,REPO,graph_hash

sys.path.insert(0,str(OFFICIAL/'src'))
from tracking_cellmot import metrics as official
sys.path.insert(0,str(REPO/'handover/annotation-selection-v1'))
from analysis import strict_summary

set_options(show_progress=False)


def make_graph(nodes,edges):
    # Mapping is explicit: rustworkx IDs are not the persisted candidate/GT IDs.
    g=td.graph.IndexedRXGraph()
    for a in 'zyx':
        g.add_node_attr_key(a,pl.Float64,-999999.)
    attrs=[dict(t=int(t),z=float(z),y=float(y),x=float(x)) for _,t,z,y,x in nodes]
    ids=g.bulk_add_nodes(attrs) if attrs else []
    mapping={int(row[0]):int(new) for row,new in zip(nodes,ids)}
    if len(edges):
        g.bulk_add_edges([dict(source_id=mapping[int(a)],target_id=mapping[int(b)]) for a,b in edges])
    return g,{v:k for k,v in mapping.items()}


def match_nodes(nodes,edges,gt_nodes,gt_edges,scale):
    g,reverse=make_graph(nodes,edges);gt,gr=make_graph(gt_nodes,gt_edges)
    if len(nodes) and len(gt_nodes):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            g.match(gt,matching=DistanceMatching(max_distance=7.,scale=tuple(scale),optimal=True))
        attrs=g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID,td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
        matches={reverse[int(i)]:gr[int(j)] for i,j in attrs.iter_rows() if j is not None and j!=-1}
    else:
        matches={}
    if len(matches)!=len(set(matches.values())) or len(matches)>min(len(nodes),len(gt_nodes)):
        raise ValueError('Matching is not one-to-one')
    return matches


def evaluate_graph(dataset,nodes,edges,gt_nodes,gt_edges,scale,estimate):
    if isinstance(estimate,bool) or not np.isfinite(estimate) or estimate<=0:
        raise ValueError('Missing/invalid total-node estimate')
    before=graph_hash(nodes,edges)
    g,reverse=make_graph(nodes,edges);gt,gr=make_graph(gt_nodes,gt_edges)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        er=official.evaluate(g,gt,scale=tuple(scale),max_distance=7.)
        # evaluate's no-edge early return does not perform node matching.
        if len(nodes) and len(gt_nodes) and not len(edges):
            g.match(gt,matching=DistanceMatching(max_distance=7.,scale=tuple(scale),optimal=True))
    if len(nodes) and len(gt_nodes):
        attrs=g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID,td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
        matches={reverse[int(i)]:gr[int(j)] for i,j in attrs.iter_rows() if j is not None and j!=-1}
    else:
        matches={}
    if len(matches)!=len(set(matches.values())) or len(matches)>min(len(nodes),len(gt_nodes)):
        raise ValueError('Non-bijective matching')
    recall=len(matches)/len(gt_nodes) if len(gt_nodes) else float('nan')
    row=official.per_sample_metrics(er,estimate,recall)
    if not all(np.isfinite(row[k]) for k in official.COUNT_COLUMNS):
        raise ValueError('Nonfinite required counts')
    if er.edge_tp+er.edge_fn!=len(gt_edges):
        raise ValueError('GT edge drift')
    if before!=graph_hash(nodes,edges):
        raise RuntimeError('Evaluation mutated persisted graph')
    gt_set={tuple(e) for e in gt_edges}
    tp_edges={tuple(map(int,e)) for e in edges if (matches.get(int(e[0]),-1),matches.get(int(e[1]),-1)) in gt_set}
    row.update(dataset=dataset,estimated_total=estimate,gt_node_recall=recall,matched_nodes=len(matches),
               graph_hash=before,metric_revision=METRIC_REV)
    return row,matches,tp_edges


def aggregate(rows,expected):
    # strict_summary checks every required sample, count, estimate and denominator.
    independent=strict_summary(rows,expected)
    derived=[official.per_sample_metrics(official.EvaluationResult(*(int(r[k]) for k in official.COUNT_COLUMNS)),
                                        float(r['estimated_total']),float(r.get('gt_node_recall',0.))) for r in rows]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result=official.summarise(derived)
    for k in ['score','edge_jaccard','adj_edge_jaccard']:
        if not np.isclose(result[k],independent[k],atol=1e-12,rtol=1e-12):
            raise ValueError(f'Aggregation drift: {k}')
    result['counts']=independent['counts']
    return result
