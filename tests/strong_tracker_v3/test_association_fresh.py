import numpy as np
import pytest

from strong_tracker_v3.association_fresh import native_v2,build_from_inputs
from strong_tracker_v3.common import load_graph
from strong_tracker_v3.context import RunContext


def test_native_v2_exact_sealed_cache_parity():
    c=RunContext.default();name='44b6_0113de3b'
    reference=c.v2/'native'/f'{name}.npz'
    if not reference.exists():pytest.skip('Optional sealed local cache fixture unavailable')
    b=load_graph(c.v1/'baseline/public'/f'{name}.npz');raw=load_graph(c.v2/'raw'/f'{name}.npz')
    pre=load_graph(c.full/'inputs'/f'pre_ilp_{name}.npz')
    actual=native_v2(b['nodes'],b['edges'],b['features'],raw['nodes'],pre,[1.625,.40625,.40625])
    expected=load_graph(reference)
    for key in ['pairs','edge_features','node_extra','native_index']:
        assert np.array_equal(actual[key],expected[key]),key


def test_explicit_fresh_evidence_feature_parity():
    c=RunContext.default();name='44b6_0113de3b'
    reference=c.out/'features'/f'{name}.npz'
    if not reference.exists():pytest.skip('Optional executed v3 local cache fixture unavailable')
    s=next(s for s in c.samples() if s['dataset']==name);b=load_graph(c.incumbent(name));expected=load_graph(reference)
    raw=load_graph(c.v2/'raw'/f'{name}.npz');pre=load_graph(c.full/'inputs'/f'pre_ilp_{name}.npz')
    old=load_graph(c.v1/'baseline/public'/f'{name}.npz');p=load_graph(c.v2/'predictions'/f'{name}.npz')
    actual,_=build_from_inputs(c,s,b['nodes'],b['edges'],raw,pre,old,p['E_native_m1.5__edges'],p['E_hgb_m1.5__edges'],expected['node_features'])
    for key in expected:assert np.array_equal(actual[key],expected[key]),key
