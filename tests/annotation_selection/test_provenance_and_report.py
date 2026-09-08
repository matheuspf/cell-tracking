import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('tracksdata',reason='Optional study runtime; see docs/annotation-selection-v1.md')
pytest.importorskip('sklearn')

from annotation_selection.common import REPO,OUT,OFFICIAL,read_json,write_json,sha
if not (OFFICIAL/'src/tracking_cellmot/metrics.py').exists():
    pytest.skip('Pinned metric checkout required for study integration tests',allow_module_level=True)
from annotation_selection.filter_graph import PreparedFilter,keep_mask
from annotation_selection.metric_adapter import aggregate
from annotation_selection.report import score_formula
from annotation_selection.train import check_source_direction


def test_inference_file_guard_in_separate_process(tmp_path):
    env={**os.environ,'PYTHONPATH':str(REPO/'tools'),'PYTHONNOUSERSITE':'1'}
    code='from annotation_selection.inference import deny_annotations; deny_annotations(); open("/tmp/forbidden.geff/zarr.json")'
    p=subprocess.run([sys.executable,'-c',code],env=env,cwd=tmp_path,text=True,capture_output=True)
    assert p.returncode!=0 and 'Annotation access forbidden' in p.stderr
    p=subprocess.run([sys.executable,'-m','annotation_selection','--help'],env=env,cwd=tmp_path,text=True,capture_output=True)
    assert p.returncode==0 and 'Resumable local' in p.stdout


def test_source_manifest_rejects_target_labels():
    d=dict(source='a',outer='b',source_samples=['a_1'],outer_samples=['b_1'])
    check_source_direction(d)
    d['source_samples'].append('b_2')
    with pytest.raises(ValueError):check_source_direction(d)


def test_vector_score_and_shapley_aggregator_parity():
    rng=np.random.default_rng(7)
    for _ in range(20):
        rows=[]
        for name in ['a','b','c']:
            values=rng.integers(0,1000,7)
            rows.append(dict(dataset=name,estimated_total=float(rng.integers(1,300)),**dict(zip(['edge_tp','edge_fp','edge_fn','division_tp','division_fp','division_fn','num_pred_nodes'],values))))
        df=pd.DataFrame(rows)
        assert np.isclose(score_formula(df)['score'],aggregate(rows,['a','b','c'])['score'],atol=1e-12)


def test_prepared_policy_parity():
    n=np.array([[i,i%5,0,0,0] for i in range(30)])
    e=np.array([[i,i+1] for i in range(30) if i%5!=4]);scores=np.random.default_rng(1).random(30)
    p=PreparedFilter(n,e)
    for policy in ['random_nodes','random_tracklets','membership_nodes','membership_tracklets','membership_tracklets_fork_protected']:
        for r in [.1,.3,.5,.7,.9,1]:
            assert np.array_equal(keep_mask(n,e,scores,r,policy),p.mask(scores,r,policy,key='a'))


def test_exact_cost_control_preserves_whole_units():
    n=np.array([[i,i,0,0,0] for i in range(13)])
    e=np.array([[0,1],[1,2],[3,4],[4,5],[5,6],[7,8],[8,9],[9,10],[10,11],[11,12]])
    p=PreparedFilter(n,e);scores=np.arange(13,dtype=float)
    # 3 + 4 is reachable; greedy highest-quality 6 cannot complete that budget.
    keep=p.exact_tracklet_cost(scores,7)
    assert keep.sum()==7
    assert all(keep[g].all() or not keep[g].any() for g in p.groups)
    with pytest.raises(ValueError):p.exact_tracklet_cost(scores,2)


def test_actual_models_source_disjoint_if_present():
    manifest_path=OUT/'fold_manifest.json'
    if not manifest_path.exists():pytest.skip('Run-local integrity test requires study artifacts')
    for d in read_json(manifest_path)['directions']:
        check_source_direction(d)
        p=OUT/'models'/d['source']/'tabular_complete.json'
        if not p.exists():pytest.skip('Training has not completed')
        m=read_json(p)['manifest']
        assert set(m['label_hashes'])==set(d['source_samples'])
        assert not set(m['label_hashes'])&set(d['outer_samples'])
        assert m['upstream_supervision']=='none' and m['provenance_lane']=='clean'


def test_public_prediction_lock_rejects_missing_or_changed_files(tmp_path,monkeypatch):
    from annotation_selection import public_analysis
    monkeypatch.setattr(public_analysis,'OUT',tmp_path)
    manifest=dict(expected_samples=['a_1','b_1'],directions=[dict(source='a',outer_samples=['b_1']),dict(source='b',outer_samples=['a_1'])])
    write_json(tmp_path/'fold_manifest.json',manifest)
    predictions={}
    for source,name in [('a','b_1'),('b','a_1')]:
        p=tmp_path/'public_predictions'/source/(name+'.npz');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixed predictions')
        predictions[str(p.relative_to(tmp_path))]=sha(p)
    graphs={}
    for name in manifest['expected_samples']:
        p=tmp_path/'baseline/public'/(name+'.npz');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixed graph')
        graphs[name]=sha(p)
    for name in ['public_selector_preregistration.json','model_lock.json','public_harmonic_full/submission.csv']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixed artifact')
    lock=dict(predictions=predictions,graph_hashes=graphs,config_hash=sha(tmp_path/'public_selector_preregistration.json'),
              source_model_lock_sha256=sha(tmp_path/'model_lock.json'),notebook_graph_csv_sha256=sha(tmp_path/'public_harmonic_full/submission.csv'))
    write_json(tmp_path/'public_prediction_lock.json',lock)
    assert public_analysis.verify_prediction_lock()==lock
    changed=tmp_path/'public_predictions/a/b_1.npz';changed.write_bytes(b'changed predictions')
    with pytest.raises(ValueError,match='prediction changed'):public_analysis.verify_prediction_lock()
    changed.write_bytes(b'fixed predictions');del lock['graph_hashes']['a_1'];write_json(tmp_path/'public_prediction_lock.json',lock)
    with pytest.raises(ValueError,match='Incomplete public graph'):public_analysis.verify_prediction_lock()


def test_fork_protection_reports_context_overshoot():
    from annotation_selection.filter_graph import filtered
    n=np.array([[0,0,0,0,0],[1,1,0,0,0],[2,2,0,0,0],[3,2,0,0,0],[4,3,0,0,0],[5,3,0,0,0]])
    e=np.array([[0,1],[1,2],[1,3],[2,4],[3,5]])
    prepared=PreparedFilter(n,e)
    plain=prepared.mask(np.arange(6),.3,'membership_tracklets',key='one')
    protected=prepared.mask(np.arange(6),.3,'membership_tracklets_fork_protected',key='one')
    assert plain.sum()==2 and protected.sum()==6
    nn,ee=filtered(n,e,protected)
    assert np.array_equal(nn,n) and np.array_equal(ee,e)


def test_public_boundary_sampling_preserves_graph_and_strict_default():
    from scipy.ndimage import map_coordinates
    from annotation_selection.public_analysis import reflected_coordinates
    from annotation_selection.features import frame_features
    from annotation_selection.inventory import validate_graph
    image=np.arange(4*8*8).reshape(4,8,8)
    xyz=np.array([[4,2,3],[0,7,7],[-1,0,0]])
    sample=reflected_coordinates(xyz,image.shape)
    assert np.array_equal(image[tuple(sample.T)],map_coordinates(image,xyz.T,order=0,mode='reflect'))
    f=frame_features(0,xyz,np.ones(3),image,image[:,::2,::2],(2,4,8,8),np.ones(3),1,sampling_xyz=sample)
    assert f[0,1]>1 and f[0,4]<0  # Geometry retains the original outside position.
    nodes=np.column_stack([np.arange(3),np.zeros(3,int),xyz]);original=nodes.copy()
    edges=np.empty((0,2),int)
    validate_graph(nodes,edges,(2,4,8,8),prediction=True,spatial_bounds=False)
    assert np.array_equal(nodes,original)
    with pytest.raises(ValueError,match='Coordinates out of bounds'):
        validate_graph(nodes,edges,(2,4,8,8),prediction=True)
    nodes[0,1]=2
    with pytest.raises(ValueError,match='Time coordinates out of bounds'):
        validate_graph(nodes,edges,(2,4,8,8),prediction=True,spatial_bounds=False)
    inside=sample
    a=frame_features(0,inside,np.ones(3),image,image[:,::2,::2],(2,4,8,8),np.ones(3),1)
    b=frame_features(0,inside,np.ones(3),image,image[:,::2,::2],(2,4,8,8),np.ones(3),1,sampling_xyz=inside)
    assert np.array_equal(a,b)
