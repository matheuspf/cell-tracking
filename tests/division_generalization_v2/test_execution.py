from collections import Counter
import copy
import numpy as np
import torch

from division_generalization_v2.dataset import SourceDataset
from division_generalization_v2.mining import supported_false,conflict_representatives
from division_generalization_v2.calibration import fit
from division_generalization_v2.contracts import lr_factor


def sample_dataset():
    d=object.__new__(SourceDataset)
    d.source='44b6';d.positive_keys=['p0','p1','p2']
    d.ordinary_keys=[f'n{i}' for i in range(57)]
    d.positive_groups={p:[p+':0',p+':1'] for p in d.positive_keys}
    d.ordinary={n:[n+':0'] for n in d.ordinary_keys}
    d.random_groups={**d.positive_groups,**d.ordinary}
    d.random_keys=sorted(d.random_groups)
    d.anchors={k:dict(group=g) for g,ks in d.random_groups.items() for k in ks}
    d.visits=Counter();d.anchor_visits=Counter();d.mined=['n0:0','n1:0']
    return d


def test_every_group_visited_and_matched_random_positive_streams():
    uniform=sample_dataset();mined=sample_dataset()
    for step in range(8):
        a=uniform.samples(step,20260916,'J_uniform')
        b=mined.samples(step,20260916,'J_mined')
        assert a[:24]==b[:24]
    assert set(uniform.visits)==set(uniform.random_groups)
    assert set(mined.visits)==set(mined.random_groups)
    assert all(x['risk_design_weight']>0 for x in a[8:24])


def test_sampling_resume_does_not_change_schedule_or_data_order():
    d=sample_dataset()
    before=[d.samples(i,314159,'J_uniform') for i in range(5)]
    checkpoint=copy.deepcopy(d)
    expected=[d.samples(i,314159,'J_uniform') for i in range(5,9)]
    actual=[checkpoint.samples(i,314159,'J_uniform') for i in range(5,9)]
    assert actual==expected
    assert d.visits==checkpoint.visits
    assert [lr_factor(i) for i in range(5,9)]==[6/128,7/128,8/128,9/128]


def test_mining_never_calls_unknown_a_negative():
    batch=dict(supported=torch.tensor([False,True,True,True]),
        utility=torch.tensor([-1.,-1.,0.,1.]),keep=torch.tensor([False,False,False,False]))
    assert supported_false(batch).tolist()==[False,True,True,False]


def test_miner_caps_transitive_conflicts_and_namespaces_clip_nodes():
    def row(key,gain,resources,dataset='clip'):
        return dict(key=key,gain=gain,resources=[('node',n) for n in resources],dataset=dataset)
    ranked=dict(a=[row('a',1.,[1])],b=[row('b',3.,[1,2])],c=[row('c',2.,[2])],
                d=[row('d',4.,[1],dataset='other')])
    chosen,n=conflict_representatives(ranked)
    assert n==4 and [r['key'] for r in chosen]==['d','b']


def test_low_support_calibration_is_explicit_identity_not_resubstitution():
    result=fit([dict(group='one',gain=4.,target=1,weight=16.),
                dict(group='two',gain=-2.,target=0,weight=16.)])
    assert result['status']=='calibration_unestablished'
    assert result['temperature']==1 and result['intercept']==0


def test_calibration_finite_with_separation_and_unpenalized_intercept():
    rows=[dict(group=f'g{i}',gain=float(i-6),target=int(i>=6),weight=16.) for i in range(12)]
    r=fit(rows)
    assert r['status']=='held_source_fitted'
    assert not r['intercept_penalized']
    assert np.isfinite([r['temperature'],r['intercept']]).all()


def test_small_head_oof_keeps_overlap_groups_intact_and_resets_pools():
    from division_generalization_v2.head_oof import fold_of,subset
    d=sample_dataset()
    for key,a in d.anchors.items():
        a.update(random_included=True,positive=a['group'].startswith('p'))
    chosen=['p0:0','p0:1','n0:0']
    a=subset(d,chosen)
    assert set(a.anchors)==set(chosen)
    assert a.positive_keys==['p0'] and a.ordinary_keys==['n0']
    assert not a.visits and not a.anchor_visits
    assert fold_of('44b6','shared_clip_group')==fold_of('44b6','shared_clip_group')
    assert {fold_of('44b6',str(i)) for i in range(50)}=={0,1,2}


def test_recommendation_requires_matched_target_controls_and_current_fresh_model():
    from pathlib import Path
    from division_generalization_v2.common import sha
    from division_generalization_v2 import model
    from division_generalization_v2.report import completion_gates
    model_hash=sha(Path(model.__file__))
    target=dict(pooled=[dict(arm=a,seed=s,clips=199) for a in ('J_uniform','J_mined')
                       for s in (20260916,314159)])
    fresh=dict(status='measured',cold_pipeline_artifacts=True,
               clips=[dict(arm='J_mined',exact_P0=True,exact_scored_candidate=True)]*2)
    gates=dict(literal_zero_path_test=True,unit_tests_passed=True)
    fits=[dict(arm='J_mined',status='complete',joint_optimizer_updates=4096,full_source_screen=True)]*10
    freeze=dict(qualified_exports=['J_uniform','J_mined'])
    image=dict(model_code_sha256=model_hash,clips=[dict(full_clip=True,exact_active_zero=True)]*2)
    matrix=dict(model_code_sha256=model_hash,image_models_exact=True)
    def check():return completion_gates(True,'J_mined',target,fresh,gates,fits,freeze,image,matrix)
    assert all(check().values())
    target['pooled']=[r for r in target['pooled'] if r['arm']=='J_mined']
    assert not check()['matched_image_controls_complete']
    fresh['clips'][0]['arm']='P0'
    assert not check()['fresh_nominated_module']
    image['model_code_sha256']='obsolete'
    assert not check()['current_image_zero']


def test_extension_matrix_is_matched_and_never_substitutes_another_seed():
    from division_generalization_v2.screen import matrix_members
    for seed in (20260916,314159):
        rows=matrix_members(seed)
        assert {a for a,_,_ in rows}=={'J_uniform','J_mined'}
        assert {s for _,s,_ in rows}=={seed}
        assert {step for _,_,step in rows}=={8192}
    assert len(matrix_members())==9


def test_queue_stops_after_a_cooperative_incomplete_training_boundary(tmp_path,monkeypatch):
    import pytest
    from division_generalization_v2 import queue
    from division_generalization_v2.common import write_json
    monkeypatch.setattr(queue,'WORK',tmp_path)
    folder=tmp_path/'training/prefix/44b6/20260916'
    def interrupted(*args):
        r=dict(status='incomplete_resumable',joint_optimizer_updates=128)
        write_json(folder/'progress.json',r);write_json(folder/'training_receipt.json',r)
    monkeypatch.setattr(queue,'call',interrupted)
    with pytest.raises(RuntimeError,match='below required boundary'):
        queue.train('44b6','prefix',20260916)
    assert len(list((folder/'invocations').glob('*.json')))==1


def test_existing_fresh_geff_is_reverified_against_complete_graph(tmp_path):
    import pytest
    from pipeline_error_training.serialization import export_geff
    from division_generalization_v2.fresh import verify_existing_geff
    nodes=np.array([[31,0,2,3,4],[17,1,3,4,5]],dtype=np.int64)
    edges=np.array([[31,17]],dtype=np.int64)
    path=tmp_path/'candidate.geff'
    export_geff(path,nodes,edges)
    receipt=verify_existing_geff(path,nodes,edges)
    assert receipt['exact_roundtrip'] and receipt['existing_export_reverified']
    changed=nodes.copy();changed[1,2]+=1
    with pytest.raises(ValueError,match='differs from the complete scored graph'):
        verify_existing_geff(path,changed,edges)
