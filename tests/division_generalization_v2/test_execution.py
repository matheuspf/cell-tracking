from collections import Counter
import copy
import numpy as np
import torch

from division_generalization_v2.dataset import SourceDataset
from division_generalization_v2.mining import supported_false
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
