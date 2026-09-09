import numpy as np
from strong_tracker_v3.common import validate
from strong_tracker_v3 import rescue_fixed as rescue

class FakeFrames:
    image=None
    def __init__(self,ctx,sample):pass
    def get(self,t):return self.image[t]
    patch=rescue.Frames.patch

def test_secondary_peak_needs_three_frames_and_preserves_boundary(monkeypatch):
    nodes=np.array([[0,0,3,10,10],[16,1,3,10,10]],np.int64);edges=np.array([[0,16]],np.int64)
    sample=dict(image_shape=[7,8,32,32],physical_scale=[1.625,.40625,.40625])
    FakeFrames.image=np.ones(sample['image_shape'],np.uint16)
    monkeypatch.setattr(rescue,'Frames',FakeFrames)
    monkeypatch.setitem(rescue.CONFIG,'point_cap_fraction',3.)
    FakeFrames.image[2,3,10,12]=1000
    n,e,ledger,_=rescue.maxima_rescue(None,sample,nodes,edges)
    assert np.array_equal(n,nodes) and np.array_equal(e,edges) and not ledger
    FakeFrames.image[3,3,10,12]=1000;FakeFrames.image[4,3,10,12]=1000
    n,e,ledger,_=rescue.maxima_rescue(None,sample,nodes,edges)
    assert len(n)==5 and len(e)==4 and len(ledger)==1
    assert np.array_equal(n[:2],nodes) and np.array_equal(e[:1],edges)
    validate(n,e,sample['image_shape'])

def test_existing_center_cannot_be_artificially_duplicated(monkeypatch):
    n=np.array([[0,0,3,10,10],[16,1,3,10,10],[20,2,3,10,12]],np.int64);e=np.array([[0,16]],np.int64)
    sample=dict(image_shape=[7,8,32,32],physical_scale=[1.625,.40625,.40625])
    FakeFrames.image=np.ones(sample['image_shape'],np.uint16)
    for t in [2,3,4]:FakeFrames.image[t,3,10,12]=1000
    monkeypatch.setattr(rescue,'Frames',FakeFrames);monkeypatch.setitem(rescue.CONFIG,'point_cap_fraction',3.)
    nn,ee,ledger,_=rescue.maxima_rescue(None,sample,n,e)
    assert np.array_equal(n,nn) and np.array_equal(e,ee) and not ledger


def test_flat_background_has_no_persistent_object(monkeypatch):
    n=np.array([[0,0,3,10,10],[16,1,3,10,10]],np.int64);e=np.array([[0,16]],np.int64)
    sample=dict(image_shape=[7,8,32,32],physical_scale=[1.625,.40625,.40625])
    FakeFrames.image=np.full(sample['image_shape'],1000,np.uint16)
    monkeypatch.setattr(rescue,'Frames',FakeFrames);monkeypatch.setitem(rescue.CONFIG,'point_cap_fraction',3.)
    nn,ee,ledger,stats=rescue.maxima_rescue(None,sample,n,e)
    assert np.array_equal(n,nn) and np.array_equal(e,ee) and not ledger
    assert stats['foreground_rejections']['flat_patch']>0


def test_background_read_noise_is_not_a_peak():
    patch=np.ones((5,17,17));patch[2,8,8]=2
    peaks,quality=rescue.foreground_peaks(patch)
    assert not len(peaks) and quality['reason']=='insufficient_contrast'


def test_last_three_available_frames_are_retained(monkeypatch):
    n=np.array([[0,2,3,10,10],[16,3,3,10,10]],np.int64);e=np.array([[0,16]],np.int64)
    sample=dict(image_shape=[7,8,32,32],physical_scale=[1.625,.40625,.40625])
    FakeFrames.image=np.ones(sample['image_shape'],np.uint16)
    for t in [4,5,6]:FakeFrames.image[t,3,10,12]=1000
    monkeypatch.setattr(rescue,'Frames',FakeFrames);monkeypatch.setitem(rescue.CONFIG,'point_cap_fraction',3.)
    nn,ee,ledger,_=rescue.maxima_rescue(None,sample,n,e)
    assert len(nn)==5 and nn[-1,1]==6 and len(ledger)==1
    assert nn[-1,0]>=rescue.CONFIG['derived_id_start']


def test_explicit_fresh_rescue_matches_cached_evidence():
    import pytest
    from strong_tracker_v3.context import RunContext
    from strong_tracker_v3.common import load_graph
    c=RunContext.default();name='44b6_0113de3b';path=c.out/'candidate_graphs/R_image_persistent'/f'{name}.npz'
    if not path.exists():pytest.skip('Optional executed local rescue fixture unavailable')
    sample=next(s for s in c.samples() if s['dataset']==name);base=load_graph(c.incumbent(name))
    raw=load_graph(c.v2/'raw'/f'{name}.npz');pre=load_graph(c.full/'inputs'/f'pre_ilp_{name}.npz')
    n,e,_,_=rescue.apply(c,sample,base['nodes'],base['edges'],raw=raw,pre_ilp=pre)
    expected=load_graph(path)
    assert np.array_equal(n,expected['nodes']) and np.array_equal(e,expected['edges'])
