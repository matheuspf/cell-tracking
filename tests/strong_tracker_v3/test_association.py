import numpy as np
import pytest

from strong_tracker_v3.association import decode,supported_labels
from strong_tracker_v3.disagreements import project_teacher
from strong_tracker_v3.features import temporal,EDGE_FEATURES
from strong_tracker_v3.common import validate,graph_hash


def fixture():
    n=np.array([[10,0,1,1,1],[20,0,1,4,1],[30,1,1,1,1],[40,1,1,4,1]],np.int64)
    e=np.array([[10,30],[20,40]],np.int64);pairs=np.array([[0,2],[0,3],[1,2],[1,3]])
    f=np.zeros((4,len(EDGE_FEATURES)),np.float32);f[:,0]=.9;f[:,25]=1;f[:,3]=3
    return n,e,dict(pairs=pairs,edge_features=f)


def test_owner_reassignment_is_complete_and_legal():
    n,e,c=fixture();out,ledger=decode(n,e,c,np.array([0,10,10,0]),margin=1.5,max_fraction=3)
    assert set(map(tuple,out))=={(10,40),(20,30)}
    assert ledger['changed_edges']==4 and ledger['edits'][0]['termination']==0
    validate(n,out,[2,8,8,8])


def test_noop_tie_preserves_exact_arrays():
    n,e,c=fixture();out,ledger=decode(n,e,c,np.ones(4),margin=0,max_fraction=3)
    assert np.array_equal(out,e) and graph_hash(n,out)==graph_hash(n,e)
    assert not ledger['edits']


def test_edit_cap_abstains():
    n,e,c=fixture();out,ledger=decode(n,e,c,np.array([0,10,10,0]),margin=1.5,max_fraction=.02)
    assert np.array_equal(out,e) and ledger['changed_edges']==0


def test_forks_are_fixed():
    n,e,c=fixture();e=np.array([[10,30],[10,40]])
    out,_=decode(n,e,c,np.array([-10,0,20,20]),margin=0,max_fraction=3)
    assert np.array_equal(out,e)


def test_sparse_unknown_is_not_negative():
    pairs=np.array([[0,1],[2,3],[0,3],[2,1]])
    y=supported_labels(pairs,np.array([100,101,-1,-1]),np.array([[100,101]]))
    assert y.tolist()==[1,-1,0,0]


def test_teacher_projection_preserves_native_identity_with_changed_centers():
    n,e,_=fixture();raw=n.astype(float);donor=n.copy();donor[0,2]+=1
    inserted=np.array([[50,0,1,5,1]]);current=np.concatenate([n,inserted]);dn=np.concatenate([donor,inserted])
    es,audit=project_teacher(current,dn,np.array([[10,30],[50,40]]),raw,np.ones(3))
    assert es=={(10,30)} and audit['mapped_nodes']==4 and audit['different_centers']==1
    assert audit['unmapped_edge_fraction']==.5


def test_graph_only_change_regenerates_temporal_features():
    n=np.array([[10,0,1,1,1],[20,1,1,1,2],[30,2,1,1,3]])
    f=np.zeros((3,16),np.float32);g=f.copy()
    temporal(n,np.array([[10,20],[20,30]]),f,np.ones(3));temporal(n,np.array([[10,20]]),g,np.ones(3))
    assert not np.array_equal(f[:,10:],g[:,10:])


def test_conflicting_alternatives_do_not_merge():
    n,e,c=fixture();out,ledger=decode(n,e,c,np.array([0,20,0,10]),margin=.1,max_fraction=3)
    assert len(set(out[:,1]))==len(out)
    validate(n,out,[2,8,8,8])


def test_timeout_abstains(monkeypatch):
    import strong_tracker_v3.association as association
    from types import SimpleNamespace
    monkeypatch.setattr(association,'milp',lambda **kw:SimpleNamespace(success=False,x=None))
    n,e,c=fixture();out,ledger=decode(n,e,c,np.array([0,10,10,0]),margin=1,max_fraction=3)
    assert np.array_equal(out,e) and ledger['abstained']>0


def test_stale_feature_cache_rejects_topology_change(tmp_path):
    from types import SimpleNamespace
    from strong_tracker_v3.features import stamp,cached
    from strong_tracker_v3.common import save_arrays,sha,write_json
    ctx=SimpleNamespace(v1=tmp_path/'v1',v2=tmp_path/'v2',full=tmp_path/'full',data=tmp_path/'data',out=tmp_path/'out')
    sample=dict(dataset='clip',physical_scale=[1,1,1])
    files=[ctx.v2/'raw/clip.npz',ctx.full/'inputs/pre_ilp_clip.npz',ctx.v1/'baseline/public/clip.npz',
        ctx.v2/'predictions/clip.npz',ctx.data/'train/clip.zarr/zarr.json']
    for p in files:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fingerprint only')
    n,e,_=fixture();cache=ctx.out/'features/clip.npz';save_arrays(cache,node_features=np.zeros((len(n),16)))
    write_json(cache.with_suffix('.json'),dict(inputs=stamp(ctx,sample,n,e),sha256=sha(cache)))
    assert len(cached(ctx,sample,n,e)['node_features'])==len(n)
    with pytest.raises(ValueError,match='Stale incumbent feature cache'):cached(ctx,sample,n,e[:1])
