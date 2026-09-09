import numpy as np
import pytest

from annotation_selection.metric_adapter import evaluate_graph
from strong_tracker_v2.common import graph_hash,validate
from strong_tracker_v2.decode import divisions
from strong_tracker_v2.hypotheses import FORK_FEATURES
from strong_tracker_v2.oracles import legal_links
from strong_tracker_v2.policy import ranked_mask


def fixture():
    n=np.array([[0,0,10,20,20],[1,1,10,20,20],[2,2,10,16,20],
                [3,2,10,24,20],[4,3,10,12,20],[5,3,10,28,20]])
    e=np.array([[0,1],[1,2],[2,4],[3,5]])
    return n,e


def test_joint_fork_fixes_real_metric_and_preserves_nodes():
    n,e=fixture();gt=np.vstack([e,[1,3]])
    features=np.zeros((1,len(FORK_FEATURES)),np.float32)
    h=dict(triples=np.array([[1,2,3]]),fork_features=features)
    before=graph_hash(n,e);out,receipt=divisions(n,e,h,np.array([.9]),.05)
    assert graph_hash(n,e)==before
    validate(n,out,(4,64,256,256))
    r,_,_=evaluate_graph('test',n,out,n,gt,(1.625,.40625,.40625),100)
    assert r['division_tp']==1 and r['edge_fn']==0 and r['edge_fp']==0
    assert receipt['solved']==1


def test_shared_daughter_competition_does_not_create_merge():
    n,e=fixture();n=np.vstack([n,[6,1,10,23,20]])
    h=dict(triples=np.array([[1,2,3],[6,2,3]]),fork_features=np.zeros((2,len(FORK_FEATURES))))
    out,_=divisions(n,e,h,np.array([.9,.8]),.05)
    validate(n,out,(4,64,256,256))
    assert len(out[out[:,1]==3])==1


def test_oracle_only_links_preserves_legal_degrees():
    n,e=fixture();gt=np.vstack([e,[1,3]])
    fixed=legal_links(n,e,n,gt,{int(i):int(i) for i in n[:,0]})
    validate(n,fixed,(4,64,256,256))
    assert set(map(tuple,fixed))==set(map(tuple,gt))


def test_fork_protection_and_bounds_exception_are_explicit():
    n,e=fixture();e=np.vstack([e,[1,3]])
    keep=ranked_mask(n,e,np.zeros(len(n)),.5,'fork_protected')
    assert keep.all()
    n[0,2]=64
    with pytest.raises(AssertionError):validate(n,e,(4,64,256,256))
    validate(n,e,(4,64,256,256),reference=n,allow_legacy_bounds=True)
    with pytest.raises(AssertionError):validate(n,np.vstack([e,e[:1]]),(4,64,256,256),reference=n,allow_legacy_bounds=True)


def test_annotation_access_guard_in_separate_process():
    import os,subprocess,sys
    script='from strong_tracker_v2.infer import deny_annotations; deny_annotations(); open("/tmp/forbidden.geff/zarr.json")'
    result=subprocess.run([sys.executable,'-c',script],env=os.environ,capture_output=True,text=True)
    assert result.returncode!=0 and 'PermissionError: Annotations unavailable' in result.stderr
