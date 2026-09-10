"""Scientific input and decoder regression tests for the v4 study."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from multidata_training_v4.proposals import build,PAIRS
from multidata_training_v4.index import supervision
from multidata_training_v4.corruptions import observations
from multidata_training_v4.decode import apply
from multidata_training_v4.adapters import native_refinement

class Contracts(unittest.TestCase):
    def test_empty_observations_have_empty_bounded_candidates(self):
        a,c,x=build(np.empty(0,int),np.empty((0,3)))
        self.assertEqual(a.shape,(0,));self.assertEqual(c.shape,(0,6));self.assertEqual(x.shape,(0,6,16))
    def test_detector_target_units_are_inverted(self):
        actual=native_refinement(np.array([[0,1/3,0]]),[1.625,.40625,.40625])
        np.testing.assert_allclose(actual,[[0,1,0]],rtol=0,atol=1e-12)
    def test_detector_physical_cap_precedes_integer_rounding(self):
        spacing=np.array([1.625,.40625,.40625])
        delta=native_refinement(np.array([[0,2/3,2/3]]),spacing)
        self.assertAlmostEqual(float(np.linalg.norm(delta*spacing)),.5)
        self.assertGreater(float(np.linalg.norm(np.rint(delta)*spacing)),.5)
    def test_censored_fork_is_unknown(self):
        a=np.array([0]);c=np.array([[1,-1,-1,-1,-1,-1]])
        y,link,w=supervision(a,c,np.array([0,1]),np.array([[0,1],[0,2]]),np.array([0,1,1]),dense=True)
        self.assertEqual(y[0],-1);self.assertEqual(link[0,0],1)
    def test_unmatched_real_parent_is_not_dense_negative(self):
        y,link,w=supervision(np.array([0]),np.array([[1,2,-1,-1,-1,-1]]),np.array([-1,-1,-1]),np.array([[0,1]]),np.array([0,1]))
        self.assertEqual(y[0],-1);self.assertTrue((link==-1).all())
    def test_missing_future_frame_is_censored(self):
        y,_,_=supervision(np.array([0]),np.full((1,6),-1),np.array([0]),np.array([[0,1]]),np.array([0,1]),dense=True)
        self.assertEqual(y[0],-1)
    def test_one_known_child_does_not_label_unknown_sibling_negative(self):
        edges=np.array([[0,1],[3,4]])
        mapping=np.array([0,1,-1,4])
        for weak in [False,True]:
            y,link,_=supervision(np.array([0]),np.array([[1,2,3,-1,-1,-1]]),mapping,edges,np.array([0,1,1,0,1]),weak=weak)
            self.assertEqual(link[0,0],1)
            self.assertEqual(link[0,1],-1)
            self.assertEqual(link[0,2],0)
            self.assertEqual(y[0],-1)
    def test_no_fake_link_from_clone_or_topology(self):
        import inspect
        self.assertEqual(list(inspect.signature(build).parameters),['t','points','anchors'])
        t=np.array([0,1,1,1]);p=np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],float)
        a,c,x=build(t,p);self.assertEqual(c.shape,(1,6));self.assertEqual(x.shape,(1,6,16));self.assertTrue((c[:,3:]==-1).all())
    def test_per_axis_unitless_scale_invariance(self):
        rng=np.random.default_rng(23);t=np.repeat(np.arange(4),30);p=rng.normal(size=(120,3))
        a,c,x=build(t,p);aa,cc,xx=build(t,p*[100,.2,3]+[5,7,4])
        np.testing.assert_array_equal(c,cc);np.testing.assert_allclose(x,xx,atol=5e-5)
    def test_duplicate_corruptions_have_no_second_truth_identity(self):
        t=np.repeat(np.arange(5),100);p=np.random.default_rng(24).normal(size=(500,3));tt,pp,m=observations(t,p,10)
        self.assertTrue((m<0).any());self.assertLess(len(m[m>=0]),len(t));self.assertEqual(len(np.unique(m[m>=0])),len(m[m>=0]))
    def test_existing_daughter_evidence_is_protected(self):
        n=np.array([[0,0,0,0,0],[1,1,0,0,0],[2,1,0,1,0],[3,2,0,0,0],[4,2,0,1,0],[5,3,0,0,0],[6,3,0,1,0]])
        e=np.array([[0,1],[0,2],[1,3],[2,4],[3,5]])
        ee,info=apply(n,e,np.array([[3,5,6]]),[100])
        np.testing.assert_array_equal(ee,e);self.assertEqual(info['selected'],0)
    def test_conflicting_forks_cannot_share_daughter(self):
        n=np.array([[0,0,0,0,0],[1,0,1,0,0],[2,1,0,0,0],[3,1,0,1,0],[4,1,0,2,0]])
        e=np.empty((0,2),int);ee,info=apply(n,e,np.array([[0,2,3],[1,3,4]]),[2,3])
        self.assertEqual(info['selected'],1);self.assertEqual(len(ee),2);self.assertEqual(len(np.unique(ee[:,1])),2)
    def test_displaced_old_child_conflicts_with_new_event(self):
        n=np.array([[0,0,0,0,0],[1,1,0,0,0],[2,1,0,1,0],[3,1,0,2,0],[4,2,0,0,0],[5,2,0,1,0]])
        e=np.array([[0,1]])
        ee,info=apply(n,e,np.array([[0,2,3],[1,4,5]]),[2,3])
        self.assertEqual(info['selected'],1)
        self.assertIn((0,1),set(map(tuple,ee)))

if __name__=='__main__':unittest.main()
