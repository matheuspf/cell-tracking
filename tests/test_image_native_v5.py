"""Behavioral contracts for the actual v5 objective and image observations."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from image_native_tracking_v5.temporal_decode import solve_window,decode,protected_context
from image_native_tracking_v5.observations import peaks,regions
from image_native_tracking_v5.event_paths import equivalence_classes

class SolverTests(unittest.TestCase):
    def fixture(self,scores):
        n=np.array([[0,0,1,2,2],[1,1,1,1,2],[2,1,1,3,2]])
        sel,e,r=solve_window(n,np.array([[0,1],[0,2]]),np.array(scores),{0,1,2},{(0,1)},np.ones(3),[])
        self.assertFalse(r['fallback']);return e
    def test_true_fork(self):self.assertEqual(self.fixture([6,6]),{(0,1),(0,2)})
    def test_continuation_plus_birth(self):self.assertEqual(self.fixture([6,-6]),{(0,1)})
    def test_empty(self):
        self.assertEqual(solve_window(np.empty((0,5)),np.empty((0,2)),np.empty(0),set(),set(),np.empty(0),[])[0],set())
    def test_no_cell(self):
        n=np.array([[0,0,1,1,1],[1,1,1,1,1]])
        sel,e,r=solve_window(n,np.array([[0,1]]),np.array([-5.]),set(),set(),np.array([.01,.01]),[])
        self.assertEqual(sel,set());self.assertEqual(e,set())
    def test_conflicting_owner(self):
        n=np.array([[0,0,1,1,1],[1,0,1,2,1],[2,1,1,1,1]])
        sel,e,r=solve_window(n,np.array([[0,2],[1,2]]),np.array([8.,4.]),{0,1,2},set(),np.ones(3),[])
        self.assertEqual(e,{(0,2)})
    def test_exclusive_two_vs_one(self):
        n=np.array([[0,0,1,1,1],[1,0,1,2,1],[2,0,1,3,1],[3,1,1,1,1],[4,1,1,3,1]])
        sel,e,r=solve_window(n,np.array([[1,3],[2,4]]),np.array([8.,8.]),{0,3,4},set(),np.array([1,.9,.9,1,1]),[(0,[1,2])])
        self.assertTrue(({0}&sel)==set() or ({1,2}&sel)==set());self.assertEqual(1 in sel,2 in sel)
    def test_scorer_context_protection(self):
        e=np.array([[0,1],[1,2],[1,3],[2,4],[3,5],[4,6]])
        p,n=protected_context(e);self.assertTrue({(0,1),(1,2),(1,3),(2,4),(3,5)}<=p)
    def test_identity_and_consecutive(self):
        n=np.array([[0,0,1,1,1],[1,1,1,1,1],[2,2,1,1,1],[3,3,1,1,1],[4,4,1,1,1],[5,5,1,1,1]])
        e=np.array([[i,i+1] for i in range(5)])
        a,b,r=decode(n,e,np.ones(5)*8,n,e,no_new_evidence=True)
        np.testing.assert_array_equal(n,a);np.testing.assert_array_equal(e,b)
        a,b,r=decode(n,e,np.ones(5)*8,n,e)
        np.testing.assert_array_equal(e,b)
    def test_duplicate_edges_do_not_change_explanation(self):
        n=np.array([[0,0,30,100,100],[1,1,30,100,100],[2,1,30,108,100]])
        e=np.array([[0,1],[0,2]]);base=np.array([[0,1]])
        a,b,_=decode(n,e,np.array([6.,7.]),n,base)
        aa,bb,_=decode(n,np.repeat(e,7,axis=0),np.repeat([6.,7.],7),n,base)
        np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
    def test_timing_alternatives_share_predicted_paths(self):
        n=np.array([[0,0,30,100,100],[1,1,30,100,100],[2,2,30,100,100],
                    [3,3,30,100,100],[11,1,30,108,100],[12,2,30,108,100],[13,3,30,108,100]])
        base=np.array([[0,1],[1,2],[2,3],[11,12],[12,13]])
        pairs=np.concatenate([base,[[0,11],[1,12]]]);scores=np.array([6.,6.,6.,.2,.2,12.,14.])
        groups=equivalence_classes(n,pairs,scores,base,set(n[:,0].astype(int)))
        self.assertTrue(any(((0,1),(0,11)) in v and ((1,2),(1,12)) in v for v in groups.values()))
        _,edges,receipt=decode(n,pairs,scores,n,base)
        forks=[a for a in set(edges[:,0]) if sum(edges[:,0]==a)==2]
        self.assertEqual(len(forks),1);self.assertGreater(receipt['event_equivalence_classes'],0)

class ObservationTests(unittest.TestCase):
    def test_flat_background(self):
        n,p,g=peaks(np.ones((16,16,16))*.9,np.zeros((16,16,16)),np.empty((0,5)),0,0)
        self.assertEqual(len(n),0)
    def test_new_peak_and_pixel_effect(self):
        x=np.indices((16,16,16));im=np.exp(-sum((x[i]-8)**2 for i in range(3))/3)
        n,p,g=peaks(im,im,np.empty((0,5)),0,0);self.assertGreater(len(n),0)
        a,b,c=peaks(im,np.zeros_like(im),np.empty((0,5)),0,0);self.assertEqual(len(a),0)
    def test_collocated_seed_not_deleted(self):
        x=np.indices((16,16,16));im=np.exp(-sum((x[i]-8)**2 for i in range(3))/3)
        n=np.array([[11,0,8,32,32],[21,0,8,32,32]])
        props,valid,collision,optical=regions(im,im,n)
        self.assertEqual(len(props),2);self.assertTrue(collision[1]);self.assertFalse(valid[1]);self.assertTrue(valid[0])

class SupervisionTests(unittest.TestCase):
    def test_empty_censored_objective_is_zero(self):
        import torch
        from image_native_tracking_v5.train_native import objective
        log=torch.zeros((3,4),requires_grad=True)
        loss=objective(log,np.empty((0,2),dtype=int));loss.backward()
        self.assertEqual(float(loss.detach()),0);self.assertEqual(float(log.grad.abs().sum()),0)
    def test_two_daughters_positive_unknown_second_daughter_masked(self):
        import torch
        from image_native_tracking_v5.train_native import objective
        log=torch.zeros((3,3),requires_grad=True)
        objective(log,np.array([[0,0],[0,1]])).backward()
        self.assertLess(float(log.grad[0,0]),0);self.assertLess(float(log.grad[0,1]),0)
        self.assertEqual(float(log.grad[:,2].abs().sum()),0)

if __name__=='__main__':unittest.main()
