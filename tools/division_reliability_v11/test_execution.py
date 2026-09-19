"""Meaningful isolation, numerical, sparse-label and export regression tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class GuardTests(unittest.TestCase):
    def test_fresh_process_denials(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'allowed').mkdir();(root/'denied').mkdir()
            (root/'denied'/'renamed.bin').write_bytes(b'not permitted')
            (root/'allowed'/'alias').symlink_to(root/'denied'/'renamed.bin')
            program='''
import os,sys,subprocess
from pathlib import Path
from division_reliability_v11.guard import install
root=Path(sys.argv[1]); fd=os.open(root/'denied',os.O_RDONLY)
receipt=install(inputs=[root/'allowed'],outputs=[],code_roots=[Path(sys.prefix),Path(sys.base_prefix)])
checks=[lambda:(root/'denied'/'renamed.bin').read_bytes(),lambda:(root/'allowed'/'alias').read_bytes(),lambda:os.open('renamed.bin',os.O_RDONLY,dir_fd=fd),lambda:subprocess.run([sys.executable,'-c','pass'])]
for f in checks:
 try:f()
 except PermissionError:continue
 raise AssertionError('Forbidden read/delegation succeeded')
assert receipt['denied']==3 and receipt['subprocess_denied']==1
'''
            subprocess.run([sys.executable,'-c',program,str(root)],check=True)


class NumericalTests(unittest.TestCase):
    def test_deterministic_adjoint(self):
        import torch
        from .deterministic import TrilinearX2
        for shape in [(2,3,3,4,5),(1,2,1,2,3)]:
            x=torch.randn(shape,requires_grad=True);g=torch.randn(tuple(shape[:2])+tuple(2*s for s in shape[2:]))
            y=torch.nn.functional.interpolate(x,scale_factor=2,mode='trilinear',align_corners=False)
            actual=TrilinearX2()(x)
            torch.testing.assert_close(actual,y,rtol=0,atol=0)
            expected=torch.autograd.grad(y,x,g)[0];got=torch.autograd.grad(actual,x,g)[0]
            torch.testing.assert_close(got,expected,rtol=1e-5,atol=4e-6)

    def test_sparse_gradients_and_negative_occurrence(self):
        import torch
        from .models import supported_losses
        for labels,target in [([-1,-1],-1),([0,-1],-1),([0,0],0),([1,0,-1],1),([1,1,0],1)]:
            a=torch.tensor(2.,requires_grad=True);b=torch.arange(len(labels),dtype=torch.float32,requires_grad=True)
            y=torch.tensor(labels)
            occ,rank,counts=supported_losses(a,b,y)
            (occ+rank).backward()
            self.assertEqual(counts['target'],target)
            self.assertTrue(torch.all(b.grad[y<0]==0))
            if target==0:self.assertGreater(abs(float(a.grad)),0)
            if target==-1:self.assertEqual(float(a.grad),0)

    def test_daughter_symmetry(self):
        import torch
        from .models import CompactPolicy
        torch.manual_seed(3);m=CompactPolicy(4,5,6).eval()
        p,a,b=(torch.randn(7,128) for _ in range(3));s=torch.randn(7,5)
        torch.testing.assert_close(m.action_scores(p,a,b,s),m.action_scores(p,b,a,s),rtol=0,atol=0)

    def test_supported_incoming_does_not_invent_second_daughter_negative(self):
        import numpy as np
        from .data import supported_incoming
        labels=supported_incoming([1,2],[3,4,5],np.array([[1,3],[1,4]]))
        np.testing.assert_array_equal(labels,[[1,1,-1],[0,0,-1]])

    def test_unknown_detection_gradient(self):
        import torch
        from .upstream import detection_loss
        x=torch.zeros(4,requires_grad=True);target=torch.tensor([1.,0.,.2,.5])
        detection_loss(x,target,torch.tensor([True,False,False,False]),torch.tensor([False,True,False,False])).backward()
        self.assertEqual(x.grad[2:].abs().sum(),0)


class GraphTests(unittest.TestCase):
    def test_real_bank_abstains_entire_parent_at_unique_action_cap(self):
        import numpy as np
        from .actions import Bank
        rng=np.random.default_rng(121)
        for _ in range(3):
            points=rng.uniform([15,50,50],[25,90,90],(4,24,3))
            nodes=np.array([[24*t+j,t,*np.rint(points[t,j])] for t in range(4) for j in range(24)],np.int64)
            edges=np.array([[24*t+j,24*(t+1)+int(k)] for t in range(3) for j,k in enumerate(rng.permutation(24)) if rng.random()<.5])
            evidence=np.array([[a[0],b[0],float(rng.normal())] for a in nodes for b in nodes if b[1]==a[1]+1])
        group=Bank(nodes,edges,evidence,4).parent(3)
        self.assertFalse(group['complete']);self.assertEqual(group['unique_forks_lower_bound'],4097)
        self.assertEqual(group['forks'],[]);self.assertEqual(group['nofork'],[])

    def test_nonfinite_predictions_fail_closed(self):
        import numpy as np
        from .graphs import peaks
        from .actions import Bank
        with self.assertRaises(ValueError):peaks(np.full((3,3,3),np.nan))
        with self.assertRaises(ValueError):Bank(np.array([[4,0,1,1,1],[8,1,1,1,1]]),np.array([[4,8]]),np.array([[4,8,np.nan]]),2)

    def test_provenance_cycles_and_exposed_parents(self):
        from .provenance import artifact,validate
        from .common import Blocked
        cyclic={'a':artifact('model',['b'],source='44b6',initialization='random'),
                'b':artifact('model',['a'],source='44b6',initialization='random')}
        with self.assertRaises(Blocked):validate(cyclic,'a','44b6')
        exposed={'raw':artifact('raw_images',source='44b6',partition='fit',exposed=True),
                 'm':artifact('model',['raw'],source='44b6',initialization='random')}
        with self.assertRaises(Blocked):validate(exposed,'m','44b6')
        from .stage_provenance import validate_stages
        leak={'cal':artifact('raw_images',source='44b6',partition='calibration'),
              'm':artifact('model',['cal'],source='44b6',initialization='random'),
              'root':artifact('calibration',['m','cal'],source='44b6')}
        with self.assertRaises(Blocked):validate_stages(leak,'root','44b6')

    def test_deployment_denominator_keeps_unknown_actions(self):
        import numpy as np
        from .actions import utilities
        from scipy.special import logsumexp
        class Bank:
            def structural(self,d):return d
        g=dict(complete=True,forks=[1.,2.,3.],nofork=[.5])
        b=np.array([2.,1.,20.])
        np.testing.assert_allclose(utilities(Bank(),g,4.,b,6),4+b-logsumexp(b)+np.arange(1,4)-.5-6)
        with self.assertRaises(ValueError):utilities(Bank(),g,4.,b[:2],6)
        from .calibration_cache import cached_utility
        for dtype in (np.float32,np.float64):
            b=np.array([-.1234,4.812,.622],dtype=dtype)
            np.testing.assert_array_equal(cached_utility(.38,b,np.array(g['forks']),np.float64(.5),1.3,-.9,6),
                utilities(Bank(),g,.38/1.3-.9,b,6))

    def test_persisted_event_is_one_unit_across_clips(self):
        import numpy as np
        from .event_training import Groups
        from .common import write,sha
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for clip in ('a','b'):
                (root/clip).mkdir()
                np.savez(root/clip/'training.npz',offset=np.array([0,1,2]),risks=np.array([1,0]),identity_pairs=np.array([[0,1]]))
                write(root/clip/'receipt.json',dict(training_sha256=sha(root/clip/'training.npz'),positive_events={'101':[0]}))
            g=Groups(root,['a','b'])
            self.assertEqual(g.positive,{'101':[('a',0),('b',0)]})
            cached={k:v.copy() for k,v in g.load('a').items()};g.load('b')
            self.assertEqual(set(g.loaded),{'a','b'})
            bounded=Groups(root,['a','b'],max_cache_bytes=1)
            bounded.load('a');bounded.load('b')
            self.assertEqual(list(bounded.loaded),['b'])
            for key,value in bounded.load('a').items():np.testing.assert_array_equal(value,cached[key])
            g.group=lambda kind,clip,key:(clip,key)
            _,r=g.choose(np.random.default_rng(1),'positive')
            self.assertEqual(r['selection_probability'],.5)
            self.assertEqual(r['event_identity'],'101')

    def test_complete_calibration_cache_resume_and_identity(self):
        import numpy as np
        from unittest.mock import patch
        from .calibration_cache import compute
        from .common import Blocked
        class Bank:
            def structural(self,d):return d
        group=dict(parent=21,forks=[1.,2.,3.],nofork=[.5],complete=True)
        conditional=np.array([2.,-.7,4.],np.float32)
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d);identity={'frozen_source_fit':'abc'}
            with patch('division_reliability_v11.policy.score_bank',return_value=iter([(group,.3,conditional)])):
                first=compute(Bank(),None,None,None,None,None,None,dest,'C11',identity)
            with patch('division_reliability_v11.policy.score_bank',side_effect=AssertionError('Cache recomputed')):
                self.assertEqual(compute(Bank(),None,None,None,None,None,None,dest,'C11',identity),first)
                with self.assertRaises(Blocked):compute(Bank(),None,None,None,None,None,None,dest,'C11',{'different':'parent'})
            with np.load(dest/'logits.npz') as values:
                self.assertEqual(values['conditional'].dtype,np.float32)
                np.testing.assert_array_equal(values['conditional'],conditional)
                np.testing.assert_array_equal(values['offset'],[0,3])
            (dest/'logits.npz').write_bytes(b'changed')
            with self.assertRaises(Blocked):compute(Bank(),None,None,None,None,None,None,dest,'C11',identity)

    def test_complete_action_optimization_parity(self):
        import numpy as np
        from .actions import Bank
        # Independent inherited event-family enumeration, with real donor
        # continuations, terminations and canonical complete-edit deduplication.
        def reference(bank,p):
            buckets={'division':{},'other':{}}
            for event in bank.events(p):
                for d in bank.alternatives.event(event):
                    if d.kind=='keep':continue
                    key=(tuple(sorted(d.add)),tuple(sorted(d.remove)))
                    dest=buckets['division' if d.kind=='division' else 'other']
                    if key not in dest or d.event<dest[key].event:dest[key]=d
                    if len(buckets['division'])>4096:return None
            return buckets
        def canonical(d):return (d.event,tuple(sorted(d.add)),tuple(sorted(d.remove)),d.owners,d.births,d.terminations,tuple(sorted(d.resources)))
        rng=np.random.default_rng(55)
        for trial in range(3):
            nodes=np.array([[1000+64*i,t,20,100+6*j,110+3*j] for t in range(4) for j,i in enumerate(range(t*4,t*4+4))])
            edges=np.array([[nodes[t*4+j,0],nodes[(t+1)*4+k,0]] for t in range(3) for j,k in enumerate(rng.permutation(4))])
            evidence=np.array([[a[0],b[0],float(rng.normal())] for a in nodes for b in nodes if b[1]==a[1]+1])
            bank=Bank(nodes,edges,evidence,4)
            for p in sorted(bank.expanded):
                expected=reference(bank,p);got=bank.parent(p)
                if expected is None:self.assertFalse(got['complete']);continue
                for key,field in [('division','forks'),('other','nofork')]:
                    self.assertEqual(sorted(canonical(d) for d in expected[key].values()),sorted(canonical(d) for d in got[field]))

    def test_provenance_rejects_cross_source_and_unknown(self):
        from .provenance import artifact,validate
        from .common import Blocked
        a={'input':artifact('raw_images',source='6bba',partition='fit'),
           'model':artifact('model',['input'],source='44b6',initialization='random')}
        with self.assertRaises(Blocked):validate(a,'model','44b6')
        with self.assertRaises(Blocked):validate({},'missing','44b6')

    def test_native_csv_ids_and_nulls(self):
        import numpy as np
        from .graphs import export_csv,read_csv,continuation
        nodes=np.array([[91,0,63.4,255.8,-.6],[303,1,63,252,0]])
        edges=np.array([[91,303]])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'graph.csv';a=export_csv(path,'renamed',nodes,edges);b=read_csv(path,'renamed')
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(a[0][0],[91,0,63,255,0])
        pairs=np.array([[0,0],[1,0]])
        chosen=continuation(np.zeros((2,3)),np.zeros((1,3)),pairs,np.array([-1.,1.]))
        np.testing.assert_array_equal(chosen,[[1,0]])

    def test_source_safety_noop_and_tie(self):
        from .calibration import select_margin
        rows=[dict(margin=m,combined_delta=0.,adjusted_edge_delta=0.,introduced_fp=0,newly_recovered_tp=0) for m in (2,4,6,8)]
        self.assertEqual(select_margin(rows)['margin'],8)
        self.assertTrue(select_margin([{**rows[0],'combined_delta':-.01}])['disabled_policy'])

    def test_official_empty_denominators_and_complete_population(self):
        from .report import clip_summary,aggregate_rows
        row=dict(dataset='control',edge_tp=0,edge_fp=0,edge_fn=0,division_tp=0,division_fp=0,division_fn=0,
            num_pred_nodes=0,estimated_total=100,gt_node_recall=None)
        self.assertIsNone(clip_summary(row,'control')['score'])
        self.assertIsNone(clip_summary(row,'control')['division_jaccard'])
        supported={**row,'dataset':'supported','edge_tp':3,'gt_node_recall':1.}
        scored=clip_summary(supported,'supported')
        self.assertIsNone(scored['division_jaccard'])
        self.assertEqual(scored['score'],scored['adj_edge_jaccard'])
        pooled=aggregate_rows([row,supported],['control','supported'])
        self.assertEqual(pooled['n'],2);self.assertEqual(pooled['n_adj'],1)
        with self.assertRaises(ValueError):aggregate_rows([supported],['control','supported'])


if __name__=='__main__':unittest.main()
