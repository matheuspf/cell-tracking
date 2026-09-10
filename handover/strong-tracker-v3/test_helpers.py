"""Synthetic/reference arithmetic checks. No v3 competition experiment is run."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from scorecard import count,summary,compare,scenarios
from graph_edits import Graph,apply_atomic,protected_external_owners
from preflight import inspect_cache,require_safe_output,sha256


def row(name='A', embryo='X', tp=90,fp=5,fn=5,nodes=100,estimate=100,dtp=3,dfp=1,dfn=1):
    return dict(dataset=name,embryo=embryo,edge_tp=tp,edge_fp=fp,edge_fn=fn,
                num_pred_nodes=nodes,estimated_total=estimate,division_tp=dtp,division_fp=dfp,division_fn=dfn)


class CountTests(unittest.TestCase):
    def test_baseline(self):
        self.assertAlmostEqual(summary([row()],['A'])['score'],.96)
    def test_count_improvement(self):
        self.assertAlmostEqual(summary([row(nodes=50)],['A'])['score'],1.005)
    def test_weight_not_macro(self):
        rows=[row(tp=9,fp=0,fn=1,nodes=50,dtp=0,dfp=0,dfn=0),row('B',tp=50,fp=25,fn=25,nodes=200,dtp=0,dfp=0,dfn=0)]
        self.assertAlmostEqual(summary(rows,['A','B'])['score'],(10*.9*1.05+100*.5*.9)/110)
    def test_division_micro(self):
        s=summary([row(dtp=1,dfp=0,dfn=0),row('B',dtp=0,dfp=0,dfn=9)],['A','B'])
        self.assertAlmostEqual(s['division_jaccard'],.1)
    def test_division_absent(self):
        s=summary([row(dtp=0,dfp=0,dfn=0)],['A'])
        self.assertIsNone(s['division_jaccard']);self.assertAlmostEqual(s['score'],.9)
    def test_clipping(self):
        self.assertAlmostEqual(summary([row(nodes=2000)],['A'])['score'],.06)
    def test_missing_duplicate_extra(self):
        for rows,ids in [([row()],['A','B']),([row(),row()],['A']),([row()],[]),([row()],['A','A']),([row('B')],['A'])]:
            with self.subTest(ids=ids),self.assertRaises(ValueError):summary(rows,ids)
    def test_bad_numbers(self):
        for x in [-1,.5,math.nan,math.inf,True,'bad']:
            with self.subTest(x=x),self.assertRaises(ValueError):count(x,'test')
    def test_large_integer_exact(self):
        self.assertEqual(count(str(2**60+1),'n'),2**60+1)
    def test_invalid_estimates(self):
        for x in [0,-1,math.nan,math.inf,True]:
            with self.subTest(x=x),self.assertRaises(ValueError):summary([row(estimate=x)],['A'])
    def test_mixed_variants(self):
        a,b=row(),row('B');a['variant']='a';b['variant']='b'
        with self.assertRaises(ValueError):summary([a,b],['A','B'])
    def test_zero_edge_sample(self):
        s=summary([row(),row('B',tp=0,fp=0,fn=0,dtp=0,dfp=0,dfn=0)],['A','B'])
        self.assertEqual(s['n_adj'],1);self.assertAlmostEqual(s['score'],.96)
    def test_empty_edge_run(self):
        with self.assertRaises(ValueError):summary([row(tp=0,fp=0,fn=0)],['A'])
    def test_compare_noop(self):
        self.assertEqual(compare([row()],[row()],['A'])['decision'],'retain_incumbent')
    def test_compare_positive(self):
        self.assertTrue(compare([row()],[row(tp=91,fn=4)],['A'])['eligible_by_counts_only'])
    def test_regression_in_one_embryo(self):
        a=[row(),row('B','Y')];b=[row(tp=95,fn=0),row('B','Y',tp=89,fn=6)]
        r=compare(a,b,['A','B']);self.assertGreater(r['delta'],0);self.assertFalse(r['eligible_by_counts_only'])
    def test_denominator_drift(self):
        with self.assertRaises(ValueError):compare([row()],[row(tp=91)],['A'])
    def test_estimate_drift(self):
        with self.assertRaises(ValueError):compare([row()],[row(estimate=101)],['A'])
    def test_embryo_drift(self):
        with self.assertRaises(ValueError):compare([row()],[row(embryo='Y')],['A'])
    def test_measured_incumbent_arithmetic(self):
        b=json.loads(Path(__file__).with_name('measured_baseline.json').read_text())['incumbent']
        self.assertAlmostEqual(b['adj_edge_jaccard']+.1*b['division_tp']/(b['division_tp']+b['division_fp']+b['division_fn']),b['score'])
        s=scenarios();self.assertAlmostEqual(s[0]['delta'],0);self.assertGreater(s[3]['delta'],.02)


class EditTests(unittest.TestCase):
    def setUp(self):
        self.times={0:0,1:1,2:1,3:0,4:2,5:2}
        self.g=Graph.build(self.times,[(0,1),(3,2),(1,4),(2,5)])
    def test_noop(self):
        self.assertEqual(apply_atomic(self.g,[],[],[]),self.g)
    def test_atomic_owner_reassignment(self):
        new=apply_atomic(self.g,[(3,2)],[(0,2)],self.times)
        self.assertIn((0,2),new.edges);self.assertIn((3,2),self.g.edges)
    def test_stolen_target_rejected(self):
        with self.assertRaises(ValueError):apply_atomic(self.g,[],[(0,2)],self.times)
    def test_invalid_graph_shapes(self):
        for es in [[(0,1),(0,1)],[(0,4)],[(1,0)],[(0,99)],[(0,1),(3,1)]]:
            with self.subTest(es=es),self.assertRaises(ValueError):Graph.build(self.times,es)
    def test_outdegree(self):
        with self.assertRaises(ValueError):Graph.build({0:0,1:1,2:1,3:1},[(0,1),(0,2),(0,3)])
    def test_boundary(self):
        with self.assertRaises(ValueError):apply_atomic(self.g,[(3,2)],[],[3])
    def test_frozen_edge(self):
        with self.assertRaises(ValueError):apply_atomic(self.g,[(3,2)],[],self.times,[(3,2)])
    def test_edits_legal_alone_conflict_together(self):
        g=Graph.build({0:0,1:0,2:1},[])
        apply_atomic(g,[],[(0,2)],dict(g.node_times));apply_atomic(g,[],[(1,2)],dict(g.node_times))
        with self.assertRaises(ValueError):apply_atomic(g,[],[(0,2),(1,2)],dict(g.node_times))
    def test_nonexistent_and_redundant_edit(self):
        for remove,add in [([(0,2)],[]),([],[(0,1)]),([(0,1)],[(0,1)])]:
            with self.subTest(remove=remove,add=add),self.assertRaises(ValueError):apply_atomic(self.g,remove,add,self.times)
    def test_negative_id(self):
        with self.assertRaises(ValueError):Graph.build({-1:0,1:1},[])
    def test_owner_guard_minimum_bug_fixture(self):
        # Old minimum-based distance gate sees min(0,1)==0 and misses owner 3.
        owners=protected_external_owners(0,[1,2],{1:3},{(3,1):dict(probability=.999,distance_um=1.)})
        self.assertEqual(owners,{3});self.assertFalse(0<min(0,1)<2.)
    def test_owner_is_parent_or_free(self):
        self.assertEqual(protected_external_owners(0,[1,2],{1:0},{}),set())
    def test_missing_owner_evidence_protected(self):
        self.assertEqual(protected_external_owners(0,[1],{1:3},{}),{3})
    def test_weak_owner_can_be_jointly_considered(self):
        self.assertEqual(protected_external_owners(0,[1],{1:3},{(3,1):dict(probability=.2,distance_um=4.)}),set())
    def test_nonfinite_owner_evidence(self):
        with self.assertRaises(ValueError):protected_external_owners(0,[1],{1:3},{(3,1):dict(probability=math.nan,distance_um=1.)})


class PreflightTests(unittest.TestCase):
    def make(self,root):
        p=root/'selected_predictions';p.mkdir()
        for n in ['a','b']:(p/(n+'.npz')).write_bytes(n.encode())
        lock=dict(variant='bypass_motion_bounds',annotation_reads=0,byte_identical_to_scored_graphs=True,
                  hashes={n:sha256(p/(n+'.npz')) for n in ['a','b']})
        (root/'selected_prediction_lock.json').write_text(json.dumps(lock))
    def test_cache_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);self.make(r);self.assertEqual(inspect_cache(r,2)['samples'],2)
    def test_changed_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);self.make(r);(r/'selected_predictions/a.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):inspect_cache(r,2)
    def test_incomplete_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);self.make(r);(r/'selected_predictions/a.npz').unlink()
            with self.assertRaises(ValueError):inspect_cache(r,2)
    def test_output_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);inp=r/'input';inp.mkdir()
            for out in [inp,inp/'x',r]:
                with self.subTest(out=out),self.assertRaises(ValueError):require_safe_output(out,[inp])
            self.assertEqual(require_safe_output(r/'out',[inp]),r/'out')
    def test_symlink_output_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);inp=r/'input';inp.mkdir();(r/'alias').symlink_to(inp,target_is_directory=True)
            with self.assertRaises(ValueError):require_safe_output(r/'alias/x',[inp])


if __name__=='__main__':
    unittest.main()
