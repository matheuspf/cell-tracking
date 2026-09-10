import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np

from strong_tracker_v3.decode import Action,BoundedActionComponents,protected_external_owners,solve_actions
from strong_tracker_v3.event_proposals import ProposalConfig,build,extract_crops


class OwnerGuardTests(unittest.TestCase):
    def test_free_sister_cannot_mask_close_confident_owner(self):
        incoming={3:8};evidence={(8,3):dict(probability=.99,distance_um=1.)}
        self.assertEqual(protected_external_owners(1,[2,3],incoming,evidence),{8})

    def test_two_owners_are_individual(self):
        incoming={2:7,3:8};evidence={(7,2):dict(probability=.5,distance_um=1.),(8,3):dict(probability=.999,distance_um=1.)}
        self.assertEqual(protected_external_owners(1,[2,3],incoming,evidence),{8})

    def test_missing_confidence_abstains(self):
        self.assertEqual(protected_external_owners(1,[2,3],{3:8},{}),{8})

    def test_permissible_donor_is_not_protected(self):
        self.assertFalse(protected_external_owners(1,[2,3],{3:8},{(8,3):dict(probability=.4,distance_um=8.)}))

    def test_protected_donor_requires_explicit_reassignment(self):
        from strong_tracker_v3.decode import divisions
        nodes=np.array([[1,0,0,5,0],[8,0,0,0,1],[2,1,0,5,1],[3,1,0,0,0],[4,1,0,0,2]])
        edges=np.array([[8,3]])
        # Fixed external links make the declared 2% clip edit cap permit one
        # four-edge donor reassignment, and exercise unchanged boundaries.
        fixed={(100+j,400+j) for j in range(200)}
        nodes=np.vstack([nodes,[[100+j,0,100,100,j] for j in range(200)],[[400+j,1,100,100,j] for j in range(200)]])
        edges=np.vstack([edges,sorted(fixed)])
        h=dict(events=np.array([[0,2,3,-1,-1]]),existing_pool=np.ones(1,bool))
        native=dict(pairs=np.array([[1,3],[0,2],[0,3]]),edge_features=np.array([[.99,0],[.8,0],[.7,0]]))
        out,r,l=divisions(nodes,edges,h,np.array([.999]),native,[1,1,1])
        np.testing.assert_array_equal(out,edges)
        self.assertEqual(r['rejected']['protected_individual_owner'],1)
        native['pairs']=np.vstack([native['pairs'],[1,4]])
        native['edge_features']=np.vstack([native['edge_features'],[.995,0]])
        out,r,l=divisions(nodes,edges,h,np.array([.999]),native,[1,1,1])
        self.assertEqual(set(map(tuple,out)),fixed|{(1,2),(1,3),(8,4)})
        self.assertEqual(l[0]['owner_alternatives'][0]['alternative'],'continuation')


class EventTests(unittest.TestCase):
    def fixture(self):
        nodes=np.array([[0,0,0,0,0],[1,1,0,0,0],[2,1,0,3,0],[3,1,0,0,3],
            [4,2,0,3,1],[5,2,0,4,0],[6,2,0,0,4],[7,2,0,1,3]],np.int64)
        edges=np.array([[0,1]],np.int64)
        native=dict(pairs=np.array([[0,1],[0,2],[0,3]]),edge_features=np.array([[.9,0],[.7,0],[.6,0]]),
                    node_features=np.zeros((8,16)),native_index=np.arange(8))
        return nodes,edges,native

    def test_wrong_current_continuation_does_not_exclude_pair(self):
        n,e,f=self.fixture();h,m=build(n,e,f,[1,1,1])
        self.assertTrue(any(tuple(row[:3])==(0,2,3) for row in h['events']))

    def test_two_candidate_paths_without_selected_successors(self):
        n,e,f=self.fixture();h,m=build(n,e,f,[1,1,1])
        rows=h['events'][np.all(h['events'][:,:3]==[0,2,3],axis=1)]
        self.assertGreaterEqual(len(set(rows[:,3])),2)
        self.assertGreaterEqual(len(set(rows[:,4])),2)
        self.assertTrue(np.all(rows[:,3]!=rows[:,4]))

    def test_reverse_neighbor_beyond_six_is_retained(self):
        n=np.array([[0,0,0,0,0]]+[[j,1,0,0,x] for j,x in enumerate([1,2,3,4,5,6,15],1)])
        e=np.array([[0,1]]);native=dict(pairs=np.array([[0,1]]),edge_features=np.array([[.9,0]]),
            native_index=np.arange(8),node_features=np.zeros((8,16)))
        h,_=build(n,e,native,[1,1,1])
        self.assertTrue(any(tuple(row[:3])==(0,1,7) for row in h['events']))

    def test_crop_boundary_mask_and_same_sampling(self):
        n=np.array([[1,0,0,0,0]]);image=np.ones((5,8,8,8),np.uint16)*20
        a=extract_crops(image,n,[0],[1,1,1]);b=extract_crops(image,n,[0],[1,1,1])
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(1,9,4,12,12));self.assertFalse(a[0,:4].any())
        self.assertLess(a[0,4,3].mean(),255)

    def test_daughter_swap_is_image_geometry_invariant(self):
        import torch
        from strong_tracker_v3.event_model import network
        n,e,f=self.fixture();h,_=build(n,e,f,[1,1,1])
        permutation=np.array([0,1,3,2,4,5,6,7]);inverse=np.argsort(permutation)
        swapped=dict(f,pairs=inverse[f['pairs']],native_index=f['native_index'][permutation],node_features=f['node_features'][permutation])
        other,_=build(n[permutation],e,swapped,[1,1,1])
        def keyed(nodes,proposals):
            result={}
            for event,values in zip(proposals['events'],proposals['event_features']):
                i,a,b,qa,qb=event
                branches=tuple(sorted([(int(nodes[a,0]),int(nodes[qa,0]) if qa>=0 else -1),
                                       (int(nodes[b,0]),int(nodes[qb,0]) if qb>=0 else -1)]))
                result[(int(nodes[i,0]),branches)]=values
            return result
        a,b=keyed(n,h),keyed(n[permutation],other)
        self.assertEqual(set(a),set(b))
        for key in a:np.testing.assert_allclose(a[key],b[key],rtol=1e-6,atol=1e-6)
        key=next(iter(a));m=network().eval();image=torch.zeros(2,9,4,12,12)
        geometry=torch.as_tensor(np.stack([a[key],b[key]]))
        with torch.no_grad():p=m(image,geometry)
        self.assertAlmostEqual(float(p[0]),float(p[1]),places=6)

    def test_parent_embedding_cache_matches_direct_cuda_predictions(self):
        import torch
        from strong_tracker_v3.event_model import network,predict_image
        if not torch.cuda.is_available():self.skipTest('CUDA parity fixture requires the event execution GPU')
        torch.manual_seed(42);model=network().cuda().eval();rng=np.random.default_rng(42)
        crops=rng.integers(0,256,(7,9,4,12,12),dtype=np.uint8)
        features=rng.normal(size=(31,40)).astype(np.float32);ids=np.arange(31)%7
        a=dict(mean=np.zeros(40),scale=np.ones(40))
        cached=predict_image((model,a),features,crops,ids,batch=5)
        with torch.inference_mode():direct=model(torch.as_tensor(crops[ids],device='cuda').float()/255.,torch.as_tensor(features,device='cuda')).sigmoid().cpu().numpy()
        np.testing.assert_allclose(cached,direct,rtol=1e-6,atol=1e-6)
        for threshold in [.1,.2,.4]:np.testing.assert_array_equal(cached>threshold,direct>threshold)


class MilpTests(unittest.TestCase):
    def test_conflict_and_noop_tie(self):
        a=[Action(0,0.,set(),set(),{'zero'},[]),Action(1,2.,set(),set(),{'target'},[]),Action(2,3.,set(),set(),{'target'},[])]
        selected,receipt=solve_actions(a)
        self.assertEqual(selected,[2])

    def test_small_solutions_match_exhaustive(self):
        a=[Action(i,v,set(),set(),set(r),[]) for i,(v,r) in enumerate([(2.,'ab'),(3.,'bc'),(2.,'cd'),(1.,'de')])]
        selected,_=solve_actions(a)
        values=[]
        for bits in itertools.product([0,1],repeat=len(a)):
            resources=[r for i,bit in enumerate(bits) if bit for r in a[i].resources]
            if len(resources)==len(set(resources)):values.append(sum(a[i].value for i,bit in enumerate(bits) if bit))
        self.assertEqual(sum(a[k].value for k in selected),max(values))

    def test_oversize_abstains(self):
        selected,r=solve_actions([Action(i,1.,set(),set(),{'all'},[]) for i in range(257)])
        self.assertFalse(selected);self.assertEqual(r['oversized_abstentions'],1)

    def test_streamed_oversize_keeps_full_resource_closure(self):
        stream=BoundedActionComponents(max_alternatives=2)
        stream.add(Action(0,1,set(),set(),{'a'},[]));stream.add(Action(1,1,set(),set(),{'a','b'},[]))
        stream.add(Action(2,1,set(),set(),{'b'},[]))
        stream.add(Action(3,1,set(),set(),{'c'},[]))
        # A late bridge must absorb the previously eligible separate component
        # into the abstained component, not accept an incomplete remainder.
        stream.add(Action(4,1,set(),set(),{'b','c'},[]))
        actions,r=stream.finish();self.assertFalse(actions)
        self.assertEqual(r['actions_in_abstained_components'],5)

    def test_early_shift_suppresses_the_displaced_daughter_fork(self):
        from strong_tracker_v3.decode import divisions
        n=np.array([[10,0,0,0,0],[11,1,0,1,0],[12,1,0,-1,0],
            [13,2,0,2,0],[14,2,0,0,0],[15,2,0,-2,0]])
        e=np.array([[10,11],[11,13],[11,14],[12,15]])
        h=dict(events=np.array([[0,1,2,3,5],[1,3,4,-1,-1]]),existing_pool=np.ones(2,bool))
        native=dict(pairs=np.array([[0,1],[0,2],[1,3],[1,4],[2,5]]),edge_features=np.array([[.9,0],[.8,0],[.6,0],[.5,0],[.9,0]]))
        unchanged,_,_=divisions(n,e,h,np.array([.999,.01]),native,[1,1,1])
        np.testing.assert_array_equal(unchanged,e)
        shifted,_,_=divisions(n,e,h,np.array([.999,.01]),native,[1,1,1],replace=True)
        self.assertEqual(set(map(tuple,shifted)),{(10,11),(10,12),(11,13),(12,15)})

    def test_late_shift_compares_and_suppresses_the_parent_fork(self):
        from strong_tracker_v3.decode import divisions
        n=np.array([[10,0,0,0,0],[11,1,0,1,0],[12,1,0,-1,0],
            [13,2,0,2,0],[14,2,0,-2,0],[15,2,0,0,0]])
        e=np.array([[10,11],[10,12],[11,13],[12,14]])
        h=dict(events=np.array([[0,1,2,3,4],[1,3,5,-1,-1]]),existing_pool=np.ones(2,bool))
        native=dict(pairs=np.array([[0,1],[0,2],[1,3],[1,5],[2,4]]),edge_features=np.array([[.9,0],[.8,0],[.6,0],[.5,0],[.9,0]]))
        unchanged,_,_=divisions(n,e,h,np.array([.01,.999]),native,[1,1,1])
        np.testing.assert_array_equal(unchanged,e)
        shifted,_,_=divisions(n,e,h,np.array([.01,.999]),native,[1,1,1],replace=True)
        self.assertEqual(set(map(tuple,shifted)),{(10,11),(11,13),(11,15),(12,14)})


class SourceEventLabelTests(unittest.TestCase):
    def fixture(self):
        # Separated annotated branches with explicit early/exact/late paths.
        n=np.array([[0,0,10,30,30],[1,1,10,30,30],[2,2,10,25,30],
            [3,2,10,35,30],[4,3,10,24,30],[5,3,10,36,30],
            [6,1,10,35,30],[7,2,10,60,30],[8,3,10,61,30],
            [9,1,10,59,30],[10,0,10,58,30]],np.int64)
        ge=np.array([[0,1],[1,2],[1,3],[2,4],[3,5],[10,9],[9,7],[7,8]],np.int64)
        pe=np.array([[0,1],[1,2],[2,4],[3,5],[10,9],[9,7],[7,8]],np.int64)
        return n,pe,n[n[:,0]!=6],ge

    def test_official_early_exact_late_positive_bag(self):
        from strong_tracker_v3.source_labels import labels
        n,e,gn,ge=self.fixture()
        events=np.array([[0,1,6,2,3],[1,2,3,4,5],[2,4,5,-1,-1]])
        l,r=labels(n,e,events,gn,ge,[1,1,1])
        np.testing.assert_array_equal(l['all_y'],[1,1,1])
        self.assertEqual(len(set(l['all_bag'])),1)
        self.assertAlmostEqual(float(l['weight'].sum()),1.)

    def test_false_neighbor_pair_is_contradictory(self):
        from strong_tracker_v3.source_labels import labels
        n,e,gn,ge=self.fixture()
        l,r=labels(n,e,np.array([[1,2,7,4,8]]),gn,ge,[1,1,1])
        self.assertEqual(int(l['all_y'][0]),0)

    def test_partial_sparse_division_is_ignored(self):
        from strong_tracker_v3.source_labels import labels
        n,e,gn,ge=self.fixture();gn=gn[~np.isin(gn[:,0],[3,5])]
        ge=ge[~np.any(np.isin(ge,[3,5]),axis=1)]
        l,r=labels(n,e,np.array([[1,2,3,4,5]]),gn,ge,[1,1,1])
        self.assertEqual(int(l['all_y'][0]),-1)


class EventResumeTests(unittest.TestCase):
    def test_identical_resume_preserves_lock_timestamp_and_changed_input_fails(self):
        from strong_tracker_v3.event_pipeline import stable_lock
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lock.json'
            stable_lock(path,dict(created='first',models={'source':'hash'}),['models'])
            result=stable_lock(path,dict(created='later',models={'source':'hash'}),['models'])
            self.assertEqual(result['created'],'first')
            with self.assertRaises(ValueError):stable_lock(path,dict(created='later',models={'source':'different'}),['models'])


if __name__=='__main__':unittest.main()
