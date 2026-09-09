import unittest
import numpy as np
import data_contracts as c


class DataContracts(unittest.TestCase):
    def graph(self):
        return np.array([10,20,30,40,50]), np.array([0,1,1,2,2]), np.zeros((5,3)), np.array([[0,1],[0,2],[1,3],[2,4]])

    def test_sequence_alignment(self):
        p,s=c.points_on_image([[4,80,120]],sequence=True)
        np.testing.assert_array_equal(p,[[4,20,30]])
        np.testing.assert_array_equal(s,[1.625]*3)

    def test_native_static(self):
        p,s=c.points_on_image([[4,80,120]],sequence=False)
        np.testing.assert_array_equal(p,[[4,80,120]])
        np.testing.assert_array_equal(s,c.NATIVE_SPACING)

    def test_physical_equivalence(self):
        p=np.array([[4.1,83.2,123.3]])
        q,s=c.points_on_image(p,sequence=True)
        np.testing.assert_allclose(q*s,p*c.NATIVE_SPACING)

    def test_static_pooled(self):
        p,s=c.points_on_image([[4,80,120]],sequence=False,pooled_static=True)
        np.testing.assert_array_equal(p,[[4,20,30]])

    def test_bad_points(self):
        for p in ([1,2,3],[[1,float('nan'),3]],[[1,2]]):
            with self.subTest(p=p),self.assertRaises(ValueError):c.points_on_image(p,sequence=True)

    def test_intensity_equivalence(self):
        x=np.array([0,1000,2000,50000],dtype=np.uint16)
        a=c.normalize_image(x,representation='uint16_scale',low_native=500,high_native=30000)
        b=c.normalize_image(x.astype(float)/65535,representation='unit',low_native=500,high_native=30000)
        np.testing.assert_allclose(a,b,atol=1e-6)

    def test_bad_intensity(self):
        for x,rep,lo,hi in [([2],'unit',0,3),([0],'auto',0,1),([0],'unit',1,1),([np.nan],'unit',0,1)]:
            with self.subTest(x=x),self.assertRaises(ValueError):c.normalize_image(x,representation=rep,low_native=lo,high_native=hi)

    def test_graph_counts(self):
        self.assertEqual(c.validate_graph(*self.graph()),{'nodes':5,'edges':4,'division_parents':1})

    def test_edges_are_rows_not_ids(self):
        ids,t,p,e=self.graph()
        with self.assertRaises(ValueError):c.validate_graph(ids,t,p,np.array([[10,20]]))

    def test_tracklets_not_clones(self):
        _,t,_,e=self.graph();u=c.tracklet_ids(t,e)
        self.assertEqual(len(np.unique(u)),3)
        self.assertEqual(u[1],u[3]);self.assertEqual(u[2],u[4]);self.assertNotEqual(u[1],u[2])

    def test_row_order_invariant_track_equivalence(self):
        _,t,_,e=self.graph();order=np.array([4,2,0,3,1]);inverse=np.argsort(order)
        u=c.tracklet_ids(t[order],inverse[e]);v=c.tracklet_ids(t,e)[order]
        np.testing.assert_array_equal(u[:,None]==u[None,:],v[:,None]==v[None,:])

    def test_duplicate_ids(self):
        ids,t,p,e=self.graph();ids[1]=10
        with self.assertRaises(ValueError):c.validate_graph(ids,t,p,e)

    def test_duplicate_edges(self):
        ids,t,p,e=self.graph()
        with self.assertRaises(ValueError):c.validate_graph(ids,t,p,np.vstack([e,e[0]]))

    def test_merge(self):
        ids,t,p,e=self.graph()
        with self.assertRaises(ValueError):c.validate_graph(ids,t,p,np.vstack([e,[2,3]]))

    def test_nonconsecutive(self):
        ids,t,p,e=self.graph();e[0]=[0,3]
        with self.assertRaises(ValueError):c.validate_graph(ids,t,p,e)

    def test_bounds(self):
        ids,t,p,e=self.graph();p[0,0]=64
        with self.assertRaises(ValueError):c.validate_graph(ids,t,p,e,shape=(3,64,64,64))

    def test_empty_graph(self):
        self.assertEqual(c.validate_graph(np.array([],int),np.array([],int),np.empty((0,3)),np.empty((0,2),int))['nodes'],0)

    def test_float_edge_rejected(self):
        ids,t,p,e=self.graph()
        with self.assertRaises(ValueError):c.validate_graph(ids,t,p,e.astype(float))

    def test_no_static_nondivision_labels(self):
        y,m=c.dense_fork_targets(np.array([0]),np.empty((0,2),int),1)
        self.assertFalse(m.any())

    def test_censored_last_frame(self):
        _,t,_,e=self.graph();y,m=c.dense_fork_targets(t,e,3)
        np.testing.assert_array_equal(y,[True,False,False,False,False])
        np.testing.assert_array_equal(m,[True,True,True,False,False])

    def test_crop_censor(self):
        _,t,_,e=self.graph();y,m=c.dense_fork_targets(t,e,3,complete_future=np.array([False,True,True,True,True]))
        self.assertTrue(y[0]);self.assertFalse(m[0])

    def test_six_frames_not_nine(self):
        x=np.ones((6,2,2,2),dtype=np.uint16)
        w,m=c.padded_time_window(x,anchor=2,before=4,after=4)
        self.assertEqual(w.shape,(9,2,2,2));self.assertEqual(m.sum(),6);self.assertEqual(w[~m].sum(),0)

    def test_temporal_masks_not_background(self):
        x=np.zeros((6,1,1,1));_,m=c.padded_time_window(x,anchor=2,before=4,after=4)
        self.assertEqual(m.sum(),6)

    def test_bad_window(self):
        with self.assertRaises(ValueError):c.padded_time_window(np.ones((6,1,1,1)),anchor=6,before=2,after=2)

    def test_sparse_unknown_edges(self):
        ids=np.array([10,20,-1,-1]);pairs=np.array([[0,1],[0,3],[2,1],[2,3]])
        y=c.edge_supervision(ids,pairs,np.array([[10,20]]),dense=False)
        np.testing.assert_array_equal(y,[1,0,0,-1])

    def test_dense_negatives(self):
        y=c.edge_supervision(np.array([-1,-1]),np.array([[0,1]]),np.empty((0,2),int),dense=True)
        self.assertEqual(y[0],0)

    def test_partition_group_leak(self):
        rows=[dict(provenance_group='zoo-fish',representation_group='a',partition='train'),dict(provenance_group='zoo-fish',representation_group='b',partition='test')]
        with self.assertRaises(ValueError):c.assert_partition(rows)

    def test_duplicate_representation_leak(self):
        rows=[dict(provenance_group='a',representation_group='same-content',partition='train'),dict(provenance_group='b',representation_group='same-content',partition='validation')]
        with self.assertRaises(ValueError):c.assert_partition(rows)

    def test_valid_partitions(self):
        self.assertTrue(c.assert_partition([dict(provenance_group='x',representation_group='a',partition='train')]))

    def test_stable_partition(self):
        self.assertEqual(c.stable_partition('seq_1'),c.stable_partition('seq_1'))
        self.assertGreater(len(set(c.stable_partition(str(i)) for i in range(100))),1)

    def test_riken_not_links(self):
        with self.assertRaises(ValueError):c.require_task('riken_points','edges')

    def test_zoo_not_images(self):
        with self.assertRaises(ValueError):c.require_task('zoo_graph','images')

    def test_mouse_not_negative_event_bank(self):
        with self.assertRaises(ValueError):c.require_task('zoo_mouse','weak_events')

    def test_static_not_events(self):
        with self.assertRaises(ValueError):c.require_task('synthetic_static','events')

    def test_supported_tasks(self):
        for source,task in [('synthetic_static','centers'),('synthetic_sequence','events'),('zoo_graph','geometry')]:self.assertTrue(c.require_task(source,task))

    def test_fork_dominated(self):
        self.assertAlmostEqual(c.raw_fork_advantage(1),-.2)
        self.assertLess(c.raw_fork_advantage(.5),0)

    def test_learned_fork_can_win(self):
        self.assertGreater(c.raw_fork_advantage(.8,learned_gain=.5),0)

    def test_reject_invalid_probability(self):
        with self.assertRaises(ValueError):c.raw_fork_advantage(1.1)


class PreflightTests(unittest.TestCase):
    def test_missing_is_not_ready(self):
        import tempfile
        from pathlib import Path
        import preflight
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            self.assertEqual(preflight.inspect(p,p,p,p/'absent')['status'],'missing_required_paths')

    def test_preflight_does_not_create(self):
        import tempfile
        from pathlib import Path
        import preflight
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);preflight.inspect(p,p/'archive',p/'prepared',p/'v3')
            self.assertEqual(list(p.iterdir()),[])


if __name__=='__main__':unittest.main()
