import unittest
import tempfile
from pathlib import Path
import numpy as np
from segmentation_tracking_v6.providers import NativeGrid
from segmentation_tracking_v6.regions import (extract,save_frame,load_frame,attach,pair_features,
    division_features,validate_hierarchy_selection)
from segmentation_tracking_v6.ultrack_adapter import export_regions,survival,recompute_native_evidence
from segmentation_tracking_v6.controls import supported_labels

class SegmentationContracts(unittest.TestCase):
    def objects(self,t=0):
        a=np.zeros((6,8,8),np.uint32);a[1:3,1:4,1:4]=17;a[1:3,4:7,4:7]=90
        return extract(a,np.arange(a.size).reshape(a.shape),clip='x',frame=t,provider='fixture',spacing=(2.,1.,1.))

    def test_real_occupancy_persistence_and_halfopen(self):
        a=self.objects();self.assertEqual(a[0].region.bbox,(1,1,1,3,4,4));self.assertEqual(a[0].region.volume_um3,36.)
        with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
            p=Path(tmp)/'frame.npz';save_frame(p,a,shape=(6,8,8),provenance={'fixture':True})
            b,receipt=load_frame(p)
            self.assertEqual([o.metadata() for o in a],[o.metadata() for o in b])
            with self.assertRaises(FileExistsError):save_frame(p,a,shape=(6,8,8),provenance={})

    def test_anisotropy_and_no_axis_guessing(self):
        grid=NativeGrid((3,4,5),(2.,1.,1.));a=np.zeros(grid.shape,np.uint32);grid.check(a,a)
        with self.assertRaises(ValueError):grid.check(a,a.transpose(2,1,0))
        with self.assertRaises(ValueError):grid.check(a,np.ones(grid.shape,float))

    def test_empty_frames(self):
        z=np.zeros((3,4,5),np.int32)
        self.assertEqual(extract(z,z,clip='x',frame=0,provider='fixture',spacing=(2,1,1)),[])
        graph=export_regions({},{});self.assertEqual(graph['nodes'].shape,(0,5))

    def test_label_permutation_changes_no_geometry(self):
        a=np.zeros((6,8,8),np.int32);a[1:3,1:4,1:4]=900;a[1:3,4:7,4:7]=3
        b=extract(a,np.zeros_like(a),clip='x',frame=0,provider='fixture',spacing=(2,1,1))
        self.assertEqual(sorted((r.region.bbox,r.region.mask_sha256()) for r in self.objects()),
                         sorted((r.region.bbox,r.region.mask_sha256()) for r in b))

    def test_ownership_preserves_merged_and_missing_nodes(self):
        nodes=np.array([[1,0,1,2,2],[2,0,2,2,2],[3,0,1,5,5],[4,0,0,0,0]])
        original=nodes.copy();owners,flags=attach(nodes,self.objects())
        self.assertEqual(set(owners),{2});self.assertEqual(flags[0],'shared_or_merged');self.assertEqual(flags[3],'missing')
        np.testing.assert_array_equal(nodes,original)

    def test_border_tile_seam_duplicate_is_not_a_cell(self):
        a=self.objects()[0].region
        from dataclasses import replace
        with self.assertRaises(ValueError):validate_hierarchy_selection([a,replace(a,instance=100)])

    def test_mask_and_bbox_are_distinct(self):
        from dataclasses import replace
        a=self.objects()[0];b=self.objects(1)[0]
        mask=b.region.mask.copy();mask[:,1,:]=False
        b=replace(b,region=replace(b.region,mask=mask))
        box=pair_features(a,b,'B0');actual=pair_features(a,b,'M0')
        self.assertEqual(box[1],1.);self.assertLess(actual[7],1.)
        self.assertEqual(pair_features(None,b,'M0').sum(),0)
        self.assertIsNone(a.confidence)

    def test_division_union_and_persistent_separation(self):
        from dataclasses import replace
        a,b=self.objects(1);parent=replace(a,region=replace(a.region,time=0))
        fa,fb=self.objects(2)
        result=division_features(parent,a,b,fa,fb)
        self.assertIsNotNone(result['persistent_separation_um'])
        self.assertEqual(result,division_features(parent,b,a,fb,fa))

    def test_unknown_second_daughter_is_not_negative(self):
        nodes=np.array([[1,0,0,0,0],[2,1,0,0,0],[3,1,0,0,0],[4,0,0,0,0]])
        pairs=np.array([[0,1],[0,2],[3,1]])
        y=supported_labels(pairs,nodes,np.array([[1,10],[2,11]]),np.array([[10,11]]))
        np.testing.assert_array_equal(y,[1,-1,0])

    def test_ultrack_legal_fork_and_gap_rejection(self):
        parent=self.objects(0)[0].region;a,b=[r.region for r in self.objects(1)]
        result=export_regions({10:parent,20:a,30:b},{20:10,30:10})
        np.testing.assert_array_equal(result['edges'],[[10,20],[10,30]])
        from dataclasses import replace
        with self.assertRaises(ValueError):export_regions({10:parent,20:replace(a,time=2)},{20:10})

    def test_hierarchy_survival_not_centroid_proximity(self):
        from dataclasses import replace
        r=self.objects()[0].region;m=r.mask.copy();m[0,0,0]=False
        result=survival([r],[replace(r,mask=m)])
        self.assertEqual(result['exact_survivors'],0)

    def test_native_mapping_rejects_unrelated_old_scores(self):
        from segmentation_tracking_v6.common import Blocked
        with self.assertRaises(Blocked):recompute_native_evidence([],[],None)
        with self.assertRaises(ValueError):
            recompute_native_evidence([self.objects()[0].region],[],lambda r,p:([],{'geometry_sha256':'wrong'}))

    def test_provider_network_denial_precedes_resolution(self):
        import subprocess,sys
        code="""from segmentation_tracking_v6.network import deny_external_network
import socket
deny_external_network()
try: socket.getaddrinfo('198.51.100.1',443)
except PermissionError: pass
else: raise AssertionError('External address was permitted')
"""
        subprocess.run([sys.executable,'-c',code],check=True)

if __name__=='__main__':unittest.main()
