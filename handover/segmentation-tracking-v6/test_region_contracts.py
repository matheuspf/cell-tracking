import unittest
from dataclasses import replace
import numpy as np
from region_contracts import (Region, from_labels, mask_iou, bbox_iou,
    intersection_voxels, translate, daughter_union_features, validate_edges)


def cube(offset=(0,0,0), instance=1, time=0):
    b=tuple(offset)+tuple(x+2 for x in offset)
    return Region('x',time,'test',instance,'g',b,np.ones((2,2,2),bool),(2.,1.,1.))


class RegionContracts(unittest.TestCase):
    def test_volume(self): self.assertEqual(cube().volume_um3,16.)
    def test_center_index_convention(self): np.testing.assert_array_equal(cube().center(),[.5,.5,.5])
    def test_physical_covariance(self): np.testing.assert_allclose(cube().covariance_um2(),np.diag([1.,.25,.25]))
    def test_identity_iou(self): self.assertEqual(mask_iou(cube(),cube()),1.)
    def test_disjoint_iou(self): self.assertEqual(mask_iou(cube(),cube((3,0,0))),0.)
    def test_touching_halfopen(self): self.assertEqual(bbox_iou(cube(),cube((2,0,0))),0.)
    def test_partial_iou(self): self.assertAlmostEqual(mask_iou(cube(),cube((1,0,0))),1/3)
    def test_intersection(self): self.assertEqual(intersection_voxels(cube(),cube((1,0,0))),4)
    def test_empty_mask(self):
        with self.assertRaises(ValueError): replace(cube(),mask=np.zeros((2,2,2),bool))
    def test_wrong_shape(self):
        with self.assertRaises(ValueError): replace(cube(),mask=np.ones((3,2,2),bool))
    def test_float_mask(self):
        with self.assertRaises(ValueError): replace(cube(),mask=np.ones((2,2,2),float))
    def test_bad_spacing(self):
        for s in [(0.,1.,1.),(float('nan'),1.,1.),(1.,1.)]:
            with self.subTest(s=s), self.assertRaises(ValueError): replace(cube(),spacing=s)
    def test_float_bbox(self):
        with self.assertRaises(ValueError): replace(cube(),bbox=(0.,0.,0.,2.,2.,2.))
    def test_readonly_mask(self):
        with self.assertRaises(ValueError): cube().mask[0,0,0]=False
    def test_copy_input(self):
        m=np.ones((2,2,2),bool); r=replace(cube(),mask=m); m[:]=False
        self.assertEqual(r.voxel_count,8)
    def test_missing_label(self):
        with self.assertRaises(ValueError): from_labels(np.zeros((3,3,3),int),1)
    def test_wrong_labels_dtype(self):
        with self.assertRaises(ValueError): from_labels(np.ones((3,3,3),float),1)
    def test_no_time_axis_as_depth(self):
        with self.assertRaises(ValueError): from_labels(np.ones((2,3,3,3),int),1)
    def test_extract_halfopen(self):
        a=np.zeros((4,5,6),int); a[1:3,2:4,3:5]=9
        r=from_labels(a,9); self.assertEqual(r.bbox,(1,2,3,3,4,5)); self.assertEqual(r.voxel_count,8)
    def test_dataset_mismatch(self):
        with self.assertRaises(ValueError): mask_iou(cube(),replace(cube(),dataset='y'))
    def test_grid_mismatch(self):
        with self.assertRaises(ValueError): mask_iou(cube(),replace(cube(),grid_id='other'))
    def test_spacing_mismatch(self):
        with self.assertRaises(ValueError): mask_iou(cube(),replace(cube(),spacing=(1.,1.,1.)))
    def test_time_and_provider_identity(self):
        self.assertNotEqual(cube().key,replace(cube(),time=1).key)
        self.assertNotEqual(cube().key,replace(cube(),provider='other').key)
    def test_same_bbox_different_mask(self):
        m=np.indices((2,2,2)).sum(0)%2==0
        a=replace(cube(),mask=m); b=replace(cube(instance=2),mask=~m)
        self.assertEqual(bbox_iou(a,b),1.); self.assertEqual(mask_iou(a,b),0.)
        np.testing.assert_array_equal(a.center(),b.center())
    def test_backward_shift(self):
        a=cube(); b=cube((3,0,0),time=1)
        self.assertEqual(mask_iou(a,b),0.); self.assertEqual(mask_iou(a,translate(b,(-3,0,0))),1.)
    def test_no_wrap(self):
        a=translate(cube(),(-9,0,0)); self.assertEqual(a.bbox[0],-9); self.assertEqual(a.voxel_count,8)
    def test_no_float_translation(self):
        with self.assertRaises(ValueError): translate(cube(),(.5,0,0))
    def test_inside_concavity(self):
        m=np.zeros((3,3,3),bool); m[0,:,:]=True; m[2,:,:]=True
        r=Region('x',0,'test',1,'g',(0,0,0,3,3,3),m,(1.,1.,1.))
        self.assertFalse(m[tuple(r.center().astype(int))]); self.assertTrue(m[tuple(r.center(inside=True))])
    def test_daughter_union_symmetric(self):
        p=Region('x',1,'test',3,'g',(0,0,0,4,2,2),np.ones((4,2,2),bool),(2.,1.,1.))
        a=cube(instance=1,time=1); b=cube((2,0,0),instance=2,time=1)
        x=daughter_union_features(p,a,b); self.assertEqual(x,daughter_union_features(p,b,a))
        self.assertEqual(x['union_iou'],1.); self.assertEqual(x['daughter_volume_ratio'],1.)
    def test_duplicate_daughter(self):
        with self.assertRaises(ValueError): daughter_union_features(cube(),cube(),cube())
    def test_wrong_frame_daughters(self):
        with self.assertRaises(ValueError): daughter_union_features(cube(),cube(),cube(instance=2,time=1))
    def test_hash_mask_content(self):
        a=cube(); m=np.ones((2,2,2),bool); m[0,0,0]=False
        self.assertNotEqual(a.mask_sha256(),replace(a,mask=m).mask_sha256())
    def test_valid_division(self): self.assertTrue(validate_edges({1:0,2:1,3:1},[(1,2),(1,3)]))
    def test_merge(self):
        with self.assertRaises(ValueError): validate_edges({1:0,2:0,3:1},[(1,3),(2,3)])
    def test_gap(self):
        with self.assertRaises(ValueError): validate_edges({1:0,2:2},[(1,2)])
    def test_duplicate_edge(self):
        with self.assertRaises(ValueError): validate_edges({1:0,2:1},[(1,2),(1,2)])
    def test_absent_endpoint(self):
        with self.assertRaises(ValueError): validate_edges({1:0,2:1},[(1,3)])
    def test_three_children(self):
        with self.assertRaises(ValueError): validate_edges({1:0,2:1,3:1,4:1},[(1,2),(1,3),(1,4)])

if __name__=='__main__': unittest.main()
