"""Synthetic integrity tests; real model/Ultrack/scorer integration runs locally."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from mask_contracts import (Key, Instance, extract_instances, describe, canonical_partition,
    bbox_iou, overlap, union_instance, division_features, grid_to_native,
    sparse_link_label, storage_gib, score_for_division_counts)
from preflight import inspect


def inst(lo=(0,0,0), shape=(2,2,2), *, frame=0, label=1, grid='native', mask=None):
    return Instance(Key('sample', frame, 'model', label), lo,
                    np.ones(shape, bool) if mask is None else mask, grid)

class MaskTests(unittest.TestCase):
    def test_key_frame_scope(self):
        self.assertNotEqual(Key('x',0,'m',1),Key('x',1,'m',1))
    def test_key_model_scope(self):
        self.assertNotEqual(Key('x',0,'a',1),Key('x',0,'b',1))
    def test_background_not_instance(self):
        with self.assertRaises(ValueError): Key('x',0,'m',0)
    def test_invalid_frame(self):
        with self.assertRaises(ValueError): Key('x',True,'m',1)
    def test_empty_mask_is_unknown_not_shape(self):
        with self.assertRaises(ValueError): inst(mask=np.zeros((2,2,2),bool))
    def test_multilabel_mask_rejected(self):
        with self.assertRaises(ValueError): inst(mask=np.ones((2,2,2),int))
    def test_invalid_origin(self):
        with self.assertRaises(ValueError): inst(lo=(-1,0,0))
    def test_support_copy_is_immutable(self):
        m=np.ones((2,2,2),bool);a=inst(mask=m);m[:]=False
        self.assertEqual(a.count,8)
        with self.assertRaises(ValueError): a.mask[0,0,0]=False
    def test_half_open_bbox(self):
        a=inst(lo=(3,4,5));np.testing.assert_array_equal(a.hi,[5,6,7])
    def test_anisotropic_volume(self):
        self.assertAlmostEqual(describe(inst(),[2,1,.5])['volume_um3'],8)
    def test_covariance_units(self):
        c=np.array(describe(inst(),[2,1,.5])['covariance_um2'])
        np.testing.assert_allclose(np.diag(c),[1,.25,.0625])
    def test_bad_spacing(self):
        for s in ([0,1,1],[1,float('nan'),1]):
            with self.assertRaises(ValueError): describe(inst(),s)
    def test_extract_bbox_and_roundtrip(self):
        x=np.zeros((8,9,10),int);x[3:5,4:7,5:9]=17
        a=extract_instances(x,'x',0,'m')[0]
        self.assertEqual(a.count,24);self.assertEqual(a.lo,(3,4,5))
        y=np.zeros_like(x);y[3:5,4:7,5:9]=a.mask*17
        np.testing.assert_array_equal(x,y)
    def test_empty_frame_is_valid(self):
        self.assertEqual(extract_instances(np.zeros((3,3,3),int),'x',0,'m'),[])
    def test_bad_label_array(self):
        with self.assertRaises(ValueError): extract_instances(np.ones((3,3,3),float),'x',0,'m')
    def test_canonical_under_arbitrary_ids(self):
        a=np.array([[[0,8,8,3],[3,0,9,9]]]);b=np.array([[[0,1,1,99],[99,0,2,2]]])
        np.testing.assert_array_equal(canonical_partition(a),canonical_partition(b))
    def test_canonical_retains_background(self):
        self.assertEqual(canonical_partition(np.array([0,9,0,4])).tolist(),[0,1,0,2])
    def test_iou_identity(self):
        a=inst();self.assertEqual(overlap(a,a)['iou'],1);self.assertEqual(bbox_iou(a,a),1)
    def test_iou_partial(self):
        a,b=inst(),inst(lo=(1,0,0),frame=1)
        self.assertAlmostEqual(overlap(a,b)['iou'],1/3)
    def test_shift_source_direction(self):
        a,b=inst(),inst(lo=(2,0,0),frame=1)
        self.assertEqual(overlap(a,b)['iou'],0)
        self.assertEqual(overlap(a,b,(2,0,0))['iou'],1)
        self.assertEqual(overlap(a,b,(-2,0,0))['iou'],0)
    def test_no_roll_wrap(self):
        a,b=inst(),inst(lo=(10,0,0),frame=1)
        self.assertEqual(overlap(a,b,(-2,0,0))['intersection_voxels'],0)
    def test_different_registered_grid_rejected(self):
        with self.assertRaises(ValueError): overlap(inst(),inst(grid='pooled'))
    def test_fractional_shift_rejected_in_integer_reference(self):
        with self.assertRaises(ValueError): overlap(inst(),inst(),(.5,0,0))
    def test_same_centroid_bbox_different_mask(self):
        a=np.zeros((3,3,3),bool);a[0,0,0]=a[2,2,2]=True
        b=np.zeros((3,3,3),bool);b[0,2,0]=b[2,0,2]=True
        x,y=inst(mask=a),inst(frame=1,mask=b)
        self.assertEqual(describe(x)['centroid_grid_zyx'],describe(y)['centroid_grid_zyx'])
        self.assertEqual(bbox_iou(x,y),1)
        self.assertEqual(overlap(x,y)['iou'],0)
    def test_interior_representative_for_hollow_object(self):
        m=np.ones((3,3,3),bool);m[1,1,1]=False;a=inst(mask=m)
        d=describe(a);self.assertEqual(d['centroid_grid_zyx'],[1,1,1])
        self.assertNotEqual(d['interior_nearest_centroid_grid_zyx'],[1,1,1])
    def test_union_not_double_counting(self):
        a,b=inst(frame=1),inst(lo=(1,0,0),frame=1,label=2)
        self.assertEqual(union_instance(a,b).count,12)
    def test_union_same_frame_required(self):
        with self.assertRaises(ValueError): union_instance(inst(),inst(frame=1))
    def test_union_memory_cap(self):
        with self.assertRaises(MemoryError): union_instance(inst(),inst(lo=(20,0,0),label=2),max_bbox_voxels=10)
    def test_division_union_not_pairwise_iou(self):
        p=inst(shape=(2,2,4));a=inst(shape=(2,2,2),frame=1)
        b=inst(lo=(0,0,2),frame=1,label=2)
        f=division_features(p,a,b)
        self.assertEqual(f['iou'],1);self.assertEqual(f['volume_ratio'],1)
        self.assertEqual(overlap(p,a)['iou'],.5)
    def test_division_rejects_overlapping_daughters(self):
        with self.assertRaises(ValueError): division_features(inst(),inst(frame=1),inst(frame=1,label=2))
    def test_division_rejects_nonconsecutive(self):
        with self.assertRaises(ValueError): division_features(inst(),inst(frame=2),inst(lo=(0,0,3),frame=2,label=2))
    def test_affine_has_explicit_center_offset(self):
        a=np.diag([1.,4.,4.,1.]);a[:3,3]=[0,1.5,1.5]
        np.testing.assert_allclose(grid_to_native([[2,3,4]],a),[[2,13.5,17.5]])
    def test_bad_affine(self):
        with self.assertRaises(ValueError): grid_to_native([[0,0,0]],np.zeros((4,4)))

class SupervisionAndBudgetTests(unittest.TestCase):
    def test_positive_link(self): self.assertEqual(sparse_link_label(1,2,[(1,2)]),1)
    def test_unobserved_second_daughter_unknown(self): self.assertEqual(sparse_link_label(1,None,[(1,2)]),-1)
    def test_incoming_contradiction(self): self.assertEqual(sparse_link_label(3,2,[(1,2)]),0)
    def test_two_children_exhaust_outdegree(self): self.assertEqual(sparse_link_label(1,4,[(1,2),(1,3)]),0)
    def test_unknown_pair(self): self.assertEqual(sparse_link_label(None,None,[(1,2)]),-1)
    def test_raw_storage(self): self.assertAlmostEqual(storage_gib(199,100,[64,256,256]),155.46875)
    def test_storage_rejects_negative(self):
        with self.assertRaises(ValueError): storage_gib(-1,100,[64,256,256])
    def test_recorded_score(self):
        self.assertAlmostEqual(score_for_division_counts(.9228682178819851,29,92,122),.934802374260586)
    def test_conditional_target(self):
        self.assertGreaterEqual(score_for_division_counts(.9228682178819851,66,92,85),.95)
    def test_score_rejects_bad_counts(self):
        with self.assertRaises(ValueError): score_for_division_counts(.9,1.,2,3)
    def test_current_budget_sum(self):
        c=json.loads((Path(__file__).parent/'config.json').read_text())
        self.assertEqual(sum(c['variant_allocation'].values()),c['max_complete_new_configurations'])

class PreflightTests(unittest.TestCase):
    def fixture(self,root):
        h=root/'handover/instance-tracking-v6';h.mkdir(parents=True)
        b=b'example';(h/'a.txt').write_bytes(b)
        (h/'MANIFEST.json').write_text(json.dumps({'files':[{'path':'a.txt','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}]}))
        return h
    def test_valid_manifest_readonly(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);h=self.fixture(root);before=set(root.rglob('*'))
            r=inspect(root,root/'new_output');self.assertTrue(r['handover_valid'])
            self.assertEqual(before,set(root.rglob('*')))
    def test_payload_drift(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);h=self.fixture(root);(h/'a.txt').write_text('changed')
            self.assertFalse(inspect(root,root/'out')['handover_valid'])
    def test_protected_old_output(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root)
            r=inspect(root,Path('/kaggle/working/cell-tracking/image-native-tracking-v5/new'))
            self.assertFalse(r['handover_valid'])

if __name__=='__main__': unittest.main()
