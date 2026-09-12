"""Synthetic helper checks, not executed detector/tracker/official score tests."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

c=load('object_contract');p=load('preflight')

def test_physical_geometry_and_half_open_box():
    a=np.ones((2,3,4),bool)
    o=c.extract(a,origin=(4,8,12),spacing=(4.,1.,1.))
    assert o.bbox==(4,8,12,6,11,16)
    assert o.volume_vox==24 and o.volume_um3==96
    assert o.centroid_vox==(4.5,9.,13.5)
    assert o.centroid_um==(18.,9.,13.5)
    np.testing.assert_allclose(np.diag(o.covariance_um2),[4.,2/3,1.25])

def test_owned_tight_crop():
    a=np.zeros((4,5,6),bool);a[1:3,2:4,3:5]=True
    o=c.extract(a);a[:]=False
    assert o.bbox==(1,2,3,3,4,5) and o.mask.sum()==8
    with pytest.raises(ValueError):o.mask[:]=False

@pytest.mark.parametrize('mask',[np.zeros((2,2,2),bool),np.ones((2,2),bool),np.ones((2,2,2),np.uint16)])
def test_reject_empty_2d_and_integer_mask(mask):
    with pytest.raises(ValueError):c.extract(mask)

@pytest.mark.parametrize('spacing',[(0,1,1),(1,-1,1),(1,float('nan'),1),(1,1)])
def test_bad_spacing(spacing):
    with pytest.raises(ValueError):c.extract(np.ones((1,1,1),bool),spacing=spacing)

@pytest.mark.parametrize('origin',[(1.2,0,0),(-1,0,0),(True,False,False),(1,2)])
def test_bad_origin(origin):
    with pytest.raises(ValueError):c.extract(np.ones((1,1,1),bool),origin=origin)

def test_same_bbox_different_shape():
    a=np.zeros((2,2,2),bool);a[0,0,0]=a[1,1,1]=True
    b=np.zeros_like(a);b[0,1,0]=b[1,0,1]=True
    x,y=c.extract(a),c.extract(b)
    assert c.bbox_iou(x.bbox,y.bbox)==1 and c.mask_iou(x,y)==0

def test_motion_restores_disjoint_overlap():
    x=c.extract(np.ones((2,2,2),bool));y=c.extract(np.ones((2,2,2),bool),origin=(3,0,0))
    assert c.mask_iou(x,y)==0 and c.mask_iou(x,y,(3,0,0))==1
    assert c.mask_iou(y,x,(-3,0,0))==1

def test_mask_partial_overlap():
    x=c.extract(np.ones((2,2,2),bool));y=c.extract(np.ones((2,2,2),bool),origin=(1,0,0))
    assert c.mask_iou(x,y)==pytest.approx(1/3)

def test_grid_mismatch_and_float_shift():
    x=c.extract(np.ones((1,1,1),bool));y=c.extract(np.ones((1,1,1),bool),spacing=(2,1,1))
    with pytest.raises(ValueError):c.mask_iou(x,y)
    with pytest.raises(ValueError):c.mask_iou(x,x,(.5,0,0))

def test_touching_box_no_overlap():
    assert c.bbox_iou([0,0,0,1,1,1],[1,0,0,2,1,1])==0
    with pytest.raises(ValueError):c.bbox_iou([0,0,0,0,1,1],[0,0,0,1,1,1])

def test_coarse_split_children_can_coexist():
    assert c.validate_selection([2,3],[1,2,3],[(1,2),(1,3)])
    with pytest.raises(ValueError):c.validate_selection([1,2],[1,2,3],[(1,2),(1,3)])

@pytest.mark.parametrize('selected',[[4],[1,1]])
def test_bad_hypothesis_selection(selected):
    with pytest.raises(ValueError):c.validate_selection(selected,[1,2,3],[(1,2)])

def nodes():return np.array([[1,0,1,2,3],[2,1,1,3,3],[3,1,1,1,3]],dtype=np.int64)

def test_real_division_and_empty_graph():
    assert c.validate_graph(nodes(),np.array([[1,2],[1,3]]),(3,5,5,5))
    assert c.validate_graph(np.empty((0,5),np.int64),np.empty((0,2),np.int64),(3,5,5,5))

@pytest.mark.parametrize('edges',[[[2,3]],[[1,99]],[[1,2],[1,2]],[[2,1]]])
def test_bad_temporal_edges(edges):
    with pytest.raises(ValueError):c.validate_graph(nodes(),np.array(edges),(3,5,5,5))

def test_more_than_two_daughters_and_merge():
    n=np.vstack([nodes(),[4,1,2,2,3]])
    with pytest.raises(ValueError):c.validate_graph(n,np.array([[1,2],[1,3],[1,4]]),(3,5,5,5))
    n=np.vstack([nodes(),[4,0,2,2,3]])
    with pytest.raises(ValueError):c.validate_graph(n,np.array([[1,2],[4,2]]),(3,5,5,5))

def test_duplicate_id_and_out_of_bounds():
    for index,value in (([1,0],1),([1,4],5)):
        n=nodes();n[tuple(index)]=value
        with pytest.raises(ValueError):c.validate_graph(n,np.array([[1,2]]),(3,5,5,5))

def test_changed_coordinate_full_output_roundtrip(tmp_path):
    n=nodes();n[1,3]=4;e=np.array([[1,2],[1,3]])
    path=tmp_path/'graph.npz';np.savez(path,nodes=n,edges=e)
    with np.load(path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['nodes'],n)
        assert z['nodes'][1,0]==2 and z['nodes'][1,3]==4
        assert c.validate_graph(z['nodes'],z['edges'],(3,5,5,5))

def test_disk_budget_and_plan():
    assert c.dense_label_bytes(clips=1)==1677721600
    assert c.dense_label_bytes()/2**30==310.9375
    assert c.dense_label_bytes(detectors=2)==2*c.dense_label_bytes()
    plan=json.loads((HERE/'experiments.json').read_text())
    assert p.validate_plan(plan)['stages']==10
    plan['stages'][0]['status']='passed'
    with pytest.raises(ValueError):p.validate_plan(plan)

def test_read_only_preflight_foreign_workdir(tmp_path):
    result=subprocess.run([sys.executable,'-s',str(HERE/'scripts/preflight.py')],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert not report['microscopy_or_gt_opened'] and not report['gpu_experiments_executed']
    assert not list(tmp_path.iterdir())
