import copy
import numpy as np
import pytest
import torch

from tests.pipeline_error_training.test_contracts import fixture
from division_generalization_v2.actions import EventBank,canonical_actions,apply_decisions
from division_generalization_v2.features import build
from division_generalization_v2.model import ActionModel,loss,sample_maps
from division_generalization_v2.contracts import lr_factor
from division_generalization_v2.labels import Counterfactual,COUNT_KEYS,SourceUtility


def packed():
    nodes,edges,native=fixture()
    native['edge_features'][:,23]=10000.
    bank=EventBank(nodes,edges,native,[1.,1.,1.])
    rows,_=canonical_actions(bank,0)
    arrays=build(bank,0,rows)
    b={k:torch.from_numpy(v) for k,v in arrays.items() if k!='query_nodes'}
    return nodes,edges,bank,rows,b


def test_zero_score_active_decoder_and_native_logit_does_not_force_fork():
    nodes,edges,bank,rows,b=packed()
    model=ActionModel(image=False)
    scores=model([b])[0]['gain']
    assert torch.count_nonzero(scores)==0
    out,receipt=apply_decisions(nodes,edges,[d for d,_ in rows],scores.detach().numpy(),max_fraction=1.)
    np.testing.assert_array_equal(out,edges)
    assert receipt.get('accepted_actions',0)==0
    assert sum(d.kind=='keep' for d,_ in rows)==1
    assert len({d.key for d,_ in rows})==len(rows)


def test_unknown_loss_zero_and_negative_only_rejection_gradient():
    _,_,_,rows,b=packed()
    n=len(rows)
    b.update(supported=torch.zeros(n,dtype=torch.bool),utility=torch.zeros(n),identity=torch.full((n,),-1))
    m=ActionModel(image=False)
    out=m([b])[0];v,_=loss(out,b);v.backward()
    assert float(v)==0 and all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in m.parameters())
    m.zero_grad(set_to_none=True)
    b['supported']=~b['keep'];b['utility']=-torch.ones(n)
    out=m([b])[0];out['gain'].retain_grad();v,_=loss(out,b);v.backward()
    assert float(v)>0
    assert (out['gain'].grad[~b['keep']]>0).all()
    assert m.score.weight.grad.norm()>0


def test_daughter_permutation_and_training_inference_scores():
    torch.manual_seed(5)
    _,_,_,rows,b=packed()
    model=ActionModel(image=True)
    torch.nn.init.normal_(model.score.weight,std=.1)
    z=torch.randn(b['incidence'].shape[-1],128)
    expected=model.scores(b,z)['gain']
    swapped=copy.deepcopy(b);swapped['event_index']=b['event_index'][:,[0,2,1]]
    torch.testing.assert_close(expected,model.scores(swapped,z)['gain'],rtol=0,atol=0)
    model.eval()
    torch.testing.assert_close(expected,model.scores(b,z)['gain'],rtol=0,atol=0)


def test_interpolation_outside_mask_and_gradient():
    torch.manual_seed(7)
    maps=torch.randn(2,4,3,5,5,requires_grad=True)
    grid=torch.rand(2,1,23,1,3)*3-1.5
    actual=sample_maps(maps,grid)
    expected=torch.nn.functional.grid_sample(maps,grid,align_corners=True)[:,:,0,:,0]
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    actual.sum().backward();a=maps.grad.clone();maps.grad.zero_()
    expected.sum().backward();torch.testing.assert_close(a,maps.grad,rtol=1e-5,atol=1e-6)
    far=torch.full_like(grid,5.)
    assert torch.count_nonzero(sample_maps(maps,far))==0


def test_complete_counterfactual_loss_and_global_fork_assignment():
    from annotation_selection.metric_adapter import evaluate_graph,aggregate
    nodes,edges,bank,rows,b=packed()
    baseline,_,_=evaluate_graph('fixture',nodes,edges,nodes,edges,[1.,1.,1.],len(nodes))
    counter=Counterfactual(nodes,edges,nodes,edges,[1.,1.,1.],baseline)
    for d,_ in rows:
        label=counter.label(d)
        actual,_=apply_decisions(nodes,edges,[d],[1.],max_fraction=10.)
        score,_,_=evaluate_graph('fixture',nodes,actual,nodes,edges,[1.,1.,1.],len(nodes))
        assert label['delta']==[score[k]-baseline[k] for k in COUNT_KEYS]
        utility=SourceUtility([baseline])
        expected=aggregate([score],['fixture'])['score']-aggregate([baseline],['fixture'])['score']
        assert utility.value(label['delta'],baseline)==pytest.approx(expected,abs=1e-12)
        if d.kind=='division':
            assert label['lost_supported_edges']>0
            assert utility.value(label['delta'],baseline)<0


def test_schedule_and_sampling_independent_of_component_ids():
    from division_generalization_v2.prepare import sampled_anchor
    assert lr_factor(0)==1/128
    assert lr_factor(127)==1
    assert lr_factor(3276)==1
    assert lr_factor(4095)==pytest.approx(.1)
    assert lr_factor(4096)==.1
    for i in range(20):
        assert sampled_anchor('clip',i)==sampled_anchor('clip',i)
    assert 35<sum(sampled_anchor('clip',i) for i in range(1000))<95


def test_stationary_physical_scene_and_missing_masks():
    from division_generalization_v2.scenes import crop
    class Images:
        shape=(3,24,96,96)
        quantiles={0:(0.,255.),1:(0.,255.),2:(0.,255.)}
        def raw(self,t):
            if not 0<=t<3:return None
            z,y,x=np.indices(self.shape[1:]);return (x+2*y+3*z).astype(np.float32)
    p=crop(Images(),0,[12,48,48])
    assert p.shape==(7,2,2,16,64,64)
    assert np.count_nonzero(p[:3])==0
    assert np.count_nonzero(p[-1])==0
    np.testing.assert_array_equal(p[3],p[4])
    assert p[3,0,1].min()==255
    assert (p[3,1,1]==0).any()


def test_no_trainable_feature_cache_across_forwards():
    torch.set_num_threads(1)
    _,_,_,_,b=packed()
    model=ActionModel(image=True)
    torch.nn.init.normal_(model.score.weight,std=.1)
    scene=torch.rand(1,7,2,2,16,64,64)
    one=model([b],scene)[0]['gain'];one.sum().backward()
    assert model.encoder.spatial[0].weight.grad.norm()>0
    with torch.no_grad():model.encoder.spatial[0].weight.add_(.01)
    two=model([b],scene)[0]['gain']
    assert model.encoder.forward_calls==2
    assert not torch.equal(one,two)
