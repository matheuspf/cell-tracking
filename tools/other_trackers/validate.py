"""Real upstream preprocessing/model parity and independent decoder contracts."""
from __future__ import annotations

import argparse
from itertools import product

import numpy as np

from .common import CONFIG, ROOT, RESULTS, Images, Lease, bank, clips, read, sha, write


def flow_contracts():
    from .decode import solve_flow,cellect_edges
    rng=np.random.default_rng(42)
    checks=[]
    for allow in (True,False):
        for _ in range(8):
            costs=rng.uniform(-2,2,4)
            nodes=[dict(id=i,features=[[2.],[0.]],appearanceFeatures=[[0.],[float(rng.uniform(0,2))]],
                        disappearanceFeatures=[[0.],[float(rng.uniform(0,2))]],divisionFeatures=[[0.],[float(rng.uniform(-1,2))]]) for i in range(4)]
            pairs=[(0,2),(0,3),(1,2),(1,3)]
            graph=dict(segmentationHypotheses=nodes,linkingHypotheses=[dict(src=a,dest=b,features=[[0.],[float(c)]]) for (a,b),c in zip(pairs,costs)])
            best=float("inf")
            for mask in product((False,True),repeat=8):
                selected=mask[:4];edges=[edge for edge,on in zip(pairs,mask[4:]) if on]
                incoming=np.zeros(4,int);outgoing=np.zeros(4,int)
                for a,b in edges:outgoing[a]+=1;incoming[b]+=1
                if any(not selected[a] or not selected[b] for a,b in edges):continue
                if incoming.max()>1 or outgoing.max()>(2 if allow else 1):continue
                cost=sum(costs[j] for j,on in enumerate(mask[4:]) if on)
                for i,h in enumerate(nodes):
                    if not selected[i]:cost+=2
                    else:
                        if incoming[i]==0:cost+=h["appearanceFeatures"][1][0]
                        if outgoing[i]==0:cost+=h["disappearanceFeatures"][1][0]
                        if outgoing[i]==2:cost+=h["divisionFeatures"][1][0]
                best=min(best,cost)
            _,_,result=solve_flow(graph,allow_divisions=allow,time_limit=15)
            assert abs(result["objective"]-best)<1e-7
            checks.append(dict(divisions=allow,exhaustive_objective=best,solver_objective=result["objective"]))
    # A dominant no-match channel prevents links even when a neighbor is close.
    test=dict(source_ids=np.array([0,1]),target_ids=np.array([[2,3,4,5,6],[2,3,4,5,6]]),
              similarity=np.array([[.9,.8,.1,.1,.1,.99],[.9,.8,.1,.1,.1,.01]]),division=np.array([.99,.99]),
              distances=np.full((2,5),2.),source_sizes=np.full(2,8.))
    assert set(map(tuple,cellect_edges(test,True)))=={(1,2),(1,3)}
    assert set(map(tuple,cellect_edges(test,False)))=={(1,2)}
    test["similarity"][0,5]=.01
    edges=cellect_edges(test,True)
    assert len(np.unique(edges[:,1]))==len(edges)
    write(RESULTS/"flow-validation.json",dict(exhaustive_cases=checks,cellect_no_match_division_and_unique_parent=True))
    print("FLOW CONTRACTS",len(checks),flush=True)


def organoid_parity():
    from .organoid import load_models,experiment,resize_patch,coord_inputs,calibrated
    models=load_models()
    import torch
    import keras
    from keras.src.backend.torch.core import device_scope
    from organoid_tracker.core import TimePoint
    from organoid_tracker.neural_network.link_detection_cnn.link_predictor import _split_into_patches as lp
    from organoid_tracker.neural_network.division_detection_cnn.division_predictor import _split_into_patches as dp
    checks=[]
    for clip in (clips()[0],clips()[3]):
        data=bank(clip["dataset"]);images=Images(clip["dataset"])
        ex,positions=experiment(data,clip["spacing_um"])
        from scipy.spatial import cKDTree
        source_indices=np.flatnonzero(data["nodes"][:,1]==0)[:4]
        target_indices=np.flatnonzero(data["nodes"][:,1]==1)
        source=data["nodes"][source_indices,0]
        _,nearest=cKDTree(data["positions"][target_indices]*clip["spacing_um"]).query(data["positions"][source_indices]*clip["spacing_um"],k=2)
        pair_ids=[(int(a),int(data["nodes"][target_indices[j],0])) for a,row in zip(source,nearest) for j in row]
        scale=tuple(np.array(clip["spacing_um"])/read(CONFIG)["organoid"]["model_spacing_zyx_um"])
        for mode,model in zip(("links","divisions"),models):
            capture={}
            if mode=="links":
                patches=list(lp(images,TimePoint(0),[(positions[a],positions[b]) for a,b in pair_ids],
                                model.time_window,patch_shape_zyx_px=model.patch_shape_zyx,scale_factors_zyx=scale,intensity_quantiles=(.01,.99)))
                for patch in patches:ex.links.add_link(patch.position_a,patch.position_b)
                class Capture:
                    def predict(self,value,verbose=0):
                        capture.update({k:keras.ops.convert_to_numpy(v) for k,v in value.items()})
                        return np.linspace(.1,.9,len(patches),dtype=np.float32)[:,None]
                fake=model._replace(keras_model=Capture());fake._predict_batch(ex,patches)
                a=np.stack([resize_patch(p.array_a,model.patch_shape_zyx) for p in patches])
                b=np.stack([resize_patch(p.array_b,model.patch_shape_zyx) for p in patches])
                distance=np.array([p.distance_zyx_px for p in patches])
                actual=coord_inputs(a,b,distance)
                error=max(float(np.max(abs(actual[k].numpy()-v))) for k,v in capture.items())
                assert error==0.
                cpu_input=actual
            else:
                patches=list(dp(images,TimePoint(0),[positions[int(a)] for a in source],model.time_window,
                                patch_shape_zyx_px=model.patch_shape_zyx,scale_factors_zyx=scale,intensity_quantiles=(.01,.99)))
                class Capture:
                    def __call__(self,value,training=False):
                        capture["array"]=value
                        return torch.from_numpy(np.linspace(.1,.9,len(patches),dtype=np.float32)[:,None])
                fake=model._replace(keras_model=Capture());fake._predict_batch(ex,patches)
                cpu_input=torch.from_numpy(np.stack([resize_patch(p.array,model.patch_shape_zyx) for p in patches]))
                error=float(np.max(abs(cpu_input.numpy()-capture["array"])));assert error==0.
            raw=np.linspace(.1,.9,len(patches),dtype=np.float32)
            probs,penalties=calibrated(raw,model)
            for i,p in enumerate(patches):
                if mode=="links":stored=ex.links.get_link_data(p.position_a,p.position_b,"link_probability")
                else:stored=ex.positions.get_position_data(p.position,"division_probability")
                assert abs(stored-probs[i])<2e-7
            with device_scope("cpu"),torch.inference_mode():
                cpu=model.keras_model(cpu_input,training=False).numpy()
            lease=Lease([model.keras_model]);lease.acquire()
            try:
                with device_scope("cuda:0"),torch.inference_mode():
                    gpu_input={k:v.cuda() for k,v in cpu_input.items()} if isinstance(cpu_input,dict) else cpu_input.cuda()
                    output=model.keras_model(gpu_input,training=False)
                    gpu=output.cpu().numpy();del gpu_input,output
            finally:lease.release()
            np.testing.assert_allclose(cpu,gpu,atol=2e-5,rtol=2e-4)
            checks.append(dict(dataset=clip["dataset"],model=mode,patches=len(patches),upstream_input_max_error=error,
                               cpu_cuda_max_probability_error=float(np.max(abs(cpu-gpu))),raw_probability_min=float(cpu.min()),raw_probability_max=float(cpu.max())))
    write(RESULTS/"organoid-validation.json",dict(checks=checks,real_images=True,annotation_reads=False))
    print("ORGANOID PARITY",checks,flush=True)


def cellect_parity():
    import torch
    from .cellect import load_models,sample_outputs,tiles,tile_owners
    model,matcher,sort_feature=load_models();torch.set_num_threads(2)
    # The sampled feature lookup is independently compared to the exact released
    # channel-transpose expression on synthetic non-symmetric channel ramps.
    shape=(1,64,7,9,5)
    feature=torch.arange(np.prod(shape),dtype=torch.float32).reshape(shape)
    div=feature[:,:2]/1000;size=feature[:,:1]/100
    xyz=torch.tensor([[0,1,2],[6,8,4],[3,4,1]])
    outputs=(None,None,feature,div,size)
    f,d,s=sample_outputs(outputs,xyz)
    y,x,z=xyz.T;batch=torch.zeros(len(xyz),dtype=torch.long)
    np.testing.assert_array_equal(f,feature.transpose(0,1)[:,batch,y,x,z].T)
    np.testing.assert_array_equal(d,torch.sigmoid(div[batch,:,y,x,z]))
    np.testing.assert_array_equal(s,size[batch,0,y,x,z])
    starts,patch,padded=tiles((256,256,64))
    assert len(starts)==3
    points=np.array(list(product((0,127,255),(0,127,255),(0,15,31,32,48,63))))
    owners=tile_owners(points,starts,patch)
    assert np.all(points-starts[owners]>=0) and np.all(points-starts[owners]<patch)
    # A real native patch verifies output gathering and CPU/CUDA FP32 parity.
    image=Images(clips()[0]["dataset"])
    raw=np.stack([image.raw(0).transpose(1,2,0),image.raw(1).transpose(1,2,0)]).astype(np.float32)
    minimum=raw[raw>0].min()
    expected=torch.log1p(torch.clamp_min(torch.from_numpy(raw),float(minimum))+1900).numpy()
    raw=np.log1p(np.maximum(raw,minimum)+1900)
    preprocess_error=float(np.max(abs(expected-raw)))
    assert preprocess_error<=1e-6
    tensor=torch.from_numpy(np.ascontiguousarray(raw[None,:,:32,:32,:32]))
    coords=torch.tensor([[8,9,10],[12,16,16],[24,20,20]])
    with torch.inference_mode():
        outputs=model(tensor);cpu=tuple(x.numpy() for x in sample_outputs(outputs,coords));del outputs
    lease=Lease([model]);lease.acquire()
    try:
        with torch.inference_mode():
            outputs=model(tensor.cuda());gpu=tuple(x.cpu().numpy() for x in sample_outputs(outputs,coords.cuda()));del outputs
    finally:lease.release()
    errors=[]
    for a,b in zip(cpu,gpu):
        np.testing.assert_allclose(a,b,atol=2e-4,rtol=3e-4)
        errors.append(float(np.max(abs(a-b))))
    write(RESULTS/"cellect-validation.json",dict(upstream_channel_lookup_exact=True,border_tile_coverage=True,
          unique_native_tiles=3,preprocess_numpy_torch_max_error=preprocess_error,real_patch_cpu_cuda_errors=errors,scope="32x32x32 real-patch numerical parity; production uses full native 256x256x32 tiles"))
    print("CELLECT PARITY",errors,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=["flow","organoid","cellect"])
    stage=parser.parse_args().stage
    {"flow":flow_contracts,"organoid":organoid_parity,"cellect":cellect_parity}[stage]()
