"""Source-only feature-unit diagnostic; never changes registered H inputs or fits."""
import inspect
import time
import torch
from skimage.measure import regionprops
from .common import *
from . import hoct_adapter as adapter, native_adapter as native, observations


SCALE=1.625
SOURCES=['44b6_24264f12','6bba_61dd1e0d']


def upstream_regions(name):
    """Run the pinned extractor on actual source-image watershed masks."""
    import tracksdata as td
    from hoct.features import create_graph
    from hoct.features.constants import REGIONPROPS
    captured=[]
    original=observations.regionprops

    def capture(labels,**kwargs):
        captured.append((labels.copy(),kwargs['intensity_image'].copy()))
        return original(labels,**kwargs)

    data=arrays(OUT/'observations'/f'{name}.npz')
    frames=native.Frames(DATA/'train'/f'{name}.zarr')
    model=native.load(device='cpu')
    with torch.no_grad():
        images=torch.stack([frames.frame(30),frames.frame(31)])[None]
        _,det=model.encode(images)
    observations.regionprops=capture
    try:
        for j,t in enumerate([30,31]):
            nodes=data['nodes'][data['nodes'][:,1]==t]
            observations.regions(frames.frame(t).numpy(),det[j][0,0].sigmoid().numpy(),nodes)
    finally:
        observations.regionprops=original
    labels=np.stack([x[0] for x in captured]);intensity=np.stack([x[1] for x in captured])
    with td.options.Options(n_workers=1):
        graph=create_graph(labels,images=intensity,normalize_images=False,
            scale=(1.,SCALE,SCALE,SCALE),distance_threshold=10.,n_neighbors=8,delta_t=1.)
    keys=['z','y','x',*[p for p in REGIONPROPS if p!='border_dist']]
    actual=graph.node_attrs(attr_keys=keys)
    # Compare feature multisets; the upstream extractor may assign different IDs.
    expected=[];physical=[]
    for lab,img in captured:
        for r in regionprops(lab,intensity_image=img):
            expected.append([*r.centroid,r.equivalent_diameter_area,r.intensity_min,
                r.intensity_max,r.intensity_mean,r.intensity_std,*r.inertia_tensor.ravel()])
        for r in regionprops(lab,intensity_image=img,spacing=(SCALE,)*3):
            physical.append([*r.centroid,r.equivalent_diameter_area,r.intensity_min,
                r.intensity_max,r.intensity_mean,r.intensity_std,*r.inertia_tensor.ravel()])
    values=np.array([[r['z'],r['y'],r['x'],r['equivalent_diameter_area'],r['intensity_min'],
        r['intensity_max'],r['intensity_mean'],r['intensity_std'],*np.asarray(r['inertia_tensor']).ravel()]
        for r in actual.to_dicts()])
    expected=np.array(expected);physical=np.array(physical)
    assert len(values)==len(expected)>0
    # Sorting each coordinate independently is enough here to verify numerical
    # feature distributions, without claiming a scored-center identity mapping.
    err=float(np.max(np.abs(np.sort(values,axis=0)-np.sort(expected,axis=0))))
    assert np.allclose(np.sort(values,axis=0),np.sort(expected,axis=0),atol=1e-5,rtol=1e-5)
    factor=np.array([SCALE]*4+[1.]*4+[SCALE**2]*9)
    assert np.allclose(physical,expected*factor,atol=1e-8,rtol=1e-8)
    return dict(dataset=name,frames=[30,31],actual_image_masks=True,regions=len(expected),
        upstream_scale_argument=[1.,SCALE,SCALE,SCALE],
        upstream_features_equal_unscaled_regionprops=True,maximum_multiset_feature_error=err,
        physical_over_voxel_centroid_and_diameter=SCALE,physical_over_voxel_inertia=SCALE**2,
        intensity_unchanged=True,registered_observation_sha256=sha(OUT/'observations'/f'{name}.npz'),
        note='Two CPU FP32 image frames for unit-contract verification; not a GPU cache-parity test. Upstream region centroids differ from the retained scored C0 centers by design.')


def model_comparison(model,name):
    from hoct.data._batching import item_from_filter,DataKeys
    from hoct.data._transforms import Standardize
    from hoct.features import REGIONPROPS
    from hoct._api import _MEAN,_STD
    data=arrays(OUT/'observations'/f'{name}.npz');old=data['oldmask']
    nodes=data['nodes'][old];props=data['properties'][old]
    physical=nodes[:,2:]*[1.625,.40625,.40625]
    eligible=(nodes[:,1]>=30)&(nodes[:,1]<=34)&data['valid_region'][old]
    tiles,counts=np.unique((physical[eligible]//32).astype(int),axis=0,return_counts=True)
    lower=tiles[np.argmax(counts)]*32.;upper=np.minimum(lower+32.,104.)
    keep=eligible&np.all((physical>=lower)&(physical<upper),axis=1)
    nodes=nodes[keep];props=props[keep];ids=set(nodes[:,0].astype(int))
    path=OUT/'banks'/name[:4]/'P0'/f'{name}.npz'
    pairs=np.array([e for e in arrays(path)['pairs'] if int(e[0]) in ids and int(e[1]) in ids],np.int64).reshape(-1,2)
    assert 2<len(nodes)<250 and len(pairs)>1
    outputs=[]
    for mode in ['registered_physical','native_voxel_reference']:
        n=nodes.astype(np.float64).copy();p=props.copy()
        if mode=='native_voxel_reference':
            n[:,2:]/=SCALE;p[:,0]/=SCALE;p[:,5:14]/=SCALE**2
        graph,_,_,reverse=adapter.feature_graph(n,pairs,p,np.ones(len(n),bool))
        batch=item_from_filter(graph.filter(node_ids=graph.node_ids()),['z','y','x'],REGIONPROPS,[],[Standardize(_MEAN,_STD)])
        tensor=lambda k:batch[k][None]
        x=tensor(DataKeys.NODE_FEATS);ei=tensor(DataKeys.EDGE_BATCH_ID)
        with torch.no_grad():
            logits,_,embeddings,orphan=model(x,tensor(DataKeys.NODE_POS),tensor(DataKeys.EDGE_POS),ei,
                torch.ones(x.shape[:2],dtype=torch.bool),torch.ones(ei.shape[:2],dtype=torch.bool))
        logits=logits[0].numpy().ravel();embeddings=embeddings[0].numpy()
        assert np.isfinite(logits).all() and np.isfinite(embeddings).all()
        index=np.array([reverse[int(i)] for i in batch[DataKeys.EDGE_ID]])
        outputs.append(dict(mode=mode,index=index,logits=logits,embeddings=embeddings,
            feature_mean=x[0].mean(0).tolist(),logit_mean=float(logits.mean()),
            logit_std=float(logits.std()),embedding_width=int(embeddings.shape[-1])))
    a,b=outputs;assert np.array_equal(a['index'],b['index'])
    pa=pairs[a['index']];targets=np.unique(pa[:,1]);changes=0
    for target in targets:
        subset=np.flatnonzero(pa[:,1]==target)
        changes+=int(subset[np.argmax(a['logits'][subset])]!=subset[np.argmax(b['logits'][subset])])
    return dict(dataset=name,source=name[:4],frames=[30,34],nodes=len(nodes),edges=len(pairs),
        ROI_selection='densest image-derived 32 micrometer tile, same objects and edges in both passes',
        optical_bounds_um=[lower.tolist(),upper.tolist()],source_bank_sha256=sha(path),
        maximum_absolute_logit_difference=float(np.max(np.abs(a['logits']-b['logits']))),
        mean_absolute_logit_difference=float(np.mean(np.abs(a['logits']-b['logits']))),
        mean_edge_embedding_l2_difference=float(np.linalg.norm(a['embeddings']-b['embeddings'],axis=1).mean()),
        candidate_parent_argmax_changes=changes,targets_with_candidates=len(targets),
        comparisons=[{k:v for k,v in row.items() if k not in ['index','logits','embeddings']} for row in outputs])


def run():
    destination=OUT/'HOCT_unit_contract_audit.json'
    if destination.exists():raise RuntimeError('Preserve the existing audit; use a new attempt namespace for any repeat.')
    torch.set_num_threads(2);adapter.imports();start=time.monotonic()
    import hoct.features.graph as upstream
    import tracksdata.nodes._regionprops as region_module
    frozen=OUT/'inference_package_validation/manifest.json'
    before=sha(frozen)
    region_result=upstream_regions(SOURCES[0])
    model=adapter.load(device='cpu');model.eval();torch._C._jit_set_bailout_depth(0)
    comparisons=[model_comparison(model,name) for name in SOURCES]
    assert sha(frozen)==before
    write(destination,dict(at=now(),complete=True,seconds=time.monotonic()-start,
        source_only=True,annotations_read=False,official_scores_computed=False,
        discovered_after_outer_scores=True,registered_features_or_models_changed=False,
        runtime='CPU FP32 for both diagnostic passes; production uses GPU BF16',
        checkpoint_sha256=sha(OUT/'models/hoct/general_v1.pt'),
        upstream_source_sha256=sha(Path(inspect.getfile(upstream))),
        RegionPropsNodes_source_sha256=sha(Path(inspect.getfile(region_module))),
        validation_bundle_manifest_sha256=before,region_contract=region_result,model_comparisons=comparisons,
        finding='Pinned create_graph accepts scale metadata but extracts default voxel-unit region features and unscaled z/y/x. V5 explicitly supplies micrometer coordinates/diameters and micrometer-squared inertia to the same published Standardize constants.',
        scope='Real pretrained HOCT and real object morphology were tested, using registered physical-unit features. This is not an exact reproduction of the upstream default voxel-feature pipeline. The late unit audit is a protocol deviation from the requested pre-score scaling audit.',
        limitations='The native-voxel reference holds C0 centers, regions, candidate edges, border features and local context fixed. It is a source-only unit sensitivity diagnostic, not a full upstream default pipeline, 199-clip counterfactual score, calibration repair or independent generalization test. No source or target score selected a replacement transform.',
        missing_comparison='No full 199-clip upstream-default-voxel HOCT arm was registered or run; retain this as an explicit limitation, not a checkpoint-quality verdict.'))
    print('Completed source-only HOCT unit-contract audit',time.monotonic()-start,flush=True)


if __name__=='__main__':run()
