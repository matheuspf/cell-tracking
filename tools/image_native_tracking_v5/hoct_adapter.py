"""Pinned official HOCT JIT + complete region features. No GT dependency in this module."""
import sys,time
import numpy as np
import torch
from .common import *

def imports():
    p=str(WORK/'python')
    if p not in sys.path:sys.path.insert(0,p)
    import hoct
    return hoct

def load(name='general_v1',device='cuda'):
    hoct=imports();p=OUT/'models/hoct'/f'{name}.pt'
    expected=read(REPO/'handover/image-native-tracking-v5/config.json')['hoct'][name+'_sha256']
    assert sha(p)==expected
    model=hoct.load_model(str(p),device=device)
    return model

def feature_graph(nodes,pairs,props,valid):
    imports()
    import polars as pl
    import tracksdata as td
    from hoct.features import REGIONPROPS,add_delta_t
    g=td.graph.IndexedRXGraph()
    for k in ['z','y','x',*REGIONPROPS]:
        dtype=pl.Array(pl.Float32,(3,3)) if k=='inertia_tensor' else pl.Float32
        default=np.zeros((3,3),np.float32) if k=='inertia_tensor' else 0.
        g.add_node_attr_key(k,dtype,default)
    indices=np.flatnonzero(valid);attrs=[]
    for i in indices:
        n=nodes[i];pr=props[i]
        attrs.append(dict(t=int(n[1]),**dict(zip('zyx',(n[2:]*[1.625,.40625,.40625]).tolist())),
            equivalent_diameter_area=float(pr[0]),intensity_min=float(pr[1]),intensity_max=float(pr[2]),
            intensity_mean=float(pr[3]),intensity_std=float(pr[4]),inertia_tensor=pr[5:14].reshape(3,3),border_dist=float(pr[14])))
    ids=g.bulk_add_nodes(attrs);mapping={int(nodes[i,0]):int(j) for i,j in zip(indices,ids)}
    keep=np.array([a in mapping and b in mapping for a,b in pairs]);good=pairs[keep]
    g.bulk_add_edges([dict(source_id=mapping[int(a)],target_id=mapping[int(b)]) for a,b in good])
    add_delta_t(g)
    g.metadata.update(scale=(1.,1.,1.,1.),was_2d=False,coordinate_units='micrometers',feature_origin='v5 image-supported native-grid watershed')
    reverse={j:int(nodes[i,0]) for i,j in zip(indices,ids)}
    pair_index={(mapping[int(a)],mapping[int(b)]):int(i) for i,(a,b) in enumerate(pairs) if keep[i]}
    keys=td.DEFAULT_ATTR_KEYS
    ea=g.edge_attrs(attr_keys=[keys.EDGE_ID,keys.EDGE_SOURCE,keys.EDGE_TARGET])
    edge_reverse={int(e):pair_index[(int(a),int(b))] for e,a,b in ea.select(keys.EDGE_ID,keys.EDGE_SOURCE,keys.EDGE_TARGET).iter_rows()}
    # The scored C0 coordinates/IDs are never replaced by region centroids or graph IDs.
    assert len(mapping)==len(indices)
    return g,mapping,reverse,edge_reverse

def iter_embeddings(model,nodes,pairs,props,valid,device='cuda'):
    """Five observed frames, spatial cores with 16um competitor halos, unique edge ownership."""
    imports()
    import polars as pl
    from hoct.data._batching import item_from_filter,DataKeys
    from hoct.data._transforms import Standardize
    from hoct.features import REGIONPROPS
    from hoct._api import _MEAN,_STD
    g,mapping,reverse,edge_reverse=feature_graph(nodes,pairs,props,valid)
    node_lookup={int(n[0]):i for i,n in enumerate(nodes)};physical=nodes[:,2:]*[1.625,.40625,.40625]
    times=nodes[:,1].astype(int);ids=nodes[:,0].astype(int)
    pair_src=np.array([node_lookup[int(p[0])] for p in pairs]);pair_tgt=np.array([node_lookup[int(p[1])] for p in pairs])
    pair_time=times[pair_src];core=np.floor(physical/52.).astype(int)
    emitted=set();model.eval();torch._C._jit_set_bailout_depth(0)
    with torch.no_grad():
        for t in range(int(times.max())):
            lo=max(0,min(t-1,int(times.max())-4));hi=min(int(times.max()),lo+4)
            for key in sorted(set(map(tuple,core[pair_src[pair_time==t]]))):
                lower=np.array(key)*52.-16.;upper=(np.array(key)+1)*52.+16.
                selected=(times>=lo)&(times<=hi)&np.all((physical>=lower)&(physical<upper),axis=1)&valid
                rowids=np.flatnonzero(selected)
                graph_ids=[mapping[int(ids[i])] for i in rowids]
                if len(graph_ids)<2:continue
                batch=item_from_filter(g.filter(node_ids=graph_ids),['z','y','x'],REGIONPROPS,[],[Standardize(_MEAN,_STD)])
                if batch is None or not len(batch[DataKeys.EDGE_ID]):continue
                eid=batch[DataKeys.EDGE_ID].numpy();original=np.array([edge_reverse[int(i)] for i in eid])
                own=(pair_time[original]==t)&np.all(core[pair_src[original]]==key,axis=1)
                if not own.any():continue
                tensor=lambda k:batch[k][None].to(device)
                x=tensor(DataKeys.NODE_FEATS);pos=tensor(DataKeys.NODE_POS);edgepos=tensor(DataKeys.EDGE_POS);edgeindex=tensor(DataKeys.EDGE_BATCH_ID)
                nm=torch.ones(x.shape[:2],dtype=torch.bool,device=device);em=torch.ones(edgeindex.shape[:2],dtype=torch.bool,device=device)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                    pred,nodefeat,edgefeat,orphan=model(x,pos,edgepos,edgeindex,nm,em)
                index=original[own]
                assert not (emitted&set(index.tolist()));emitted.update(index.tolist())
                values=edgefeat[0,torch.as_tensor(own,device=device)].float().cpu().numpy()
                logits=pred[0,torch.as_tensor(own,device=device)].float().cpu().numpy().ravel()
                assert np.isfinite(values).all() and np.isfinite(logits).all()
                yield index,logits,values,dict(t=t,nodes=int(x.shape[1]),edges=int(edgeindex.shape[1]),context_start=lo,context_end=hi)

def pilot(name):
    c=arrays(OUT/'observations'/f'{name}.npz');m=c['oldmask'];nodes=c['nodes'][m];pairs=c['fixed_pairs'];props=c['properties'][m];valid=c['valid_region'][m]
    model=load();start=time.monotonic();count=0;sample=[]
    for index,logit,feat,meta in iter_embeddings(model,nodes,pairs,props,valid):
        count+=len(index)
        if len(sample)<5:sample.append(dict(**meta,embedding_shape=list(feat.shape),logit_min=float(logit.min()),logit_max=float(logit.max())))
        if len(sample)==5:break
    write(OUT/'hoct_interface_pilot.json',dict(dataset=name,seconds=time.monotonic()-start,parameters=sum(p.numel() for p in model.parameters()),
        keys_shapes={k:list(v.shape) for k,v in model.state_dict().items()},sample_batches=sample,covered_edges=count,
        region_valid=int(valid.sum()),region_total=len(valid),fixed_node_ids_coordinates_preserved=True,
        property_order=['t','z_um','y_um','x_um','equivalent_diameter_area','intensity_min','intensity_max','intensity_mean','intensity_std','inertia_tensor_3x3','border_dist'],
        source_pin='2ccc5040823bc944ab67790abd1f56eea7cd4f05',license='MIT',weights_verified=True,
        point_stub_not_called=True,input_feature_width=19,true_embedding_width=288,full_backbone_adaptation=False))
    print('HOCT interface',name,count,time.monotonic()-start,flush=True)

if __name__=='__main__':
    import sys
    pilot(sys.argv[1] if len(sys.argv)>1 else '44b6_24264f12')
