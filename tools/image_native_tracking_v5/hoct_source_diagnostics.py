"""Source-only alternative checkpoint and original HOCT decoder diagnostics."""
import time
import torch
from .common import *
from .hoct_adapter import load,feature_graph,imports
from .score_models import hoct_scores
from .calibrate import apply
from .temporal_decode import decode


def original_decoder(name,dense_roi=False):
    imports()
    import polars as pl
    import tracksdata as td
    from hoct.data._batching import item_from_filter,DataKeys
    from hoct.data._transforms import Standardize
    from hoct.features import REGIONPROPS
    from hoct._api import _MEAN,_STD
    from hoct.tracking import ILPSolverConfig,solve_tracking
    c=arrays(OUT/'observations'/f'{name}.npz');source=name.split('_')[0]
    m=c['oldmask'];n=c['nodes'][m];physical=n[:,2:]*[1.625,.40625,.40625]
    eligible=(n[:,1]>=30)&(n[:,1]<=34)&c['valid_region'][m]
    lower=np.array([36.,36.,36.]);upper=np.array([68.,68.,68.])
    if dense_roi:
        tiles,counts=np.unique((physical[eligible]//32).astype(int),axis=0,return_counts=True)
        lower=tiles[np.argmax(counts)]*32.;upper=np.minimum(lower+32.,104.)
    keep=eligible&np.all((physical>=lower)&(physical<upper),axis=1)
    n=n[keep];ids=set(n[:,0].astype(int));bank=arrays(OUT/'banks'/source/'P0'/f'{name}.npz')
    pairs=np.asarray([e for e in bank['pairs'] if int(e[0]) in ids and int(e[1]) in ids],np.int64).reshape(-1,2)
    assert len(n)>2 and len(pairs)>1
    graph,mapping,reverse,edge_reverse=feature_graph(n,pairs,c['properties'][m][keep],np.ones(len(n),bool))
    batch=item_from_filter(graph.filter(node_ids=graph.node_ids()),['z','y','x'],REGIONPROPS,[],[Standardize(_MEAN,_STD)])
    model=load();tensor=lambda k:batch[k][None].cuda()
    x=tensor(DataKeys.NODE_FEATS);pos=tensor(DataKeys.NODE_POS);ep=tensor(DataKeys.EDGE_POS);ei=tensor(DataKeys.EDGE_BATCH_ID)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits,_,features,orphan=model(x,pos,ep,ei,torch.ones(x.shape[:2],dtype=torch.bool,device='cuda'),torch.ones(ei.shape[:2],dtype=torch.bool,device='cuda'))
    raw=logits[0].float().clamp(max=20).exp().cpu().numpy().ravel()
    oph=orphan[0].float().clamp(max=20).exp().cpu().numpy().ravel()
    assert len(oph)==len(batch[DataKeys.NODE_ID])
    node_exp=dict(zip(map(int,batch[DataKeys.NODE_ID]),oph));keys=td.DEFAULT_ATTR_KEYS
    attrs=graph.edge_attrs(attr_keys=[]);targets=dict((int(e),int(b)) for e,b in attrs.select(keys.EDGE_ID,keys.EDGE_TARGET).iter_rows())
    eids=list(map(int,batch[DataKeys.EDGE_ID]));denom={}
    for eid,v in zip(eids,raw):denom[targets[eid]]=denom.get(targets[eid],0)+float(v)
    similarity=[float(v/(denom[targets[eid]]+node_exp[targets[eid]])) for eid,v in zip(eids,raw)]
    node_ids=graph.node_ids();orphan_prob=[float(node_exp[i]/(denom.get(i,0)+node_exp[i])) for i in node_ids]
    graph.add_edge_attr_key('similarity',pl.Float32,0.);graph.add_node_attr_key('orphan_prob',pl.Float32,0.)
    graph.update_edge_attrs(edge_ids=eids,attrs={'similarity':similarity});graph.update_node_attrs(node_ids=node_ids,attrs={'orphan_prob':orphan_prob})
    cfg=ILPSolverConfig.default().model_copy(update={'timeout':30.});start=time.monotonic()
    result=solve_tracking(graph,cfg,return_solution=True)
    selected_edges=result.edge_attrs(attr_keys=[keys.SOLUTION])
    selected_nodes=result.node_attrs(attr_keys=[keys.SOLUTION])
    suffix='_dense_roi' if dense_roi else ''
    write(OUT/'source_pilots'/f'{name}_HOCT_original_decoder{suffix}.json',dict(dataset=name,source=source,source_only=True,
        optical_bounds_um=[lower.tolist(),upper.tolist()],frames=[30,34],input_nodes=len(n),input_edges=len(pairs),
        solution_nodes=int(selected_nodes[keys.SOLUTION].sum()),solution_edges=int(selected_edges[keys.SOLUTION].sum()),
        seconds=time.monotonic()-start,solver_config=cfg.model_dump(),actual_pretrained_backbone=True,actual_original_HOCT_solver=True,
        orphan_head_executed=True,incoming_parental_normalization=True,embedding_width=int(features.shape[-1]),
        ROI_selection='densest image-derived 32um grid tile, frames30..34; no GT' if dense_roi else 'central source tile',
        restriction='five-frame source spatial tile; consecutive common candidates, not upstream gap<=3/distance300; no complete competition score claimed'))


def checkpoint_pilot(name):
    from annotation_selection.metric_adapter import evaluate_graph
    from strong_tracker_v3.common import validate
    source=name.split('_')[0];c=arrays(OUT/'observations'/f'{name}.npz');m=c['oldmask'];base=graph(name)
    common=arrays(OUT/'banks'/source/'P0'/f'{name}.npz');model=load('ctc_v0');start=time.monotonic()
    values,coverage=hoct_scores(model,c['nodes'][m],common['pairs'],c['properties'][m],c['valid_region'][m])
    joint=read(OUT/'calibration'/f'{source}_J.json');cal=read(OUT/'calibration'/f'{source}_Hctc.json')['calibration']
    scores=apply(cal,values['H0']);n,e,receipt=decode(c['nodes'][m],common['pairs'],scores,base['nodes'],base['edges'],config=joint['config'])
    row=next(r for r in inventory() if r['dataset']==name);validate(n,e,row['image_shape']);gt=arrays(V1/'evaluation/gt'/f'{name}.npz')
    result,_,_=evaluate_graph(name,n,e,gt['nodes'],gt['edges'],row['physical_scale'],row['estimated_total'])
    write(OUT/'source_pilots'/f'{name}_HOCT_ctc_J.json',dict(dataset=name,source=source,source_only=True,score=result,
        coverage=coverage,decode=receipt,seconds=time.monotonic()-start,checkpoint_sha256=sha(OUT/'models/hoct/ctc_v0.pt'),
        restriction='source-pilot checkpoint comparison; not a complete 199-clip variant'))


def run():
    torch.set_num_threads(2)
    with gpu_aux():
        for name in ['44b6_24264f12','6bba_61dd1e0d']:
            dest=OUT/'source_pilots'/f'{name}_HOCT_original_decoder.json'
            if not dest.exists():
                try:original_decoder(name)
                except Exception as exc:
                    write(dest,dict(dataset=name,source_only=True,status='blocked diagnostic',error=repr(exc),actual_original_HOCT_solver_completed=False))
                    print('Original decoder diagnostic blocked',name,repr(exc),flush=True)
            if not (OUT/'source_pilots'/f'{name}_HOCT_ctc_J.json').exists():checkpoint_pilot(name)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--retry-sparse-roi',action='store_true');a=p.parse_args()
    if a.retry_sparse_roi:
        torch.set_num_threads(2)
        with gpu_aux():original_decoder('6bba_61dd1e0d',True)
    else:run()
