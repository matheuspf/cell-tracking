"""One streamed native pass; full-field P1 proposals, morphology and genuine native features."""
import time,resource
from concurrent.futures import ThreadPoolExecutor
import torch
from .common import *
from . import native_adapter as native
from .observations import peaks,regions
from .candidates import bank,motion_config

def score_features(model,nodes,pairs,fs,ft):
    lookup={int(n[0]):i for i,n in enumerate(nodes)};out=np.full(len(pairs),np.nan,np.float32)
    srcrow=np.array([lookup[int(a)] for a,b in pairs]);tgtrow=np.array([lookup[int(b)] for a,b in pairs])
    with torch.no_grad():
        for t in range(99):
            ai=np.flatnonzero(nodes[:,1]==t);bi=np.flatnonzero(nodes[:,1]==t+1);pi=np.flatnonzero(nodes[srcrow,1]==t)
            if not len(pi):continue
            log=native.logits(model,nodes[ai],nodes[bi],features=(fs[ai],ft[bi]))
            p=log.softmax(0).clamp(1e-6,1-1e-6);logodds=torch.logit(p).cpu().numpy()
            am={i:j for j,i in enumerate(ai)};bm={i:j for j,i in enumerate(bi)}
            out[pi]=logodds[[am[i] for i in srcrow[pi]],[bm[i] for i in tgtrow[pi]]]
    assert np.isfinite(out).all();return out

def one(model,row):
    name=row['dataset'];dest=OUT/'observations'/f'{name}.npz';fp=dict(c0=sha(V3/'selected_predictions'/f'{name}.npz'),
        native=sha(NATIVE/'weights/unet_transformer/split_0/edge_predictor_best.pth'),code={k:sha(Path(__file__).parent/f'{k}.py') for k in ['cache','observations','native_adapter','candidates']})
    if dest.exists():assert read(dest.with_suffix('.json'))['fingerprint']==fp;return
    start=time.monotonic();frames=native.Frames(DATA/'train'/f'{name}.zarr');base=graph(name);old=base['nodes']
    node_by_t={};strengths={};groups={};props={};val={};collisions={};opt={};source_feat={};target_feat={}
    nextid=int(old[:,0].max())+1
    with torch.no_grad(),ThreadPoolExecutor(max_workers=2) as pool:
        region_jobs={}
        for t in range(99):
            images=frames.pair(t);features,det=model.encode(images)
            for j in range(2):
                tt=t+j
                if tt not in node_by_t:
                    old_t=old[old[:,1]==tt];prob=det[j][0,0].sigmoid().cpu().numpy();image=frames.frame(tt).numpy()
                    new,confidence,group=peaks(prob,image,old_t,nextid,tt);nextid+=len(new)
                    n=np.concatenate([old_t,new]);node_by_t[tt]=n
                    strengths[tt]=np.concatenate([np.ones(len(old_t),np.float32),confidence])
                    groups[tt]=np.concatenate([np.full(len(old_t),-1,np.int64),group])
                    region_jobs[tt]=pool.submit(regions,image,prob,n)
                n=node_by_t[tt];inp=native.pair_inputs(node_by_t[t],node_by_t[t+1]) if t+1 in node_by_t else None
            inp=native.pair_inputs(node_by_t[t],node_by_t[t+1])
            source_feat[t]=model._index_features(features[:,0],inp[0][0],inp[0][2])[0].cpu().numpy()
            target_feat[t+1]=model._index_features(features[:,1],inp[1][0],inp[1][2])[0].cpu().numpy()
            del features,det,images
        for tt,f in region_jobs.items():props[tt],val[tt],collisions[tt],opt[tt]=f.result()
    nodes=np.concatenate([node_by_t[t] for t in range(100)])
    fs=np.concatenate([source_feat.get(t,np.zeros((len(node_by_t[t]),32),np.float32)) for t in range(100)])
    ft=np.concatenate([target_feat.get(t,np.zeros((len(node_by_t[t]),32),np.float32)) for t in range(100)])
    oldmask=np.isin(nodes[:,0],old[:,0]);motion=motion_config();source='6bba' if row['embryo']=='44b6' else '44b6'
    # Store uncapped complete physical alternatives once; source-specific cap is frozen above.
    distance=motion[source]['distance_um'];pairs=bank(nodes,base['edges'],distance)
    fixedpairs=pairs[np.isin(pairs[:,0],old[:,0])&np.isin(pairs[:,1],old[:,0])]
    fixedlog=score_features(model,nodes[oldmask],fixedpairs,fs[oldmask],ft[oldmask])
    unionlog=score_features(model,nodes,pairs,fs,ft)
    save(dest,nodes=nodes,oldmask=oldmask,features_source=fs.astype(np.float16),features_target=ft.astype(np.float16),
        properties=np.concatenate([props[t] for t in range(100)]),valid_region=np.concatenate([val[t] for t in range(100)]),
        collisions=np.concatenate([collisions[t] for t in range(100)]),optical_centroids=np.concatenate([opt[t] for t in range(100)]),
        confidence=np.concatenate([strengths[t] for t in range(100)]),split_owner=np.concatenate([groups[t] for t in range(100)]),
        pairs=pairs,native_union_logits=unionlog,fixed_pairs=fixedpairs,native_fixed_logits=fixedlog)
    write(dest.with_suffix('.json'),dict(dataset=name,fingerprint=fp,created=now(),seconds=time.monotonic()-start,
        c0_nodes=len(old),novel_peaks=int((~oldmask).sum()),nodes=len(nodes),pairs=len(pairs),fixed_pairs=len(fixedpairs),
        valid_regions=int(sum(v.sum() for v in val.values())),collisions=int(sum(v.sum() for v in collisions.values())),
        full_frame_search=True,frames=100,spacing=[1.625,.40625,.40625],source_motion_configuration=motion[source],
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,sha256=sha(dest)))
    print('native/observation cache',name,read(dest.with_suffix('.json'))['seconds'],'seconds',int((~oldmask).sum()),'new peaks',flush=True)

def run(names=None):
    torch.set_num_threads(2);model=native.load();write(OUT/'native_architecture.json',native.receipt(model));motion_config()
    rows=inventory()
    # First two complete optical pilots were chosen by C0 image-derived density.
    pilots=[r['dataset'] for r in read(OUT/'density_selection.json')['pilots']]
    rows=sorted(rows,key=lambda r:(r['dataset'] not in pilots,r['dataset']))
    for r in rows:
        if names and r['dataset'] not in names:continue
        one(model,r)

if __name__=='__main__':
    import sys
    run(sys.argv[1:] or None)
