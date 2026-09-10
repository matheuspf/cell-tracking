"""Explicit synthetic optical patches from eligible Zoo trajectories; no native Zoo images."""
from scipy.spatial import cKDTree
from .common import *
from .corruptions import observations
from .proposals import build

def source_statistics(source):
    patches=[];distances=[]
    for row in inputs():
        if row['embryo']!=source:continue
        data=arrays(OUT/'cache/real'/f'{row["dataset"]}.npz')
        ids=np.linspace(0,len(data['patch'])-1,min(16,len(data['patch']))).astype(int)
        patches.append(data['patch'][ids,0,1].astype(float)/255)
        graph=arrays(V3/'selected_predictions'/f'{row["dataset"]}.npz')['nodes']
        points=graph[graph[:,1]==int(np.median(graph[:,1])),2:]*row['physical_scale']
        if len(points)>1:distances.extend(cKDTree(points).query(points,k=2)[0][:,1])
    p=np.concatenate(patches);median=np.median(p,axis=0);background=float(np.quantile(median,.1))
    weight=np.maximum(median-background,0);yy,xx=np.meshgrid(np.arange(12)-5.5,np.arange(12)-5.5,indexing='ij')
    sigma=float(np.sqrt((weight*(yy*yy+xx*xx)).sum()/max(2*weight.sum(),1e-8)))
    return dict(source=source,patches=len(p),background=float(np.clip(background,0,.2)),
        amplitude=float(np.clip(median.max()-background,.2,.9)),sigma_pixels=float(np.clip(sigma,.5,2.5)),
        noise_std=float(np.clip(np.std(p[:,:,:2,:]),.005,.08)),nearest_cell_um=float(np.median(distances)),
        pixel_step_um=1.625,source_only=True,label_values_used_for_statistics=False,
        caveat='Source incumbent-centered patch statistics; phenomenological Gaussian rendering, not measured Zoo optics')

def render(clean_t,clean_points,query_t,query_points,stats):
    import torch
    torch.manual_seed(SEED+int(stats['source'],16));torch.set_num_threads(2)
    out=np.zeros((len(query_t),3,3,12,12),np.uint8);valid=np.zeros((len(query_t),3),np.uint8)
    yy,xx=np.meshgrid(np.arange(12)-5.5,np.arange(12)-5.5,indexing='ij');zero=np.zeros_like(yy)
    grid=np.stack([np.stack([zero,yy,xx],-1),np.stack([yy,zero,xx],-1),np.stack([yy,xx,zero],-1)]).reshape(3,144,3)
    grid=torch.as_tensor(grid,device='cuda',dtype=torch.float32)
    brightness=np.random.default_rng(777).uniform(.6,1.4,len(clean_t)).astype(np.float32)*stats['amplitude']
    for f in np.unique(clean_t):
        physical=np.flatnonzero(clean_t==f);points=clean_points[physical];tree=cKDTree(points)
        for j,offset in enumerate([-1,0,1]):
            ids=np.flatnonzero(query_t+offset==f)
            if not len(ids):continue
            dist,near=tree.query(query_points[ids],k=16,distance_upper_bound=14.)
            present=np.isfinite(dist);near=np.minimum(near,len(points)-1)
            for start in range(0,len(ids),256):
                ix=ids[start:start+256];ni=near[start:start+256];mask=present[start:start+256]
                local=torch.as_tensor(points[ni]-query_points[ix,None],device='cuda',dtype=torch.float32)
                amp=torch.as_tensor(brightness[physical[ni]]*mask,device='cuda')
                delta=grid[None,:,None]-local[:,None,:,None]
                p=(torch.exp(-.5*delta.square().sum(-1)/stats['sigma_pixels']**2)*amp[:,None,:,None]).sum(2)
                p=p+stats['background']+torch.randn_like(p)*stats['noise_std']
                out[ix,j]=torch.round(p.clamp(0,1)*255).byte().reshape(-1,3,12,12).cpu().numpy();valid[ix,j]=1
    return out,valid

def one(source):
    done=OUT/f'rendered_zoo_{source}_receipt.json'
    if done.exists():
        for r in read(done)['partitions']:assert sha(r['cache'])==r['cache_sha256']
        return
    stats=source_statistics(source);lab=arrays(PREPARED/'zoo/zebrafish_graph.npz');t=lab['t'];p=lab['zyx_source']
    tt,pp,m=observations(t,p,SEED+len('zebrafish'));reverse={int(g):i for i,g in enumerate(m) if g>=0}
    median=np.median(pp,axis=0);iqr=np.maximum(np.quantile(pp,.75,axis=0)-np.quantile(pp,.25,axis=0),1e-6)
    q=(p-median)/iqr;qq=(pp-median)/iqr;mid=q[t==int(np.median(t))]
    unit_nn=float(np.median(cKDTree(mid).query(mid,k=2)[0][:,1]));factor=stats['nearest_cell_um']/1.625/max(unit_nn,1e-6)
    clean=q*factor;query=qq*factor;records=[]
    for partition in ['train','validation','test']:
        data=arrays(OUT/'cache/zoo'/f'zebrafish_{partition}.npz');anchors=np.array([reverse[int(g)] for g in data['group']])
        a,c,x=build(tt,pp,anchors);np.testing.assert_array_equal(x,data['x'])
        ids=np.c_[a,c];used=np.unique(ids[ids>=0]);patch,valid=render(t,clean,tt[used],query[used],stats)
        rev=np.full(len(tt),-1,int);rev[used]=np.arange(len(used));lookup=rev[np.maximum(ids,0)]
        data['patch']=patch[lookup];data['valid']=valid[lookup];data['valid'][ids<0]=0
        assert np.isfinite(data['patch']).all() and data['valid'].sum(2).max()<=3
        path=OUT/'cache'/f'rendered_zoo_{source}'/f'{partition}.npz';save(path,**data)
        records.append(dict(source=f'rendered_zoo_{source}',partition=partition,cache=str(path),cache_sha256=sha(path),
            source_graph_sha256=sha(PREPARED/'zoo/zebrafish_graph.npz'),bags=len(a),positive_groups=int((data['target']>0).sum()),
            label_kind='weak_Zoo_tracks',image_representation='explicit_synthetic_Gaussian_triplanes',
            source_use_status='derived_from_eligible_zebrafish',provenance_group=f'zebrafish_time_{partition}',
            independent_acquisition=False,rendered_not_experimental=True))
        print('rendered',source,partition,len(a),flush=True)
    write(done,dict(completed=True,source_statistics=stats,partitions=records,renderer_code_sha256=sha(Path(__file__)),
        trajectory_normalization='whole unlabeled acquisition per-axis IQR; spacing remapped from source-only incumbent detections',
        intensity_labels_independent=True,renderer_neighbors_per_query=16,clean_points_rendered_with_corrupted_query_centers=True))

def run():
    for source in ['44b6','6bba']:one(source)
    records=sum([read(OUT/f'rendered_zoo_{s}_receipt.json')['partitions'] for s in ['44b6','6bba']],[])
    write(OUT/'rendered_dataset_index.json',dict(records=records,primary_dataset_index_unchanged=True))
