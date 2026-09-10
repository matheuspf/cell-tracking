"""Source-specific training caches; no target outcome or target-derived thresholds."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from scipy.spatial import cKDTree
from .common import *
from .proposals import build, PAIRS
from .adapters import synthetic_record, temporal_patches, planes
from .corruptions import observations

def metadata_contract(record):
    record=dict(record,source_id=record['source'],source_use_status='eligible_selected')
    if record['source']=='synthetic_static':
        record.update(frame_valid='one actual native frame; no temporal duplication',
            future_observed='not applicable to static centers',candidate_visibility='native center queries and dense simulated background')
    return record

def supervision(anchors,candidates,mapping,truth_edges,truth_t,dense=False,weak=False):
    children=defaultdict(list);parents={}
    for a,b in truth_edges:children[int(a)].append(int(b));parents[int(b)]=int(a)
    targets=np.full(len(anchors),-1,np.int64);ly=np.full(candidates.shape,-1,np.int8);w=np.ones(len(anchors),np.float32)
    for row,(a,cc) in enumerate(zip(anchors,candidates)):
        g=int(mapping[a]); known=children.get(g,[])
        if not (cc>=0).any():continue  # No observed next-frame evidence, including missing frames.
        for j,c in enumerate(cc):
            if c<0:continue
            h=int(mapping[c])
            if h in known:ly[row,j]=1
            # A single recorded child does not rule out an unannotated second
            # daughter. Only dense truth, a complete binary fork, or a known
            # different predecessor can contradict this candidate edge.
            elif dense or len(known)==2 or (h in parents and parents[h]!=g):ly[row,j]=0
        if len(known)==2:
            for k,(u,v) in enumerate(PAIRS):
                if cc[u]>=0 and cc[v]>=0 and set(mapping[cc[[u,v]]])==set(known):targets[row]=k+1;break
        elif dense:
            if g>=0 and truth_t[g]<truth_t.max():targets[row]=0
        elif g>=0:
            quiet=True
            cur=g
            for _ in range(2):
                if len(children.get(cur,[]))!=1:quiet=False;break
                cur=children[cur][0]
            cur=g
            for _ in range(2):
                if cur not in parents:quiet=False;break
                cur=parents[cur]
                if len(children[cur])!=1:quiet=False;break
            if quiet:targets[row]=0
        if weak:w[row]=.6 if targets[row]>0 else .2
    return targets,ly,w

def select_anchors(t,mapping,edges,seed,train=True,limit=24):
    rng=np.random.default_rng(seed);degree=np.bincount(edges[:,0],minlength=int(edges.max())+1) if len(edges) else np.zeros(0)
    possible=np.flatnonzero((mapping>=0)&(t<t.max()))
    if not train:return np.sort(rng.choice(possible,min(len(possible),128),replace=False))
    g=mapping[possible];div=possible[(g<len(degree))&(degree[np.minimum(g,len(degree)-1)]==2)]
    non=np.setdiff1d(possible,div)
    return np.sort(np.r_[rng.choice(div,min(limit,len(div)),False),rng.choice(non,min(limit,len(non)),False)])

def cache_bags(dest,t,points,mapping,edges,truth_t,anchors,image=None,spacing=None,dense=False,weak=False):
    a,c,x=build(t,points,anchors);target,ly,w=supervision(a,c,mapping,edges,truth_t,dense,weak)
    data=dict(x=x,target=target,link_y=ly,weight=w,group=mapping[a],anchor_t=t[a])
    if image is not None:
        ix=np.c_[a,c];used=np.unique(ix[ix>=0]);p,v=temporal_patches(image,t[used],points[used],spacing)
        reverse=np.full(len(t),-1,np.int64);reverse[used]=np.arange(len(used));ind=reverse[np.maximum(ix,0)]
        data['patch']=p[ind];data['valid']=v[ind];data['valid'][ix<0]=0
    save(dest,**data)
    return dict(bags=len(a),positive_groups=int((target>0).sum()),negative_groups=int((target==0).sum()),
                censored_or_unreachable=int((target<0).sum()),candidate_alternatives=int((x[:,:,11]>0).sum()))

def detection_data(image,points,seed,real=False,spacing=(1,1,1)):
    rng=np.random.default_rng(seed);frame=np.asarray(image,np.float32)
    lo,hi=np.quantile(frame[::2,::4,::4],[.02,.995]);im=np.clip((frame-lo)/max(hi-lo,1),0,1)
    n=min(16,len(points));sel=rng.choice(len(points),n,False);offset=rng.uniform(-2,2,(n,3));loc=points[sel]+offset
    bg=rng.uniform([1,1,1],np.array(frame.shape)-2,(128,3))
    nearest=cKDTree(points).query(bg)[0]
    if real:
        patches=planes(im,bg)
        bg=bg[(np.quantile(patches,.95,axis=(1,2,3))<.08)&(nearest>5)]
    else:bg=bg[nearest>4]
    bg=bg[:16];coords=np.vstack([loc,bg]);patch=planes(im,coords)
    target=np.r_[np.exp(-np.sum(offset**2,axis=1)/4),np.zeros(len(bg))].astype(np.float32)
    return dict(patch=np.rint(patch*255).astype(np.uint8),target=target,
        offset=np.vstack([-offset/3,np.zeros((len(bg),3))]).astype(np.float32),positive=np.r_[np.ones(n),np.zeros(len(bg))].astype(np.float32))

def synthetic_one(rec):
    sid=rec['sample_id'];dest=OUT/'cache/synthetic'/f'{sid}.npz';meta=dest.with_suffix('.json')
    if meta.exists():return read(meta)
    rawpath=PREPARED.parent/rec['source'];labelpath=PREPARED.parent/rec['labels']
    raw=arrays(rawpath);lab=arrays(labelpath);image,points,spacing=synthetic_record(raw,lab,rec['kind'])
    seed=SEED+int(sid.split('_')[1]);train=rec['split']=='synthetic_train'
    partition='train' if train else ('validation' if int(sha(labelpath)[:8],16)%2==0 else 'test')
    if rec['kind']=='sequence':
        t,p,m=observations(lab['t'],points,seed)
        anchors=select_anchors(t,m,lab['edges'],seed,train)
        stats=cache_bags(dest,t,p,m,lab['edges'],lab['t'],anchors,image,spacing,dense=True)
    else:
        data=detection_data(image[0],points,seed);save(dest,**data);stats=dict(patches=len(data['target']),positive_centers=int(data['positive'].sum()))
    record=dict(source='synthetic_'+rec['kind'],sample=sid,partition=partition,provenance_group=sid,representation_group=sid,
        source_sha256=sha(rawpath),label_sha256=sha(labelpath),label_kind='dense_simulated',task_mask=['centers'] if rec['kind']=='static' else ['edges','events','paired_images'],
        image_representation='uint8_local_quantile_normalized_from_uint16',grid=list(image.shape),voxel_spacing=spacing,spacing_verified=True,
        corruption_seed=seed,frame_valid='explicit per node, offsets -1/0/1',future_observed='only t < last observed; missing daughters censor pair',
        candidate_visibility='six observed nearest daughters; absent alternatives masked',cache=dest,**stats)
    write(meta,record);return record

def zoo(species):
    p=PREPARED/'zoo'/f'{species}_graph.npz';lab=arrays(p)
    t0=lab['t'];points=lab['zyx_source'];edges=lab['edges'];n=len(t0)
    assert np.all(t0[edges[:,1]]==t0[edges[:,0]]+1)
    assert np.max(np.bincount(edges[:,1],minlength=n))<=1
    assert np.max(np.bincount(edges[:,0],minlength=n))<=2
    # Reconstruct direct parent endpoint semantics, independently of stored edges.
    tr=lab['tracklet_id'];first=np.r_[0,np.flatnonzero(np.diff(tr))+1];last=np.r_[first[1:]-1,n-1]
    ptr=lab['parent_tracklet'];children=np.flatnonzero(ptr>=0)
    cross=edges[tr[edges[:,0]]!=tr[edges[:,1]]]
    expected=np.c_[last[ptr[children]],first[children]]
    assert set(map(tuple,cross))==set(map(tuple,expected))
    tt,pp,m=observations(t0,points,SEED+len(species))
    rng=np.random.default_rng(SEED);anchors=[];maxf=int(t0.max());cut1=int(maxf*.7);cut2=int(maxf*.85)
    degrees=np.bincount(edges[:,0],minlength=n)
    for f in np.unique(tt):
        ix=np.flatnonzero((tt==f)&(m>=0))
        for label in [True,False]:
            selected=ix[(degrees[m[ix]]==2)==label]
            anchors.extend(rng.choice(selected,min(64,len(selected)),False))
    anchors=np.array(anchors,np.int64)
    a,c,x=build(tt,pp,anchors);y,ly,w=supervision(a,c,m,edges,t0,weak=True)
    records=[]
    for part,mask in [('train',tt[a]<cut1-6),('validation',(tt[a]>=cut1+6)&(tt[a]<cut2-6)),('test',tt[a]>=cut2+6)]:
        dest=OUT/'cache/zoo'/f'{species}_{part}.npz'
        save(dest,x=x[mask],target=y[mask],link_y=ly[mask],weight=w[mask],group=m[a[mask]],anchor_t=tt[a[mask]])
        rec=dict(source='zoo_'+species,sample=species+'_'+part,partition=part,provenance_group=species+'_time_'+part,
            acquisition=species,independent_acquisition=False,representation_group=species+'_time_'+part,source_sha256=sha(p),
            label_kind='weak_experimental_tracking',task_mask=['geometry','weak_edges','weak_events'],images=False,
            image_representation='absent',grid='source ZYX',voxel_spacing=None,spacing_verified=False,
            frame_valid='nearest observed frame offsets',future_observed='purged contiguous blocks; missing daughter censored',candidate_visibility='observation corruption mask',
            bags=int(mask.sum()),positive_groups=int(((y>0)&mask).sum()),negative_groups=int(((y==0)&mask).sum()),
            censored_or_unreachable=int(((y<0)&mask).sum()),cache=dest)
        records.append(rec)
    write(OUT/'cache/zoo'/f'{species}_audit.json',dict(nodes=n,edges=len(edges),direct_parent_edges_checked=len(cross),
        raw_sha256=sha(p),purge=6,records=records))
    return records

def real_one(row):
    import zarr
    name=row['dataset'];dest=OUT/'cache/real'/f'{name}.npz';meta=dest.with_suffix('.json')
    if meta.exists():return read(meta)
    g=arrays(V3/'selected_predictions'/f'{name}.npz');gt=arrays(V1/'evaluation/gt'/f'{name}.npz')
    matches=arrays(OUT/'evaluation_matches/C0'/f'{name}.npz')['matches'];mt=dict(matches)
    nodes=g['nodes'];mapping=np.array([mt.get(int(i),-1) for i in nodes[:,0]])
    gtindex={int(n):i for i,n in enumerate(gt['nodes'][:,0])}
    mapping=np.array([gtindex.get(int(i),-1) for i in mapping])
    edge=np.array([[gtindex[int(a)],gtindex[int(b)]] for a,b in gt['edges']],np.int64).reshape(-1,2)
    degree=np.bincount(edge[:,0],minlength=len(gt['nodes']))
    possible=np.flatnonzero((mapping>=0)&(nodes[:,1]<row['image_shape'][0]-1))
    div=possible[degree[mapping[possible]]==2];non=np.setdiff1d(possible,div)
    rng=np.random.default_rng(SEED+int(name.split('_')[1],16));a=np.sort(np.r_[div,rng.choice(non,min(128,len(non)),False)])
    image=zarr.open_group(row['image_path'],mode='r')['0']
    stats=cache_bags(dest,nodes[:,1],nodes[:,2:].astype(float),mapping,edge,gt['nodes'][:,1],a,image,row['physical_scale'])
    # Native-grid positive centers plus validated dark background, not unmatched cells.
    det=[]
    for f in np.unique(gt['nodes'][:,1])[::max(1,len(np.unique(gt['nodes'][:,1]))//8)]:
        pts=gt['nodes'][gt['nodes'][:,1]==f,2:]
        if len(pts):det.append(detection_data(image[int(f)],pts,SEED+int(f),real=True,spacing=row['physical_scale']))
    save(OUT/'cache/dreal'/f'{name}.npz',**{k:np.concatenate([d[k] for d in det]) for k in det[0]})
    rec=dict(source='biohub_'+row['embryo'],sample=name,partition='source_only_adaptation',provenance_group=name,
        image_representation='uint8_local_quantile_normalized_from_uint16',grid=row['image_shape'],voxel_spacing=row['physical_scale'],spacing_verified=True,
        label_kind='sparse_observed',task_mask=['positive_centers','supported_edges','supported_events'],
        frame_valid='explicit per node offsets -1/0/1',future_observed='quiet continuity requires 2 observed transitions each side',candidate_visibility='six nearest current detections',
        source_sha256=sha(V1/'evaluation/gt'/f'{name}.npz'),cache=dest,**stats)
    write(meta,rec);return rec

def run():
    status('W410',state='building compact source caches')
    manifest=read(PREPARED/'synthetic_manifest.json');records=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i,r in enumerate(pool.map(synthetic_one,manifest,chunksize=2)):
            records.append(r)
            if i%100==0:print('paired source',i+1,'/',len(manifest),flush=True)
    for species in ['zebrafish','ascidian']:
        ap=OUT/'cache/zoo'/f'{species}_audit.json'
        records += read(ap)['records'] if ap.exists() else zoo(species)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i,r in enumerate(pool.map(real_one,inputs(),chunksize=1)):
            records.append(r)
            if i%10==0:print('sparse source',i+1,'/199',flush=True)
    # Separate domains persist in each record; real samples are never pooled in a fit.
    records=[metadata_contract(r) for r in records]
    with (OUT/'runtime_dataset_index.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
    summary=pd.DataFrame(records).groupby(['source','partition']).agg(samples=('sample','size'),bags=('bags','sum'),positive_groups=('positive_groups','sum'),negative_groups=('negative_groups','sum'))
    summary.to_csv(OUT/'source_coverage.csv');print(summary,flush=True)
    status('W410',state='source caches complete')
