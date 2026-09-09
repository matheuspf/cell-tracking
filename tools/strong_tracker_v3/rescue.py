"""Bounded image-triggered localization and missing-native-point rescue.

No annotations, estimates, source labels or evaluation imports. Local alternatives
keep external edges and integer IDs. The unchanged graph is always available.
"""
from collections import OrderedDict,Counter
import time
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from .common import *

CONFIG=dict(native_probability=.8,min_native_persistence_frames=3,trigger_radius_um=10.,
    point_cap_fraction=.002,refine_cap_fraction=.01,refine_max_shift_um=.75,
    secondary_maxima_quantile=.999,secondary_min_distance_um=3.,secondary_persistence_radius_um=2.,
    secondary_maxima_per_gap=1,version='rescue_v3_1')

class Frames:
    def __init__(self,ctx,sample):
        import zarr
        self.ctx,self.name=ctx,sample['dataset'];self.a=zarr.open_group(sample['image_path'],mode='r')['0'];self.cache=OrderedDict()
    def get(self,t):
        if t not in self.cache:
            self.cache[t]=np.asarray(self.a[t]);
            if len(self.cache)>4:self.cache.popitem(last=False)
        return self.cache[t]
    def patch(self,t,center,radius=(1,4,4)):
        image=self.get(t);c=np.rint(center).astype(int);lo=np.maximum(c-radius,0);hi=np.minimum(c+np.asarray(radius)+1,image.shape)
        return image[tuple(slice(a,b) for a,b in zip(lo,hi))].astype(float),lo

def candidate_restore(ctx,sample,nodes,edges):
    name=sample['dataset'];scale=np.asarray(sample['physical_scale']);ix,pred,succ=adjacency(nodes,edges)
    raw=load_graph(ctx.v2/'raw'/f'{name}.npz');rn,re=raw['nodes'],raw['edges']
    with np.load(ctx.full/'inputs'/f'pre_ilp_{name}.npz',allow_pickle=False) as f:conf=f['node_probabilities']
    ri,rp,rs=adjacency(rn,re);existing=set(map(int,nodes[:,0]));missing=set()
    triggers={}
    for t in np.unique(nodes[:,1]):
        ids=[i for i in np.flatnonzero(nodes[:,1]==t) if (len(succ[i])!=1 or not pred[i]) and 0<t<sample['image_shape'][0]-1]
        if ids:triggers[int(t)]=cKDTree(nodes[ids,2:]*scale)
    eligible=0
    for i,r in enumerate(rn):
        k,t=int(r[0]),int(r[1])
        if k in existing or conf[k]<CONFIG['native_probability'] or t not in triggers:continue
        eligible+=1
        if triggers[t].query(r[2:]*scale)[0]<=CONFIG['trigger_radius_um']:missing.add(i)
    groups=[];seen=set()
    for i in sorted(missing):
        if i in seen:continue
        component=set();todo=[i]
        while todo:
            j=todo.pop()
            if j in component:continue
            component.add(j);todo.extend(k for k in rp[j]+rs[j] if k in missing and k not in component)
        seen|=component
        if len(component)>=3 and all(len(rs[j])<=1 and len(rp[j])<=1 for j in component):groups.append(component)
    cap=max(1,int(len(nodes)*CONFIG['point_cap_fraction']));added=[];add_edges=[];ledger=[];used=0
    occupied_in={int(b) for a,b in edges};occupied_out=Counter(map(int,edges[:,0]))
    for group in sorted(groups,key=lambda g:(-min(conf[int(rn[i,0])] for i in g),min(g))):
        if used+len(group)>cap:continue
        ids={int(rn[i,0]) for i in group};new_edges=[];valid=True
        for a,b in re:
            a,b=int(a),int(b)
            if a not in ids and b not in ids:continue
            if a not in ids|existing or b not in ids|existing:continue
            if b in occupied_in or occupied_out[a]>=1:valid=False;break
            new_edges.append((a,b))
        if not valid:continue
        nn=rn[sorted(group)].copy();nn[:,2:]=np.clip(np.rint(nn[:,2:]),0,np.array(sample['image_shape'][1:])-1)
        # The same stable native ID cannot be introduced twice or moved onto an existing center.
        if any(np.any(np.all(nodes[:,1:]==r[1:].astype(np.int64),axis=1)) for r in nn):continue
        added.extend(nn.astype(np.int64));add_edges.extend(new_edges);used+=len(group);existing|=ids
        for a,b in new_edges:occupied_in.add(b);occupied_out[a]+=1
        ledger.append(dict(kind='native_reappearance',node_ids=sorted(ids),added_edges=new_edges,removed_edges=[],
            minimum_native_confidence=float(min(conf[int(rn[i,0])] for i in group)),persistent_frames=len({rn[i,1] for i in group}),
            boundary_policy='Only previously unowned boundary endpoints; no external removal'))
    n=np.vstack([nodes,np.array(added).reshape(-1,5)]).astype(np.int64) if added else nodes.copy()
    e=np.vstack([edges,np.array(add_edges).reshape(-1,2)]).astype(np.int64) if add_edges else edges.copy()
    return n,e,ledger,dict(missing_native_above_confidence=eligible,triggered_missing=len(missing),persistent_groups=len(groups),restored_nodes=used,point_cap=cap)

def refine(ctx,sample,nodes,edges):
    n=nodes.copy();scale=np.asarray(sample['physical_scale']);ix,pred,succ=adjacency(n,edges)
    raw=load_graph(ctx.v2/'raw'/f'{sample["dataset"]}.npz')
    prob={tuple(map(int,p)):float(q) for p,q in zip(raw['edges'],raw['edge_prob'])}
    triggers=set()
    for i,links in enumerate(succ):
        if len(links)==2:triggers.update([i,*links])
        elif len(links)==1:
            j=links[0]
            if prob.get((int(n[i,0]),int(n[j,0])),0)<.7 and np.linalg.norm((n[j,2:]-n[i,2:])*scale)>4:
                triggers.update([i,j])
    frames=Frames(ctx,sample);cap=max(1,int(len(n)*CONFIG['refine_cap_fraction']));ledger=[]
    occupied={tuple(r[1:]) for r in n}
    for i in sorted(triggers,key=lambda i:(n[i,1],n[i,0])):
        if len(ledger)>=cap:break
        patch,lo=frames.patch(int(n[i,1]),n[i,2:]);w=np.maximum(patch-np.quantile(patch,.35),0)
        if not w.sum():continue
        grid=np.indices(w.shape).reshape(3,-1).T+lo
        center=np.average(grid,axis=0,weights=w.ravel());delta=center-n[i,2:]
        distance=np.linalg.norm(delta*scale)
        if distance>CONFIG['refine_max_shift_um']:delta*=CONFIG['refine_max_shift_um']/distance
        coord=np.rint(n[i,2:]+delta).astype(int);key=(int(n[i,1]),*map(int,coord))
        if np.array_equal(coord,n[i,2:]) or key in occupied:continue
        occupied.remove(tuple(n[i,1:]));occupied.add(key)
        ledger.append(dict(kind='local_image_centroid',node_id=int(n[i,0]),old=n[i,2:].tolist(),new=coord.tolist(),
            image_time=int(n[i,1]),shift_um=float(np.linalg.norm((coord-n[i,2:])*scale)),removed_edges=[],added_edges=[]))
        n[i,2:]=coord
    return n,ledger,dict(image_triggered_nodes=len(triggers),relocated=len(ledger),refinement_cap=cap)

def maxima_rescue(ctx,sample,nodes,edges):
    """Probe detached temporal maxima next to actual terminated trajectories."""
    n=nodes.copy();e=edges.copy();ix,pred,succ=adjacency(n,e);shape=sample['image_shape'];scale=np.asarray(sample['physical_scale'])
    frames=Frames(ctx,sample);endpoints=[i for i in range(len(n)) if not succ[i] and pred[i] and n[i,1]<shape[0]-4]
    cap=max(1,int(len(n)*CONFIG['point_cap_fraction']));ledger=[];added=[];ee=[];tested=0
    by_t={int(t):n[n[:,1]==t,2:]*scale for t in np.unique(n[:,1])};new_by_t={}
    nextid=int(n[:,0].max())+1
    # Fixed spatial/time hash subsampling bounds cost independently of GT.
    endpoints=[i for i in endpoints if int(n[i,0])%16==0]
    for i in sorted(endpoints,key=lambda i:(n[i,1],n[i,0])):
        if len(added)+3>cap:break
        tested+=1;pos=n[i,2:].astype(float);chain=[]
        for dt in [1,2,3]:
            t=int(n[i,1])+dt;patch,lo=frames.patch(t,pos,(2,8,8));threshold=np.quantile(patch,CONFIG['secondary_maxima_quantile'])
            peaks=np.argwhere((patch==maximum_filter(patch,size=3))&(patch>=threshold));accepted=[]
            for p0 in peaks:
                coord=(p0+lo).astype(int);dist=np.linalg.norm((coord-pos)*scale)
                if dist>CONFIG['secondary_persistence_radius_um']:continue
                known=np.vstack([by_t.get(t,np.empty((0,3))),np.asarray(new_by_t.get(t,[])).reshape(-1,3)])
                if len(known) and np.linalg.norm(known-coord*scale,axis=1).min()<CONFIG['secondary_min_distance_um']:continue
                accepted.append((float(patch[tuple(p0)]),coord))
            if not accepted:break
            _,pos=max(accepted,key=lambda x:x[0]);chain.append((t,pos.copy()))
        if len(chain)!=3:continue
        ids=list(range(nextid,nextid+3));nextid+=3
        for k,(t,p0) in zip(ids,chain):added.append([k,t,*p0]);new_by_t.setdefault(t,[]).append(p0*scale)
        links=[(int(n[i,0]),ids[0]),(ids[0],ids[1]),(ids[1],ids[2])];ee.extend(links)
        ledger.append(dict(kind='persistent_secondary_maxima',anchor_id=int(n[i,0]),node_ids=ids,added_edges=links,
            removed_edges=[],persistent_frames=3,appearance='image local maxima, three consecutive frames; no GT centers'))
    if added:n=np.vstack([n,np.asarray(added,np.int64)]);e=np.vstack([e,np.asarray(ee,np.int64)])
    return n,e,ledger,dict(triggered_endpoints=len(endpoints),tested= tested,secondary_inserted_nodes=len(added),point_cap=cap)

def apply(ctx,sample,nodes,edges,mode):
    n,e,ledger,stats=candidate_restore(ctx,sample,nodes,edges)
    if mode=='R_image_local':
        n,ls,st=refine(ctx,sample,n,e);ledger+=ls;stats.update(st)
        n,e,ls,st=maxima_rescue(ctx,sample,n,e);ledger+=ls;stats.update(st)
    validate(n,e,sample['image_shape'])
    return n,e,ledger,stats

def one(task):
    ctx,sample=task;name=sample['dataset'];b=load_graph(ctx.incumbent(name));n,e=b['nodes'],b['edges']
    stamp=dict(graph=graph_hash(n,e),native=sha(ctx.full/'inputs'/f'pre_ilp_{name}.npz'),raw=sha(ctx.v2/'raw'/f'{name}.npz'),
        image_metadata=sample['metadata_hash'],code=sha(__file__),config=digest(CONFIG))
    dest=ctx.out/'rescue'/f'{name}.json'
    if dest.exists():
        if read_json(dest)['inputs']!=stamp:raise RuntimeError('Rescue cache drift')
        return name+' cached'
    start=time.perf_counter();variants={}
    for mode in ['R_native_restore','R_image_local']:
        nn,ee,ledger,stats=apply(ctx,sample,n,e,mode);path=ctx.out/'candidate_graphs'/mode/f'{name}.npz'
        save_graph(path,nn,ee);variants[mode]=dict(stats=stats,ledger=ledger,graph_hash=graph_hash(nn,ee),sha256=sha(path))
    write_json(dest,dict(inputs=stamp,variants=variants,seconds=time.perf_counter()-start));return name+' rescue'

def run(ctx,args):
    census=read_json(ctx.out/'census_summary.json')
    missing=sum(v for k,v in census['edge_reasons'].items() if 'missing' in k)
    if not missing:raise RuntimeError('V350 point-rescue trigger absent; document omitted')
    lock=ctx.out/'rescue_config_lock.json'
    old=read_json(lock) if lock.exists() else {}
    write_json(lock,dict(created=old.get('created',now()),config=CONFIG,code_sha256=sha(__file__),
        variants=['R_native_restore','R_image_local'],eligible_evidence='Incumbent census missing endpoints; fixed image-triggered policy'),immutable=True)
    list(run_pool(one,[(ctx,s) for s in ctx.samples()],args.workers))
    records=[read_json(ctx.out/'rescue'/f'{s["dataset"]}.json') for s in ctx.samples()]
    summary={v:dict(Counter({k:sum(r['variants'][v]['stats'].get(k,0) for r in records)
        for k in set().union(*(r['variants'][v]['stats'] for r in records))})) for v in ['R_native_restore','R_image_local']}
    write_json(ctx.out/'rescue_summary.json',dict(samples=len(records),variants=summary,config=CONFIG,
        focus='Optional local FOCUS checkpoint not needed; no new model dependency',
        cached_heatmaps='Native confidence and raw volumes used; no full-volume masks generated'))
