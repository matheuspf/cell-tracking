"""Image-only translation checks and qualified biological-event deduplication."""
from collections import defaultdict,deque

import numpy as np
import zarr

from .common import DATA,OUT,V1,SCALE,inventory,load_graph,read_json,write_json


def run(args):
    regs=read_json(V1/'overlap_registration.json');adj=defaultdict(list)
    verified=[]
    for r in regs:
        a,b=r['dataset_a'],r['dataset_b'];off=np.array(r['offset_tzyx'])
        aa={x[0]:np.array(x[1:]) for x in read_json(V1/'fingerprints'/f'{a}.json')}
        bb={x[0]:np.array(x[1:]) for x in read_json(V1/'fingerprints'/f'{b}.json')}
        shared=[h for h in sorted(aa.keys()&bb.keys()) if np.array_equal(aa[h]-bb[h],off)]
        checks=[]
        za=zarr.open_group(DATA/'train'/f'{a}.zarr',mode='r')['0'];zb=zarr.open_group(DATA/'train'/f'{b}.zarr',mode='r')['0']
        for h in shared[:3]:
            pa,pb=aa[h],bb[h]
            cuts=[]
            for arr,p in [(za,pa),(zb,pb)]:
                cuts.append(np.asarray(arr[int(p[0]),int(p[1])-1:int(p[1])+2,int(p[2])-2:int(p[2])+3,int(p[3])-2:int(p[3])+3]))
            checks.append(np.array_equal(*cuts))
        passed=bool(checks) and all(checks)
        verified.append(dict(dataset_a=a,dataset_b=b,patch_checks=len(checks),passed=passed,offset_tzyx=off.tolist()))
        if passed:
            adj[a].append((b,off));adj[b].append((a,-off))
    origins={};components={};cycles=[]
    for name in sorted(adj):
        if name in origins:continue
        origins[name]=np.zeros(4,int);components[name]=name;q=deque([name])
        while q:
            a=q.popleft()
            for b,off in adj[a]:
                expected=origins[a]+off
                if b in origins:
                    if not np.array_equal(origins[b],expected):cycles.append(dict(a=a,b=b,residual=(origins[b]-expected).tolist()))
                else:origins[b]=expected;components[b]=name;q.append(b)
    events=[]
    for r in inventory():
        name=r['dataset'];gt=load_graph(V1/'evaluation/gt'/f'{name}.npz')
        ids,count=np.unique(gt['edges'][:,0],return_counts=True)
        for ident in ids[count>=2]:
            node=gt['nodes'][gt['nodes'][:,0]==ident][0]
            events.append(dict(dataset=name,id=int(ident),embryo=r['embryo'],component=components.get(name,name),
                global_tzyx=(node[1:]+origins.get(name,np.zeros(4,int))).tolist(),registered=name in origins))
    parent=list(range(len(events)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(events):
        for j,b in enumerate(events[:i]):
            if a['component']!=b['component'] or a['dataset']==b['dataset']:continue
            delta=np.array(a['global_tzyx'])-b['global_tzyx']
            if abs(delta[0])<=1 and np.linalg.norm(delta[1:]*SCALE)<=3:
                parent[root(i)]=root(j)
    for i,e in enumerate(events):e['provisional_event_group']=root(i)
    write_json(OUT/'evaluation/registered_events.json',events)
    write_json(OUT/'registration_validation.json',dict(registered_pairs=len(regs),fresh_raw_patch_checks=sum(r['patch_checks'] for r in verified),
        all_patch_checks_passed=all(r['passed'] for r in verified),registered_samples=len(origins),samples=199,
        inconsistent_cycles=cycles,registered_components=len(set(components.values())),
        annotated_division_observations=len(events),provisional_distinct_event_groups=len({e['provisional_event_group'] for e in events}),
        by_embryo={s:dict(samples=sum(r['embryo']==s for r in inventory()),registered_samples=sum(n.startswith(s) for n in origins),
            division_observations=sum(e['embryo']==s for e in events),provisional_event_groups=len({e['provisional_event_group'] for e in events if e['embryo']==s})) for s in ['44b6','6bba']},
        purged_independence_established=False,
        reason='Image translations cover only part of the clips; unknown relationships remain possible; neither unseen overlaps nor feature-support purges can be certified across all source blocks',
        tuning='Fixed limited source configurations; no fabricated bootstrap confidence intervals',pairs=verified))
