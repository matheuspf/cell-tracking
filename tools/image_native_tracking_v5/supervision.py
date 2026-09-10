"""Source-only, conservative supported incoming-parent targets. Never imported by inference."""
from .common import *

def supported_pairs(name,nodes,pairs=None):
    match=dict(map(tuple,arrays(OUT/'evaluation_matches/C0'/f'{name}.npz')['matches']))
    truth=arrays(V1/'evaluation/gt'/f'{name}.npz');reverse={int(g):int(p) for p,g in match.items()}
    nodeids=set(map(int,nodes[:,0]));positive=[];missing=0
    for a,b in truth['edges']:
        pa,pb=reverse.get(int(a)),reverse.get(int(b))
        if pa in nodeids and pb in nodeids:positive.append((pa,pb))
        else:missing+=1
    positive=set(positive)
    if pairs is None:return positive,missing
    covered=positive & set(map(tuple,pairs));targets={b for a,b in covered}
    labels=np.array([1 if tuple(e) in covered else 0 if int(e[1]) in targets else -1 for e in pairs],np.int8)
    return labels,dict(gt_edges=len(truth['edges']),endpoints_present=len(positive),candidate_positives=len(covered),missing_endpoints=missing,
        supported_negative=int((labels==0).sum()),unknown=int((labels<0).sum()))

def native_catalog(source):
    catalog=[];total=0;missing=0
    for row in inventory():
        if row['embryo']!=source:continue
        name=row['dataset'];nodes=graph(name)['nodes'];positive,censored=supported_pairs(name,nodes)
        times={int(n[0]):int(n[1]) for n in nodes};groups={}
        for a,b in positive:groups.setdefault(times[a],[]).append([a,b])
        for t,edges in sorted(groups.items()):catalog.append(dict(dataset=name,t=int(t),positive=sorted(edges)))
        total+=len(positive);missing+=censored
    write(OUT/'training_labels'/f'{source}_native_catalog.json',dict(source=source,rows=catalog,positive_edges=total,censored_edges=missing,
        selection='all source transitions with both C0 endpoints; full predicted frame supplies competing parents; absent predecessor censored',
        inherited_exposure='public native checkpoints and C0 teachers retain prior embryo exposure'))
    return catalog
