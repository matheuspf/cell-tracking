"""Prediction-only coarse peaks, continuation assignment, and integer CSV."""
import csv
import hashlib
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def peaks(logits):
    if not np.isfinite(logits).all():raise ValueError('Nonfinite detection logits')
    grid=np.stack(np.meshgrid(*[np.arange(-2,3)]*3,indexing='ij'),-1)
    footprint=np.linalg.norm(grid*1.625,axis=-1)<=3.
    candidates=np.argwhere((logits>0)&(logits==maximum_filter(logits,footprint=footprint,mode='constant',cval=-np.inf)))
    if not len(candidates):return np.empty((0,3),np.int64),np.empty(0,np.float32)
    value=logits[tuple(candidates.T)]
    order=np.lexsort((candidates[:,2],candidates[:,1],candidates[:,0],-value))
    tree=cKDTree(candidates*1.625);suppressed=set();keep=[]
    for i in order:
        if int(i) not in suppressed:
            keep.append(int(i));suppressed.update(tree.query_ball_point(candidates[i]*1.625,3.))
    keep=sorted(keep,key=lambda i:tuple(candidates[i]))
    return candidates[keep]*np.array([1,4,4]),value[keep]


def candidate_union(a,b,*,neighbors=6,reverse_neighbors=6,gate=15.):
    pairs=set();scale=np.array([1.625,.40625,.40625])
    for first,second,n,reverse in [(a,b,neighbors,False),(b,a,reverse_neighbors,True)]:
        if not len(first) or not len(second):continue
        distance,index=cKDTree(second*scale).query(first*scale,k=min(n,len(second)))
        for i,(dd,jj) in enumerate(zip(np.asarray(distance).reshape(len(first),-1),np.asarray(index).reshape(len(first),-1))):
            for d,j in zip(dd,jj):
                if d<=gate:pairs.add((int(j),i) if reverse else (i,int(j)))
    return np.asarray(sorted(pairs),np.int64).reshape(-1,2)


def continuation(a,b,pairs,logits):
    if not len(a) or not len(b):return np.empty((0,2),np.int64)
    if len(pairs)!=len(logits) or not np.isfinite(logits).all():raise ValueError('Missing/nonfinite association score')
    # Every source has an explicit zero-utility private null column.
    utility=np.full((len(a),len(b)+len(a)),-np.inf,dtype=np.float64)
    utility[np.arange(len(a)),len(b)+np.arange(len(a))]=0
    for (i,j),value in zip(pairs,logits):
        if value>0:utility[i,j]=value
    row,col=linear_sum_assignment(utility,maximize=True)
    return np.array([(i,j) for i,j in zip(row,col) if j<len(b)],np.int64).reshape(-1,2)


def export_csv(path,dataset,nodes,edges,shape=(100,64,256,256)):
    nodes=np.asarray(nodes).copy()
    if len(nodes):nodes[:,1:]=np.clip(np.rint(nodes[:,1:]),0,np.array(shape)-1)
    nodes=nodes.astype(np.int64);edges=np.asarray(edges,np.int64).reshape(-1,2)
    validate(nodes,edges)
    with path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','dataset','row_type','node_id','t','z','y','x','source_id','target_id'])
        index=0
        for n in nodes:writer.writerow([index,dataset,'node',*n,-1,-1]);index+=1
        for a,b in edges:writer.writerow([index,dataset,'edge',-1,-1,-1,-1,-1,a,b]);index+=1
    return nodes,edges


def read_csv(path,dataset):
    nodes=[];edges=[]
    with path.open() as f:
        for row in csv.DictReader(f):
            if row['dataset']!=dataset:raise ValueError('CSV dataset mismatch')
            if row['row_type']=='node':nodes.append([int(row[k]) for k in ('node_id','t','z','y','x')])
            elif row['row_type']=='edge':edges.append([int(row[k]) for k in ('source_id','target_id')])
            else:raise ValueError('Unknown CSV row type')
    nodes=np.asarray(nodes,np.int64).reshape(-1,5);edges=np.asarray(edges,np.int64).reshape(-1,2)
    validate(nodes,edges)
    return nodes,edges


def validate(nodes,edges):
    ids={int(n[0]):int(n[1]) for n in nodes}
    if len(ids)!=len(nodes):raise ValueError('Duplicate persisted node ID')
    incoming={};outgoing={}
    if len(set(map(tuple,edges)))!=len(edges):raise ValueError('Duplicate edge')
    for a,b in edges:
        if a not in ids or b not in ids or ids[b]!=ids[a]+1:raise ValueError('Invalid edge ID/time')
        incoming[b]=incoming.get(b,0)+1;outgoing[a]=outgoing.get(a,0)+1
    if any(n>1 for n in incoming.values()) or any(n>2 for n in outgoing.values()):raise ValueError('Invalid ownership/degree')


def graph_hash(nodes,edges):
    digest=hashlib.sha256()
    for array in (nodes,edges):digest.update(np.asarray(array,dtype='<i8').tobytes())
    return digest.hexdigest()
