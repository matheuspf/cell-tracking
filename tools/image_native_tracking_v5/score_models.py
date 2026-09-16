"""Feature and edge scoring functions shared by batch and fresh-image prediction."""
import torch
from .common import *
from . import native_adapter as native
from .hoct_adapter import iter_embeddings
from .cache import score_features

def encode_at_nodes(model,nodes,path):
    frames=native.Frames(path);fs=np.zeros((len(nodes),32),np.float32);ft=np.zeros_like(fs)
    with torch.no_grad():
        for t in range(99):
            ai=np.flatnonzero(nodes[:,1]==t);bi=np.flatnonzero(nodes[:,1]==t+1)
            feat,_=model.encode(frames.pair(t));inp=native.pair_inputs(nodes[ai],nodes[bi])
            if len(ai):fs[ai]=model._index_features(feat[:,0],inp[0][0],inp[0][2])[0].cpu().numpy()
            if len(bi):ft[bi]=model._index_features(feat[:,1],inp[1][0],inp[1][2])[0].cpu().numpy()
    # Quantization is identical for cached and fresh runs and all native controls.
    return fs.astype(np.float16).astype(np.float32),ft.astype(np.float16).astype(np.float32)

def native_scores(model,nodes,pairs,fs,ft):return score_features(model,nodes,pairs,fs,ft)

def hoct_scores(model,nodes,pairs,properties,valid,heads=()):
    out={'H0':np.full(len(pairs),np.nan,np.float32)}
    for name,head in heads:out[name]=np.full(len(pairs),np.nan,np.float32)
    windows=[]
    for ii,raw,feat,meta in iter_embeddings(model,nodes,pairs,properties,valid):
        out['H0'][ii]=raw;windows.append(meta)
        for name,head in heads:
            out[name][ii]=(feat@head['weight'].numpy().ravel()+float(head['bias'].item())).astype(np.float32)
    return out,dict(edges=len(pairs),valid_edges=int(np.isfinite(out['H0']).sum()),windows=len(windows),
        max_nodes=max([r['nodes'] for r in windows],default=0),max_edges=max([r['edges'] for r in windows],default=0),
        genuine_pretrained_backbone_executed=True)
