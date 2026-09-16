"""The installed temporal 3D U-Net and native transformer; no replacement heads."""
import sys, functools
import torch
import numpy as np
import zarr
from .common import NATIVE, V1, OUT, DATA, read, sha, write

def interfaces():
    for p in [NATIVE/'src',NATIVE/'scripts']:
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    import predict_unet_transformer as native
    import train_unet_transformer as train
    return native,train

def load(checkpoint=None,device='cuda'):
    native,_=interfaces()
    original=NATIVE/'weights/unet_transformer/split_0/edge_predictor_best.pth'
    model,window,ds=native.load_model(original,torch.device(device))
    assert window==2 and tuple(ds)==(1,4,4)
    if checkpoint is not None:
        payload=torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(payload['state'])
    model.eval()
    return model

class Frames:
    def __init__(self,path):
        self.path=path;self.array=zarr.open_group(str(path),mode='r')['0']
        meta=read(path/'zarr.json');self.shape=self.array.shape
        self.quantiles=meta['attributes']['image_statistics']['quantiles']
        self.low=float(self.quantiles['0.001']);self.high=float(self.quantiles['0.999'])
    @functools.lru_cache(maxsize=6)
    def frame(self,t):
        native,_=interfaces()
        raw=native._load_frame(self.array,t,[64,64,64],(1,4,4))
        return ((raw-self.low)/(self.high-self.low+1e-6)).clamp(0.)
    def pair(self,t):return torch.stack([self.frame(t),self.frame(t+1)])[None].cuda()

def node_inputs(nodes,shape=(100,64,256,256)):
    _,train=interfaces()
    grid=nodes[:,2:].astype(np.float32)/[1,4,4]
    # The installed extractor truncates floating coordinates; keep that behavior.
    coords=torch.tensor(grid,dtype=torch.float32,device='cuda')[None]
    relative=np.column_stack([nodes[:,1]-nodes[:,1].min(),grid])
    pos=torch.tensor(train.extract_pos_features(relative,(2,64,64,64)),device='cuda')[None]
    mask=torch.ones((1,len(nodes)),device='cuda',dtype=torch.bool)
    return coords,pos,mask

def pair_inputs(a,b):
    _,train=interfaces();ds=np.array([1,4,4],np.float32)
    result=[]
    for time,nodes in enumerate([a,b]):
        coords=torch.tensor(nodes[:,2:]/ds,dtype=torch.float32,device='cuda')[None]
        rel=np.column_stack([np.full(len(nodes),time),nodes[:,2:]/ds])
        pos=torch.tensor(train.extract_pos_features(rel,(2,64,64,64)),device='cuda')[None]
        mask=torch.ones((1,len(nodes)),device='cuda',dtype=torch.bool)
        result.append((coords,pos,mask))
    return result

def logits(model,a,b,features=None,images=None):
    inp=pair_inputs(a,b)
    if features is None:
        feat,_=model.encode(images)
        features=[model._index_features(feat[:,t],inp[t][0],inp[t][2]) for t in range(2)]
    elif not isinstance(features[0],torch.Tensor):
        features=[torch.tensor(f,dtype=torch.float32,device='cuda')[None] for f in features]
    ds=torch.tensor([1,4,4],device='cuda')
    return model.predict_edges(features[0],features[1],inp[0][0]*ds,inp[1][0]*ds,
        inp[0][1],inp[1][1],inp[0][2],inp[1][2])[0]

def receipt(model):
    return dict(architecture='installed UNetNodeTransformer / TemporalUNet3D / SimpleNodeTransformer',
        parameters=sum(p.numel() for p in model.parameters()),encoder_parameters=sum(p.numel() for p in model.unet.parameters()),
        module_files={str(p.relative_to(NATIVE)):sha(p) for p in [NATIVE/'scripts/train_unet_transformer.py',
            NATIVE/'src/biohub_tracking/models/temporal_unet.py',NATIVE/'src/biohub_tracking/models/simple_node_transformer.py']},
        keys_shapes={k:list(v.shape) for k,v in model.state_dict().items()},downsample=[1,4,4],
        spacing_um=[1.625,1.625,1.625],image_context_frames=2,decoder_context_frames=5,
        receptive_field='Full 64 cubed downsampled frame; cross-time voxel attention, global frame-node competitors',
        normalization='stored 0.001/0.999 quantiles; clamp minimum zero, no upper clipping',
        native_changed_tensors='N comparisons use primary checkpoint only, no 8-view detection TTA or secondary harmonic mixture; C0 remains full unchanged control')
