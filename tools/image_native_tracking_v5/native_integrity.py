"""Additional actual-network optical test and architecture receptive-field accounting."""
import torch
from .common import *
from . import native_adapter as native


def convolution_field():
    def conv(x):return [set().union(*x[max(0,i-1):min(len(x),i+2)]) for i in range(len(x))]
    def block(x):return conv(conv(x))
    def pool(x):return [x[i]|x[i+1] for i in range(0,len(x),2)]
    def up(x):
        result=[]
        for i in range(len(x)*2):
            position=(i+.5)/2-.5;left=max(0,min(len(x)-1,int(np.floor(position))));right=max(0,min(len(x)-1,int(np.ceil(position))))
            result.append(x[left]|x[right])
        return result
    a=block([{i} for i in range(64)]);b=block(pool(a));c=block(pool(b))
    d=block([u|s for u,s in zip(up(c),b)]);e=block([u|s for u,s in zip(up(d),a)])
    center={str(i):dict(first_input=min(e[i]),last_input=max(e[i]),voxels=len(e[i]),voxel_support_width_um=len(e[i])*1.625) for i in range(30,34)}
    write(OUT/'architecture_details.json',dict(created=now(),native_encode_input_shape=[1,2,64,64,64],unet_internal_shape=[1,2,1,64,64,64],input_field_width_um=[104,104,104],
        local_feature_support=center,derivation='exact 1D index union through two 3x3 convs per stage, two 2x pools, trilinear x2 upsamples and decoder convs; applies independently in ZYX',
        temporal_attention='two real frames, per-voxel attention at lower resolutions',
        global_association='cross-attention over all predicted nodes in both full frames; association can therefore depend on the full observed field',
        interpretation='theoretical convolutional support, not empirical nonzero-gradient area; padding truncates support near borders',
        native_source_sha256=sha(NATIVE/'src/biohub_tracking/models/temporal_unet.py')))


def optical():
    torch.set_num_threads(2)
    with gpu_aux():
        model=native.load();grid=torch.meshgrid(*(torch.arange(64,device='cuda') for _ in range(3)),indexing='ij')
        empty=torch.zeros((1,2,64,64,64),device='cuda');blob=empty.clone()
        for t in range(2):blob[0,t]=3*torch.exp(-sum((grid[k]-(30+t if k==2 else 32))**2 for k in range(3))/8.)
        with torch.no_grad():
            fa,da=model.encode(empty);fb,db=model.encode(blob)
        feature_difference=float((fa-fb).abs().max());heat_difference=max(float((a.sigmoid()-b.sigmoid()).abs().max()) for a,b in zip(da,db))
        assert feature_difference>1e-5 and heat_difference>1e-5
        write(OUT/'actual_native_pixel_test.json',dict(at=now(),passed=True,actual_pretrained_3D_network_executed=True,
            synthetic_fixture='zero image versus moving Gaussian optical source, two full 64-cubed frames',
            feature_max_abs_difference=feature_difference,heatmap_max_abs_difference=heat_difference,
            checkpoint_sha256=sha(NATIVE/'weights/unet_transformer/split_0/edge_predictor_best.pth'),
            labels_read=False,changed_model_parameters=False))


if __name__=='__main__':convolution_field();optical()
