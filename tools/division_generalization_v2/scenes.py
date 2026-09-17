"""Stationary prediction-centered seven-frame scenes; only raw uint8 is cached."""
from collections import OrderedDict
import time
import numpy as np
from scipy.ndimage import map_coordinates

from pipeline_error_training.training_images import FrameStatistics, TrainingImages
from .common import WORK, save_arrays, sha, read_json

SHAPE = (16,64,64)
OFFSETS = tuple(range(-3,4))
GRID = np.stack(np.meshgrid(*[np.arange(n)-(n-1)/2 for n in SHAPE],indexing='ij'))


def crop(images, t, position):
    result = np.zeros((7,2,2,*SHAPE),np.uint8)
    for j,dt in enumerate(OFFSETS):
        frame = images.raw(t+dt)
        if frame is None:
            continue
        lo,hi = images.quantiles[t+dt]
        for s,factor in enumerate((1.,2.)):
            coords = np.asarray(position)[:,None,None,None] + factor*GRID
            valid = np.all((coords>=0)&(coords<=np.asarray(frame.shape)[:,None,None,None]-1),axis=0)
            value = map_coordinates(frame.astype(np.float32,copy=False),coords,order=1,mode='constant',cval=0)
            result[j,s,0] = np.rint(255*np.clip((value-lo)/max(1.,hi-lo),0,1)*valid).astype(np.uint8)
            result[j,s,1] = valid.astype(np.uint8)*255
    return result


class FloatImages(TrainingImages):
    def raw(self,t):
        frame=super().raw(t)
        if frame is not None and frame.dtype!=np.float32:
            frame=frame.astype(np.float32)
            self.frames[t]=frame
        return frame


class Scenes:
    def __init__(self, rows, cache_root=None, persist=True):
        self.rows = {r['dataset']:r for r in rows}
        self.root = cache_root or WORK/'scenes'
        self.persist=persist
        self.stats = FrameStatistics()
        self.images = OrderedDict()
        self.memory = OrderedDict()
        self.cold_seconds = self.warm_seconds = 0.
        self.image_digests={}
        if persist:
            for row in rows:
                path=WORK/'image_hashes'/(row['dataset']+'.json')
                if path.exists():self.image_digests[row['dataset']]=read_json(path)['content_sha256']

    def get(self,name,anchor,position,t):
        start = time.monotonic()
        key = (name,int(anchor))
        if key in self.memory:
            self.memory.move_to_end(key)
            self.warm_seconds += time.monotonic()-start
            return self.memory[key]
        row = self.rows[name]
        path = self.root/name/(str(anchor)+'.npz')
        fingerprint = (sha(__import__('pathlib').Path(__file__)) + row['metadata_sha256'] +
                       self.image_digests.get(name,''))
        if self.persist and path.exists():
            with np.load(path,allow_pickle=False) as f:
                if str(f['fingerprint']) != fingerprint:
                    raise ValueError('Raw scene cache code/image metadata drift')
                result = f['scene']
            self.warm_seconds += time.monotonic()-start
        else:
            if name not in self.images:
                self.images[name] = FloatImages(row['image_path'],self.stats,max_frames=9)
                while len(self.images)>2:
                    self.images.popitem(last=False)
            self.images.move_to_end(name)
            result = crop(self.images[name],t,position)
            if self.persist:
                save_arrays(path,scene=result,fingerprint=np.array(fingerprint))
            self.cold_seconds += time.monotonic()-start
        self.memory[key] = result
        while len(self.memory)>256:
            self.memory.popitem(last=False)
        return result


def augment(scene, query, seed):
    """Identical physical transforms for all frames, scales and point queries."""
    import torch
    import torch.nn.functional as F
    g = torch.Generator(device=scene.device).manual_seed(int(seed))
    draws = torch.rand(7,generator=g,device=scene.device)
    q = query.clone()
    x = scene
    for axis in (1,2):
        if draws[axis-1] < .5:
            x = x.flip(-3+axis)
            q[:,axis] *= -1
    turns = int(draws[2]*4)
    x = torch.rot90(x,turns,(-2,-1))
    for _ in range(turns):
        q = torch.stack((q[:,0],-q[:,2],q[:,1]),-1)
    jitter = (draws[3:6]*2-1)*scene.new_tensor([1.,2.,2.])
    sizes = scene.new_tensor(SHAPE)-1
    grids = []
    for scale in (1.,2.):
        theta = torch.eye(3,4,device=scene.device)
        theta[:,3] = (2*jitter/(sizes*scale)).flip(0)
        grids.append(F.affine_grid(theta[None].expand(7,-1,-1),(7,2,*SHAPE),align_corners=True))
    grid = torch.stack(grids,1).flatten(0,1)
    x = F.grid_sample(x.flatten(0,1),grid,align_corners=True).reshape_as(x)
    q -= jitter
    image, mask = x[:,:,0], x[:,:,1]
    gamma = .85+.3*draws[6]
    brightness = .9+.2*draws[0]
    image = (image.clamp_min(0).pow(gamma)*brightness).clamp(0,1)
    noise = torch.randn((1,1,*SHAPE),generator=g,device=x.device)*.01
    image = (image+noise).clamp(0,1)
    if draws[1] < .25:
        image = F.avg_pool3d(image.flatten(0,1)[:,None],(1,3,3),1,(0,1,1))[:,0].reshape_as(image)
    return torch.stack((image*mask,mask),2),q
