"""Offline provider boundary. Missing runtimes fail before any model download.

FOCUS invocation is explicitly bound to a local infer_volume signature; no
unobserved upstream call signature or coordinate transform is guessed. This
binding has NOT been validated against FOCUS in the recorded environment.
"""
from dataclasses import dataclass
import inspect
from pathlib import Path
import time
import numpy as np
from .common import Blocked, sha, digest
from .regions import extract
from .network import deny_external_network

@dataclass(frozen=True)
class NativeGrid:
    shape: tuple[int,int,int]
    spacing_um: tuple[float,float,float]
    origin: tuple[float,float,float]=(0.,0.,0.)
    axes: str='zyx'

    def check(self,image,labels):
        if self.axes!='zyx' or self.origin!=(0.,0.,0.): raise ValueError('Unverified provider lattice')
        if np.shape(image)!=self.shape or np.shape(labels)!=self.shape: raise ValueError('Native ZYX shape changed')
        if len(self.spacing_um)!=3 or not np.isfinite(self.spacing_um).all() or min(self.spacing_um)<=0:
            raise ValueError('Invalid physical spacing')
        a=np.asarray(labels)
        if a.dtype.kind not in 'iu' or (a<0).any(): raise ValueError('Expected instance labels, not logits or foreground')

class FocusAdapter:
    """Bind a resident, authorized backend after local API inspection.

    parameter_map maps image/spacing/diameter/checkpoint to explicit backend
    keyword names. A backend which cannot return native-grid labels is rejected.
    No callable construction, dependency installation, or URL resolution occurs.
    """
    provider='FOCUS-3D-nuclei'
    def __init__(self, infer_volume, checkpoint, parameter_map, fixed_kwargs=None, labels_key=None):
        deny_external_network()
        self.checkpoint=Path(checkpoint)
        if not self.checkpoint.is_file(): raise Blocked('FOCUS nuclei checkpoint absent: '+str(checkpoint))
        if not callable(infer_volume) or infer_volume.__name__!='infer_volume':
            raise Blocked('A locally inspected headless infer_volume callable is required')
        if set(parameter_map)!={'image','spacing','diameter','checkpoint'}:
            raise ValueError('Explicit image, spacing, diameter and checkpoint mapping required')
        self.fn=infer_volume;self.mapping=dict(parameter_map);self.fixed=dict(fixed_kwargs or {});self.key=labels_key
        if set(self.mapping.values()) & set(self.fixed): raise ValueError('Duplicate backend binding')
        inspect.signature(self.fn).bind(**{**self.fixed,**dict.fromkeys(self.mapping.values())})
        source=Path(inspect.getfile(self.fn))
        self.provenance=dict(checkpoint_sha256=sha(self.checkpoint),source_sha256=sha(source),
                             signature=str(inspect.signature(self.fn)),parameter_map=self.mapping,
                             config_sha256=digest(self.fixed),compartment='nuclei')

    def infer(self,image,grid,diameter_um,clip,frame):
        if np.ndim(image)!=3: raise ValueError('T is not a depth axis')
        kwargs=dict(self.fixed)
        kwargs.update({self.mapping[k]:v for k,v in dict(image=image,spacing=grid.spacing_um,
                       diameter=diameter_um,checkpoint=str(self.checkpoint)).items()})
        start=time.monotonic();output=self.fn(**kwargs)
        labels=output[self.key] if self.key is not None else output
        grid.check(image,labels)
        return labels,dict(**self.provenance,seconds=time.monotonic()-start,output_shape=list(np.shape(labels)),
                           diameter_um=diameter_um,spacing_um=list(grid.spacing_um),origin=list(grid.origin),
                           actual_learned_execution=True)

class CellposeAdapter:
    """Only a supplied local checkpoint; the installed API must expose 3D eval."""
    provider='Cellpose-nuclei'
    def __init__(self, checkpoint):
        deny_external_network()
        checkpoint=Path(checkpoint)
        if not checkpoint.is_file(): raise Blocked('Local Cellpose volumetric checkpoint absent: '+str(checkpoint))
        try:
            from cellpose import models
        except ImportError as exc: raise Blocked('cellpose is not installed') from exc
        required={'do_3D','anisotropy','diameter','z_axis','channel_axis'}
        if not required<=set(inspect.signature(models.CellposeModel.eval).parameters):
            raise Blocked('Installed Cellpose volumetric API has not passed signature validation')
        self.model=models.CellposeModel(gpu=True,pretrained_model=str(checkpoint))
        self.provenance=dict(checkpoint_sha256=sha(checkpoint),signature=str(inspect.signature(self.model.eval)))

    def infer(self,image,grid,diameter_um,clip,frame):
        if np.ndim(image)!=3 or grid.spacing_um[1]!=grid.spacing_um[2]: raise ValueError('Unsupported voxel grid')
        start=time.monotonic()
        output=self.model.eval(image,do_3D=True,anisotropy=grid.spacing_um[0]/grid.spacing_um[1],
                               diameter=diameter_um/grid.spacing_um[1],z_axis=0,channel_axis=None)
        labels=output[0];grid.check(image,labels)
        # Cellpose segmentation flows never enter inter-frame association.
        return labels,dict(**self.provenance,seconds=time.monotonic()-start,output_shape=list(np.shape(labels)),
                           diameter_um=diameter_um,actual_learned_execution=True)

def frame_objects(adapter,image,grid,diameter_um,clip,frame):
    labels,receipt=adapter.infer(image,grid,diameter_um,clip,frame)
    return extract(labels,image,clip=clip,frame=frame,provider=adapter.provider,
                   spacing=grid.spacing_um),receipt
