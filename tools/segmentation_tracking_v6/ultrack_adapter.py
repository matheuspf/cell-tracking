"""Actual Ultrack entry boundary and strict region/lineage export contracts.

Runtime integration is UNVALIDATED when ultrack is absent. A SCIP installation
or passing these export contracts is not evidence that Ultrack ran.
"""
import inspect
import numpy as np
from .common import Blocked, digest
from .contracts import validate_edges
from .regions import validate_hierarchy_selection
from .network import deny_external_network

def geometry_fingerprint(regions):
    return digest([dict(key=r.key,bbox=r.bbox,mask=r.mask_sha256(),spacing=r.spacing,
                        center=r.center().tolist()) for r in regions])

def recompute_native_evidence(regions, pairs, image_native_evaluator):
    """Full native recomputation at the exact proposed centers, no NN copying.

    The image evaluator receives masks/centers and must attest to the complete
    primary, secondary and augmentation recipe. U1 cannot use a single model.
    """
    if image_native_evaluator is None:
        raise Blocked('U1 requires a full image-native evaluator at Ultrack region centers')
    expected=geometry_fingerprint(regions)
    scores,receipt=image_native_evaluator(regions,pairs)
    required={'primary','secondary','eight_view_augmentation'}
    if receipt.get('geometry_sha256')!=expected or set(receipt.get('components',[]))!=required:
        raise ValueError('Native evidence geometry or full ensemble provenance mismatch')
    scores=np.asarray(scores)
    if scores.shape!=(len(pairs),) or not np.isfinite(scores).all(): raise ValueError('Invalid native link evidence')
    return scores,receipt

def export_regions(selected, parent_ids):
    """selected maps actual Ultrack observation IDs to retained Region objects.

    parent_ids must be direct observation ancestry, never track-ID equality or
    nearest-center reconstruction. Hierarchy conflicts and gap edges fail closed.
    """
    if not set(parent_ids)<=set(selected): raise ValueError('Parent row absent from selected observations')
    validate_hierarchy_selection(list(selected.values()))
    nodes=[];edges=[]
    for node_id,r in sorted(selected.items()):
        point=r.center(inside=True)
        nodes.append([int(node_id),r.time,*map(int,point)])
        parent=parent_ids.get(node_id)
        if parent is not None and int(parent)>=0: edges.append([int(parent),int(node_id)])
    times={int(k):r.time for k,r in selected.items()}
    if times: validate_edges(times,edges)
    elif edges: raise ValueError('Edges without regions')
    return dict(nodes=np.asarray(nodes,np.int64).reshape(-1,5),edges=np.asarray(edges,np.int64).reshape(-1,2))

def survival(original, hierarchy):
    """Exact mask survival, distinct from overlap/centroid approximation."""
    expected={(r.time,r.grid_id,r.bbox,r.mask_sha256()) for r in original}
    present={(r.time,r.grid_id,r.bbox,r.mask_sha256()) for r in hierarchy}
    return dict(input_masks=len(expected),hierarchy_masks=len(present),
                exact_survivors=len(expected&present),lost_input_masks=len(expected-present))

class UltrackAdapter:
    def __init__(self):
        deny_external_network()
        try:
            import ultrack
            from ultrack.utils import labels_to_contours
        except ImportError as exc: raise Blocked('ultrack package/source absent; actual hierarchy and lineage solver cannot run') from exc
        self.api=ultrack;self.convert=labels_to_contours
        self.signatures={name:str(inspect.signature(getattr(ultrack,name)))
                         for name in ['segment','link','solve','to_tracks_layer']}

    def run(self, labels, config, *, images, native_link_writer=None, native_evidence=None):
        """Use installed segmentation/link/solve calls; validate before mutation.

        A native link writer must update the actual Ultrack candidate IDs and
        return a receipt. No guessed SQL schema or licensed solver is installed.
        The resulting node/mask export still requires an inspected installed
        hierarchy reader before this arm can be considered complete.
        """
        labels=np.asarray(labels)
        if labels.ndim!=4 or labels.dtype.kind not in 'iu' or (labels<0).any():
            raise ValueError('Ultrack labels must be TZYX integer instances')
        if np.shape(images)!=labels.shape: raise ValueError('Image/label lattice mismatch')
        seg_kwargs=dict(config=config,foreground=None,contours=None)
        link_kwargs=dict(config=config,images=images)
        for name,kwargs in [('segment',seg_kwargs),('link',link_kwargs),('solve',dict(config=config)),
                            ('to_tracks_layer',dict(config=config))]:
            try: inspect.signature(getattr(self.api,name)).bind(**kwargs)
            except TypeError as exc: raise Blocked(f'Unvalidated installed Ultrack {name} signature: {exc}') from exc
        if native_evidence is not None and native_link_writer is None:
            raise Blocked('U1 needs inspected Ultrack candidate-ID link writer before solving')
        foreground,contours=self.convert(labels)
        if np.shape(foreground)!=labels.shape or np.shape(contours)!=labels.shape:
            raise ValueError('Ultrack conversion changed TZYX lattice')
        self.api.segment(config=config,foreground=foreground,contours=contours)
        self.api.link(**link_kwargs)
        native_receipt=None
        if native_evidence is not None:
            native_receipt=native_link_writer(config,native_evidence)
            if not native_receipt.get('exact_candidate_ids'):
                raise ValueError('U1 native link mapping was not exact')
        self.api.solve(config=config)
        return self.api.to_tracks_layer(config=config),dict(signatures=self.signatures,
            native=native_receipt,actual_ultrack_execution=True,export_validated=False)
