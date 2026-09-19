"""Source-only full-fit mask coverage, faint-center and lattice collision audit."""
from pathlib import Path
import sys,time
from collections import Counter
from .common import REPO,WORK,Blocked,read,write,sha,now


def run(source):
    clips=read(WORK/'source_partitions.json')[source]['fit'];cache=WORK/'preprocessed'/source
    folder=WORK/'mask_audit'/source;folder.mkdir(parents=True,exist_ok=True)
    from .guard import install
    sys.dont_write_bytecode=True
    guard=install(inputs=[cache],outputs=[folder],code_roots=[REPO/'tools',Path(sys.prefix),Path(sys.base_prefix),*[Path(p) for p in sys.path if 'site-packages' in p]])
    import numpy as np
    from scipy.spatial import cKDTree
    from .preprocess import SourcePairs
    from .data import centers_target,SPACING
    reader=SourcePairs(cache,clips);total=Counter();by_clip={};tick=time.monotonic()
    for name in clips:
        c=reader.cache[name];counts=Counter()
        for t in range(100):
            points=c['nodes'][c['nodes'][:,1]==t,2:]
            image=c['images'][t];bg=c['background'][t];_,positive=centers_target((64,64,64),points.astype(np.float32))
            counts['coarse_voxels']+=int(image.size);counts['positive_support_voxels']+=int(positive.sum())
            counts['background_eligible_before_positive_override']+=int(bg.sum())
            counts['supervised_background_voxels']+=int((bg&~positive).sum())
            counts['unknown_zero_supervised_detection_voxels']+=int((~positive&~bg).sum())
            counts['annotated_centers']+=len(points)
            if len(points):
                q=np.clip(np.rint(points/np.array([1,4,4])).astype(int),0,63)
                counts['centers_sharing_nearest_coarse_bin_beyond_first']+=len(q)-len(np.unique(q,axis=0))
                counts['annotated_centers_with_image_background_flag_before_positive_override']+=int(bg[tuple(q.T)].sum())
                counts['annotated_centers_at_nearest_lattice_intensity_below_0_05']+=int((image[tuple(q.T)]<.05).sum())
                counts['annotated_pairs_within_NMS_3um']+=len(cKDTree(points*SPACING).query_pairs(3.))
        total.update(counts);by_clip[name]=dict(counts)
    result=dict(status='complete',source=source,fit_clips=len(clips),guard=guard,counts=dict(total),per_clip=by_clip,
        image_threshold_is_descriptive_only=.05,background_is_not_certified_absence=True,
        all_positive_support_overrides_background=True,wall_seconds=time.monotonic()-tick,
        cache_manifest_sha256=sha(cache/'manifest.json'),code_sha256=sha(Path(__file__)),finished_utc=now())
    write(folder/'receipt.json',result,immutable=True);return result
