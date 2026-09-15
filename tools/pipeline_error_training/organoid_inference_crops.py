"""Released Organoid extraction with verified per-frame quantile reuse.

Only inference uses this adapter. The released extractor, nearest resizing,
time-window filling and float32 normalization order remain unchanged.
"""
import numpy as np

from .common import read_json
from .organoid_adapter import ORG_MODEL, setup


def patches(images, nodes, indices):
    setup()
    from organoid_tracker.core import TimePoint
    from organoid_tracker.neural_network.image_loading import fill_none_images_with_copies, extract_patch_array
    from other_trackers.organoid import resize_patch
    settings = read_json(ORG_MODEL/'settings.json')
    shape = tuple(settings['patch_shape_zyx'])
    scale = tuple(images.scale/np.array([2., .32, .32]))
    native_shape = tuple(int(n/s) for n,s in zip(shape,scale))
    result = {}
    for t in sorted({int(nodes[i,1]) for i in indices}):
        frames = {TimePoint(t+dt):images.get_image(TimePoint(t+dt))
                  for dt in range(settings['time_window'][0],settings['time_window'][1]+1)}
        present = [key.time_point_number() for key,value in frames.items() if value is not None]
        fill_none_images_with_copies(frames)
        low = float(min(images.quantiles[k][0] for k in present))
        high = float(max(images.quantiles[k][1] for k in present))
        for i in indices:
            if int(nodes[i,1])!=t:
                continue
            starts = tuple(int(round(float(c)-n/2)) for c,n in zip(nodes[i,2:],native_shape))
            patch = extract_patch_array(frames,starts,native_shape)
            patch /= high-low
            patch -= low/(high-low)
            np.clip(patch,0.,1.,out=patch)
            result[int(i)] = resize_patch(patch,shape)
    return np.stack([result[int(i)] for i in indices])
