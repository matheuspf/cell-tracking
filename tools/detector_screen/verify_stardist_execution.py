"""Check CPU/GPU separation and smaller tile counts without consulting labels."""
import fcntl
import json
import time

import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .stardist_adapter import ROOT, LOCK, StarDist3D, normalize, tf


def main():
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.set_logical_device_configuration(gpu, [
            tf.config.LogicalDeviceConfiguration(memory_limit=4000)])
    folder = ROOT / 'assets/xenopus'
    model = StarDist3D(None, name=folder.name, basedir=str(folder.parent))
    rows = [r for r in json.loads((ROOT/'panel.json').read_text())['frames'] if r['role'] == 'pilot']
    results = []
    for row in rows:
        raw = np.load(row['image_path']).astype(np.float32)
        image = zoom(raw, [4/model.config.anisotropy[0],1,1],
                     order=1, grid_mode=True, mode='nearest')
        scale = np.array(image.shape)/np.array(raw.shape)
        image = normalize(image,1,99.8,axis=(0,1,2))
        reference = np.load(ROOT/'predictions/xenopus'/(row['key']+'.npz'))
        for tiles in ([(4,2,2),(2,2,2)] if row == rows[0] else [(2,2,2)]):
            start = time.monotonic()
            with LOCK.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                start = time.monotonic()
                generator = model._predict_instances_generator(
                    image,axes='ZYX',n_tiles=tiles,prob_thresh=float(model.thresholds.prob),
                    nms_thresh=.3,return_labels=False,show_tile_progress=False)
                for stage in generator:
                    if isinstance(stage,str) and stage=='nms':break
                neural = time.monotonic()-start
            start = time.monotonic()
            for result in generator:pass
            _, details = result
            nms = time.monotonic()-start
            centers = (np.asarray(details['points'])+.5)/scale-.5
            scores = np.asarray(details['prob'])
            d = cdist(reference['centers_zyx']*[1.625,.40625,.40625],
                      centers*[1.625,.40625,.40625])
            i,j = linear_sum_assignment(d)
            item = dict(key=row['key'],n_tiles=tiles,neural_seconds=neural,nms_seconds=nms,
                        reference_candidates=len(reference['scores']),candidates=len(scores),
                        center_max_error_um=float(d[i,j].max()),
                        score_max_error=float(np.abs(reference['scores'][i]-scores[j]).max()))
            item['passes'] = (len(scores)==len(reference['scores']) and
                              item['center_max_error_um']<1e-10 and item['score_max_error']<2e-5)
            results.append(item)
            print(json.dumps(item),flush=True)
    out=ROOT/'xenopus/default-nms-tile-verification.json'
    out.write_text(json.dumps(dict(results=results,scope='No GT labels read. Default threshold versus filtered permissive NMS; unchanged network with fewer overlapping tiles.'),indent=2)+'\n')
    assert results[0]['passes'], 'Default-only NMS must preserve the upstream filtered predictions.'


if __name__=='__main__':main()
