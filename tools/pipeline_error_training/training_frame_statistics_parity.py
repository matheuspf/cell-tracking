"""Cold source crops compare original quantiles with bounded scalar reuse."""
from collections import OrderedDict
import os
from pathlib import Path
import time


def run():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from .guard import install
    install(source='44b6')
    import numpy as np
    import torch
    from .common import RESULTS,WORK,adjacency,digest,inputs,sha,verified_graph,write_json
    from .crops import Images
    from .fast_crops import sample_native
    from .resources import Monitor
    from .training_images import FrameStatistics,TrainingImages
    torch.set_num_threads(2)
    rows = sorted([r for r in inputs() if r['embryo']=='44b6'],key=lambda r:r['baselines']['P0']['nodes'])[:8]
    graphs = {}
    sequence = []
    for row in rows:
        graph = verified_graph(row)
        _,pred,succ = adjacency(graph['nodes'],graph['edges'])
        graphs[row['dataset']] = (graph['nodes'],pred,succ)
    for frame in [0,50,99]:
        for row in rows:
            nodes = graphs[row['dataset']][0]
            index = int(np.argmin(abs(nodes[:,1]-frame)))
            sequence.append((row,index))
    sequence = sequence*2
    statistics = FrameStatistics()
    root = WORK/'training_frame_statistics_parity'
    times,reference,hashes = {},[],[]
    with Monitor(root/'resources.json'):
        for kind in ['original','retained']:
            pool = OrderedDict()
            began = time.monotonic()
            for k,(row,index) in enumerate(sequence):
                name = row['dataset']
                if name not in pool:
                    pool[name] = Images(row['image_path']) if kind=='original' else TrainingImages(row['image_path'],statistics)
                    while len(pool)>2:
                        pool.popitem(last=False)
                pool.move_to_end(name)
                pixels,valid = sample_native(pool[name],*graphs[name],index)
                if kind=='original':
                    reference.append((pixels,valid))
                    hashes.append(dict(dataset=name,node_index=index,pixels_sha256=digest(pixels),valid_sha256=digest(valid)))
                else:
                    np.testing.assert_array_equal(pixels,reference[k][0])
                    np.testing.assert_array_equal(valid,reference[k][1])
            times[kind] = time.monotonic()-began
        if not statistics.reused or len(statistics.entries)>statistics.max_entries:
            raise ValueError('Bounded scalar-statistic reuse was not exercised')
        if torch.cuda.is_initialized():
            raise ValueError('Cold training crop proof unexpectedly initialized CUDA')
    result = dict(status='measured',source='44b6',clips=len(rows),cold_crop_calls_per_path=len(sequence),
        source_frames=[0,50,99],image_reader_limit_unchanged=2,raw_frame_limit_per_reader_unchanged=9,
        quantile_calculations=statistics.computed,reused_quantiles=statistics.reused,
        scalar_cache_entries=len(statistics.entries),scalar_entry_limit=statistics.max_entries,
        timings_seconds=times,exact_all_pixels_and_masks=True,CUDA_hidden=True,
        crop_file_cache_bypassed_in_both_paths=True,source_records=hashes,
        implementation_sha256=sha(Path(__file__).with_name('training_images.py')),
        original_images_sha256=sha(Path(__file__).with_name('crops.py')),
        normalization_unchanged=True,training_recipe_changed=False,new_target_metrics_read=False)
    write_json(RESULTS/'training_frame_statistics_parity.json',result,immutable=True)
    return result


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget()
    print(run(),flush=True)
