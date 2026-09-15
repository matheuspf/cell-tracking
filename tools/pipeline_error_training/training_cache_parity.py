"""Real source batches must be identical under forced eviction and retained RAM."""
from pathlib import Path
import time


def run():
    from .guard import install
    install(source='44b6')
    import numpy as np
    import torch
    from .common import RESULTS, WORK, digest, sha, write_json
    from .dataset import SourceDataset
    from .resources import Monitor
    torch.set_num_threads(2)
    root = WORK/'training_cache_parity'
    root.mkdir(parents=True,exist_ok=True)
    began = time.monotonic()
    with Monitor(root/'resources.json') as monitor:
        reference = SourceDataset('44b6',graph_cache_size=1,crop_cache_size=2)
        retained = SourceDataset('44b6')
        rngs = [np.random.default_rng(1701),np.random.default_rng(1701)]
        rows = []
        for index in range(64):
            event = index>=32 and index%2==0
            samples = [d.sample(r,event=event) for d,r in zip([reference,retained],rngs)]
            if digest(samples[0])!=digest(samples[1]):
                raise ValueError('RAM retention changed source sampling or biological groups')
            batches = [d.batch(s,'temporal') for d,s in zip([reference,retained],samples)]
            a,b = batches
            for key in ['patch','valid','pairs','edge_features','native_offset']:
                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            for key in ['records','pair_map','node_map','pair_labels','supervised_pair_indices',
                        'sampling_probability','group']:
                if digest(a[key])!=digest(b[key]):
                    raise ValueError('RAM retention changed source labels, alternatives or normalization')
            if a['decisions']!=b['decisions']:
                raise ValueError('RAM retention changed complete ownership decisions')
            rows.append(dict(index=index,event=event,node_tokens=len(a['patch']),group_sha256=digest(a['group'])))
            del batches,a,b
            monitor.check()
        if dict(reference.visits)!=dict(retained.visits) or rngs[0].bit_generator.state!=rngs[1].bit_generator.state:
            raise ValueError('Source group visits or RNG state diverged')
        occupancy = dict(reference_graphs=len(reference.graphs),retained_graphs=len(retained.graphs),
                         reference_crops=len(reference.crop_cache),retained_crops=len(retained.crop_cache))
    receipt = dict(status='measured',source='44b6',batches=64,source_sampling_seed=1701,
        exact_training_pixels_masks_features_labels_decisions=True,exact_group_visits_and_rng=True,
        reference_limits=dict(graphs=1,crops=2),previous_production_limits=dict(graphs=4,crops=512),
        new_limits=dict(graphs=16,crops=8192),observed_occupancy=occupancy,
        retained_native_crop_upper_bound_gib=8192*7*2*16*64*64/2**30,
        seconds=time.monotonic()-began,rows=rows,
        implementation_sha256=sha(Path(__file__).with_name('dataset.py')),
        previous_implementation_sha256=sha(WORK/'execution_versions/training_cache_limits/dataset.py'),
        optimization='Only bounded in-memory retention; model, optimizer, sampling and augmentation recipes unchanged',
        existing_fits_and_inputs_invalidated=False,existing_fit_not_restarted=True,
        no_new_training_update_budget=True,new_target_metrics_read=False)
    write_json(RESULTS/'training_cache_parity.json',receipt,immutable=True)
    print({k:v for k,v in receipt.items() if k!='rows'},flush=True)


if __name__=='__main__':
    from .resources import cpu_budget
    cpu_budget(4)
    run()
