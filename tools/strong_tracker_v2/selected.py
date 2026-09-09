"""Export the measured winning pipeline with annotations unavailable."""
from __future__ import annotations

import numpy as np

from .common import DATA,FULL,OUT,V1,graph_hash,load_graph,now,read_json,save_arrays,sha,validate,write_json


def run(args):
    from .infer import deny_annotations
    deny_annotations()
    names=[p.stem for p in sorted((V1/'baseline/public').glob('*.npz'))]
    hashes={};records=[]
    for name in names:
        r=load_graph(OUT/'replay/no_motion'/name/'final_export.npz')
        source_receipt=read_json(OUT/'replay/no_motion'/name/'complete.json')
        assert source_receipt['annotation_access_blocked']
        shape=read_json(DATA/'train'/f'{name}.zarr/0/zarr.json')['shape']
        n=r['nodes'].copy();n[:,2:]=np.clip(n[:,2:],0,np.array(shape[1:])-1);e=r['edges']
        valid=validate(n,e,shape)
        expected=load_graph(OUT/'candidate_graphs/bypass_motion_bounds'/f'{name}.npz')
        assert graph_hash(n,e)==graph_hash(expected['nodes'],expected['edges'])
        p=OUT/'selected_predictions'/f'{name}.npz';save_arrays(p,nodes=n,edges=e)
        assert sha(p)==sha(OUT/'candidate_graphs/bypass_motion_bounds'/f'{name}.npz')
        hashes[name]=sha(p);records.append(dict(dataset=name,graph_hash=graph_hash(n,e),**valid))
    lock=OUT/'selected_prediction_lock.json'
    created=read_json(lock)['created'] if lock.exists() else now()
    config=dict(created=created,variant='bypass_motion_bounds',notebook_sha256=sha(FULL/'harmonic_isolated.py'),
        original_neural_predictions='Reused sealed Harmonic Fusion neural GEFF graphs, exported into v2 raw NPZ cache',
        changes=['Set OUTPUT_MOTION_RELINK=False; retain original learned associations',
                 'Clip serialized z/y/x to the image bounds; separately rescored with no score change'],
        original_other_settings='Unchanged: edge validity, single-parent guard, gap/gap2 closing, DeepCenter-gated safe divisions, pruning, smoothing',
        selection='Exploratory stage ablation, chosen after diagnostic census; not an untouched prespecified primary winner',
        provenance='Public upstream checkpoint contamination remains',annotation_reads=0,
        full_sample_set=len(names)==199,strict_coordinates=True,forbidden_merges=0,duplicate_edges=0,
        consecutive_frames=True,byte_identical_to_scored_graphs=True,
        hashes=hashes,graphs=records)
    write_json(OUT/'selected_prediction_lock.json',config,immutable=True)
    write_json(OUT/'winning_config.json',{k:v for k,v in config.items() if k not in ['graphs','hashes']},immutable=True)
    print(f'Exported {len(names)} strict, annotation-unavailable graphs; exact byte parity with scored winner',flush=True)
