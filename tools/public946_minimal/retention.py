"""Retain compact neural evidence within the registered scratch allocation.

Prediction arithmetic is untouched. Full B0 probability matrices and every
pilot matrix remain available. Other completed arms retain exact candidates,
offsets, node probabilities, source/target universes, and full-matrix hashes.
"""
from __future__ import annotations

from pathlib import Path

from .common import array_hash, digest, read_json, save_arrays, sha, write_json


def compact(directory):
    import numpy as np
    directory=Path(directory)
    receipt_path=directory/'neural.json'
    if not receipt_path.exists():
        return None  # This arm references the unchanged B0 neural cache.
    if not (directory/'complete.json').exists():
        raise ValueError('Cannot compact evidence while inference is active')
    graph=read_json(directory/'complete.json')
    if graph['arm']=='B0':
        return None
    path=directory/'evidence.npz'
    journal=directory/'evidence_compaction.json'
    temporary=directory/'evidence_compact_pending.npz'
    if journal.exists():
        record=read_json(journal)
    else:
        receipt=read_json(receipt_path)
        if sha(path)!=receipt['evidence_sha256']:
            raise ValueError('Neural evidence changed before compaction')
        with np.load(path,allow_pickle=False) as data:
            omitted=[k for k in data.files if k.startswith('p_')]
            if not omitted:
                return None
            arrays={k:data[k] for k in data.files if k not in omitted}
        save_arrays(temporary,**arrays)
        with np.load(temporary,allow_pickle=False) as data:
            for key,value in arrays.items():
                if array_hash(value)!=array_hash(data[key]):
                    raise ValueError('Compaction changed retained tensor values')
        record=dict(format='exact_candidates_without_dense_probabilities_v1',
            original_sha256=sha(path),compact_sha256=sha(temporary),
            original_bytes=path.stat().st_size,compact_bytes=temporary.stat().st_size,
            retained_arrays=sorted(arrays),omitted_arrays=omitted,
            original_neural_receipt=receipt,
            allowed_cache_use='All registered replay recipes except E01. E01 uses full B0 evidence, which is never compacted.',
            reason='Bounded scratch retention after completed inference; no quantization and no changed prediction values.')
        write_json(journal,record,immutable=True)
    current=sha(path)
    if current==record['original_sha256']:
        if not temporary.exists() or sha(temporary)!=record['compact_sha256']:
            raise ValueError('Interrupted compaction has no verified replacement')
        temporary.replace(path)
    elif current!=record['compact_sha256']:
        raise ValueError('Evidence differs from both compaction journal states')
    receipt=dict(record['original_neural_receipt'])
    receipt.update(evidence_sha256=record['compact_sha256'],
        original_dense_evidence_sha256=record['original_sha256'],evidence_format=record['format'])
    write_json(receipt_path,receipt)
    return record


def finish(job,scope):
    if scope not in ('full','fresh_finalists'):
        return
    directory=Path(job['directory'])
    compact(directory)
    if scope!='fresh_finalists':
        return
    complete=read_json(directory/'complete.json')
    cache_id=digest(dict(image=job['row']['image_sha256'],transform=job.get('transform'),
                        source=job['original_source_sha256'],model=job['model_hashes'][job['deepcenter_weights']]))
    root=Path(job['out'])/'heatmaps'/cache_id
    if not root.exists():
        return
    if root.is_symlink() or 'fresh_finalist_cache' not in root.parts:
        raise ValueError('Refusing to remove an unowned or shared heatmap cache')
    files=list(root.iterdir())
    allowed={str(t)+suffix for t in range(job['row']['image_shape'][0]) for suffix in ('.npy','.json')}
    if any(not p.is_file() or p.name not in allowed for p in files):
        raise ValueError('Unexpected file in clip-local fresh heatmap cache')
    for p in files:
        if p.suffix=='.npy' and sha(p)!=complete['heatmaps']['hashes'].get(p.stem):
            raise ValueError('Fresh heatmap hash changed before compact retention')
    journal=directory/'heatmap_retention.json'
    if journal.exists():
        if read_json(journal)['cache_id']!=cache_id:
            raise ValueError('Heatmap retention journal belongs to another clip')
    else:
        write_json(journal,dict(cache_id=cache_id,removed_bytes=sum(p.stat().st_size for p in files),
            retained_frame_hashes=complete['heatmaps']['hashes'],
            policy='Fresh-finalist heatmaps are clip-local, recomputable intermediate tensors; retained original images/models and all graph outputs are unchanged.'),immutable=True)
    for p in files:
        p.unlink()
    root.rmdir()
