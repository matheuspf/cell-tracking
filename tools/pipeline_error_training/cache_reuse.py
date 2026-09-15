"""Reuse immutable model- and image-specific embeddings across graph policies."""
import os
from pathlib import Path
import shutil

from .common import read_json, sha, write_json


def continuation(source, target, model_sha256, metadata_sha256):
    """The ordinary embedding loader subsequently checks graph, queries and frames."""
    source, target = Path(source), Path(target)
    if not source.exists():
        return None
    receipt = read_json(source.with_suffix('.json'))
    if sha(source)!=receipt['sha256'] or receipt['inputs']['model_sha256']!=model_sha256 \
            or receipt['inputs']['image_metadata_sha256']!=metadata_sha256:
        raise ValueError('Continuation embedding provenance changed')
    if target.exists() and sha(target)!=receipt['sha256']:
        raise ValueError('Existing continuation embedding differs from its verified source')
    if target.with_suffix('.json').exists() and sha(target.with_suffix('.json'))!=sha(source.with_suffix('.json')):
        raise ValueError('Existing continuation embedding receipt changed')
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        os.link(source,target)
    if not target.with_suffix('.json').exists():
        shutil.copyfile(source.with_suffix('.json'),target.with_suffix('.json'))
    result = dict(status='measured',kind='Distinct frozen continuation encoder on identical full P0 tracklets',
        source_sha256=receipt['sha256'],receipt_sha256=sha(source.with_suffix('.json')),
        model_sha256=model_sha256,metadata_sha256=metadata_sha256,annotations_copied=False,
        graph_queries_and_frames_verified_by_ordinary_loader=True,fresh_end_to_end_claim=False)
    write_json(target.with_suffix('.reuse.json'),result,immutable=True)
    return result


def observation(source_cache, destination_cache, model_sha256, metadata_sha256):
    source_cache, destination_cache = Path(source_cache), Path(destination_cache)
    records = []
    for name in ['P0.npz','raw.npz']:
        source = source_cache/name
        if not source.exists() or not source.with_suffix('.json').exists():
            return None
        receipt = read_json(source.with_suffix('.json'))
        if sha(source)!=receipt['sha256'] or receipt['inputs']['model_sha256']!=model_sha256 \
                or receipt['inputs']['image_metadata_sha256']!=metadata_sha256:
            raise ValueError('Observation-policy image embedding provenance changed')
        records.append(dict(name=name,sha256=receipt['sha256'],receipt_sha256=sha(source.with_suffix('.json'))))
    destination_cache.mkdir(parents=True,exist_ok=True)
    for record in records:
        source, target = source_cache/record['name'], destination_cache/record['name']
        if target.exists():
            if sha(target)!=record['sha256']:
                raise ValueError('Existing policy embedding differs from its verified source')
        else:
            # Both files are immutable study outputs on the same filesystem.
            # A second directory entry avoids duplicating a large image cache.
            os.link(source,target)
        if target.with_suffix('.json').exists():
            if sha(target.with_suffix('.json'))!=record['receipt_sha256']:
                raise ValueError('Existing policy embedding receipt changed')
        else:
            shutil.copyfile(source.with_suffix('.json'),target.with_suffix('.json'))
    result = dict(status='measured',kind='Same frozen selector, same P0/raw observations, distinct graph policies',
        files=records,model_sha256=model_sha256,metadata_sha256=metadata_sha256,
        annotations_copied=False,raw_images_still_verified_by_embedding_loader=True,
        fresh_end_to_end_claim=False)
    write_json(destination_cache/'policy_reuse.json',result,immutable=True)
    return result
