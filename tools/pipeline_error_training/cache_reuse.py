"""Reuse immutable model- and image-specific embeddings across graph policies."""
import os
from pathlib import Path
import shutil

from .common import digest, read_json, sha, write_json


def native_reference(source, target, metadata_sha256):
    """Copy only the two immutable query files, never old graph feature rows."""
    source,target = Path(source),Path(target)
    path = source/'query.npz'
    if not path.exists():
        return None
    receipt = read_json(source/'query.json')
    if sha(path)!=receipt['sha256'] or receipt['inputs']['image_metadata_sha256']!=metadata_sha256:
        raise ValueError('Native query reference provenance changed')
    for name in ['query.npz','query.json']:
        if (target/name).exists() and sha(target/name)!=sha(source/name):
            raise ValueError('Existing native query reference changed')
    target.mkdir(parents=True,exist_ok=True)
    if not (target/'query.npz').exists():
        os.link(path,target/'query.npz')
    if not (target/'query.json').exists():
        shutil.copyfile(source/'query.json',target/'query.json')
    return receipt


def native_query(reference, destination, nodes, metadata_sha256):
    """Exact IDs/order/coordinates are required; different queries run cold."""
    reference,destination = Path(reference),Path(destination)
    if not (reference/'query.npz').exists():
        return None
    receipt = read_json(reference/'query.json')
    if receipt['inputs']['observations_sha256']!=digest(nodes):
        return None
    native_reference(reference,destination,metadata_sha256)
    result = dict(status='measured',same_observation_ids_order_and_coordinates=True,
        source_query_sha256=receipt['sha256'],source_query_receipt_sha256=sha(reference/'query.json'),
        full_query_provenance_rechecked_by_original_loader=True,all_38_graph_features_recomputed=True,
        copied_prior_graph_features=False,annotation_reads=0,fresh_end_to_end_claim=False)
    write_json(destination/'query_reuse.json',result,immutable=True)
    return result


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
    native_reference(source_cache.parent/'fresh_native',destination_cache/'native_query_reference',metadata_sha256)
    return result
