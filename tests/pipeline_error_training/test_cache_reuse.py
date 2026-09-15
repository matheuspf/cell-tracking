import json

import pytest

from pipeline_error_training.cache_reuse import continuation, native_query, native_reference, observation
from pipeline_error_training.common import digest, sha


def source_cache(tmp_path):
    root = tmp_path/'source'
    root.mkdir()
    for name in ['P0.npz','raw.npz']:
        path = root/name
        path.write_bytes(name.encode())
        path.with_suffix('.json').write_text(json.dumps(dict(sha256=sha(path),
            inputs=dict(model_sha256='weights',image_metadata_sha256='images'))))
    return root


def test_observation_reuse_rejects_changed_model_and_bytes(tmp_path):
    source = source_cache(tmp_path)
    destination = tmp_path/'destination'
    with pytest.raises(ValueError, match='provenance'):
        observation(source,destination,'different','images')
    assert not destination.exists()
    (source/'raw.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='provenance'):
        observation(source,destination,'weights','images')
    assert not destination.exists()


def test_observation_reuse_keeps_immutable_files_and_receipts(tmp_path):
    source = source_cache(tmp_path)
    destination = tmp_path/'destination'
    receipt = observation(source,destination,'weights','images')
    for name in ['P0.npz','raw.npz']:
        assert (source/name).stat().st_ino==(destination/name).stat().st_ino
        assert sha((source/name).with_suffix('.json'))==sha((destination/name).with_suffix('.json'))
    assert receipt['annotations_copied'] is False
    assert receipt['fresh_end_to_end_claim'] is False
    observation(source,destination,'weights','images')


def test_continuation_reuse_rejects_changed_image_before_writing(tmp_path):
    source = source_cache(tmp_path)/'P0.npz'
    destination = tmp_path/'destination/model/clip.npz'
    with pytest.raises(ValueError,match='provenance'):
        continuation(source,destination,'weights','different images')
    assert not destination.parent.exists()
    receipt = continuation(source,destination,'weights','images')
    assert receipt['annotations_copied'] is False
    assert destination.stat().st_ino==source.stat().st_ino
    assert sha(destination.with_suffix('.json'))==sha(source.with_suffix('.json'))
    destination.with_suffix('.json').write_text('{}')
    with pytest.raises(ValueError,match='receipt changed'):
        continuation(source,destination,'weights','images')


def test_native_query_reuse_requires_exact_observations(tmp_path):
    source = tmp_path/'fresh'
    source.mkdir()
    nodes = [[7,2,3,4,5]]
    (source/'query.npz').write_bytes(b'neural query')
    (source/'current_features.npz').write_bytes(b'graph features must not be copied')
    (source/'query.json').write_text(json.dumps(dict(sha256=sha(source/'query.npz'),
        inputs=dict(image_metadata_sha256='image',observations_sha256=digest(nodes)))))
    reference = tmp_path/'reference'
    native_reference(source,reference,'image')
    assert not (reference/'current_features.npz').exists()
    target = tmp_path/'target'
    assert native_query(reference,target,[[7,2,3,4,6]],'image') is None
    assert native_query(reference,target,[[8,2,3,4,5]],'image') is None
    assert not target.exists()
    result = native_query(reference,target,nodes,'image')
    assert result['copied_prior_graph_features'] is False
    assert (target/'query.npz').stat().st_ino==(source/'query.npz').stat().st_ino
    (reference/'query.json').write_text(json.dumps(dict(sha256='changed',
        inputs=dict(image_metadata_sha256='image',observations_sha256=digest(nodes)))))
    with pytest.raises(ValueError,match='provenance'):
        native_query(reference,target,nodes,'image')
