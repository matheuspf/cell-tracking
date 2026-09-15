import json

import pytest

from pipeline_error_training.cache_reuse import observation
from pipeline_error_training.common import sha


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
