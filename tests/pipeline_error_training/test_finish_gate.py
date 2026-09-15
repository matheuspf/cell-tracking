import pytest

from pipeline_error_training import finish_queue
from pipeline_error_training.common import sha, write_json


def test_incomplete_primary_requires_a_concrete_recorded_disposition(tmp_path, monkeypatch):
    monkeypatch.setattr(finish_queue, 'WORK', tmp_path/'work')
    monkeypatch.setattr(finish_queue, 'RESULTS', tmp_path/'results')
    from pipeline_error_training import freeze
    monkeypatch.setattr(freeze, 'PRIMARY', ['D20_temporal'])
    for source in ['44b6', '6bba']:
        write_json(tmp_path/'work/training/D20_temporal'/source/'20260915/frozen_package.json', {})
    write_json(tmp_path/'work/source_screen/D20_temporal/44b6/20260915/summary.json', {})
    with pytest.raises(RuntimeError, match='D20_temporal/6bba'):
        finish_queue.verify_primary_availability()
    write_json(tmp_path/'results/lane_blockers.json', dict(lanes=[
        dict(arm='D20_temporal', source='6bba', status='blocked', reason='Fixed GPU reservation exhausted')]))
    finish_queue.verify_primary_availability()


def test_existing_freeze_rejects_changed_checkpoint_before_resumption(tmp_path, monkeypatch):
    monkeypatch.setattr(finish_queue, 'RESULTS', tmp_path)
    weights = tmp_path/'model.pt'
    weights.write_bytes(b'frozen checkpoint')
    for name in ['manifest', 'nomination', 'replication']:
        write_json(tmp_path/f'{name}.json', dict(frozen=True))
    write_json(tmp_path/'target_freeze.json', dict(packages=[dict(
        manifest_path=str(tmp_path/'manifest.json'), manifest_sha256=sha(tmp_path/'manifest.json'),
        weights_path=str(weights), weights_sha256=sha(weights))],
        nomination_sha256=sha(tmp_path/'nomination.json'), replication_sha256=sha(tmp_path/'replication.json')))
    finish_queue.verify_existing_freeze()
    weights.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='Frozen directional model changed'):
        finish_queue.verify_existing_freeze()
