import pytest

from pipeline_error_training.budget import assert_room, category, hours
from pipeline_error_training.common import Blocked


def test_gpu_budget_charges_only_study_intervals_and_enforces_reserved_lane():
    ledger = dict(completed=[dict(category='first_seed_training', seconds=3600)], active={
        'lease': dict(category='replication', started_epoch=100)})
    total = hours(ledger, timestamp=1900)
    assert total['first_seed_training'] == 1
    assert total['replication'] == .5
    assert_room(total, 'replication')
    total['final_inference_and_package'] = 8
    with pytest.raises(Blocked, match='budget reached'):
        assert_room(total, 'final_inference_and_package')
    assert category(['prediction_entry', '/study/source_screen/D20_temporal/44b6/314159/job.json']) == 'replication'
    assert category(['prediction_entry', '/study/final_point_inference/A10/job.json']) == 'final_inference_and_package'
