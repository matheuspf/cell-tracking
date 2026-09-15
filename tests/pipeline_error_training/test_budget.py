import pytest

from pipeline_error_training.budget import assert_room, category, effective_caps, hours
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
    assert category(['fresh_entry','/study/fresh_candidate_A10/job.json']) == 'final_inference_and_package'


def test_unused_training_reservation_preserves_replication_and_total_cap():
    total = dict(preflight_and_pilots=1.,first_seed_training=10.,replication=0.,final_inference_and_package=21.)
    assert effective_caps(total,transfer=True)['final_inference_and_package']==22.
    assert_room(total,'final_inference_and_package',transfer=True)
    total['final_inference_and_package']=22.
    with pytest.raises(Blocked,match='budget reached'):
        assert_room(total,'final_inference_and_package',transfer=True)
    # Later same-recipe source repair also competes for the same 32-hour pool.
    with pytest.raises(Blocked,match='budget reached'):
        assert_room(total,'first_seed_training',transfer=True)
    total['final_inference_and_package']=20.
    total['replication']=12.
    with pytest.raises(Blocked,match='budget reached'):
        assert_room(total,'replication',transfer=True)
    total.update(preflight_and_pilots=4.,first_seed_training=24.,replication=12.,final_inference_and_package=8.)
    with pytest.raises(Blocked,match='budget reached'):
        assert_room(total,'final_inference_and_package',transfer=True)
