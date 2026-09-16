import numpy as np
import pytest

from pipeline_error_training import validation
from pipeline_error_training.common import save_arrays, sha


def test_cold_candidate_must_match_the_scored_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, 'WORK', tmp_path)
    nodes = np.array([[0, 0, 1, 2, 3], [1, 1, 1, 2, 3]], dtype=np.int64)
    edges = np.array([[0, 1]], dtype=np.int64)
    clips = []
    for dataset, renamed in [('44b6_example', 'alder'), ('6bba_example', 'birch')]:
        cold = tmp_path/'fresh_candidates/A10'/renamed/'output/module.npz'
        scored = tmp_path/'predictions/A10'/f'{dataset}.npz'
        save_arrays(cold, nodes=nodes, edges=edges)
        save_arrays(scored, nodes=nodes, edges=edges)
        clips.append(dict(dataset=dataset, unfamiliar_name=renamed, module_sha256=sha(cold)))
    receipt = dict(status='measured', clips=clips)
    assert validation.candidate_replay_parity('A10', receipt)['status']=='measured'
    save_arrays(scored, nodes=nodes, edges=np.empty((0, 2), dtype=np.int64))
    with pytest.raises(AssertionError, match='Cold-image candidate differs from scored graph'):
        validation.candidate_replay_parity('A10', receipt)


def test_cold_proof_requires_both_registered_clips():
    assert validation.candidate_replay_parity('A10', dict(status='measured', clips=[]))['status']=='failed'
    assert validation.candidate_replay_parity('A10', {})['status']=='not run'
