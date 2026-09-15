import numpy as np

from pipeline_error_training.observations import ObservationBank
from pipeline_error_training.observation_infer import construct_actions, apply_observations
from pipeline_error_training.serialization import export_geff


def fixture_bank():
    # The frozen two-percent budget permits three nodes in this 200-node graph.
    nodes = np.array([[100+k, k % 3, 4, 4 if k < 3 else 200+k, 4] for k in range(200)])
    raw = dict(nodes=np.array([[k, k, 4, 5, 4] for k in range(3)]),
               edges=np.array([[0, 1], [1, 2]]), node_probabilities=np.ones(3)*.8,
               original_graph_sha256='fixture')
    row = dict(physical_scale=[1., 1., 1.], image_shape=[3, 512, 512, 512],
               baselines={'P0': {'sha256': 'fixture'}}, raw={'sha256': 'fixture'})
    return ObservationBank(row, dict(nodes=nodes, edges=np.array([[100, 101], [101, 102]])), raw)


def test_one_for_one_is_atomic_and_raw_ids_are_not_reused():
    bank = fixture_bank()
    gain = np.ones((len(bank.pairs), 2))
    actions, receipt = construct_actions(bank, gain, False)
    actions = [a for a in actions if a.owners[0]['changed_node_cost'] == 3]
    result, ledger = apply_observations(bank, actions, receipt, False)
    assert len(result['nodes']) == len(bank.nodes)
    assert ledger['accepted_observation_actions'] == 1
    assert set(result['nodes'][:, 0]) == (set(bank.nodes[:, 0])-{100, 101, 102}) | {300, 301, 302}
    np.testing.assert_array_equal(result['edges'], [[300, 301], [301, 302]])
    assert ledger['selected_node_cost'] == 3


def test_restoration_preserves_old_track_and_counts_every_new_observation():
    bank = fixture_bank()
    actions, receipt = construct_actions(bank, np.ones((len(bank.pairs), 2)), True)
    actions = [a for a in actions if a.owners[0]['changed_node_cost'] == 3]
    result, ledger = apply_observations(bank, actions, receipt, True)
    assert len(result['nodes']) == len(bank.nodes)+3
    assert set(map(tuple, bank.graph['edges'])) <= set(map(tuple, result['edges']))
    assert ledger['selected_node_cost'] == 3
    assert ledger['accepted_observation_actions'] == 1


def test_observation_zero_logits_preserve_exact_arrays_and_geff(tmp_path):
    bank = fixture_bank()
    actions, receipt = construct_actions(bank, np.zeros((len(bank.pairs), 2)), False)
    result, _ = apply_observations(bank, actions, receipt, False)
    for key in ['nodes', 'edges']:
        np.testing.assert_array_equal(result[key], bank.graph[key])
    assert export_geff(tmp_path/'graph.geff', result['nodes'], result['edges'])['exact_roundtrip']
