import numpy as np

from pipeline_error_training.observations import ObservationBank
from pipeline_error_training.observation_infer import construct_actions, apply_observations
from pipeline_error_training.serialization import export_geff


def fixture_bank():
    # Both independent 2% budgets permit this three-node, four-edge swap.
    nodes = np.array([[100+k, k % 3, 4, 4 if k < 3 else 200+k, 4] for k in range(300)])
    edges = np.array([[100+k+a,100+k+a+1] for k in range(0,300,3) for a in [0,1]])
    raw = dict(nodes=np.array([[k, k, 4, 5, 4] for k in range(3)]),
               edges=np.array([[0, 1], [1, 2]]), node_probabilities=np.ones(3)*.8,
               original_graph_sha256='fixture')
    row = dict(physical_scale=[1., 1., 1.], image_shape=[3, 512, 512, 512],
               baselines={'P0': {'sha256': 'fixture'}}, raw={'sha256': 'fixture'})
    return ObservationBank(row, dict(nodes=nodes, edges=edges), raw)


def test_one_for_one_is_atomic_and_raw_ids_are_not_reused():
    bank = fixture_bank()
    gain = np.ones((len(bank.pairs), 2))
    actions, receipt = construct_actions(bank, gain, False)
    actions = [a for a in actions if a.owners[0]['changed_node_cost'] == 3]
    result, ledger = apply_observations(bank, actions, receipt, False)
    assert len(result['nodes']) == len(bank.nodes)
    assert ledger['accepted_observation_actions'] == 1
    assert set(result['nodes'][:, 0]) == (set(bank.nodes[:, 0])-{100, 101, 102}) | {400, 401, 402}
    assert set(map(tuple,result['edges'])) == (set(map(tuple,bank.graph['edges']))-{(100,101),(101,102)}) | {(400,401),(401,402)}
    assert ledger['selected_node_cost'] == 3
    assert ledger['observation_changed_edges'] == ledger['changed_edge_cap'] == 4


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


def test_node_allowance_does_not_override_the_separate_edge_budget():
    bank = fixture_bank()
    bank = ObservationBank(bank.row, dict(nodes=bank.nodes,edges=bank.graph['edges'][:2]), bank.raw)
    actions, receipt = construct_actions(bank, np.ones((len(bank.pairs),2)), False)
    actions = [a for a in actions if a.owners[0]['changed_node_cost']==3]
    result, ledger = apply_observations(bank, actions, receipt, False)
    assert ledger['changed_node_cap']==6 and ledger['changed_edge_cap']==0
    assert ledger['accepted_observation_actions']==0
    np.testing.assert_array_equal(result['nodes'],bank.nodes)
    np.testing.assert_array_equal(result['edges'],bank.graph['edges'])


def test_followup_association_cannot_spend_an_exhausted_observation_budget():
    from pipeline_error_training.association import decode
    bank = fixture_bank()
    edges, receipt = decode(bank.nodes,bank.graph['edges'],{},[],max_changed_edges=0)
    np.testing.assert_array_equal(edges,bank.graph['edges'])
    assert receipt['no_remaining_edge_budget']


def test_known_same_identity_does_not_turn_localization_ties_into_negatives():
    from pipeline_error_training.observations import localization_preference
    assert localization_preference(1.,1.)==-1
    assert localization_preference(1.,1.25)==-1
    assert localization_preference(1.1,1.)==-1
    assert localization_preference(2.,1.)==1
    assert localization_preference(1.,2.)==0


def test_remaining_budget_limits_a_real_complete_owner_reassignment():
    from pipeline_error_training.association import decode
    bank = fixture_bank()
    pairs = np.concatenate([bank.graph['edges']-100, np.array([[0,4],[3,1]])])
    features = np.zeros((len(pairs),38),np.float32)
    features[:,0] = .9
    scores = np.zeros(len(pairs))
    scores[-2:] = 20.
    native = dict(pairs=pairs,edge_features=features)
    before = set(map(tuple,bank.graph['edges']))
    full, _ = decode(bank.nodes,bank.graph['edges'],native,scores)
    bounded, receipt = decode(bank.nodes,bank.graph['edges'],native,scores,max_changed_edges=3)
    assert len(before ^ set(map(tuple,full)))==4
    assert len(before ^ set(map(tuple,bounded)))<=3
    assert receipt['explicit_edge_budget']==3
