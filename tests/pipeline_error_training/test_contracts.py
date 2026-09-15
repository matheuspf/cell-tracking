import numpy as np
import pytest

from pipeline_error_training.actions import Alternatives, Decision, apply_decisions, fork_support
from pipeline_error_training.bank import EventBank
from pipeline_error_training.labels import supported_incoming, Topology
from pipeline_error_training.splits import split_groups
from strong_tracker_v3.decode import Action, BoundedActionComponents, solve_actions
from strong_tracker_v3.event_proposals import build


def fixture():
    nodes = np.array([[101, 0, 10, 20, 20], [205, 1, 10, 21, 20], [999, 1, 10, 16, 20],
                      [47, 2, 10, 22, 20], [88, 2, 10, 15, 20], [170, 0, 10, 16, 20]])
    edges = np.array([[101, 205], [205, 47], [170, 999], [999, 88]])
    pairs = np.array([[0, 1], [0, 2], [1, 3], [2, 4], [5, 2], [5, 1]])
    native = dict(pairs=pairs, edge_features=np.column_stack([np.full(6, .9), np.zeros((6, 37))]),
                  node_features=np.ones((6, 16)))
    return nodes, edges, native


def test_streaming_exact_v3_union():
    nodes, edges, native = fixture()
    expected, _ = build(nodes, edges, native, [1., 1., 1.])
    actual = EventBank(nodes, edges, native, [1., 1., 1.]).materialize()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_noncontiguous_ids_zero_head_exact_and_owner_closure():
    nodes, edges, native = fixture()
    bank = EventBank(nodes, edges, native, [1., 1., 1.])
    decisions = list(Alternatives(bank).event((0, 1, 2, 3, 4)))
    # continuation_a is already the incumbent, so it is the canonical keep row.
    assert {'keep', 'division', 'continuation_b'} <= {d.kind for d in decisions}
    out, _ = apply_decisions(nodes, edges, decisions, np.zeros(len(decisions)), max_fraction=1)
    np.testing.assert_array_equal(out, edges)
    fork = next(d for d in decisions if d.kind == 'division')
    assert (5, 2) in fork.remove
    assert fork.owners and 5 in fork.terminations
    out, receipt = apply_decisions(nodes, edges, [fork], [10.], max_fraction=1)
    assert {(101, 205), (101, 999)} <= set(map(tuple, out))
    assert (170, 999) not in set(map(tuple, out))
    assert receipt['exact_unchanged_outside_components']


def test_missing_sparse_branch_and_known_competitor():
    pairs = [(1, 2), (1, 3), (4, 2), (9, 2)]
    matches = {1: 11, 2: 12, 4: 14}
    assert supported_incoming(pairs, matches, [(11, 12)]).tolist() == [1, -1, 0, -1]


def test_full_fork_window_protection():
    nodes = np.array([[i*13+9, i//2, 0, i, 0] for i in range(8)])
    edges = np.array([[9, 35], [35, 61], [35, 74], [61, 87]])
    protected = fork_support(nodes, edges)
    assert {0, 2, 4, 5, 6} <= protected


def test_late_bridge_cannot_resurrect_abstained_component():
    collector = BoundedActionComponents(max_alternatives=2)
    for k, resources in enumerate([{0}, {0, 1}, {1, 2}, {4}, {2, 4}]):
        collector.add(Action(k, 1., set(), set(), {('n', r) for r in resources}, []))
    actions, receipt = collector.finish()
    assert not actions
    assert receipt['actions_in_abstained_components'] == 5


def test_actual_solver_ties_and_conflicts():
    actions = [Action(0, 0., set(), set(), {('n', 0)}, []),
               Action(1, 3., set(), set(), {('n', 1)}, []),
               Action(2, 2., set(), set(), {('n', 1)}, [])]
    chosen, receipt = solve_actions(actions)
    assert chosen == [1]
    assert receipt['accepted_actions'] == 1


def test_shared_clips_and_alternatives_never_split():
    assignment = split_groups(['a', 'b', 'c'], {'frame1': ['a', 'b'], 'frame2': ['b', 'c']})
    assert len({r['partition'] for r in assignment.values()}) == 1
    assert len({r['overlap_group'] for r in assignment.values()}) == 1


def test_topology_removes_displaced_owner():
    topology = Topology([[], [], [0], [1]], [[2], [3], [], []], {(1, 3)}, {(0, 3)})
    assert topology.predecessors(3) == [0]
    assert topology.successors(1) == []


def test_nonfinite_decision_rejected():
    nodes, edges, _ = fixture()
    keep = Decision('keep', (0, 1, 2, 3, 4), frozenset(), frozenset(), (), (), (), frozenset())
    with pytest.raises(ValueError, match='Nonfinite'):
        apply_decisions(nodes, edges, [keep], [np.nan])


def test_actual_milp_timeout_abstains_complete_component():
    rng = np.random.default_rng(9182)
    actions = [Action(k, float(rng.uniform(1, 3)), set(), set(),
        {('n', int(n)) for n in rng.choice(70, 8, replace=False)}, []) for k in range(180)]
    chosen, receipt = solve_actions(actions, time_limit=0.)
    assert not chosen
    assert receipt['solver_abstentions'] == 1
