"""Scientific contracts: feature units, parental normalization and exact decoding."""
from itertools import product

import numpy as np

from tools.hoct_reassessment.decode import objective, transition
from tools.hoct_reassessment.legacy import converted_features
from tools.hoct_reassessment.cellpose import candidate_pairs


def test_unit_conversion_does_not_move_output_nodes_or_change_intensity():
    nodes = np.array([[9, 30, 10, 24, 32], [11, 31, 13, 27, 28]])
    props = np.arange(30, dtype=np.float32).reshape(2, 15)
    original_n, original_p = nodes.copy(), props.copy()
    physical, p = converted_features(nodes, props, "physical")
    voxels, v = converted_features(nodes, props, "voxel")
    np.testing.assert_allclose(voxels, nodes[:, 2:] / [1, 4, 4])
    np.testing.assert_allclose(physical / 1.625, voxels)
    np.testing.assert_allclose(v[:, 0], p[:, 0] / 1.625)
    np.testing.assert_allclose(v[:, 5:14], p[:, 5:14] / 1.625**2)
    np.testing.assert_array_equal(v[:, [1, 2, 3, 4, 14]], p[:, [1, 2, 3, 4, 14]])
    np.testing.assert_array_equal(nodes, original_n)
    np.testing.assert_array_equal(props, original_p)


def test_candidates_use_physical_distance_and_do_not_link_across_missing_frames():
    nodes = np.array([[4, 0, 0, 0, 0], [8, 1, 10, 0, 0], [12, 1, 0, 0, 10], [13, 3, 0, 0, 0]])
    positions = nodes[:, 2:] * [1.625, .40625, .40625]
    assert candidate_pairs(nodes, positions).tolist() == [[4, 12]]


def test_assignment_matches_exhaustive_legal_graphs_with_and_without_divisions():
    rng = np.random.default_rng(740573)
    sources, targets = np.array([7, 11, 19]), np.array([31, 32, 35, 40])
    all_pairs = np.array(list(product(sources, targets)))
    for repetition in range(12):
        keep = rng.random(len(all_pairs)) > .2
        pairs = all_pairs[keep]
        similarity = rng.random(len(pairs))
        orphan = rng.random(len(targets))
        for division in (False, True):
            actual, cost = transition(sources, targets, pairs, similarity, orphan, allow_division=division)
            available = set(map(tuple, pairs))
            best = np.inf
            for assignments in product([-1, *sources], repeat=len(targets)):
                edges = [(int(a), int(b)) for a, b in zip(assignments, targets) if a != -1]
                if any(e not in available for e in edges):
                    continue
                if any(sum(e[0] == a for e in edges) > (2 if division else 1) for a in sources):
                    continue
                best = min(best, objective(edges, pairs, similarity, sources, targets, orphan))
            np.testing.assert_allclose(cost, best, atol=1e-12)
            np.testing.assert_allclose(objective(actual, pairs, similarity, sources, targets, orphan), best, atol=1e-12)


def test_no_parent_probability_can_prevent_a_false_link():
    sources, targets, pairs = np.array([4]), np.array([9]), np.array([[4, 9]])
    # Weak parent evidence with a likely orphan costs more than a birth and death.
    assert len(transition(sources, targets, pairs, [.01], [.99])[0]) == 0
    assert transition(sources, targets, pairs, [.99], [.01])[0].tolist() == [[4, 9]]


def test_division_control_suppresses_second_child_without_creating_a_merge():
    sources, targets = np.array([4]), np.array([9, 10])
    pairs = np.array([[4, 9], [4, 10]])
    assert len(transition(sources, targets, pairs, [.95, .90], [.05, .10])[0]) == 2
    assert len(transition(sources, targets, pairs, [.95, .90], [.05, .10], allow_division=False)[0]) == 1


def test_residual_preserves_zero_update_and_parent_orphan_normalization():
    from tools.hoct_reassessment.adapt import posterior
    p = np.array([.2, .3, .6, .1])
    q = np.array([.5, .3])
    groups = np.array([0, 0, 1, 1])
    pp, qq, _ = posterior(np.log(p), np.log(q), groups, np.zeros(4))
    np.testing.assert_allclose(pp, p)
    np.testing.assert_allclose(qq, q)
    pp, qq, _ = posterior(np.log(p), np.log(q), groups, np.array([1000., -1000., 5., -5.]))
    assert np.isfinite(pp).all() and np.isfinite(qq).all()
    np.testing.assert_allclose(np.bincount(groups, weights=pp) + qq, 1)


def test_residual_gradient_matches_finite_differences():
    from tools.hoct_reassessment.adapt import objective
    rng = np.random.default_rng(913586)
    X = rng.normal(size=(6, 3))
    groups = np.array([0, 0, 0, 1, 1, 1])
    lp = np.log(np.array([.1, .3, .2, .4, .1, .2]))
    lq = np.log(np.array([.4, .3]))
    y = np.array([0., 1., 0., 1., 0., 0.])
    weights = rng.normal(size=4)
    _, analytic = objective(weights, X, lp, lq, groups, y, .1)
    numerical = []
    for i in range(4):
        step = np.zeros(4)
        step[i] = 1e-6
        numerical.append((objective(weights+step, X, lp, lq, groups, y, .1)[0] -
                          objective(weights-step, X, lp, lq, groups, y, .1)[0]) / 2e-6)
    np.testing.assert_allclose(analytic, numerical, atol=1e-8)


def test_source_guard_blocks_other_embryo_in_a_separate_process():
    import subprocess
    import sys
    script = """from tools.hoct_reassessment.adapt import source_guard
source_guard('44b6')
try:
    open('/tmp/6bba_unseen-example.npz', 'rb')
except PermissionError:
    pass
else:
    raise AssertionError('Other embryo access was not blocked')
"""
    subprocess.run([sys.executable, "-c", script], check=True)
