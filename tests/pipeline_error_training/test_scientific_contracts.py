import numpy as np
import torch

from pipeline_error_training.actions import Alternatives, Decision, apply_decisions
from pipeline_error_training.bank import EventBank
from pipeline_error_training.labels import SourceLabels
from pipeline_error_training.models import DecisionModel
from pipeline_error_training.scoring import edge_features, score
from pipeline_error_training.train import decision_scores
from tests.pipeline_error_training.test_contracts import fixture


def test_removed_supported_continuation_cannot_hide_behind_retained_edge():
    nodes, edges, native = fixture()
    labels = SourceLabels(nodes, edges, nodes, edges, [1., 1., 1.])
    d = Decision('continuation_a', (0, 1, 2, 3, 4), frozenset({(5, 2)}), frozenset(),
                 (), (2,), (5,), frozenset())
    target = labels.decision(d)
    assert target['supported_edges'] > 0
    assert target['lost_supported_edges'] == 1
    assert target['identity'] == 0


def test_official_early_on_time_late_event_set_agrees_with_serialized_score():
    from annotation_selection.metric_adapter import evaluate_graph
    n = np.array([[0, 0, 10, 30, 30], [1, 1, 10, 30, 30], [2, 2, 10, 25, 30],
                  [3, 2, 10, 35, 30], [4, 3, 10, 24, 30], [5, 3, 10, 36, 30],
                  [6, 1, 10, 35, 30], [7, 2, 10, 60, 30], [8, 3, 10, 61, 30],
                  [9, 1, 10, 59, 30], [10, 0, 10, 58, 30]])
    ge = np.array([[0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [10, 9], [9, 7], [7, 8]])
    e = ge[~np.all(ge == [1, 3], axis=1)]
    gn = n[n[:, 0] != 6]
    pairs = np.array([(i, j) for i in range(len(n)) for j in range(len(n)) if n[j, 1] == n[i, 1]+1])
    native = dict(pairs=pairs, edge_features=np.column_stack([np.full(len(pairs), .9), np.zeros((len(pairs), 37))]))
    bank = EventBank(n, e, native, [1., 1., 1.])
    labels = SourceLabels(n, e, gn, ge, [1., 1., 1.])
    groups = set()
    for event in [(0, 1, 6, 2, 3), (1, 2, 3, 4, 5), (2, 4, 5, -1, -1)]:
        decisions = [d for d in Alternatives(bank, replace=True).event(event) if d.kind == 'division']
        good = [d for d in decisions if labels.decision(d)['biological'] == 1]
        assert good
        d = good[0]
        target = labels.decision(d)
        groups.update(target['compatible_events'])
        out, _ = apply_decisions(n, e, [d], [1.], max_fraction=1.)
        measured, _, _ = evaluate_graph('timing', n, out, gn, ge, [1., 1., 1.], len(n))
        assert measured['division_tp'] == 1
        assert target['metric_fork_target'] == 1
    assert len(groups) == 1


def test_partial_single_child_is_unknown_biology_but_evaluable_metric_risk():
    nodes, edges, native = fixture()
    gt_nodes = nodes[:2]
    gt_edges = edges[:1]
    labels = SourceLabels(nodes, edges, gt_nodes, gt_edges, [1., 1., 1.])
    d = next(d for d in Alternatives(EventBank(nodes, edges, native, [1., 1., 1.])).event((0, 1, 2, 3, 4))
             if d.kind == 'division')
    target = labels.decision(d)
    assert target['biological'] == -1
    assert target['metric_fork_target'] == 0


def test_vectorized_prediction_matches_training_complete_owner_scores():
    torch.manual_seed(4)
    nodes, edges, native = fixture()
    bank = EventBank(nodes, edges, native, [1., 1., 1.])
    decisions = list(Alternatives(bank).event((0, 1, 2, 3, 4)))
    features = [np.arange(40, dtype=np.float32)/40 for _ in decisions]
    model = DecisionModel('compact').eval()
    z = torch.randn(len(nodes), 128)
    valid = np.ones((len(nodes), 3, 4), np.float32)
    pairs = sorted({p for d in decisions for p in d.remove | d.add})
    x = edge_features(nodes, native, pairs, [1., 1., 1.])
    link = model.link_scores(z, torch.tensor(pairs), torch.tensor(x))+torch.tensor(x[:, 23])
    batch = dict(decisions=decisions, node_map={i: i for i in range(len(nodes))},
                 pair_map={p: i for i, p in enumerate(pairs)}, valid=torch.tensor(valid),
                 records=[dict(event_features=f) for f in features])
    expected, risk = decision_scores(model, batch, z, link)
    actual, actual_risk = score(model, dict(mean=np.zeros(38), scale=np.ones(38)), nodes, native,
                                z.numpy(), valid, decisions, features, [1., 1., 1.])
    np.testing.assert_allclose(actual, expected.detach().numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual_risk, risk.detach().numpy(), atol=1e-6, rtol=1e-6)


def test_sampling_perturbations_preserve_shape_time_mask_and_finite_values():
    from pipeline_error_training.augment import view
    torch.set_num_threads(1)
    for family, shape, t in [('compact', (2, 3, 3, 12, 12), 3),
                             ('temporal', (2, 7, 2, 16, 64, 64), 7),
                             ('organoid', (2, 12, 64, 64, 3), 7)]:
        torch.manual_seed(17)
        patch = torch.rand(shape)
        valid = torch.ones(2, t, 4)
        valid[:, -1] = 0
        result = view(dict(patch=patch, valid=valid), family)
        assert result['patch'].shape == patch.shape
        assert torch.isfinite(result['patch']).all()
        assert ((result['patch'] >= 0) & (result['patch'] <= 1)).all()
        if family != 'organoid':
            assert torch.count_nonzero(result['patch'][:, -1]) == 0
def test_empty_source_decision_partition_remains_boolean_and_unknown():
    from pipeline_error_training.calibration import targets
    good, known = targets([])
    assert good.dtype == known.dtype == np.dtype(bool)
    assert (~known).sum() == 0
    assert not len(good)
