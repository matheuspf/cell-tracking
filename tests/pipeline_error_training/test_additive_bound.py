import numpy as np
import torch

from pipeline_error_training.actions import Alternatives
from pipeline_error_training.bank import EventBank
from pipeline_error_training.fast_infer import AdditiveScores, event_values
from pipeline_error_training.models import DecisionModel
from pipeline_error_training.scoring import score
from tests.pipeline_error_training.test_contracts import fixture


def test_bound_preserves_every_improving_complete_donor_assignment():
    torch.set_num_threads(1)
    nodes, edges, native = fixture()
    bank = EventBank(nodes, edges, native, [1., 1., 1.])
    torch.manual_seed(8)
    models = {'a': DecisionModel('compact').eval(), 'b': DecisionModel('compact').eval()}
    spec = dict(mean=np.zeros(38), scale=np.ones(38), calibration=dict(temperature=1.3, intercept=-.1))
    specs = {n: spec for n in models}
    z = {n: np.random.default_rng(15).normal(size=(len(nodes), 128)).astype(np.float32) for n in models}
    valid = {n: np.ones((len(nodes), 3, 4), np.float32) for n in models}
    scorer = AdditiveScores(models, specs, bank, z, valid)
    bounded = Alternatives(bank, possible=scorer.possible)
    unbounded = Alternatives(bank)
    for event, features, _ in bank:
        ev = event_values(models, z, [(event, features)])[0]
        scorer.set_event(event, ev)
        full = [d for d in unbounded.event(event) if d.kind != 'keep']
        retained = {d.key for d in bounded.event(event)}
        if not full:
            continue
        values = np.stack([scorer.decision(d) for d in full])
        for m, name in enumerate(models):
            expected, _ = score(models[name], spec, nodes, native, z[name], valid[name], full,
                                  [features]*len(full), [1., 1., 1.])
            expected = expected/1.3-.1
            np.testing.assert_allclose(values[:, m], expected, atol=2e-6, rtol=2e-6)
        assert {d.key for d, v in zip(full, values) if (v > 1e-9).any()} <= retained


def test_high_confidence_other_parent_beats_a_mitotic_parent_score():
    nodes, edges, native = fixture()
    native['edge_features'][4, 23] = 20.
    bank = EventBank(nodes, edges, native, [1., 1., 1.])
    model = DecisionModel('compact').eval()
    for p in model.parameters():
        p.data.zero_()
    spec = dict(mean=np.zeros(38), scale=np.ones(38), calibration=dict(temperature=1., intercept=0.))
    scorer = AdditiveScores({'model': model}, {'model': spec}, bank,
        {'model': np.zeros((len(nodes), 128), np.float32)}, {'model': np.ones((len(nodes), 3, 4), np.float32)})
    event = (0, 1, 2, 3, 4)
    scorer.set_event(event, [10.])
    d = next(d for d in Alternatives(bank).event(event) if d.kind == 'division')
    assert scorer.decision(d)[0] < 0
