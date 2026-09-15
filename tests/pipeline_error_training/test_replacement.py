from types import SimpleNamespace
import numpy as np

from pipeline_error_training.actions import Decision
from pipeline_error_training.replacement import complete_value, fork_signatures, ReplacementAlternatives


def test_fork_removal_pays_full_existing_event_and_daughter_birth():
    bank = SimpleNamespace(succ=[[1, 2], [3], [4], [], []], pred=[[], [0], [0], [1], [2]])
    decision = Decision('continuation_a', (0, 1, 2, 3, 4), frozenset({(0, 2)}), frozenset(), (), (2,), (), frozenset())
    signature = (0, 1, 2, 3, 4)
    assert fork_signatures(bank, decision) == ({signature}, set())
    scalar = SimpleNamespace(bank=bank, names=['x'], specs={'x': {'calibration': {'temperature': 1., 'intercept': 0.}}},
        costs=np.ones((5, 2, 1)), sum_links=lambda es: np.array([2.*len(es)]))
    assert complete_value(scalar, decision, {signature: np.array([10.])})[0] == -13.
    assert complete_value(scalar, decision, {signature: np.array([-10.])})[0] == 7.
    assert complete_value(scalar, decision, {}) is None


def test_changed_grandchild_support_replaces_parent_event_score():
    bank = SimpleNamespace(succ=[[1, 2], [3], [4], [], [], []], pred=[[], [0], [0], [1], [2], []])
    decision = Decision('continuation_a', (1, 3, 5, -1, -1), frozenset({(1, 3)}), frozenset({(1, 5)}), (), (3,), (), frozenset())
    assert fork_signatures(bank, decision) == ({(0, 1, 2, 3, 4)}, {(0, 1, 2, 5, 4)})


def test_additive_bound_is_disabled_for_every_affected_old_fork():
    decoder = ReplacementAlternatives.__new__(ReplacementAlternatives)
    decoder.old_forks = {0}
    decoder.pred = [[], [0], [0], [1], [], []]
    decoder.scalar = SimpleNamespace(possible=lambda *args: False)
    event = (4, 3, 5, -1, -1)
    # Direct fork edits, a displaced fork owner, and changed grandchild support
    # can improve by removing negative old energy even when the additive bound
    # alone says no. Every such family must reach full before/after scoring.
    assert decoder.possible_replacement(event, {0: (1,)}, [], [], 'continuation_a')
    assert decoder.possible_replacement(event, {4: (2,)}, [0], [[()]], 'continuation_a')
    assert decoder.possible_replacement(event, {1: (5,)}, [], [], 'continuation_a')
    assert not decoder.possible_replacement(event, {4: (5,)}, [], [], 'continuation_a')
