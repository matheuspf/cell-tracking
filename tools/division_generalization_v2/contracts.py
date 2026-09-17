"""Immutable scientific settings and zero-based optimizer schedule."""
import math

UPDATES = 4096
PREFIX = 2048
BATCH = 32
LR = 3e-4
SAMPLE_PROBABILITY = 1 / 16
SAMPLING_SEED = 20260916


def lr_factor(step):
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < 8192:
        raise ValueError('Expected zero-based joint update in [0, 8192)')
    if step >= UPDATES:
        return .1
    if step < 128:
        return (step+1)/128
    if step < 3277:
        return 1.
    return .1 + .45*(1+math.cos(math.pi*(step-3277)/(4095-3277)))


def scientific_recipe():
    return dict(joint_updates=UPDATES, common_prefix_updates=PREFIX,
                branch_updates=2048, effective_batch_groups=BATCH,
                positive_slots=8, random_supported_slots=16, confuser_slots=8,
                mining_refresh_updates=[2048, 3072], learning_rate=LR,
                weight_decay=1e-4, gradient_clip=1., warmup_updates=128,
                decay_start=3277, lr_floor_ratio=.1, optional_max_joint_updates=8192,
                random_anchor_probability=SAMPLE_PROBABILITY,
                random_anchor_seed=SAMPLING_SEED, max_changed_edge_fraction=.02,
                decision_boundary=0., keep_on_ties=True,
                unknown_as_negative=False, automatic_budget_shrink=False,
                extension='Both branches: held loss improves >=1% at 4096 vs 3072 and full source score does not regress',
                selection='Best nonregressing full source calibration score, then fewer lost supported edges, then earlier update',
                calibration='Decision-unit held source; >=5 positive overlap groups and >=5 negative overlap groups, otherwise unestablished zero gain',
                normalization='Per-frame fixed 1/99 quantiles; no fitted target statistics',
                sampling='Bernoulli anchors independent of labels; source positive augmentation is a separate nonpopulation stream')
