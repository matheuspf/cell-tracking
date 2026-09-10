# Evidence-driven review of completed v2

Inspected through the GitHub app on 2026-09-09 at commit
`a58c80b041db0a05262a58b3d1b5f34045b8d2ff`. These are repository-reported measured
results; this authoring session did not rerun microscopy inference.

## Preserve the actual success

The full 199-clip `bypass_motion_bounds` pipeline improved the exact original
score **0.9117740142186423** to
**0.9342063149703403**, a gain of **0.0224323007516980**.
It keeps neural associations, disables OUTPUT_MOTION_RELINK, preserves other
repairs, and enforces output coordinate bounds. The bounds-only difference was
rescored as zero. Both embryos improve. Source: v2 summary.json, v2_report.md,
winning_config.json and selected.py. All new gains must be relative to this winner.

Winner edge TP/FP/FN = **123023 / 4930 / 5860**;
division TP/FP/FN = **29 / 92 / 122**; predicted nodes = **4108943**.
Its adjusted edge contribution is 0.9222721585917396 and D=29/243=0.11934156.
Division recall is 29/151=19.2053%; eliminating all 92 current division FPs while
keeping TP fixed improves the combined score by only ~0.00727114. Significant
division gains require additional true recoveries, not only suppression.

Conditional examples holding the adjusted edge contribution exactly fixed:
60 division TP / 60 FP gives score 0.95070818 (gain ~0.01650187);
75 TP / 50 FP gives 0.95958559 (gain ~0.02537928).
They are arithmetic scenarios, NOT measured outcomes or forecasts. Actual edits
change edge counts and must use the full evaluator. Regenerate with scorecard.py.

## What did not work

The strongest v2 association arm scored 0.922810, below the new winner. The
fixed division p=.05 arm had 35 TP but 220 FP, score 0.911501. Native membership
filtering peaked at 0.912791 in hindsight; risk filtering scored 0.899132. The
larger temporal-membership image study did use all source positives, six 1.42M
parameter fits and 30,000 steps total; its best full-image filter scored 0.911598.
Do not repeat the earlier claim that visual selection has only had an eight-second
probe. V2 tested it substantially more; it did NOT train an image division detector.

## Concrete source-level gaps worth testing

**Wrong graph context.** training_data.py, fit.py and infer.py load
`V1/baseline/public`. native.py computes displacement, velocity, adjacency, origin
and candidate ranks on that graph. hypotheses.py follows its successor paths.
The winner was a later replay ablation. Learned gains from the old graph do not
add to the winner's gain; both candidate construction and learned components must
be rebuilt or explicitly labeled frozen-transfer controls.

**Complementary errors, not a reason to restore global relinking.** Relative to
v1, the winner retains 120567 old TP predicted pairs, loses 1634, recovers 2456,
removes 3088 FP pairs and introduces 1133. This suggests local arbitration between
alternatives. These counters use predicted-pair identities; they are not necessarily
1634 distinct newly missing GT edges. Reconcile by GT-edge identity after fresh
matching before calling any part recoverable. Different smoothing/inserted nodes
also mean graphs do not initially share a coordinate universe.

**Restricted fork candidates.** hypotheses.build takes four ranked outgoing
options, requires intersection with an existing successor, applies 12um/16um
parent/sister gates, and drops proposals lacking selected daughter persistence.
This can exclude a correct two-daughter alternative when the current continuation
is wrong or fragmented. The old pool covered 102/151 event observations; this is
a property of that pool and graph, not a biological detectability ceiling.
The old census (99 two-daughter-present misses) must not be reused as the new
winner's residual census.

**Owner protection has a concrete counterexample.** decode.divisions checks
`features[k,25]`, the MINIMUM of two owner distances, repeatedly for each external
owner. An unowned daughter contributes zero, so an external owner 1um away plus
one unowned daughter yields min=0 and bypasses that distance test. The code does
not check that external edge's confidence there. Construct this exact fixture;
replace the aggregate check with per-daughter actual ownership and score evidence.
Do not call the bug the established cause of all division FPs. A stricter guard
can also reject true splits, so owner displacement must ultimately be a scored
complete alternative with explicit boundary constraints.

**Wrong-daughter supervision and priors.** training_data.py masks essentially all
nonpositive hypotheses near known divisions; most negatives are quiet, fully
observed linear contexts. Thus it supplies weak discrimination among rival pairs
at an actual split. Add negatives ONLY when the official sparse-aware local rules
really contradict that alternative. Unknown wrong-looking hypotheses remain
unknown. Normalize positives by event, distinguish pair/timing choices from event
presence, and account for candidate selection before applying absolute thresholds.
Source fork weighted prevalences differ (~0.000866 vs ~0.001603); a fixed .05
probability threshold is not a universal calibrated graph-benefit threshold.

**Pure geometry cannot create missing nuclei.** Stage census must separate absent
candidate points from bad linking. The raw neural graph has no forks in the v2
summary: trace the scorer and decoder's row/column normalization and degree limits
before assuming pretrained logits express two-daughter likelihood. This absence
alone does not prove the exact cause. Freeze detection first, then benchmark a
bounded local detector rescue on the failure mode it can actually address.

## Boundaries on inference

Public upstream training/selection used supplied embryos; only two embryos exist,
and their outcomes have been repeatedly viewed. Source-only learning and fixed
rounds reduce additional leakage but cannot undo this. Do not fabricate independent
clip folds from missing crop origins. Keep oracles local, retrospective and separate
from inference; do not optimize through target-label oracles. No leaderboard claim
or inference about competitors is supported by these local results.
