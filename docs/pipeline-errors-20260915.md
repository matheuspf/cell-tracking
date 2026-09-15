# Current best solution and pipeline error priorities

## Recommendation

**Focus first on division decisions using the incumbent observations, then on
observation selection and temporal identity.** The strongest direct evidence is
that **97 of 121 missed divisions already have the required local parent and
daughter matches, but no predicted fork near the parent**. This leaves a large
opportunity after detection. Improving detector recall alone cannot repair this
group.

Open the [Pipeline report](http://localhost:8767/#tab=report) for the interactive
breakdown and [Tracking & divisions](http://localhost:8767/#tab=tracking) for the
individual mistakes. Both use the same generated
[evidence summary](../results/pipeline-errors-20260915/summary.json).

## What is currently best?

- **C4_m6: 0.935178370257**, the highest recorded pooled score among complete
  non-oracle local pipelines, evaluated on all 199 clips. It uses the incumbent
  TemporalUNet-based detection/association pipeline and the v4 learned division
  refinement with the margin-6 policy.
- **P0: 0.934864986413**, the retained pipeline after adoption checks. It uses
  the same final observations and a conservative learned association residual.
- **C0 / selected v3: 0.934802374261**, the underlying comparison baseline.

C4_m6 was not adopted: its 6bba result regressed slightly versus C0 and the
required replication did not qualify. It exceeds P0 by only **0.000313384**:
one additional recovered division outweighs its slightly worse edge term.
P0 is still the appropriate adopted reference; keep C4_m6 alongside it when
comparing new experiments.

The newer Cellpose/ultrack, HOCT and other tracker experiments are smaller
pilot comparisons. They do not establish a replacement for these complete
199-clip results. A notebook's public leaderboard score also measures a
different population and is not directly comparable to these local scores.

Sources: [v4 results](../results/multidata-training-v4/final_report.md),
[P0 results](../results/segmentation-tracking-v6/final_report.md), and the
[recent ultrack error analysis](cellpose-ultrack-current-errors-20260914.md).

## Scope and verification

The new audit rechecked **398 complete graphs**: C4_m6 and P0 on every training
clip, covering **19,900 frames**, **133,318 sparse annotations**, and **128,883
annotated edges**. All node matches, edge statuses and division counts agree
with the frozen evaluation receipts. Predictions and evaluation files were
checked by SHA256, canonical GEFF graphs were reloaded, and native image/graph
scales were verified. No model, prediction or source data was changed.

This is an exploratory, training-exposed diagnosis. The dominant released
detector checkpoint's documented training list includes all 199 clips. Both
embryos have also been reused during development. These scores are neither
independent generalization estimates nor measured leaderboard gains. See the
[checkpoint provenance audit](incumbent-provenance-20260914.md).

The audit uses official metric revision
`075fc5f5a52d11077f9dc2b074644618f26939e2`. Its combined score is the weighted
adjusted edge Jaccard plus 0.1 times division Jaccard. The
[official metric description](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md)
explains sparse edge validity, independent local division matching and timing
tolerance. The live official description was also consulted on 2026-09-15;
the numerical results here remain tied to the pinned implementation.

## 1. Division decisions: the first priority

C4_m6 recovers **30 / 151 divisions**:

- 30 true positives (TP), 121 false negatives (FN), 92 false positives (FP).
- Recall **19.9%**; precision **24.6%**.
- Division Jaccard **0.123457**; contribution to the combined score **0.012346**
  out of a possible 0.1.

The 121 misses separate cleanly using the official division-window matching:

- **97 (80.2%) have sufficient parent and daughter matches but no parent-side
  predicted fork.** This includes the official allowance for an earlier parent
  or a fork at its immediate successor; it is not just an exact-frame check.
- **24 (19.8%) lack sufficient local parent or daughter matches.** Detection,
  localization or observation selection can matter here.
- No remaining miss in this audit had a locally proposed final-graph fork that
  reached the topology or one-to-one pairing failure groups.

Of the 92 false divisions, **87 are evaluable forks that do not recover an
annotated division**, and **five have daughter-branch evidence in different
annotated connected components**. These are official FP decisions. They are
not inferred by calling every unmatched fork false.

**Next experiment:** freeze the incumbent observations and compare a learned
division-versus-continuation-plus-birth decision against a matched control.
Record whether the correct daughters enter the candidate bank, survive the
geometric gate, pass the learned score/margin, and survive graph conflict
resolution. The final-graph diagnosis establishes absence of a fork; it cannot
alone distinguish candidate generation, scoring, acceptance and decoding.

Do not lower a global division threshold solely to increase fork count: the
current 92 FP already exceed the 30 TP. Preserve correct continuations and
measure the full edge/division score, not just division recall. P0 has the same
pattern: **98** matched-window/no-fork misses and **24** observation-limited
misses.

### A concrete example

In [44b6_12dfb391, frame 66](http://localhost:8767/#tab=tracking&tmodel=pooled&tfilter=division_fn&tstage=division_no_fork&tcase=pooled%3A44b6_12dfb391%3Adivision_fn%3A172000000050),
the parent and both daughters are detected. The parent connects correctly to
daughter C, but the expected **B → D** branch is missing. D instead receives a
scored incorrect link from an unmatched predecessor, while both daughters
continue correctly afterward. The viewer retains the correct branch alongside
the incorrect one. This illustrates why division recovery and observation
identity are related, and why the same event can appear in multiple score flags.

## 2. Observation selection and identity: the biggest link-error pattern

The final graph matches **130,836 / 133,318 annotations (98.14%)**. Its 2,482
unmatched annotations consist of:

- **1,765 (71.1%)** with a raw detector candidate within 7 µm but no nearby final
  center. This is evidence of a downstream selection/position gap, not proof
  that the nearby candidate is the same biological cell.
- **670 (27.0%)** with neither a raw proposal nor a final center within 7 µm.
- **47 (1.9%)** with a final center nearby that loses the one-to-one assignment.

These endpoint failures block **3,040 / 5,750 missed temporal edges (52.9%)**:

- **1,970** have nearby detector proposals but unavailable final observations.
- **1,009** have at least one missing endpoint without a nearby detector proposal.
- **61** have nearby final centers assigned elsewhere.

These are disjoint edge groups. For mixed failures at two endpoints, the
classification checks a raw-proposal gap first, then an assignment conflict,
then the selection gap. An annotation can affect multiple incident edges.

Incorrect links reinforce the identity problem:

- **4,907 / 4,968 (98.8%)** touch an unmatched prediction.
- **2,467 / 4,968 (49.7%)** use an unmatched prediction within 7 µm of the expected
  annotated cell, while a different prediction owns that cell's assignment.
- Another **2,440** touch an unmatched prediction without that specific
  competing-center evidence.
- **61** connect two matched cells incorrectly.

**Next experiment:** inspect these trajectories, then compare observation
selection and continuation identity scores on a fixed candidate bank. Preserve
true nearby cells; the evidence does not justify indiscriminate deduplication.
Because annotations are sparse, the millions of unmatched whole-field
predictions are not millions of proven false detections.

## 3. Continuation linking with both endpoints available

**2,710 / 5,750 missed edges (47.1%)** have both endpoint observations matched.
They identify where association or graph decoding can improve the result
without adding a detection. P0 has **2,671** such misses.

This is distinct from the 3,040 endpoint-blocked links. It is still not a
complete causal attribution to a specific model: final graphs cannot say
whether a link was absent from the exact pipeline's candidate bank or rejected
by its score, constraints or optimizer. Log those stages in the next comparison.

Wrong edges can simultaneously produce an FP and an FN. A division mistake
can also produce edge errors. **Do not sum these groups as independent biological
mistakes or independent repair opportunities.**

## How much can each improvement matter?

C4_m6's observed score decomposes as:

- Raw edge Jaccard: **0.919925887741**.
- Adjusted edge contribution: **0.922832691244**.
- Division contribution: **0.012345679012**.
- Combined score: **0.935178370257**.

The net node-count adjustment is **+0.002906804**, so it is not currently a net
score loss. It uses coarse per-clip count estimates; it does not identify false
cells individually.

The report includes accounting scenarios that alter only the named counts and
recompute the official clip weights with node totals held fixed:

- Perfect division term: **+0.087654**.
- Perfect edge matching at the same node counts: **+0.080046**.
- Recover all missed divisions, keeping the current division FP: **+0.049794**.
- Remove all incorrect edges: **+0.035572**.
- Recover all endpoint-blocked links: **+0.022552**.
- Recover all missing links with matched endpoints: **+0.020271**.
- Remove all false divisions, keeping the current division recall: **+0.007522**.

These are **not rescored repaired graphs, forecasts or additive gains**.
Actual graph changes affect several counts together, and repaired detections
can change the one-to-one matching. As a scale illustration, recovering 37
additional divisions with no other count change would produce **0.950405**;
this calculation is not evidence that those repairs are achievable.

## Where to review first

Embryo **6bba contributes 80.9% of edge error flags** and 103 of 121 division
misses. It also contains most annotations, so this is an absolute workload
measure, not proof of intrinsically harder images. The largest edge-error clips
are `6bba_57b7cc1e` (634 FP+FN), `6bba_ebff6e76` (480),
`6bba_786893ac` (463), and `6bba_a90a0b9c` (430).

The priority differs by embryo: 44b6 has lower raw edge Jaccard (**0.902161**)
than 6bba (**0.923215**), and larger edge-term accounting headroom than
division-term headroom. The UI exposes each embryo separately and preserves
the full-population comparison.

## UI and reproduction

The added **Pipeline report** tab includes score decomposition, best/adopted
status, evidence partitions, accounting scenarios, next experiments, embryo
comparisons, top error clips and a downloadable JSON report.

The added **Tracking & divisions** tab exposes **10,931 C4_m6** and **10,921 P0**
official error flags. It supports category, evidence, embryo and clip filters;
pagination and direct case links; expected/actual temporal graphs; microscopy
from the exact event frames in XY/XZ/YZ; synchronized physical crops and depth
projections; endpoint links to Detection review; and local notes/export.

Division FN scenes use their independently computed official local-window
assignments. Edge scenes and center-check tables use whole-clip assignments.
The UI labels this distinction explicitly. Unknown edges stay unevaluated.
Missing source images are reported rather than replaced with adjacent frames.

Rebuild the report and index using the existing metric runtime:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=tools \
  /kaggle/envs/cell-tracking-annotation-selection-v1/bin/python \
  -m center_comparison index-tracking
```

This requires the existing complete `index-detection` export and frozen source
graphs. Derived graphs and SQLite catalogs stay under
`/kaggle/working/cell-tracking/center-comparison`; only concise evidence and code
are kept in the repository. The source detection-index hash is checked by the
API, so rebuilding detections requires rebuilding this audit too.

Serve using the main environment:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=tools \
  /home/mpf/.conda/envs/cell-tracking/bin/python -m center_comparison serve
```

Use localhost port 8767, with SSH forwarding when needed. The existing
Detection review and 42-frame FOCUS comparison remain available in their tabs.

### Validation

- **26 tests passed** across tracking review, detection review and the existing
  center comparison, including sparse-label attribution and nonadditive score
  scenarios.
- **398 fresh full-graph audits** reproduce the frozen metric counts.
- **469 scenes checked**, including every division error for both pipelines:
  all displayed coordinates and edges occur in the source graphs.
- Browser checks passed for six report scope/model combinations, all ten
  tracking category/model combinations, last-case pagination, direct links,
  detection links, notes/export, empty-filter recovery and existing tabs.
- Native frame hashes, XY/XZ/YZ views and identical physical crops were checked.
  Desktop, tablet and 390-pixel mobile layouts passed without document overflow;
  there were no browser errors, failed requests or external requests.

See the [validation receipt](../results/pipeline-errors-20260915/validation.json).
