# Evidence used to choose this study

Snapshot: 2026-09-15. Base commit: bd844731c0e93b401cf94e67c92350d4109e12df.
All repository paths below are relative to the repository root. Read the measured
reports rather than assuming a handover's presence proves execution.

## Baseline and dominant errors

Sources: docs/code-only-planning-20260915.md;
docs/pipeline-errors-20260915.md;
results/pipeline-errors-20260915/summary.json;
results/multidata-training-v4/final_report.md;
results/segmentation-tracking-v6/final_report.md.

| Complete 199-clip pipeline | Combined score | Status |
|---|---:|---|
| P0 | 0.934864986413134 | Adopted reference |
| C4_m6 | 0.935178370257 | Highest pooled result; not adopted |
| C0 / selected v3 | 0.934802374260586 | Underlying reference |

All three share the 4,108,943 final observations. These are 199 clips from TWO
repeatedly used training embryos, not 199 independent embryos. Public LB 0.946
and pilot-panel results describe different populations.

C4_m6 has division TP/FP/FN = 30/92/121. Of its misses, 97 have the required
local parent/daughter matches but no fork, while 24 lack local observations.
P0 has 98 and 24 in those respective groups. Final-graph evidence alone does
not identify the responsible candidate, score, gate or optimizer stage.

C4_m6 has edge TP/FP/FN = 123133/4968/5750. Of FN, 3040 lack a matched endpoint
and 2710 have both endpoints matched. Of 2482 unmatched annotations, 1765 have
a raw proposal within 7 um without a nearby final center; 670 lack a raw proposal
and 47 lose one-to-one matching to another final center. Proximity does not prove
biological identity. Of edge FP, 4907 touch an unmatched prediction; 2467 have
specific nearby competing-center evidence. Sparse unannotated cells are not
negative detector examples.

The audit's '+37 divisions gives 0.950405' is a count-only illustration for
C4_m6, holding other counts fixed. It is neither a deployable oracle nor a
forecast. Error categories overlap across metrics and their gains are not additive.

## What must not be repeated without a substantive change

| Existing evidence | Outcome | Consequence for this plan |
|---|---|---|
| v3 learned event models | Best pooled event arm D_existing_p020 scored 0.915381141505; 69 TP but 1738 division FP | Parent/event classification with class-balanced scores was insufficient. Train complete competing decisions and their identity evidence. |
| v3 candidate expansion | Roughly 190 million alternatives; 98.40% unknown; only six additional supported event observations | Do not solve the problem by indiscriminately enlarging candidate banks. Stream compact local hypotheses and record coverage separately. |
| v3 label-informed expanded event repair | 0.967205187591 versus then-v2 baseline | Historical source-label feasibility only. It motivates learning, but must be rerun on current P0 before claiming current feasible headroom. |
| v4 external G/I/D training | 804000 optimizer updates; no qualifying adopted candidate; C4 seed 2 regressed | Repeating synthetic pretraining and the same compact event classifier is not a new approach. |
| v4 C4_m6 | 0.935178370257; a small 6bba regression and no qualifying replication | Preserve as a challenger, not as an adopted or generalized improvement. |
| v4 continuation rebuild | C4_relinked 0.922258418418; C1_relinked 0.907559949709 | Preserve strong continuations outside explicitly scored local edits. |
| v6 P0 | +0.000062612152548; 38-feature logistic residual | Baseline already includes a learned point control, not learned segmentation. |
| Cellpose + ultrack optical/persistence | 0.82415820 versus matched no-image 0.82813536 on six clips | Do not reuse the frozen v4 optical head and call it a solved mitosis model. |
| OrganoidTracker2 on Cellpose | 2/5 divisions, 1 FP; 0.83126292, with worse edge term than ultrack | A division-head-only transfer at incumbent observations is a justified small probe. Replacing the full tracker is not. |
| Cellpose/refiner unions | 0.63825793 / 0.60722698 on six clips in a controlled native pipeline | More proposals can damage identity and counting. Observation changes need a trained selector and fresh association features. |
| Native-resolution / public946 handovers | Several planned or incomplete arms | Do not report them as completed improvements or silently resume them. |

Sources for these rows:
results/strong-tracker-v3/final_report.md;
results/multidata-training-v4/final_report.md;
results/segmentation-tracking-v6/final_report.md;
docs/cellpose-ultrack-error-analysis-20260914.md;
docs/other-trackers-20260914.md;
docs/incumbent-comparison-20260914.md.

## Source-level findings

- tools/multidata_training_v4/proposals.py keeps six daughter candidates after
  acquisition-wide per-axis IQR normalization. This geometry is not the same as
  physical-distance candidate coverage in the current full pipeline.
- tools/multidata_training_v4/models.py: ImageEvent uses three-frame, 12x12
  triplanes, compressed with mean/moment pooling. The new model tests retained
  temporal/spatial information and abundant continuation pretraining, not just
  a larger MLP on the same compressed features.
- tools/multidata_training_v4/decode.py protects existing forks and two surrounding
  generations. Its local repair path cannot remove those false forks. v3's
  decoder DOES already support complete edits/replacement; reuse its contracts.
- tools/strong_tracker_v3/event_proposals.py already has forward/reverse neighbors,
  alternative daughter paths and +/-1 anchor expansion. These are not novel work
  to reimplement under a new name.
- tools/strong_tracker_v3/decode.py provides Action, BoundedActionComponents and
  solve_actions, with complete owner accounting, conflict closure and no-op.
- tools/other_trackers/organoid.py has an existing Keras-with-torch-backend
  adapter, physical patch rescaling and released calibration checks. It is
  hard-wired to the pilot Cellpose bank: parameterize a NEW adapter for incumbent
  points rather than reusing its cached predictions or candidate linker.
- tools/strong_tracker_v3/inference.py and tools/segmentation_tracking_v6/infer.py
  are the fresh-image reference routes. Features at changed points must be
  regenerated; copying features by nearest old node is invalid.

## Exposure and external evidence

Read docs/incumbent-provenance-20260914.md. The secondary detection checkpoint
contributing 80% of detection logits has all 199 clips in its published training
manifest. This percentage does not describe the association ensemble. v2 E_hgb
teachers carry further target-label exposure, as documented in the v3 report.
A head fitted only on the opposite embryo does not clean those upstream inputs.

Read docs/external-data-guide/README.md before using external data. Zoo exports
lack paired microscopy in the downloaded inventory; the inspected RIKEN sample
has no verified parent-child links. Do not manufacture image/lineage supervision.
The historical synthetic simulator was calibrated on 44b6, including when that
embryo was used as the transfer target. Exclude it from the clean evidence lane.

Official metric source (pinned, verified during planning):
https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md

OrganoidTracker source used by the existing adapter:
https://github.com/jvzonlab/OrganoidTracker/tree/db28ff26584ac6d1230ea90750a3324a2778fb31

Checkpoint release checked on 2026-09-15:
https://zenodo.org/records/18479952
The release is v3, published 2026-02-04, describes mouse intestinal organoid
H2B-mCherry microscopy at XYZ spacing 0.32/0.32/2 um, and corrects an improperly
scaled PyTorch division-model conversion in v2. Verify the exact existing v3
weights and preprocessing; do not silently use v2. Released scaling is a
reproduction reference, not calibrated Biohub division prevalence.

Training/manual entry point:
https://jvzonlab.github.io/OrganoidTracker/
Current competition rules and runtime requirements must be checked from official
references by the local agent before new external assets or notebook packaging.
