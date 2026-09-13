# Evaluation, selection and anti-overfitting contract

## What the data can and cannot establish

All 199 clips and both supplied embryos are reused, correlated local data with
upstream checkpoint exposure. Full paired rescoring is necessary but not clean OOF.
Known crop overlaps must be carried in diagnostic grouping metadata; unknown overlaps
remain unknown. Do not bootstrap 199 clips as independent embryos, issue false
confidence intervals, or call the second embryo untouched. The eight recipes,
three combinations and two transfers are a bounded exploratory program, not a
proof that multiple-comparison bias has disappeared.

Official run-level scoring includes fresh node/edge rematching, division matching
and FP-dependent weights. Recompute the official metric; never average clip scores,
reuse another arm's matching or compare counts with stale weights. Report raw counts
and division Jaccard `TP/(TP+FP+FN)` separately. Do not clip scores to [0,1]; the
historical count-adjusted metric can exceed one in oracle interventions.

## Required per-arm record

Every arm has planned modules and actual source hashes, parent ID, applicability
proof, full expected/completed/failed clip lists, input/model/cache/metric fingerprints,
raw/final stage hashes, full and per-embryo scores, edge/division TP/FP/FN, original
matched-edge survival and new TP after rematching, node/edge edit counts, GPU/RSS/disk
cost, fresh-versus-replay scope, robustness results and an explicit decision reason.
Use null for unmeasured values. Record statuses in outputs, not by mutating the plan
registry; the plan freeze remains verifiable. No missing arm or blank result may be
interpreted as zero performance or a successful experiment.

## Eligibility gate (fixed before new comparative scores)

A single or combination is eligible for selection only when:
1. Correct public identity, allowed scientific diff and valid annotation-free inference.
2. Full expected population is scored under one official metric revision.
3. Positive full-cohort and positive per-embryo deltas versus B0 beyond observed
   A/A numerical variability, and no lower pooled division Jaccard beyond that range.
4. No unexplained numerical/portability issue; runtime/memory fit the predeclared
   local envelope and the verified Kaggle envelope for a packaged candidate.

Estimate numerical ranges from exactly two fresh baseline processes on the four
pilots. Per metric use the maximum absolute paired repeat delta observed (zero if
identical); this is an engineering noise screen, not a statistical uncertainty band.
If a full-cohort improvement is within this screen, label inconclusive, not successful.
If repeats actually differ, repeat full baseline scoring before final selection and
use the maximum observed range; do not search seeds until the preferred arm wins.
The historical 1e-6 rounded-score tolerance is for reproducing published numbers,
not a tunable improvement threshold. A small gain can qualify; local >=0.95 is never
a requirement. This conservative gate may reject a true gain; report that tradeoff.

For X transfers, apply the same delta gate versus B1, not merely versus B0. Thus
B1's old +0.022432 cannot be misattributed to a newly added mechanism. Independently
report B0 differences. The intervention may be useful on B0 and harmful on B1;
retain that interaction result rather than pooling them or claiming transfer success.

## Fixed ranking, combinations and transfer resolution

Among eligible B0-family recipes, rank by descending minimum of the two per-embryo
score deltas versus B0, then descending official full-cohort delta, then fewer
mechanisms, then measured runtime, then lexicographic ID. This is a fixed selection
policy, not a claim of statistical significance. Never deploy per-embryo routing.

Only the three registered combinations run; both constituents must independently
qualify. After their scores are complete, lock the top two eligible non-E01 recipes
for X01/X02 once; fewer are allowed when fewer qualify. Do not promote duplicate
recipes or replace a failed X slot. E01 may still be the best standalone B0-family
candidate, but cannot transfer to B1. The contract helper checks resolution structure;
the executor must attach the actual measured eligibility receipts.

Final package slots: top eligible B0-family recipe under that ranking, plus top
eligible X recipe ranked by the analogous per-embryo/full-cohort deltas versus B1.
Compare their absolute scores too, but do not erase the distinction between these
public-faithful and no-motion lineages. Keep B0 and B1 control notebooks regardless.
No arbitrary three/four-way stack or ensemble of final graph CSVs.

## Fixed robustness and portability checks

On the same four full pilots, after recipes are frozen:
- Rename stems and reverse dataset and batch enumeration. Undo names, canonicalize
  IDs, and compare graphs. Verify child workers also cannot open annotation paths.
- Repeat in a fresh process. Distinguish arithmetic variation from changing models,
  cached tensors, mutable global state or a hidden source/config override.
- Reflect X and separately Y, preserving time and physical spacing; invert coordinates
  for comparison and transform labels only in the separate evaluator. Report both
  official paired deltas and label-free matched-prediction disagreement. Use unchanged
  matching conventions, never adjust tolerances to make equivariance appear improved.
- For E04, also test its coordinate inverse with an impulse/known-peak fixture and
  measure one-voxel non-wrapping image-shift consistency on common valid interior as
  a **diagnostic**, not a replacement official population or a tuned threshold.

X reflection is part of E05's mechanism, so success there is not independent
confirmation; report Y and the original data separately. These nuisance tests are
not new embryos and cannot certify biological generalization. For packaging, require
no worse paired reflection delta beyond measured numerical variability in either
reflected pilot aggregate; report each embryo as well. A failure marks that package
not recommended under this study, not permission to optimize to the reflection.

Validate graph endpoints, consecutive-time edges, IDs, finite coordinates, bounds,
duplicates, merge/degree limits and exact node/edge CSV schema. Reject absent/partial
clips and NaNs. Shared deployment sanitation must be identical and separately scored.
Use true input metadata, not hardcoded 100x64x256x256 assumptions or known test stems.

## LB protocol and reporting

No automatic submission. Freeze at most two novel manual candidate tests and all
notebooks/configs before any candidate LB feedback. Their order is the B0-family
candidate, then the eligible B1-transfer candidate; missing slots remain empty.
B0/B1 controls are separate references, not extra free tuning slots. Record exact
notebook/version, evaluator context, score and completion date for any later user-run
submission. No knob revision from the first LB result, no extrapolated LB score and
no assertion of 0.95+ without a completed Kaggle measurement. Visible example test
clips are deployment smoke tests, not independent evaluation.

The final report must distinguish previously known results, new paired local results,
source-proven no-ops, scientific failures, engineering/resource blockers, clean-data
availability, package readiness and actual LB evidence. Include which mechanisms
changed predictions, why they helped/hurt, worst affected groups and the minimal
selected diff. More experiments are useful only if their outcomes remain auditable.
