# V6: integrate segmentation tools into tracking

This is the **only execution plan** for `handover/segmentation-tracking-v6`.
It replaces the earlier broad v6 proposal and the separate downloadable plan.
Codex implements and executes it locally; this commit contains no new experiments.

## Goal and evidence

Use real 3D instance masks and bounding boxes to improve cell associations and
lineages. Retain C0 = **0.934802374260586**, targeting **>=0.95** on the complete
local evaluation. An attractive segmentation or improved point recall alone is
not success.

The inherited [v5 continuation](../image-native-tracking-v5/CONTINUATION.md) and
[v5 results](../../results/image-native-tracking-v5/ablation_scores.csv) describe
an in-progress snapshot at `0b6ce2d7a77baed54385d4c9b08acbefb6aa7c03`, not a final
negative study. Read actual current local receipts before work. The existing
[region extractor](../../tools/image_native_tracking_v5/observations.py) returns
watershed properties rather than persistent masks. This iteration adds the missing
region representation and tool integration; it does not rerun native training.

## Scope

Use **one learned segmenter plus Ultrack**. Prefer locally available, authorized
FOCUS-3D weights; use an available Cellpose volumetric model as the one fallback.
Choose the observed compartment correctly: nuclear fluorescence supports nucleus
masks, not invisible cell membranes. Do not run a StarDist/HOCT/Trackastra survey,
new external-data training, segmenter fine-tuning, or an optical-flow research arm.

All delivery stays in this branch and its existing directory layout. Reuse local
source checkouts, installed environments, cached dependencies and authorized
weights. **No automatic external downloads, new datasets, hosted inference,
license acceptance, contact sharing, ZIPs, patches or user setup instructions.**
If a required asset is absent, record its exact path/package/checkpoint blocker;
continue independent runnable work. If neither learned segmenter is runnable,
report the segmentation experiment blocked, not completed using substitute masks.
If Ultrack is absent, complete the native-mask comparison and report Ultrack blocked.
Do not let optional tooling failures trigger another open-ended setup campaign.

## S600 — Reuse and establish the baseline

Read AGENTS.md and the existing competition skills. Keep the current data,
evaluation, model-loading and CSV routines. Verify the cached C0 graph scores,
all 199 expected clips, and the complete primary/secondary/augmentation evidence;
a weakened single-model control is not C0. If a newer fully validated incumbent
exists locally, lock it before v6 scores and report both comparisons.

Use `tools/segmentation_tracking_v6/` for the implementation and
`scripts/run_segmentation_tracking_v6.sh` for one entry point, following the old
wrappers. Neither exists yet: implement them, do not present planned commands as
already working. Store outputs in
`/kaggle/working/cell-tracking/segmentation-tracking-v6/`.
Keep prior outputs read-only. Do not stop old jobs, change their source checkout,
or reuse expired deadlines. An isolated worktree is appropriate when needed.

## S610 — Integrate the segmenter and retain objects

Start on two image-selected clips per embryo, eight consecutive frames each.
Select settings using only the source embryo for each direction; target images
or labels must not choose that direction's settings. Use at most two sensible
physical-size recipes for the available backend, then select one per direction.

For FOCUS, reuse its headless `infer_volume` backend. For Cellpose, use its actual
installed volumetric API. Run individual ZYX frames, preserving T separately.
Check the model output against actual image spacing, axis order and resampling
origin. Keep model/checkpoint/config hashes and real output shape in the receipt.
The input scale recorded by the repository is Z/Y/X = 1.625/0.40625/0.40625 um;
verify it locally rather than treating isotropic provider coordinates as native.

Retain frame-local label maps or compressed bbox-local masks, not just centroids.
Each object needs `(clip, frame, provider, instance_id)`, mask reference, half-open
ZYX bbox, native center, physical volume, shape, inside-mask intensity and border
flags. Model confidence is nullable; missing confidence does not mean background.
Frame-local labels are not tracking identities. Inspect real orthogonal overlays
for merged cells, duplicate instances and tile seams. Sparse point containment
and one-to-one node recall are useful checks, but not dense segmentation accuracy.

Pilot time and storage before all-frame inference. Keep the model resident where
possible and cache each selected mask once for all later comparisons.

## S620 — Test mask information without replacing the strong tracker

Attach masks to C0 detections through image-only same-frame correspondence.
Use one-to-one ownership and flag shared/merged masks. Missing or ambiguous masks
leave the original native evidence intact; do not delete detections or fabricate
sphere masks. Preserve the C0 centers, candidate edge pool and full native scores.

Compare the same source-trained regularized link scorer and the same existing
decoder in three arms: point features, then bbox features, then actual masks.
Use a single simple scorer family and matched fit budgets, not another search.
Mask features include overlap, directional coverage, physical volume ratio,
shape change and masked appearance. Boxes prune comparisons but do not substitute
for occupancy. If existing image-registration code is already available and tested,
its fixed displacement can align masks in every matched arm; do not build a new
flow model. Cellpose segmentation flows are not inter-frame motion.

Include parent-to-daughter-union overlap and persistent daughter separation as
soft division evidence within the same graph decision. Compare against continuation
plus independent birth and competing parents. Do not require exact volume or
intensity conservation, and do not create a separate event-model campaign.

Fit only on source-embryo supported transitions. Unknown cells/links stay unknown;
one recorded daughter does not make every other daughter negative. No GT IDs,
count estimates or target filenames enter model features or routing.

## S630 — Integrate actual Ultrack on the same masks

Run the available Ultrack labels/foreground/contour interface and actual lineage
solver on the selected segmenter's outputs. Validate installed signatures and
coordinate/ID mapping on the pilot first. Use its locally available noncommercial
solver or an already licensed solver; do not acquire a new license.

Keep competing single-region/split hypotheses mutually exclusive. Measure which
original masks survive contour/hierarchy construction; do not assume every input
mask remains a selectable object. Do not union duplicates into additional cells.
Use instance overlap, appearance and temporal consistency for links. Run a second
Ultrack arm adding the full native association evidence. Recompute native evidence
at changed region centers as necessary; do not attach unrelated old-node scores
by an undocumented nearest-neighbor copy.

Use the same hierarchy and solver settings for the two Ultrack arms. Validate
birth, continuation, division, competing ownership and boundary cases. Export
one observation per selected region/frame and valid consecutive-frame links.
The competition still receives center/edge CSV rows, but only after mask-based
tracking. Never send t-to-t+2 gap links or allow two parents for one observation.

## S680 — Compare, retain improvements and deliver

Keep the initial comparison to **six complete configurations**, not twenty:

| ID | Comparison |
|---|---|
| C0 | Unchanged full incumbent. |
| P0 | Matched source scorer/decoder with native and point information. |
| B0 | P0 plus real bounding-box descriptors. |
| M0 | B0 plus actual mask, appearance and region-division evidence. |
| U0 | Standalone Ultrack on the selected instance masks. |
| U1 | Same Ultrack setup plus correctly mapped full native evidence. |

These separate tool benefits from a new decoder or scorer. Freeze both source
recipes before revealing new target results. Every complete comparison covers
all 199 clips with fresh official node matching, division scoring and the exact
run-level aggregation. Reuse the pinned local evaluator; do not substitute mask
IoU for the competition score. Report each embryo, pooled score, TP/FP/FN for
edges/divisions, node counts and actual runtime. Mark incomplete arms explicitly;
never silently fill failed clips with C0 and call that a pure tool result.

Keep C0 unless a candidate improves pooled score without embryo regression beyond
1e-8. Repeat the same recipe with a second seed for a learned finalist; reserve
at most two extra complete runs for replication. Fixed pretrained/deterministic
routes need repeatability checks rather than pretend retraining. Smaller valid
gains are retained, but >=0.95 alone is target attainment. Reused embryos and
inherited checkpoint exposure remain exploratory validation, not a hidden-LB claim.

Use one 4090 worker, bounded CPU workers and clip-at-a-time masks/databases.
Inspect current resources; respect the previous 20 GiB GPU / 24 GiB process-RSS
ceilings or lower available limits and keep 8 GiB disk free. Start with a shared
24 GPU-hour cap including segmentation and training, not one budget per arm.
No all-dataset uncompressed mask copies. Preserve all prior data and artifacts;
only disposable v6 scratch may be reclaimed. Report uncompleted runs at the cap.

Before selecting a package, run the actual image-to-mask-to-graph entry point on
two complete fresh clips, one per embryo, including an unfamiliar filename.
Old predicted masks/graphs, GT and network access must be unavailable. Check
semantic graph/CSV parity, runtime and the unchanged C0 fallback.

Deliver `results/segmentation-tracking-v6/` with one `final_report.md`, aggregate
`ablation_scores.csv`, `status.json` and a small offline `dashboard.html` reusing
existing reporting tools. Keep real image overlays and detailed masks locally.
The report answers whether bbox/mask evidence helped and whether Ultrack helped;
it is not another proposal. Write `CONTINUATION.md` in this handover with exact
commands, selected artifacts and blockers. Keep weights, raw images, masks,
detailed labels and credentials outside Git. Commit/push sanitized code, configs
and results to the same branch; no PR merge or Kaggle submission.

## Supporting checks and references

Retain the existing NumPy mask contracts. Locally run their tests plus real adapter
checks for anisotropy, half-open boxes, empty frames, label permutation, tile seams,
mask ownership, hierarchy conflicts and legal fork export. Helper tests are not
evidence that a segmenter ran or that scores improved.

Tool sources from the preceding review: [FOCUS-3D](https://github.com/yu-lab-vt/FOCUS-3D)
(`5c4b53f743a0fbbae056e2c1a139895ae819f069`) and
[Ultrack](https://github.com/royerlab/ultrack)
(`5c94d845eb0a7b78c8dc24492ef00f218a467995`). Consult local copies and record actual
installed versions; these references are not instructions to download assets.
