# Current branch: models, measured results and data interfaces

Prepared on 2026-09-09 against `handover/strong-tracker-v3`, starting at
`f67ba60`. This is a factual handover, not a measured external-data experiment.
**No external-data training or transfer evaluation was performed in this
preparation work.** Their files and prepared labels exist on the original machine;
the [dataset guide](README.md) describes access, provenance and verification.

**The next agent decides whether and how to include each dataset in training.**
This document records the current models, measured failures, available supervision
and interface mismatches. It does not select datasets, architectures, training
order, ablations or a future promotion rule.

## What the next agent is improving

Read the committed [v3 report](../../results/strong-tracker-v3/final_report.md),
[completed handover](../../results/strong-tracker-v3/NEXT_AGENT.md),
[winning policy](../../results/strong-tracker-v3/winning_config.json) and
[execution contract](../strong-tracker-v3.md). Core conclusions are available in
Git; large graphs, caches, checkpoint weights and upstream notebook sources are
separate dependencies and are not restored by checking out the branch.

The selected policy is **`A_residual_m3.0`**, pooled score
**0.934802374260586**, a gain of **0.0005960592902456** over v2
`bypass_motion_bounds` (**0.9342063149703403**). Its target-embryo scores are:

- **44b6, 71 clips:** 0.931664468721842; gain over v2 +0.000052081944511.
- **6bba, 128 clips:** 0.935221784097327; gain over v2 +0.000696534227794.

These are official local evaluations of 199 supplied clips, not leaderboard
results. V3 selected 4,108,943 predicted nodes; edge TP/FP/FN are
123,135 / 4,965 / 5,748 and division TP/FP/FN are 29 / 92 / 122.
It recovered 112 GT edges and lost none relative to v2, while adding 35 FP edges.
See [family outcomes](../../results/strong-tracker-v3/family_outcomes.json) and
[association regret](../../results/strong-tracker-v3/association_regret_summary.json).

The selected inference computation is:

1. The preserved Harmonic Fusion U-Net/transformer predictor, primary and
   secondary checkpoints, and its graph solver produce native image-coordinate
   detections and learned association evidence.
2. The v2 repair pipeline disables the original motion relinker. It retains
   edge validity, the single-parent guard, gap/gap2 recovery, DeepCenter-gated
   safe divisions, pruning, smoothing, and integer bounds-checked serialization.
3. V3 reconstructs raw/old-final/frozen E-teacher evidence at current centers and
   applies a regularized logistic residual to native edge log odds. The selected
   margin is 3.0; bounded local graph edits preserve its existing constraints.
   The newly trained v3 event and rescue arms are **not** selected additions.

The exact integration points are
[repair order](../../tools/strong_tracker_v3/replay.py),
[association model/decoder](../../tools/strong_tracker_v3/association.py),
[feature builder](../../tools/strong_tracker_v3/features.py) and
[fresh inference policy](../../tools/strong_tracker_v3/inference.py).
V1's classical candidate filter is a different model from the branch's strongest
tracker: the filter improved 0.674116 to 0.676907, but its frozen primary filter
reduced the stronger public tracker from 0.911774 to 0.873322. Dense synthetic
cell labels do not directly identify the sparse-annotation membership target of
that earlier study. See the [v1 report](../../results/annotation-selection-v1/report.md).

## Measured failures relevant to the available labels

**Division selection.** The v2 census found
98 missed annotated divisions whose two daughter lineages were already matched
but had no surviving fork; 18 had missing daughter candidates, four had competing
assignment, and two lacked parent-side evidence. Among edge FN, 3,040 involved
missing points or assignment, 2,643 had a rejected available alternative, and
177 had no alternative link. These are census categories of the v2 incumbent,
not a fresh causal decomposition of a newly trained external-data model.
Source: [census summary](../../results/strong-tracker-v3/census_summary.json).

The wider event pool covered 112 of 151 annotated division observations but
created 190,139,632 alternatives, 98.40% with unknown sparse labels. Four real
image fits ran 10,000 optimizer steps each, using only 20 positive source groups
for 44b6 and 92 for 6bba. All ten learned event policies lost in both embryos.
The best, `D_existing_p020`, scored **0.9153811415049362**: it added 40 division TP
but 1,646 division FP relative to v2. The annotation-assisted expanded source
oracle reached **0.9672051875911127** (+0.0329988726207724 versus v2), with 110
division TP / 94 FP / 41 FN. That oracle proves some candidate feasibility on
observed labels; it is neither deployable nor an upper bound. More alternatives
alone failed. See the [measured report](../../results/strong-tracker-v3/final_report.md)
and [event coverage](../../results/strong-tracker-v3/event_candidate_coverage.json).

**Raw decoder objective.**
The preserved raw neural ILP charges division cost 1.2, gives any second edge
reward at most 1, and permits a free daughter birth. Thus forks are strictly
dominated, even when candidate links exist. This was verified using the actual
solver. A change in learned edge probabilities within those bounds leaves this
objective property intact. Whether to change that decoder is a separate model
decision, outside this handover. See
[raw decoder evidence](../../results/strong-tracker-v3/raw_decoder_trace.json).

**Image rescue outcomes.** Corrected image rescue
`R_image_persistent` scored 0.9342333345063806, but its 6bba delta was negative;
the heatmap rescue control equaled v2. These results do not establish a gain from
external center supervision or from a larger detector. The scored output includes
all detections and links, with duplicate/incorrect-link and count effects.
The branch contains flat-background foreground-persistence regression checks
from the rescue correction.

## Dataset-to-model interfaces

### Synthetic static volumes: center supervision, no temporal ground truth

There are **1,539** native uint16 volumes of shape `(64,256,256)` with fractional
ZYX center labels and native spacing `(1.625,0.40625,0.40625)` µm. No dense masks
are supplied. Center heatmaps and localization targets can be derived from the
fractional coordinates; the existing reader implements a heatmap example. They cannot
provide real temporal links or cell divisions merely by repeating a frame.

The existing [load_synthetic / points_to_heatmap](../../tools/biohub_external_data/data_adapter.py)
functions provide a reader and Gaussian center-target generator. The target is
not a segmentation mask or a fixed evaluator target. The U-Net/transformer and
DeepCenter have their own training interfaces in their pinned source;
this repository's external-data reader does not train either.

`load_synthetic(grid="native")` preserves static resolution;
`grid="pooled"` uses XY stride-4 sampling. That reader uses slicing, **not mean
pooling**. V3's image-feature builder separately computes DoG responses after
XY mean pooling by two. These transformations serve different interfaces and
produce different image statistics and spatial grids.

### Synthetic sequences: dense graph supervision, six-frame context

There are **2,174** six-frame clips, **4,056,226 node observations** and
**165,267 division parents**. Images are already `(6,64,64,64)`, but source ZYX
labels remain on the native `(64,256,256)` grid. The prepared reader supplies:

- `image`: float32 TZYX, source uint16 divided by 65,535.
- `zyx_image`: fractional ZYX aligned to the chosen image grid; sequence Y/X are
  divided by four and sequence spacing becomes `(1.625,1.625,1.625)` µm.
- `node_id`, `t`, graph `edges`, `division_parent_ids`, and graph-derived
  `tracklet_id`. Edges index node rows; source `track_id` is preserved as
  `source_clone_id` because daughters inherit it. It is not a unique tracklet.
- `division_target_observed`: masks last-frame cells, whose future is absent.
  Longer path/timing objectives require additional context-specific masks.

Available supervision includes centers, parent/daughter identities, temporal
associations and nondivision relationships in the dense source graph. It does not
include an annotation-selection label for the real sparse-label process. GT
adjacency, GT fork flags and clone IDs are targets/provenance, not evidence that
the deployed predictor can observe.

The existing [event proposal builder](../../tools/strong_tracker_v3/event_proposals.py)
expects graph nodes `[node_id,t,z,y,x]`, edges referring to node IDs, physical
spacing, and **prediction-derived** native/node features. Its proposals are
`[parent,daughter_a,daughter_b,path_a,path_b]` row indices, with missing paths
represented by `-1`. `EVENT_FEATURES` contains native scores, unordered daughter
geometry, motion, image responses, density, owner evidence, path persistence and
boundary information. Image features and labels remain separate.

The [event image model](../../tools/strong_tracker_v3/event_model.py) consumes
parent-centered crops `(N,9,4,12,12)`: nine frames, three triplanar image channels
plus a validity channel, 2 µm sampling over a 24 µm field. It flattens time and
channel to 36 input channels. `extract_crops` expects raw intensity-scale images.
The reader's `[0,1]` output does not match its contrast-normalization floor without
an adaptation. The original uint16 images remain available separately. Pooled
synthetic sequence centers and pooled physical spacing align with the 64³ image;
native centers and native spacing describe the unreleased full-resolution grid.

**Six frames cannot supply nine observed frames.** Existing zero padding and
validity channels represent absent context; they do not recover missing temporal
observations. How to reconcile this mismatch is a future model-design decision.
The existing event model is daughter-permutation invariant. The prepared
synthetic split holds complete examples together, not individual nodes/crops.

The published generator calibrated tissue shape/density on ten clips from the
first sorted embryo, **44b6**. Its fixed motion/render assumptions and synthetic
division frequency also differ from real data. A synthetic holdout checks the
generator domain, not biological transfer. The prepared synthetic division count
is not a real prevalence calibration target: official sparse GEFFs contain 151
annotated parent forks, and missing annotations plus censoring prevent inferring
the true division prior. The v3 event objectives balance classes/groups and
their probabilities are explicitly uncalibrated. The recorded parent-group
sampling fraction is not a row inclusion probability, and the source audit does
not validate it as an inverse-probability weight. A forum-derived prevalence
multiplier or thresholds transferred to a new training domain have no measured
calibration in this branch. See
[sampling audit](../../results/strong-tracker-v3/source_sampling_metadata_audit.json).

### Six Zoo exports: geometry and trajectories without image features

Prepared graphs exist for **zebrafish, drosophila, mouse, ascidian, elegans and
tribolium**. These are experimental tracking outputs with no downloaded raw
microscopy, no dense masks, and no blanket claim of manual label correctness.
Their original base ZIPs and enriched stores overlap and are not independent
training examples. Source study/species and acquisition identity are relevant
provenance; independence between different releases has not been established.

Prepared arrays include node IDs, time, `zyx_source`, tracklet IDs, direct parent
tracklets, adjacent-frame edges and graph-derived fork indices. Source positions
are not assured µm, and frame cadence still needs source-specific confirmation.
The preparation decodes direct parents from metadata; the stored
`tracks_to_tracks` relation is a reachability closure, not temporal edges. The
viewer `divisions` field is a color encoding. Gap counts and absent correspondence
remain explicit. Mouse has **no recorded forks in this export**; this does not
establish representative biological nondivision labels.

The graphs contain motion and branch geometry, subject to their source quality
and calibration. They cannot directly populate the selected residual model's
full feature vector: native logits, teacher votes, image responses, detector
confidence and relocation provenance are missing. The current residual even
uses native log odds as a fixed coefficient-one offset. Zero-filling those
columns does not reproduce that model's training distribution. A compatible
training interface has not been implemented. Source adjacency used as a target
cannot simultaneously count as independent prediction evidence of that target.

The zebrafish export has Zebrahub organizer clearance, which does not establish
physical calibration or label quality. The five other species are not extra
independent zebrafish embryos. Each original dataset has its own terms; the viewer
software license does not license all datasets. Source links and clearance are
in the [dataset guide](README.md).

### Seven RIKEN archives: positions and features, links unresolved

Downloaded archives cover **animal A, animal B, animal C, dorsal, in toto MZoep,
in toto WT and ventral** zebrafish acquisitions. Only animal C received a full
structural count audit and ten-frame prepared export: **821 frames**, **3,447,269
position observations**, µm coordinates, 90-second cadence, and 14 intensity/
size feature types. The other six archives have not had the same label inspection.

No explicit temporal or parent-child links were found in animal C. Its IDs are
measurement IDs, not established persistent cell identities. There are no paired
raw images in these downloads. Source positions/intensity/extent measurements
describe spatial layout, signal and size. Nearest-neighbor correspondences would
be pseudo-labels, not newly recovered GT. FWHM features can be missing or unusually
broad and are not cell boundaries. Source CC BY-NC-SA
terms require their own use-case assessment; technical preparation alone does
not establish permission. This does not amount to an automatic conflict with
Kaggle's winner-code terms, which exempt input data/models from relicensing.

## Historical evaluation protocol and its limitations

- **The executed study used both source directions:** fit using 44b6's 71 source
  clips and predict 6bba's 128; fit using 6bba and predict 44b6. Both directions
  were frozen before comparative target scoring. Reproducing that study requires
  those directions; random clip/frame splitting is not its validation protocol.
- The published simulator calibrated density/tissue shape on 44b6. Reusing that
  synthetic
  data when 44b6 is the target would introduce target-derived distribution
  information. Whether to recalibrate, retain and disclose that exposure, or
  exclude that source is a decision for the next agent.
- Even with direct source-only fits, these two embryos have already been reused.
  Public checkpoints and the frozen v2 E-hgb teachers carry upstream target-label
  exposure. Unknown cross-clip overlap prevents independent inner folds or
  bootstrap confidence intervals. The study is explicitly **operational
  exploratory**; new external pretraining would not erase that history.
- V3's historical gate was positive pooled gain versus v2 and nonnegative gain
  in each embryo within `1e-10`. This describes its recorded selection. For a new
  experiment, the next agent defines and records its own protocol. The current
  branch incumbent is **selected v3**: beating v2 alone does not establish an
  improvement over that current solution. This document adds no promotion rule.
- The executed evaluation used scorer revision
  `075fc5f5a52d11077f9dc2b074644618f26939e2`, 199-clip coverage, fixed supplied
  count estimates and fresh official node matching/division assignment. The
  score is adjusted edge Jaccard plus 0.1 times division Jaccard; 7 µm is a node
  matching tolerance, not a universal temporal-link radius. Timing-window path
  evidence matters: preserving immediate fork edges did not prevent division
  loss in the branch's association oracle. See
  [metric verification](../../results/strong-tracker-v3/metric_verification.json)
  and [division regret](../../results/strong-tracker-v3/source_oracle_division_regret_summary.json).
- Real sparse unlabeled points/edges/events are unknown under the supported-label
  rules. Dense synthetic truth can supervise synthetic negatives, but does not
  turn unmatched real candidates into background. The branch's prediction contract
  excludes direct use of held-out labels, source oracles, match tables and
  evaluation count estimates as inference features; inherited teacher exposure
  remains documented above.
- Prepared coordinates are floating point. The submission contract uses native
  integer ZYX, in-bounds positions, exact dataset names, unique node IDs per
  dataset, consecutive temporal links, indegree ≤1 and outdegree ≤2.
  Prepared edge row indices require mapping if node IDs are renumbered.
  The external CSV demo is not an inference
  runner or a complete submission validator.
- V1/v2/v3 raw inputs, checkpoints, outputs, failed controls and locks describe
  the historical studies. Their locations and identities are recorded in the
  committed [artifact manifest](../../results/strong-tracker-v3/artifact_manifest.json) and
  [inference dependency manifest](../../results/strong-tracker-v3/inference_package_manifest.json)
  but large artifacts are absent from Git. Historical `/root/...` paths in
  receipts identify their original execution host, not the present checkout.

Use the `cell-tracking` Conda environment and `PYTHONNOUSERSITE=1` for external
data inspection/preparation, with the isolated preparation dependencies documented
in [ACCESS.md](ACCESS.md). The winning tracker has a separately pinned CUDA study
runtime; the preparation dependency directory was isolated from that runtime.
This handover adds no training,
checkpoint changes, Kaggle submission, or measured score improvement.
