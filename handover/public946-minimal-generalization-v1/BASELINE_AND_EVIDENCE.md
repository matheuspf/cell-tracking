# Baseline mechanics and evidence boundary

## What was actually inspected

The planner read the committed repository at main
`fb5521629eb41c8c485b291a5bcf344944c113ae`, including agent instructions, public
notebook provenance, inference adapters, architecture receipts, measured v2
results and the v6 continuation. Raw microscopy, public notebook originals and
weights are deliberately ignored in Git and were not available in this session.
The local source review in P000 is therefore mandatory; this document does not
pretend to be a line-by-line audit of an unavailable original notebook.

The user's current public-score report is 0.946. Repository verification is
dated **2026-09-08**, not today. Web access to the Kaggle leaderboard/evaluation
pages did not expose usable score/metric contents during planning. Local Codex
must verify the published notebook version and current official metric through
the existing throttled reference tools; do not silently choose a newer notebook.

## Public notebook to preserve

`docs/notebooks.md` and `configs/notebooks.json` identify:

- `flexonafft/biohub-harmonic-fusion`, v29, scriptVersionId 347965685, LB 0.946.
- `redoctopusk/biohub-942tta`, v1, scriptVersionId 347821442, also LB 0.946.

Use the first, because it has the complete existing local replay. Do not claim
it uniquely leads the competition. Older Code-list scores were stale after a
metric rescore; compare published versions and one evaluator revision.

Original local archive:
`/kaggle/notebooks/biohub-cell-tracking-during-development/flexonafft/biohub-harmonic-fusion/`.
Readable local export: `/kaggle/working/biohub-harmonic-fusion.py`.
The archived .ipynb, metadata and local-verification.json are provenance sources.

Its inputs are the competition plus the existing pilkwang datasets:
`biohub-tracking-support-pack-50ep-v1`,
`biohub-deepcenter-unet3d-center-prior-v1`, and
`biohub-temporal-unet3d-seed314159-v1`. Preserve actual dataset versions and
checkpoint hashes; no extra data/model downloads or training in this study.

## How the existing solution works

1. **Image preparation.** Read 3D images over time. The committed native adapter
   uses `(T,Z,Y,X)` input, spatial downsampling `(1,4,4)` and stored 0.001/0.999
   intensity quantiles. This maps the typical 64x256x256 volume onto a 64-cubed
   grid. Native voxel spacing is anisotropic; physical distances must not be
   confused with original or downsampled voxel distances. P000 must verify the
   complete public path, including any notebook-specific preprocessing patches.
2. **Learned detection and association.** The installed model is
   `UNetNodeTransformer / TemporalUNet3D / SimpleNodeTransformer`. The architecture
   receipt records 2,076,706 parameters for the inspected native model, not the
   sum of all models in the notebook. The image encoder consumes two-frame
   context; downstream decoding has five-frame context. It produces detection
   evidence and features for scoring possible temporal links. Public inference
   includes eight-view detection TTA and a secondary harmonic mixture. Keep
   both checkpoints and these operations; the exact transform set, inverse
   mapping, fusion formula, location and coefficients must be transcribed from
   the archived source, not invented from the notebook title.
3. **Candidate graph and constrained selection.** The committed native trace
   records parent-column normalization, `softmax(raw, dim=0)`, a probability
   threshold >0.48, and ILP graph selection with indegree <=1 and outdegree <=2.
   Its objective uses edge cost `-edge_probability`, appearance 0,
   disappearance 2 and division 1.2. The trace proves that a raw fork is dominated
   by detaching a daughter as a free appearance under those settings. This
   explains why later safe-division repair matters. Do not tune these costs here.
4. **Heuristic output repair.** The actual public call graph is: raw neural graph
   -> edge validity / motion reassignment -> one-frame and strict gap2 repair
   -> safe-division additions -> isolated/short-component pruning -> line-fit
   smoothing -> integer serialization. DeepCenter evidence is used in repair
   through the existing detector/heatmap helpers. Preserve its epoch-2 checkpoint
   and exact use; this is not a request to add cell segmentation.
5. **Submission.** Export node and edge rows to a CSV generated from whatever
   test-image stems Kaggle provides, without reading annotation graphs or using
   local per-embryo fitted models.

The key distinction is between the **learned association / ILP graph** and the
**later motion relinker**. Disabling the latter does not turn off tracking, the
neural model, the ILP, gap completion or division repair.

## Measured evidence motivating this one change

The v2 study scored all 199 supplied clips with the same official run-level
aggregation. Its reported progression is:

| Graph | Local score | Interpretation |
|---|---:|---|
| Raw neural graph | 0.914903 | Before heuristic output repair |
| After motion relinking | 0.892978 | Descriptive intermediate phase |
| Full original public pipeline | 0.911774 | B0 identity reference |
| Full pipeline, motion relinking bypassed | 0.934206 | Complete ablation, not just a phase comparison |
| v3 retained residual pipeline | 0.934802374260586 | More complex local-only reference; not this candidate |
| v6 P0 point residual | 0.9348649864131336 | More complex local-only reference; not this candidate |

The complete no-motion ablation improved both recorded embryos:

| Embryo | Original | No-motion | Delta |
|---|---:|---:|---:|
| 44b6 | 0.912521 | 0.931612 | +0.019091 |
| 6bba | 0.911597 | 0.934525 | +0.022928 |

Pooled edge counts changed from **122,201 TP / 6,885 FP / 6,682 FN** to
**123,023 TP / 4,930 FP / 5,860 FN**. Division counts changed from
**23 TP / 97 FP / 128 FN** to **29 TP / 92 FP / 122 FN**.
These are historical measurements, not fresh results from this branch.

Removing smoothing or safe divisions did not show the same benefit in v2.
The v4 external-data experiment adopted no candidate. V6's measured point-only
gain was tiny and its learned-segmentation/Ultrack arms were blocked. None of
that justifies bundling additional changes into this minimal public-baseline test.

## Why this is plausible, and what it does not establish

Hypothesis: the learned association model already incorporates useful image and
competition context; an additional motion heuristic can overwrite that evidence
with assumptions that transfer less reliably. Removing the heuristic reduces
one source of hand-designed assumptions without fitting anything new.

Counter-hypothesis: the motion prior helps genuinely unseen embryos even though
it hurts these supplied embryos. A one-line stage removal can be a large
behavioral change. The existing ablation was selected after 104 variants and
repeated examination of two embryos; low parameter count does not erase that
selection bias. Inherited checkpoints were also trained/selected using supplied
embryos. New splits or renamed clips cannot turn this into clean OOF.

Thus the plan freezes the already known hypothesis, adds fresh deployment and
failure-mode checks, and prepares one manual hidden-LB trial. It does not promise
that removing the stage improves generalization. In particular:
`0.946 + (0.934206 - 0.911774)` is NOT a valid LB forecast.

## Repository source map (all at the pinned base)

- `AGENTS.md`; `.agents/skills/competition-data/SKILL.md`: environment and data discipline.
- `docs/notebooks.md`; `configs/notebooks.json`: public notebook and model provenance.
- `docs/competition.md`: sparse annotation, physical units, embryo split and submission snapshot.
- `tools/annotation_selection/public_lane.py`: path-only original adaptation; excludes the GT-reading validator.
- `tools/image_native_tracking_v5/native_adapter.py`;
  `results/image-native-tracking-v5/native_architecture.json`: native network interface and public-vs-v5 distinctions.
- `tools/strong_tracker_v3/fresh.py`: actual raw decoder trace and fresh-inference preparation.
- `tools/strong_tracker_v2/replay.py`: real repair call graph and OUTPUT_MOTION_RELINK ablation.
- `tools/strong_tracker_v3/replay.py`: repair namespace; beware its hard-coded no-motion default.
- `results/strong-tracker-v2/v2_report.md`; `docs/strong-tracker-v2.md`: historical counts, scores, caveats and paths.
- `README.md`; `handover/segmentation-tracking-v6/CONTINUATION.md`: later retained results and scope boundary.

These sources are sufficient to nominate the minimal experiment, not to replace
P000's audit of the original, fully patched notebook and prediction script.
