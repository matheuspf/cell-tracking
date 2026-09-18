# Implementation and training specification

## 1. New package and boundaries

Implement `tools/clean_validation_v10/` with explicit configuration, artifacts/provenance, data, source splits, upstream model/loss/training, graph inference, event sampling/training, calibration, tracing, evaluation, resource accounting and reporting modules. Prefer a small clear package over copying every historical queue. Historical modules remain unchanged unless a separately documented correctness fix is unavoidable; do not rewrite historical reports after such a fix.

Expose this CLI after implementation:

```sh
PYTHONNOUSERSITE=1 PYTHONPATH=tools:. python -m clean_validation_v10 preflight --config handover/clean-validation-division-v10/study.json
PYTHONNOUSERSITE=1 PYTHONPATH=tools:. python -m clean_validation_v10 pilot
PYTHONNOUSERSITE=1 PYTHONPATH=tools:. python -m clean_validation_v10 lock
PYTHONNOUSERSITE=1 PYTHONPATH=tools:. python -m clean_validation_v10 run --resume
PYTHONNOUSERSITE=1 PYTHONPATH=tools:. python -m clean_validation_v10 report
```

Also implement independently resumable `train-upstream`, `prepare-events`, `train-events`, `calibrate`, `predict`, `freeze-predictions`, `evaluate`, `validate` and `package` stages with explicit source/seed/lane arguments. Provide `infer --model-package PATH --images PATH --output PATH` accepting unfamiliar image names and no GT argument. Every stage checks parent/config hashes before reuse; `run --resume` must not execute target scoring before the global freeze gate. These commands are a required interface to implement, not existing functionality at handover creation.

The root configuration records paths discovered locally. Runtime and model inputs are explicit CLI/manifests, not `WORK`, `DATA` or model paths borrowed from another experiment. Reuse pure official scoring, graph-ID, coordinate, candidate and solver routines only after auditing imports.

## 2. Clean upstream model: new sparse-safe reference

Use architecture source for the existing two-frame TemporalUNet3D plus UNetNodeTransformer association family: U-Net channels [32,64,128], image feature width 32, native sampling stride (1,4,4), two consecutive frames. Instantiate the complete model from random initialization. Inspect local released architecture/trainer source and record hashes/licensing; loading released weights even transiently is prohibited. `tools/incumbent_comparison/train_native.py` shows construction and resumable training patterns, but its old output roots, 400-epoch lock and checkpoint-loading workflow are not v10's implementation.

This coarse model is an intentionally affordable source-isolated reference, not the unexecuted native-resolution detector. Native images remain inputs. Normalize each frame using its fixed 1st/99th intensity percentiles, clip to [0,1], then apply zero-origin XY striding. Preserve true native coordinates and physical spacing. No learned normalization or test-time fitting. Test normalization, stride origins, inverse transforms and native roundtrips on real data.

### Detection supervision

Use a heatmap head on the coarse lattice. Build physical Gaussian center targets with sigma 1.625 micrometers and compact support radius 3.25 micrometers around source-fit annotations. Evaluate targets at actual coarse lattice coordinates, rather than discarding the fractional location of a native annotation. Overlapping supports use the maximum, not averaged fictitious cells. Log unresolvable coarse-bin collisions and close daughters.

Supervise known center supports and conservative image-only background, with separate normalization. Positive weight 1; background weight 0.01. Unknown regions have exactly zero supervised detection gradient. Background requires BOTH intensity below the frame's 10th percentile and local contrast below its 20th percentile, excluding center supports, detected local peaks, saturated pixels and uncertain image boundaries. Define local contrast as a 3x3x3 coarse-neighborhood standard deviation, with boundary padding excluded from eligible background. Human positive support overrides every background rule. This background is heuristic, not certified biological absence. Audit faint source annotations and source image crops; report false-negative heuristic risks and mask coverage.

Do not add incumbent pseudo-labels, external masks, target self-supervision or EMA in this study. The native-resolution/sparse-EMA campaign remains deferred. Do not restore a missing-as-negative dense loss to make a failed overfit look better without recording a new symmetric pre-lock scientific amendment.

### Association supervision

Use the same source encoder at supplied centers. Known annotated temporal edges are positive; a competing incoming parent is a biological negative only with supported identity contradiction. Other pairs, including possible unrecorded second daughters, stay unknown. Do not reuse the older `strong_tracker_v3.association.supported_labels` outgoing-edge negative rule. Separate biological identity labels from official metric-risk labels.

Bootstrap with source-fit annotated center queries and bounded physical jitter <=1.5 micrometers; never GT-route inference. During training, mix equal query-group counts from those supervised centers and current model proposals where supported matches exist. During the first 10% of upstream updates, or when proposals contain no supported group, use only supervised center queries and log the fallback. The detector always trains on the sampled full image, not annotation-centered inference crops. Record proposal coverage and the source-train/target-prediction distribution limitation.

The association head predicts independent edge logits trained with masked binary cross-entropy over supported incoming-choice groups; do not force two genuine daughters to compete in a single-child training softmax. Keep one source-supported group as the weighting unit rather than allowing thousands of negatives to dominate. Sample image pairs uniformly across source-fit clips and frame pairs; include images without labels for conservative background supervision, and use zero association loss when no supported group exists.

Default AdamW lr=1e-4, weight decay=0.01, batch 8, gradient norm clip 1. Use BF16 autocast when supported, otherwise FP16 autocast with gradient scaler; lock precision from source-only pilots. Retain batch 8 for the original normalization architecture. An OOM cannot silently substitute a different microbatch/normalization regime; profile and document a matched pre-lock change or report the blocker.

For U updates, warm up linearly for W=min(500, floor(0.05*U)); then cosine-decay from the base rate to 1e-6 over the remaining updates. Save/resume model, optimizer, scheduler, scaler, all RNGs and data-sampler state. This is a new finite-update recipe, not an exact replay of the old trainer. Both directions and seeds get the same U, chosen from [24000,16000,8000] by the locked cost projection.

### Clean C00 graph

Decode deterministic heatmap peaks with probability >0.5 and physical NMS radius 3 micrometers; deterministic ties use native lattice order. No target threshold tuning, top-K count matching or count prior. Fuse predictions for a frame that appears in two adjacent input windows by averaging aligned logits; endpoints use their available window. Use the frame's corresponding encoder features and record the exact feature-fusion rule.

Form adjacent-frame candidate edges from the union of six forward and six reverse spatial neighbors within 15 micrometers. Sample the clean encoder at the actual selected centers and score every candidate. Solve per-transition one-to-one continuation assignment with explicit null choices of zero utility and real-edge utility equal to the independent learned logit; only positive-utility real edges are selected. Resolve equal optima deterministically. A valid empty frame yields births/terminations, not cached substitute detections.

C00 is continuation-only by design. Its graph has at most one parent and one child; C10's event actions can add the second child. Inference uses no E teachers, secondary/harmonic checkpoint, DeepCenter refinement or P0 residual. Report this control's scope prominently and do not compare C10-C00 as the isolated effect of changing P0's division head.

## 3. Event bank, features and complete actions

For OP, reuse the exact pinned P0 observations and native evidence. For clean models, generate source and target observations/evidence independently from each direction/seed's C00. Never reuse another seed's encoder features or use the other source's package for a target prediction.

Use the latest `EventBank` / `Alternatives` design in `tools/pipeline_error_training/{bank,actions}.py`: prediction-only anchor/candidate construction, parent/daughter alternatives, daughter continuations, other-parent alternatives, explicit births/terminations and atomic ownership closure. Do not regress to v4's top-six pair bank or pretend complete alternatives are a new idea. Snapshot the actual `ProposalConfig`, candidate gates and enumeration hash in the execution lock. They stay fixed for all corresponding controls; do not widen gates after target FN inspection.

Write clean-native feature adapters with versioned schemas. Compute actual probabilities, displacement, image evidence and boundary/missingness from the clean model/images. Teacher-vote fields and inherited-native features are unavailable in clean mode: remove them and adjust network input dimensions, or represent documented missingness explicitly. Never fill unavailable teacher votes with agreement, fake probabilities, or import a historical graph to satisfy a shape assertion. The clean and OP heads have independently initialized schema-specific weights; R/N within OP retain exactly the same schema.

Retain stable graph IDs and prediction-only proposal hashes. Group exact duplicate complete edits by canonical removed/added edges while preserving their compatible timing labels. Use a single keep alternative per local decision group. Do not multiply the weight of one biological event because it has many daughter pairs or time anchors. For solver/resource pruning, retain exact rejection reasons, including protection, oversized components, timeout and edit cap.

Preserve existing fork evidence in OP as the primary protected-window policy. C00 has no initial forks. Use the inherited atomic conflict solver and a 2% changed-edge cap, with conservative abstention on unsolved components. No global fork-penalty or node-count sweep is registered. A disabled-edit control must be byte/graph identical to the baseline; this is an explicit no-op path, not an assertion that an arbitrarily zero-initialized network's native objective cannot edit.

## 4. Compact model and controlled sampling

Reuse the compact triplanar encoder and complete-decision model structure from `tools/pipeline_error_training/models.py`: compact image evidence, symmetric daughter-pair event score, association residual, distinct metric-risk score and boundary costs. Use prediction-centered crops and fixed physical support from the audited compact-view adapter. Freeze image preprocessing/crop hashes. Never reuse trained v4/event weights. No Organoid, temporal-volume, external pretraining or larger-backbone arm.

R and N differ only in event sampling/mining, not architecture, initialized weights, optimizer, augmentation, calibration access or update budget. The first 10% of event updates uses 32 identity groups in both. Thereafter:

- R: 16 identity groups + 16 positive-containing event groups, preserving the current sampler's exposure pattern.
- N: 16 identity groups + 8 positive-containing event groups + 4 uniformly sampled supported negative-only event groups + 4 supported hard-negative event groups.

Negative-only groups are deployment-bank neighborhoods containing supported wrong-identity or evaluable false-fork alternatives, not arbitrary unlabeled neighborhoods. Preserve separate `biological`, `identity`, `metric_risk` and unknown masks. A known single child does not certify no second daughter. A safe keep/continuation alternative may be positive only when its source evidence supports that label. If a negative-only group has no supported compatible positive action, use its masked binary risk/identity terms; do not fabricate a positive keep label or divide an empty listwise loss.

Enumerate source-fit predictions across complete source-fit clips for the sampling bank. At 50% of the locked horizon, mine once using each N model's current frozen snapshot: select high-scoring supported negative decisions from source-fit only. Until mining, use the uniform supported-negative pool for the hard-negative slots. Keep an equally bounded source inference pass for the R control to report/match mining access without changing R's sampler. Calibration and target groups are excluded from mining. Unknown high-score events are logged, not negative-labeled. Stratify selected negatives by clip and cap repeated parent/time neighborhoods so a single hard acquisition does not dominate.

Sample groups with an explicit seeded distribution and save visited group identities/probabilities, event support, number of real images, unique observations and token counts. Event-positive quotas imply intentional reweighting: do not call that population prevalence. A source pool with no supported positives is a training blocker for that head, not permission to train a constant classifier and claim an event model. Insufficient hard-negative groups may reuse uniform supported negatives with a named count; do not silently substitute positives.

Use masked compatible-set ranking, supported incoming identity loss and separate binary metric-risk loss. Unknown alternatives must not receive a negative gradient through the normalization denominator; a group containing only unknown labels has zero supervised loss. Same-image augmentation consistency remains a separate label-free term and must not be reported as supervised identity. Protect padding and daughter-swap invariance. Ensure late/early timing-compatible positives remain a set, not contradictory exclusive classes.

Default AdamW lr=3e-4, weight decay=1e-4, 32 accumulated groups/update, norm clip 1. For E in [4000,2000], warm up for min(200,floor(0.05*E)), then cosine to 1e-6. The 10% identity prefix does not reset or restart the LR schedule. Log actual LR endpoints/peak, supervised event-update count, parameter gradients and source curves. Save a single fixed final checkpoint; no target-selected epoch. Count optimizer updates, not microbatches.

## 5. Calibration comparison and source selection

Both methods consume complete source-calibration clips and exactly the same supported candidate population for a given model. Do not recreate deterministic `frame % 9` sampling or ninefold pseudo-propensity weights. Stream the census of deployed anchors/alternatives; unknown support remains censored. A calibration-population census is not dense biological ground truth.

ROW is the existing regularized scalar temperature/intercept binary decision calibration, adapted to explicit v10 inputs. It is a schedule/sampler control, not a claim to reproduce the historical 178-update outcome. Preserve the mathematical objective and report full-source row counts, raw support prevalence and fit diagnostics.

DEPLOY calibrates complete actions within deployment decision groups, against a single keep alternative. Use shared temperature plus regularized edit and fork intercepts; keep score remains zero. Minimize a group-weighted masked compatible-set objective, with a separate supported binary-risk term for negative-only groups without a known positive. Use one unit per eligible group, regularization 0.01, temperature bounds exp([-2,2]) and intercept bounds [-12,12]. Unknown alternatives are censored in fitting, present at actual inference, and reported separately. Do not present these scores as an identifiable biological division posterior.

Evaluate both calibrators through the same complete-action decoder on all source-calibration clips. Select a margin once from [2,4,6,8] using official combined source score, subject to: nonnegative combined delta, adjusted edge contribution delta >=-0.001, and introduced evaluable division FP <=max(2, 2*newly recovered division TP). Rank eligible margins by combined delta, ties within 1e-9 favor the larger margin. Count newly recovered/lost event identities, not only net TP. This is source-only selection on reused/uncertified inner data, not independent validation.

If fewer than two supported positive groups or 100 supported negative groups exist, do not fit free calibration parameters or search margins. Use T=1, edit/fork intercepts=0, margin=6, record insufficient support, and still run the full source safety check. If no margin/fallback passes safety, freeze that package with edits disabled and report an explicit abstaining C10/OP policy. Preserve raw score/rank and rejection diagnostics. A zero-edit output is not evidence that the division model improved; do not replace a failing direction by an unreported different model.

Keep raw-model, calibrated, selected-action and final-graph reliability separate. Record top-score tails, candidate multiplicity, safe/unsafe selected actions, unknown fraction and official source score. The OP 2x2 matrix identifies sampler effects at each calibration method and calibration effects at each fixed weight set. All OP arms use the corrected optimization schedule; a separate convergence-causal claim versus the historical model is not supported.

C10 uses N+DEPLOY in all four clean cells regardless of OP outcomes. Calibration and safety selection may differ by permitted training source because the fitted packages differ; hidden filenames must not choose between packages or thresholds. In outer evaluation, the caller explicitly supplies the source-appropriate package.

## 6. Efficiency and completion

Cache only deterministic, source/lane/seed-hashed preprocessing and embeddings from frozen encoders. Trainable encoders require fresh differentiable forwards; do not speed up training by accidentally freezing image features. Reuse quantiles/crops only after parity tests and include their footprint in the budget. No all-field dense logits need be permanently retained; preserve compact candidate evidence and reproducible hashes.

Small source tests, full source graph checks and actual finite final fits are required. If the clean graph score is low, still finish its registered comparison and explain detection, edge and division limitations. Do not divert mid-study into an unregistered native detector to rescue the result. After target evaluation, select the next research direction in REPORT_BACK.md, not by changing this experiment.
