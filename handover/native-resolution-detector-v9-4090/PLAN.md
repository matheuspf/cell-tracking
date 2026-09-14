# V9: native-resolution detector on one RTX 4090

**Authored:** 2026-09-14. **Status:** planned, not trained.  
**Branch:** `handover/native-resolution-detector-v9-4090`.  
**Parent:** `e654c50c7543b3486556e2a1cd472f669f07a625` (unexecuted v8); inherited main is `fb5521629eb41c8c485b291a5bcf344944c113ae`.

## 1. Deliverable and differences from v8

Build and train a new, independently callable, source-isolated nuclear-center detector from raw competition images. Preserve acquired `(Z,Y,X)=(64,256,256)` pixels. Compare against both clean scratch controls and the exact operational incumbent. Then test the replacement with association/repair held fixed. A detector improvement is not automatically a tracking improvement, and neither establishes a new hidden leaderboard score.

V9 turns the approved review into one local execution contract. It inherits v8 without altering old experiments. It fixes the integer-offset-target issue, makes outer validation fixed rather than adaptively selected, orders native finals before broad sweeps, measures 4090/RAM capacity instead of assuming a remote `/kaggle` host, and supplies read-only preflight and contract tests. Do not execute v8 and v9 as parallel studies.

The primary milestone is **two independent source-to-opposite-embryo native models with complete detection evaluation**, followed by same-recipe second-seed replication. The final deliverable is actual checkpoints, inference code, measurements and a usable continuation—not another plan, library integration, or training-loss-only result.

## 2. D900: repository, rules and exact baseline audit

Read AGENTS and competition skills; refresh official references through the maintained extractor with its normal throttles. The committed instructions use `PYTHONPATH=tools python -m kaggle_extract pages` for pages or `sync` for the broader mirror. Inspect help before execution. Do not accept new terms, alter account/team state, or submit. Record source timestamps, metric implementation hash, relevant rule excerpts, and online/offline verification status. If refresh is temporarily unavailable, use the existing dated official mirror for compliant source-only development and record the limitation; unresolved rule questions block dependent external/deployment actions, not all source-only training.

Inspect `docs/notebooks.md`, downloaded notebook exports and installed support packs. Record exact checkpoint SHA-256, inference source hashes, initialization/training evidence, TTA, ensemble fusion, thresholds and postprocessing. The maintained v5 adapter loads an upstream checkpoint before applying a replacement state, asserts `(1,4,4)` downsampling, and documents that its experimental primary-only path omits the full incumbent's eight-view TTA and secondary harmonic mixture. **Do not reuse its load() in the clean trainer.** Architecture code reuse without weights is allowed and recorded.

The reviewed upstream family is a temporal 3D U-Net plus detection and association heads; the strided XY loader turns native 64x256x256 into 64 cubed. Existing local reports describe exposure to both embryos. Names such as split_0 and 50ep-v1 are not provenance. Classify each installed asset as verified_source_only, verified_external_no_target_exposure, known_target_exposure, or unknown. Treat unproven assets conservatively.

Preserve raw neural detections separately from repaired graph nodes. The full public notebook is the operational detector comparator; a primary-only adapter is a separately named diagnostic. Missing incumbent assets cannot prevent clean scratch/native training. Historical C0=0.934802374260586 and P0=0.934864986413134 are full-tracker controls, not detector scores or clean embryo validation. The public 0.946 is also a full-pipeline leaderboard result.

## 3. Validation and dependency isolation: lock before fitting

Expected inventory: 199 training clips, prefix 44b6 with 71 and 6bba with 128. Verify actual Zarr metadata and GEFF pairing; do not silently drop samples if the inventory differs. Each held-out clip is inferred across every frame. Visible test copies are not validation data.

| Direction | Train images/labels/teachers | Held-out inference/evaluation |
|---|---|---|
| A | 44b6 only | 6bba only |
| B | 6bba only | 44b6 only |

Use a new random initialization per direction and seed. Source-only applies recursively to normalization fitting, warm starts, pseudo-labels, EMA, calibration and cached features. No incumbent teacher, old target-calibrated synthetic images, external pretrained model, target-image self-supervision, or test-time parameter update enters the primary study. Image-local normalization using a fixed rule during inference is allowed. Train and predict in separate processes with explicit manifests; inference receives image paths only, evaluation alone receives target GEFF paths. Embryo names route folds, never model features, thresholds or clip-specific policy.

Historical audits found overlapping clips and could not certify independent inner groups. **V9 does not use within-embryo validation for model selection.** All recipes, fixed final schedules, decoder rules, seeds and resource-based omissions are locked before outer scoring. Source diagnostics may repair broken code/coordinates or adjust resource allocation before the lock; any scientific recipe change is logged, applied symmetrically, and cannot depend on target performance. Do not manufacture a random clip/node/frame split.

Primary=N_ema; supervised anchor=N_base. Other arms are prespecified descriptive comparisons. Run both directions and all retained seeds, write checkpoint and configuration hashes, freeze prediction tables, then unlock outer scores. Do not evaluate A, alter the plan, and then call B independent confirmation. Model selection after seeing the sweep is exploratory and cannot overwrite the primary result. Seeds are optimization replication, not new biological subjects. Both embryos were used in past research, so this is source-isolated fitting but not a pristine new external test set.

A frozen sweep manifest lists each retained run and both directions before evaluation. Complete-data checks compare exact clip-ID sets and per-clip frame sets; an intersection of available result directories is forbidden. Zero detections in a frame is a valid measured output, not an absent result.

## 4. D910: machine and input readiness

Use the existing suitable notebook/PyTorch environment after inspection; do not broadly upgrade packages or destroy other environments. Record Python, torch, CUDA, driver, zarr/codecs and metric versions. Set PYTHONNOUSERSITE=1. The host has one RTX 4090 and approximately **65 GB**, not a promise of 65 GiB free. Measure installed and available memory locally.

Default data discovery: explicit `--data-root`, then repo `data`, then the canonical path in AGENTS. Outputs default to ignored `work/native-resolution-detector-v9-4090`; an explicit durable output mount is supported. Do not create `/kaggle` aliases or use sudo as a prerequisite. Reuse actual data, never copy the entire dataset into RAM or duplicate the archive.

One active GPU job, microbatch one. Mixed precision (BF16 when supported; otherwise FP16 with scaler), GroupNorm, activation checkpointing, gradient accumulation four. Measure peak CUDA allocated AND reserved memory, process memory, nvidia-smi utilization and total host process-tree RSS, including loaders and EMA. Pilot real backward, optimizer step, teacher inference and checkpoint save—not just a no-grad forward.

Budgets: aim <=20 GiB CUDA reserved, maintain >=2 GiB device free; <=44 GiB combined study process RSS and >=10 GiB system MemAvailable. Reduce limits when other applications require space. Start two loader workers, prefetch two, at most four workers after measurement. Limit decoded-frame/cache memory to **4 GiB total**, not per worker. Start conservative CPU thread counts and avoid worker x BLAS oversubscription. Log the actual CPU/IO bottleneck before adding workers.

Keep >=10 GiB durable free disk, plus measured room for two atomic checkpoint generations, compact predictions and reports. The 10 GiB reserve alone is not proof of sufficient capacity. Do not save all-frame dense heatmaps; stream decoding to compact scored point tables. Clean only explicitly owned, reproducible v9 cache files. Never delete previous studies to make space.

Pilot full native input with width16 first. Fixed fallback ladder, based on memory/seam/source checks only:
1. Full 64x256x256 input.
2. Native XY tiles: 64x192x192 input including 32-pixel XY halos; 64x128x128 valid core.
3. Native XY tiles: 64x128x128 input including 32-pixel XY halos; 64x64x64 valid core.

At physical boundaries pad explicitly and crop to valid output. Tiling never rescales XY or Z. These halos are starting engineering choices, not proof of receptive-field containment. Quantify seams and common-core shifts versus larger context; fix one tiling recipe for N_base/N_ema/N_lowpass and both directions before real fits. GroupNorm makes exact full/tile equivalence inappropriate; exact same-recipe replay and coordinate round trips are required. Memory fallback does not silently shrink model width; width changes are separate labeled arms. Do not turn native learning into a proposal-centered-only refiner.

## 5. D920: implement the native model and sparse objective

Create `tools/native_detector_v9/` with data, targets, models, losses, train, infer, evaluate, integrate, report and one module CLI. Reuse maintained schema/metric utilities explicitly, not functions that load old weights. Add focused tests under `tests/native_detector_v9/` or this module's isolated test directory.

**Native architecture:** residual anisotropic 3D U-Net, channels [16,32,64,96,128]. First two encoder blocks predominantly 1x3x3; first two strides (1,2,2). After feature spacing becomes approximately isotropic, use 3x3x3 and (2,2,2). Decode to the input grid with original-resolution skips. Early native details must reach learned features before any XY reduction. No artificial 256-cubed Z upsampling. Plain control: residual 3D U-Net [16,32,64,96] with ordinary isotropic convolutions/pooling. Context arm: lightweight image-only coarse full-field branch fused with the native path.

**Localization correction from v8:** default ALL mandatory arms to a heatmap-only head, with targets centered at the exact transformed coordinates. The native GEFF reference points are integer-valued; residual-to-nearest-native-voxel targets at annotated centers are otherwise identically zero. Do not attach a zero-trained offset head and claim subvoxel localization learning. Use deterministic peak decoding and a fixed local interpolation rule, with integer argmax results also reported. Interpolation is an estimator, not additional measured ground truth.

The single optional N_flow ablation uses meaningful local vectors d(q)=spacing*(p-q), for voxels q assigned unambiguously to annotated p within a compact physical support. Supervise only that support; fractional augmentation and neighboring assigned voxels provide nonzero targets. Verify distributions/gradients and one-to-one assignment. Do not average incompatible cell targets or give nearest-GT vectors to the entire volume. No external mask teacher is required. Coarse cells colliding into one bin are counted explicitly; those losses are not resolved by inventing an average cell.

Native spacing is expected [1.625,0.40625,0.40625] um; verify against metadata. Coordinate transforms must retain resampling origins: striding maps q->s*q, averaging has a center offset. Round once at competition export. Test flips, XY quarter turns, translations, tile origins, coarse/native mapping, anisotropic physical distances and boundaries.

**Masked supervision:** human center supports use Gaussian sigma0.8 um (compact support up to 1.5 sigma); conservative image-only background uses low intensity AND low local contrast and excludes annotated supports, visible peaks, uncertain boundaries and saturation. Unknown regions have exactly zero supervised gradient. Human positives override background/pseudo supports. Define overlap handling explicitly without suppressing adjacent daughters. Normalize human and background terms separately, weights 1.0/0.1. Audit the background heuristic on faint source annotations and sampled source regions; it is not certified background. Do not estimate a PU class prior from annotation-density metadata.

Sample both uniform all-field and positive-biased source crops (initial 50/50 when tiled), balancing clips and track groups. Full-frame mode samples source clips/frames with recorded balancing. Missing annotations never imply an empty image. Track unique source observations/frames and physical valid-core exposure, not only steps. Positive-biased training crops do not authorize GT-routed inference.

**EMA primary:** same source-only R_masked warmup for 500 optimizer updates; EMA decay .99 thereafter, pseudo-positive weight .25 ramped over the next 500 updates. Teacher views and student views see only the source. Require initial confidence .9, image support, inverse-transform agreement <=1.5 um and consistency on a fixed source probe bank across teacher snapshots. Pseudo absence is NOT a negative. Exclude ambiguous duplicates, retain missed human positives, and cap pseudo support per frame/group. Do not require temporal persistence through all frames of a division. Frozen teacher inference uses eval/no_grad; student consistency remains differentiable. An empty pseudo set leaves valid supervised training and must be reported; do not lower thresholds using the target.

Compare N_base/N_ema with identical initialization, source sampling and optimizer schedule; record the additional EMA/view compute. A compute-matched supervised comparison uses no pseudo loss while consuming the same view budget. Masked-only, EMA and missing-as-negative losses have explicit collapse tests (constant fields, counts, plateau peaks, faint-cell suppression). Top-K is a diagnostic, not a way to conceal a collapsed detector.

Before a real fit, run a tiny 2-4 source-crop overfit, source-only gradient/coordinate tests, native input hooks, collision fixtures and deterministic decode/replay checks. Then move to real runs; synthetic tests alone do not satisfy D920.

## 6. D930-D950: fixed experiment order and resource allocation

`study.json` is the authoritative registry. Defaults: AdamW lr3e-4, weight decay1e-4, effective batch4, gradient norm clip1, 200-update warmup then cosine to 1e-6. Seeds 20260914 and 314159. Define updates as optimizer steps, not microbatches. Save initialization/config/sampling hashes. Optimizer, scheduler, scaler, EMA, RNG and sampler state are part of an exact-resume checkpoint.

Pilot estimates allocate a default cap of **48 active study GPU-hours**, including teacher inference, training, failures and detector inference. This is a resource cap, not a promised duration or a calendar deadline. CPU official scoring is logged separately; count GPU idle time blocked by this study's IO as study usage. Do not reset the clock after a restart. Reserve enough budget for mandatory full inference and packaging before starting optional fits. Dry planning uses measured seconds/update and seconds/frame from both directions and worst-case precision/tiling.

**P0 mandatory finals:** N_base and N_ema, both directions, both seeds (8 fits). Register a fresh 8000-update final cosine schedule; never resume a 1500-update cosine screen into it. If pilot predicts the core fits/evaluation will exceed their allocation, choose the largest common supported schedule in [8000,4000,2000] BEFORE fits/outer results. The choice applies to all matched P0/P1 finals. Report undertraining and the chosen horizon; do not claim 2000 updates equals convergence. If even 2000 cannot fit, checkpoint a clearly partial study rather than fake completion.

**P1 mandatory resolution controls:** B64_stride and N_lowpass, both directions, both seeds (8 fits), matched final schedule, sampling and physical field. B64_stride is a clean scratch control of coarse sampling, NOT a faithful retrain of the original temporal joint model. N_lowpass uses area XY4 reduction and explicit native-lattice reconstruction with the SAME anisotropic model as N_base. It isolates retained information more closely than architecture-changing controls.

Core has 16 fits before optional breadth. A 1500-update prefix checkpoint of a final run is useful for learning curves, but is not a separate short-cosine experiment. Complete first-seed P0/P1 pairs, then second seeds, then extra arms. Queue order, allocations and omissions depend on measured resources/source integrity only, never outer performance.

**P2 registered secondary screens:** B64_AA, B128_AA, N_plain, N_dense_diagnostic, N_context; each both directions at primary seed, 1500-update short-cosine schedules. Run in this order while preserving core/evaluation reserves. They are explicitly screening results, not fully converged comparisons against 8000-update models. Compare their common-prefix exposure and compute curves to controls; distinguish schedules. N_dense treats unannotated voxels as lightly negative only as a diagnostic, never primary eligible.

**P3 source-diagnostic probes, capacity permitting:** N_lr1e4, N_sigma1p2, N_flow, each both directions/primary seed with one changed factor. No width24 expansion in this iteration; native detail and reliable supervision take precedence on 24GB-class hardware. No external teachers or new dataset campaigns. These probes are descriptive and cannot replace the fixed primary after outer scoring. At most 32 retained fits total (16 finals + 10 secondary + 6 probes); failed/resumed work remains in the ledger and resource accounting.

Suggested allocation envelope: <=2 GPU-hours pilots/integrity; <=22 P0 finals; <=12 P1 controls; >=8 reserved for inference/integration/export verification; <=4 P2/P3. These are initial partitions to revise once, from source-only timing before the lock. Full model-training and inference feasibility is measured, not asserted. If inference needs more than eight, reduce optional breadth first and then choose the common final horizon at the pre-run lock. Never train every screen and leave no capacity for actual validation.

## 7. D960: full detector evaluation and honest success criteria

Freeze all retained directional/seed checkpoints, decoding, NMS, thresholds and frame-complete predictions BEFORE opening new target scores. Target images are available to inference only, not adaptive fitting. Evaluate the exact 199-clip set with metadata-derived frame ranges (expected 100 each, but read rather than hard-code). Reuse `tools/annotation_selection/metric_adapter.py` for the pinned official matcher; independently test diagnostic match radii. Compare one-to-one physical distances at the correct time; multiple predictions cannot all recover the same GT node.

Fixed initial decoder: physical local-maximum radius1.7 um, greedy physical NMS radius1.7 um, diagnostic peak floor .01 and operational sigmoid threshold .5. These are prespecified starting choices, not calibrated probabilities. Use axiswise three-point logit-parabola interpolation clamped to half a voxel; use zero shift on flat curvature, missing neighbors or boundaries. Rank by score then lexicographic ZYX; perform physical NMS after interpolation. Apply identical rules to scratch/native comparisons, retain the incumbent's original decoder for its operational reference, and do not tune any of them on target outcomes. The registry input shapes describe full-frame grids; native tiling/control crops must cover the same physical field and record their actual tensor sizes.

Report annotated-node recall at 1,2,3,5,7 um, per embryo and pooled; fixed peak-count budgets K=50,100,200,400,800 per frame; actual counts when fewer genuine peaks exist; and the locked operational decoder. Primary Q is mean annotated recall@3um at K=100,200,400. Decode real local maxima with deterministic plateau/tie handling, no artificial points to fill K. Report both float/interpolated and final integer outputs.

Report localization median/p90, signed Z/Y/X errors, misses with a fixed censored-error convention, matched-GT intersections, density/duplicates, close-pair recovery, and faint/deep/border/division-neighborhood strata. Held-out annotation-defined strata belong only in evaluation. Show source and target learning/evaluation namespaces separately.

Unmatched detections are unclassified under sparse GT, not ordinary false positives. Do not calculate conventional AP/precision/F1 or segmentation Dice without independent dense truth. Prediction count penalties from the official graph metric remain legitimate official graph terms; do not confuse them with dense per-cell precision.

Claim native-detail evidence only if N_base improves over matched B64_stride and N_lowpass on both embryos and both seeds at the declared horizons, with resolution/control caveats. Claim an EMA improvement only if N_ema beats N_base in Q for both embryos and both seeds and recall@7 does not drop >0.002 absolute at the same budgets. Always present actual deltas, including ties and mixed outcomes. These pragmatic gates are not a statistical proof or hidden-LB guarantee. Same-seed paired comparisons are informative, but two seeds do not increase the embryo count.

Compare the new detector to the exact exposed/uncertain incumbent in a separately labeled table. An isolated new detector may be scientifically better controlled yet operationally worse. No favorable threshold selected using target outcomes can count as a preregistered result. Avoid tight clip-bootstrap claims of biological generalization with two subjects.

## 8. D970: fixed-linker integration, not a new association study

Use the frozen existing linker/repair configuration for the operational baseline, N_base and N_ema (primary seed, both directional models). At most six operational configurations including unchanged C0/P0. If a second seed is integrated, label it and stay within the cap. Build new node features/geometry at the NEW centers, rebuild candidate edges and recompute association probabilities. Do not nearest-copy old edge scores or reuse stale node-indexed arrays. Preserve integer sampler behavior for the primary fixed-linker comparison; a trilinear sampler is a separate named interface change, not a silent improvement mixed into all arms.

Keep graph thresholds, division penalties, motion repair and selection fixed. Measure official adjusted edge/division components, count effects, valid graph constraints, per-clip completeness and total runtime. A clean detector with an exposed inherited linker is **operational inherited-linker**, not clean full-system OOF. A 0.95 local graph score is not a claim of 0.95 hidden LB. Negative graph results do not invalidate detector measurements and must not erase them.

Do not replace incumbent artifacts automatically or submit. Deliver new candidates separately. A useful scratch detector and a non-promoted tracker is a valid outcome. A missing local linker closes only integration; detector training/evaluation/export remain required.

## 9. D980: export, replay and final artifacts

Package a detector that processes a complete raw Zarr into scored points without upstream weights, old graphs or GT. Run a fresh process on one renamed complete clip per held-out embryo with annotation and prediction-cache paths unavailable and network disabled where the local environment supports it. Record actual read guards and their limits. Compare to the same-recipe saved predictions, including integer outputs. This proves inference independence, not checkpoint training provenance.

Save the full source-isolated two-fold/two-seed checkpoint family locally with SHA-256, manifest, configs and pinned environment receipt. No all-embryo refit is needed for the primary study. After the complete lock and evaluation, an optional all-training-data deployment fit may be documented as a separate future step with no held-out local score; it is not authorized to consume the core budget here.

Required tracked summaries under `results/native-resolution-detector-v9-4090/`:
`final_report.md`, `dashboard.html`, `detection_metrics.csv`, `resolution_regime_comparison.csv`, `learning_curves.csv`, `graph_comparison.csv`, `provenance.json`, `experiment_ledger.json`, `environment.json`, `status.json`. Keep large points, GEFF assignments, images and checkpoints in ignored durable output storage. CSVs identify recipe, seed, source/target, horizon, compute, clip/frame counts and status. Missing arms are explicit rows, not absent evidence. The HTML shows genuine source/held-out crop examples, recall/count curves, localization, supervision masks, per-embryo deltas and runtime—never fabricated microscopy or successful numbers for blocked arms.

Write `handover/native-resolution-detector-v9-4090/CONTINUATION.md` with exact last state, usable checkpoint locations/hashes, what improved/failed, commands to resume, and remaining resource budget. Commit/push only new code/tests/configs and sanitized summaries on this branch with an explicit allowlist. Do not merge branches, rewrite inherited results, publish weights to Git or upload competition data.

## 10. Runtime CLI contract to implement locally

The branch ships planning/preflight helpers, NOT the following training commands yet. Implement them with argparse, proper help, path/config checks and resumable state:

```bash
PYTHONPATH=tools PYTHONNOUSERSITE=1 python -m native_detector_v9 audit --study handover/native-resolution-detector-v9-4090/study.json
PYTHONPATH=tools PYTHONNOUSERSITE=1 python -m native_detector_v9 pilot --study handover/native-resolution-detector-v9-4090/study.json
PYTHONPATH=tools PYTHONNOUSERSITE=1 python -m native_detector_v9 run --study handover/native-resolution-detector-v9-4090/study.json --resume
PYTHONPATH=tools PYTHONNOUSERSITE=1 python -m native_detector_v9 infer --checkpoint CHECKPOINT --images RAW_CLIP.zarr --output POINTS_FILE
PYTHONPATH=tools PYTHONNOUSERSITE=1 python -m native_detector_v9 evaluate --frozen-manifest FROZEN_MANIFEST
PYTHONPATH=tools PYTHONNOUSERSITE=1 python -m native_detector_v9 report --run-root RUN_ROOT
```

`run` executes the locked queue serially and writes atomic status after each stage/run. Expose `--dry-run` for projected jobs/resources, `--only-stage`, `--resume`, `--data-root`, `--output-root`, and explicit no-outer-score behavior before freezing. Avoid hidden reliance on shell cwd or PYTHONPATH beyond the documented module entry. Make help work from a fresh checkout and test execution from a foreign working directory via the correct entry point. Do not require users to patch activation blocks.

## 11. Required tests beyond the included CPU helpers

Geometry: actual input first-convolution tensor; every native XY sample can influence a nearby feature; coordinate/spacing/pooling origins; anisotropic physical NMS; tile seams and coverage; true zero-detection frames; all-field inference without proposals; integer export bounds.

Loss: unknown gradients exactly zero; empty support finite; human overrides pseudo/background; nearby daughters not silently merged; Gaussian targets on exact transformed centers; offset ablation has nonzero supported vector targets/gradients; no vectors learned in unknown regions; pseudo absence never creates negatives; collapse visible before top-K.

Isolation/reproducibility: recursive teacher/cache ancestry; forbidden target read during train/calibration; no implicit old checkpoint load; exact RNG/optimizer/EMA resume; deterministic same-recipe inference; both-fold freeze before evaluator; exact expected clip AND frame sets; one-to-one assignment fixtures; hashes agree with exported checkpoints.

Resource: measured complete backward plus EMA and checkpoint peaks; bounded total cache/process RSS; sufficient durable disk before save; atomic checkpoint survives simulated interruption; measured CUDA/host limits; no parallel GPU trainers.

The supplied stdlib tests validate only registry/preflight/manifest helper behavior. Their success is not evidence of native model gradients, GPU fit, competition compliance or learning.

## 12. Failure and continuation policy

Do not terminate because a native short screen scores worse, an optional package is missing, the synthetic helper tests pass, or the initial full-volume backward OOMs. Follow native tiling fallbacks and run core learning. No magic local-score hard stop gates all training. A technical fault may receive up to three logged repairs; afterward block that dependent arm and proceed with independent feasible work. A discovered leakage/coordinate bug invalidates affected results and requires repaired re-execution, not a footnote.

Stop cleanly for actual resource exhaustion, unavailable authorized raw data/runtime with no viable local recovery, unrecoverable integrity failure, or completion. Save durable last state and honest partial metrics. If the whole budget cannot cover both-direction completeness, label it incomplete; never substitute source training accuracy for missing outer results. Core success and failure both deserve a usable checkpoint/inference deliverable where training was feasible.

## 13. Evidence boundaries

This handover was authored through repository inspection and CPU-only helper tests. No image training, GPU memory pilot, official score, Kaggle runtime validation or new LB result was performed remotely. Hardware capabilities and budgets above are measured locally before training; all unobserved training outcomes remain null. V9's hypotheses can fail.

## 14. Pinned repository evidence and official references

Read these inherited paths at base `e654c50c7543b3486556e2a1cd472f669f07a625` (most experiment source derives from main `fb5521629eb41c8c485b291a5bcf344944c113ae`):
- `AGENTS.md`, `.agents/skills/competition-{rules,data,overview,forum,status}/SKILL.md`, `docs/competition.md`, `docs/notebooks.md`: workflow, official-reference routing, data/runtime contracts and installed artifacts.
- `tools/image_native_tracking_v5/native_adapter.py`, `results/image-native-tracking-v5/native_architecture.json`: actual coarse input, weight-loading caveat, model family and primary-only ensemble limitation.
- `results/annotation-selection-v1/report.md`: exposure/overlap evidence; `results/multidata-training-v4/final_report.md`: earlier proposal refinement and source-calibration caveats.
- `handover/native-resolution-detector-v8/{PLAN.md,study.json,reference.py,test_reference.py}`: inherited rationale/helpers, not a second execution instruction. V9 makes no new claims about v8 training; it was planned only at the branch point.
- `tools/annotation_selection/metric_adapter.py`: maintained official-metric adapter to inspect and pin before reuse.

Official pages: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules and competition overview/data pages. At authoring, search-index excerpts supported offline/12-hour notebook requirements and external-data wording, but the complete live rules page did not yield readable text. Local Codex must use/refresh the official mirror and record exact evidence; this is not a complete legal/rule certification. Primary fits use only supplied source images/annotations and random weights.

The approved native-detector review in the preceding conversation informed v9's integer-offset correction and validation boundaries. No inaccessible conversation detail is needed to execute this file.
