# Strong tracker v2 — measured results

Decision: **significant_local_gain**. The selected exploratory pipeline scores **0.911774 → 0.934206 (+0.022432)** on all 199 clips. It bypasses the original motion relinker, retains the learned neural associations, and runs the remaining Harmonic Fusion repairs unchanged. The exported winner enforces image bounds; the separately rescored bounds correction changes the score by zero.

Both embryo directions improve: **44b6 0.912521 → 0.931612 (+0.019091)**; **6bba 0.911597 → 0.934525 (+0.022928)**. This exceeds the proposed +0.02 pooled local target. Selection followed the failure census and phase ablations, so the winner is exploratory and selected after inspecting results.

The public checkpoints were trained or selected using supplied embryos, and both embryos had already been examined. Source-only repair/selector training does not remove upstream contamination. These are local operational results, not clean OOF estimates or a hidden-test forecast. The historical 0.946 public leaderboard score is a different evaluation population; no local delta is added to it.

## V200–V210: exact baseline, real stages, failure census

The fresh official baseline reproduces 122,201 edge TP, 6,885 FP, 6,682 FN, and 23 division TP / 97 FP / 128 FN. All 199 final input hashes and the original two replay pilots were verified. Instrumentation follows six actual coarse phases, rather than treating 27 helper functions as 27 transformations.

All 199 repaired graphs reproduce the sealed baseline after a documented serializer compatibility adjustment at two floating-point half-integer ties. Native replay floats and the two failed parity receipts are retained. All edges and IDs already agreed. Native rounding was independently scored and gives the same combined baseline score. The six original out-of-bounds points are retained for baseline identity; the separate bounds-only experiment also leaves the score unchanged.

The raw neural graph scores **0.914903**. Motion relinking lowers this to **0.892978**; gap closing gives **0.894511**, safe divisions **0.903685**, pruning **0.903789**, and smoothing **0.911774**. Cumulative changes are descriptive, not proof that a stage should be removed. Complete bypass experiments establish that skipping safe divisions scores **0.902672**, skipping smoothing **0.903789**, and skipping motion relinking **0.934206**.

The census covers all **151 GT division observations, 97 division FP, 6,682 edge FN and 6,885 edge FP**, using fresh official local division assignments at every phase. Initially, 99 missed divisions had both daughter lineages locally matched but no surviving fork; 21 lacked daughter candidates, four involved assignment competition, three lacked parent evidence, and one had incorrect local topology. The final census adds explicit stage attribution and flags for correct forks lost or relocalized by later repair. There are **3,327** FN edges in the available alternative pool; **1,437** have exported native pre-ILP scores. These counts are diagnostics, not guaranteed recoverable gains.

Stable raw node IDs map exactly to pre-ILP coordinates, one-to-one. Only **37,822** final nodes are inserted relative to the raw graph. The old nearest-center confidence mapping marked 383,339 nodes missing, including many relocated original detections; v2 carries original confidence and displacement explicitly instead of treating missing confidence as false-cell evidence.

Three impossible-inference interventions were actually rescored: fixed-node GT-guided legal rewiring **1.040780**, GT-centered candidate injection plus rewiring **1.102278**, and GT-selected pruning **1.046218**. Their search is heuristic and incomplete, not a rigorous global upper bound. Scores above one are possible under the official count adjustment. Oracle code, graphs and labels are isolated under evaluation paths, and inference blocks those paths.

## V220–V230: constrained divisions and associations

The bounded fork pool covers **102 of 151** annotated division observations and supplies **204** positive hypotheses. Source 44b6 contributes 39 hypotheses from 18 covered event observations; source 6bba contributes 165 from 84. All available positive hypotheses are used. Uncovered events remain explicit coverage failures, not silent negative labels. Contradictory negatives are sampled by predicted tracklet, with inverse inclusion weights; incomplete or time-shifted division neighborhoods are masked.

The native edge learner uses **125,528** positive candidate transitions. Fork features include native probabilities/margins, physical distances, barycenter motion, image intensity, daughter separation across neighboring frames, and boundary masks. Continuation/fork choices are solved jointly over competing source/target components using a bounded binary program; association assignments keep confident external links and existing forks fixed. Every output is checked for merges, duplicate edges, degree limits, integer coordinates and consecutive frames.

The fixed temporal division arm at p=0.05 recovers 35 divisions but raises FP to 220 and scores **0.911501**. This negative result is retained. The strongest measured association setting scores **0.922810 (+0.011036)** with gains in both directions. It recovers 605 previously missed TP edges and loses 31 baseline TP edges. Its FP count falls from 6,885 to 5,941. The more conservative prespecified E setting scores **0.918673**; prespecified D+E scores **0.918469**.

The selected complete motion-bypass pipeline has **123,023 edge TP, 4,930 FP, 5,860 FN**, and **29 division TP / 92 FP / 122 FN**. Its main improvement is association accuracy. Division recovery remains a substantial unresolved opportunity.

## V240: native selectors and observed deletion risk

Native logistic and seven-leaf boosted models use every source matched-positive node: **19,980** in 44b6 and **110,979** in 6bba. Geometry-free quality/temporal features are primary; normalized geometry is an ablation. Negative sampling is stratified within tracklets by origin, depth and density, with recorded inclusion weights. Metrics are evaluated on the full natural node population. Labels mean matched sparse annotation, not biological cell truth.

Node, whole-tracklet, five-frame segment and predicted-fork-protected actions are tested at requested retention 1, .995, .99, .98, .95, .9, .8, .7 and .5. Realized budgets, TP survival, new TP, counts and divisions are recomputed exactly. The best native selection setting in hindsight is **F_hgb_geometry_tracklet_r0.98**, scoring **0.912791 (+0.001017)**. The historical DoG transfer control remains **0.873322**, consistent with the preserved v1 failure.

The learned risk model uses all **164,765** source action groups, including **6,729** groups incident to matched TP edges, plus **1,577 actual group-deletion rescoring experiments**. Actual run-level loss includes rematching, division changes, FP changes and count weights. Rematching changes the naive TP-loss estimate in **227** rescored actions. The primary risk threshold can abstain and caps removal at 10%; it does not force deletion in every clip. Its score is **0.899132 (-0.012642)**.

Quality profiles and explicit low-response/missing-confidence summaries are retained. There is no dense biological true/false-cell audit, so neither native selection nor the risk target establishes latent annotator preference independently of ordinary detection quality.

## V250: full temporal image study

Six **1,421,057-parameter** five-frame triplanar fits were completed on the RTX 4090: 10%, 30% and 100% of positive tracklet groups in each source embryo, **5,000 optimization steps per fit (30,000 total)**. Both full fits used every positive observation and every positive source tracklet group: 19,980 / 981 groups and 110,979 / 6,344 groups. The encoder combines 13 µm local and 26 µm context views; panel geometry and temporal boundary masks are explicit. Shared intensity gain is the only augmentation; no inconsistent panel flips or rolled borders are used.

Learning curves record actual steps, unique positives/groups, weighted BCE, memory and synchronized time. All six final checkpoints were frozen before comparative image scoring. Source-prior correction is used for probabilities, with no target-label calibration. Classification results for all three training fractions use the full opposite-embryo population. The best full-image selector in hindsight is **F_temporal_tracklet_r0.995**, scoring **0.911598 (-0.000176)**. It does not beat the selected graph pipeline.

The six fits consumed **0.098134 GPU-synchronized training hours**; actual steps and positive coverage establish the fit size, not elapsed time alone. FOCUS was not downloaded or run. No authorized local FOCUS checkpoint was present, and the census showed greater immediate headroom from existing nodes and links. This image arm tests selection risk; it cannot restore absent daughter candidates.

## V260: combinations and validation limits

There are **104 complete variants and 20,696 per-sample score rows**. Identity, D-only, E-only, F-only, D+E and D+E+F are all evaluated. The prespecified D+E+learned-risk combination scores **0.906140 (-0.005634)**. Additional E+F combinations are explicitly exploratory. Filters calibrated on the original graph are labelled as transfer arms when applied after repair; their new action groups and fork protection are recomputed from predicted structure.

Every graph variant uses fresh official matching and division evaluation. Every aggregation requires the full expected sample set and is checked against the pinned official run-level weighting. FP changes alter weights, so neither average clip scores nor count-only approximations are used as results. The selected 199 graph files also reproduce byte-for-byte in annotation-unavailable export, with strict image bounds and graph validity.

All **219** fresh image-patch checks for the 73 known crop translations pass, with no inconsistent cycles. Transforms cover 65 clips of 44b6 and none of 6bba. They identify 149 provisional event groups among 151 observations, but unknown overlaps remain. Independent purged source blocks cannot be certified, so fixed limited settings replace inner tuning; no bootstrap confidence intervals are fabricated. A clean upstream source-trained replication remains unavailable. The completed v1 clean classical gain (+0.002791) and failed public transfer remain historical evidence, unchanged.

## V270: resources, artifacts and reproduction

The original GPU study interpreter, pinned metric revision `075fc5f5a52d11077f9dc2b074644618f26939e2`, and documented PyTorch metadata exception are preserved. No package stack or driver was upgraded. CUDA library-path initialization follows `scripts/root_remote_env.sh`. Jobs run in tmux with per-stage logs/checkpoints, at most about 40 active CPU threads, and GPU 0.

Recorded peak aggregate process RSS is **21.46 GiB** and peak GPU use is **3.55 GiB**. New v2 disk use at report generation is **13.74 GiB**, within the 60 GiB cap. Resource logs remain in `resource_samples.csv`; training checkpoints and exact source/config/model hashes are retained.

The complete local artifact root is `/kaggle/working/cell-tracking/strong-tracker-v2/`. Start with `dashboard.html`, `score_rows.csv`, `operating_points.csv`, `stage_scores.csv`, both failure censuses, `oracle_diagnostics.json`, `native_classifier_metrics.csv`, `learning_curves.csv`, `winning_config.json`, `selected_prediction_lock.json`, `validation_receipt.json` and `artifact_manifest.json`. Detailed event coordinates, predictions, image stores, weights and matching files remain local. Reproduction commands are in `docs/strong-tracker-v2.md` and `reproduce.sh`.

Before this instance is destroyed, copy out the **entire v2 output root**, `work/strong-tracker-v2/`, and the new Git commit. Reproduction also needs the preserved v1 store, pinned evaluator and original competition inputs already mirrored on this host. The instance disk is not a persistent attached volume. No Kaggle submission, forum post, paid API, external model download or remote publication was performed.
