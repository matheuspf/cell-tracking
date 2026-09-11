# V6 measured execution results

**Learned segmentation and Ultrack integration are blocked by missing local runtimes.** The independent C0 and P0 comparisons completed on all 199 clips. No learned masks were generated; B0, M0, U0 and U1 have no competition scores. Classical watershed was not used as a replacement.

C0 remains intact at **0.934802374260586**. P0 scores **0.934864986413134**, a pooled gain of **+0.000062612152548**. The small point-control improvement is retained after both-embryo, deterministic-repeat and fresh-image gates passed. **The 0.95 target is not attained.** This gain does not measure bbox, mask or Ultrack benefit.

## Complete measurements

All rows use fresh pinned official node assignment, edge/division scoring and exact run-level aggregation. Scores are not means of per-clip scores. Both arms retain 4,108,943 C0 observations.

**C0**

- 44b6: 71 clips; score 0.931664468721842; edge TP/FP/FN 18,866/1,085/960; division TP/FP/FN 7/22/19; 1,972,750 nodes.
- 6bba: 128 clips; score 0.935221784097327; edge TP/FP/FN 104,269/3,880/4,788; division TP/FP/FN 22/70/103; 2,136,193 nodes.
- pooled: 199 clips; score 0.934802374260586; edge TP/FP/FN 123,135/4,965/5,748; division TP/FP/FN 29/92/122; 4,108,943 nodes.

**P0**

- 44b6: 71 clips; score 0.931727256413358; edge TP/FP/FN 18,869/1,087/957; division TP/FP/FN 7/22/19; 1,972,750 nodes.
- 6bba: 128 clips; score 0.935284191136704; edge TP/FP/FN 104,303/3,909/4,754; division TP/FP/FN 22/70/103; 2,136,193 nodes.
- pooled: 199 clips; score 0.934864986413134; edge TP/FP/FN 123,172/4,996/5,711; division TP/FP/FN 29/92/122; 4,108,943 nodes.

## What ran

- S600: all 199 C0 graph hashes, cached scores, image axes/spacing and full fresh native evidence were checked. Primary, secondary, native source and frozen teacher checkpoints passed their manifest hashes. The latest v5 local snapshot had 12/18 scored configurations, six/eight native fits, and selected C0. No v5 process was active at preflight; the supervisor JSON was stale. No prior job was stopped or restarted.
- P0: the full 38-field incumbent evidence and identical candidate bank feed one regularized logistic residual with a fixed native-logit offset (L2=1, at most 250 L-BFGS iterations). Each source embryo trains its opposite direction. Only supported incoming-parent contradictions are negatives. An unrecorded second daughter remains unknown. The unchanged v3 association decoder uses margin 3, a 2% edit cap and frozen existing forks. This is a new point control applied to C0, not the weakened v5 primary-only control.
- Both deterministic fits reproduced their coefficients exactly; all 199 repeated P0 graphs matched. The fitter and decoder have no random sampling, so a nominal second seed would not be a distinct experiment.
- The source44 optimizer converged after 180 iterations. The source6 optimizer used its full 250-iteration budget and reported an iteration-limit stop (not convergence). The fixed-budget coefficients were finite and exactly reproducible. The budget was not increased after target results.
- Two complete fresh C0 clips passed exact graph/CSV parity in 220.09 seconds. Both used unfamiliar image names and explicit source-model arguments. The early Python audit guard denied real GT, old graph/observation reads and external DNS; successful inference audits recorded no blocked access attempts.
- Fresh P0 checks rebuild the full native point features from those new image-derived arrays, then check exact candidate/features, graph and CSV parity. The first attempt exposed an absent packaged input-hash manifest in the optional heatmap-sharing helper. The v6 caller now uses the original independent regeneration path. The failure log is preserved locally.
- 32 actual image frames were read across four image-density-selected clips, two per embryo, eight consecutive frames each. Orthogonal raw-image/C0-center views were inspected. Nuclear fluorescence supports nuclei segmentation. These views contain no learned masks and cannot validate mask merges, duplicate instances or segmentation tile seams.
- The C0/P0 fit/decode/score stage took 196.55 wall seconds with three bounded CPU workers; deterministic repeat checks took 94.26 seconds. Source fits used no GPU training.
- The delivered `infer --arm P0` CLI independently repeated both full fresh clips in 330.36 seconds. Both emitted C0 and P0 CSVs matched the complete batch graphs; startup guards passed. The blocked M0 CLI check exited explicitly before baseline execution.

## Blocked stages and implementation limits

- FOCUS: /root/FOCUS-3D has checkpoints but no Python source or headless infer_volume backend.
- FOCUS runtime: detectron2 package absent in the annotation runtime.
- Cellpose fallback: cellpose package and local volumetric checkpoint absent.
- Ultrack: ultrack package/source absent; installed pyscipopt alone is not Ultrack.
- Persistent disk has 6.838 GiB free, below 8 GiB floor; no persistent mask/database allocation.

The three FOCUS weight SHA256 values match `/root/FOCUS-3D/SHA256SUMS`; nuclei weights are the appropriate compartment. Weights alone do not supply `infer_volume`, its preprocessing or its runtime. No downloads, license acceptance, external data collection, fine-tuning or other model survey was performed.

The maintained adapters implement native ZYX shape/origin validation, anisotropic size conversion, compressed bbox-local occupancy with hashes, physical shape/volume, inside-mask intensity and nullable confidence. Strict one-to-one containment flags shared masks and leaves unmatched C0 observations intact. Bbox and occupancy features are distinct. Daughter-union/persistent-separation features, hierarchy exclusion, exact mask survival, explicit Ultrack observation ancestry and full-native geometry provenance have executable contracts.

**Those contracts are not completed tool integration.** FOCUS signature binding, Cellpose volumetric execution, Ultrack conversion/hierarchy/solver signatures, hierarchy-mask extraction, native link writing and mask-division terms in the graph objective remain unvalidated/unexecuted without their runtimes and real masks. No physical recipe was selected. B0/M0 fits, U0/U1, learned-mask timing/storage pilot, sparse containment checks and fresh image-to-mask-to-graph tests were not run. `infer --arm B0/M0/U0/U1` fails explicitly before C0 execution. There is no silently filled tool score or newly selected segmentation package.

## Resources, checks and preservation

The initial persistent filesystem had 6.838 GiB free, already below the plan's 8 GiB floor. No old artifacts were deleted. Only small v6 code/models/metadata were persisted; transient control graphs, fresh exports and images used `/dev/shm/cell-tracking-segmentation-v6`, with a separate 8 GiB admission floor. RAM-backed artifacts are local but do not survive a reboot. Persistent mask/database allocation stayed blocked.

One 4090 inference worker ran at a time. C0 fresh process-tree RSS and total GPU peaks were sampled against 24/20 GiB limits. CPU control peak RSS was not sampled, and the measured wall runtimes must not be described as measured GPU-kernel hours. The 24 GPU-hour cap was not approached. No new neural-network training was run.

The 38 inherited NumPy region-contract tests and 13 new tests passed. They cover external-network denial, half-open boxes, spacing, empty frames, label permutation, packed occupancy, ownership, duplicate/hierarchy conflicts, sparse unknowns, daughter union, legal forks/gaps and rejection of unrelated native scores. Synthetic seam/conflict fixtures are distinct from unrun real segmenter seam checks.

Code and the small offline dashboard reuse the repository scorer, decoder, native predictor, CSV routines and reporting/browser conventions. The score CSV includes explicit blocked rows with blank score/count fields. Prior source/results and the uncommitted plan edit are preserved. Weights, raw images, masks, per-clip labels, model coefficients and credentials are outside Git. No merge or Kaggle submission occurred.

Both embryos have been reused and upstream checkpoints retain prior exposure. These are exploratory local comparisons, not independent biological validation or evidence of a hidden-leaderboard gain.

Measured fresh/CLI peak process-tree RSS: 2.620 GiB; peak total GPU memory: 1.492 GiB.
