# HOCT reassessment — 14 September 2026

**Best standalone HOCT result: 0.757105 on six complete clips.**

Best of the six standalone HOCT configurations: Cellpose + HOCT general_v1, no divisions, 0.75710496. The matched ultrack no-division control scores 0.82987357; v3 scores 0.97178168.

Changing only legacy model units changes the full pilot score by -0.00044592. Keeping the old calibration and baseline-dependent decoder does not produce a gain on this panel.

The standalone experiment uses actual unchanged Cellpose masks and HOCT's parent-versus-orphan probabilities. It uses neither baseline links nor native detector features. Two frozen checkpoints are followed by one fixed source-only residual-head recipe, trained on the other embryo.

The old adapted HOCT graph (H_probe_J) shares 98.66% of predicted edges with v3; its decoder explicitly protects 1,033,995 baseline edges across 199 clips. Its old score is not an independent tracking baseline.

Only six reused public clips are measured here. The incumbent has known training overlap; exact HOCT and Cellpose pretraining lists remain unresolved. No result establishes performance on unseen embryos.

## Full competition metric

Each value uses all nodes and edges in the same six 100-frame clips. These are not 199-clip or leaderboard scores.

- **v3 reference (training exposed):** 0.97178168; edge TP/FP/FN 3923/91/103; division TP/FP/FN 1/3/4; 84,831 nodes.
- **Previous HOCT + baseline structure:** 0.96771706; edge TP/FP/FN 3913/98/113; division TP/FP/FN 1/3/4; 84,831 nodes.
- **HOCT physical-unit control + baseline:** 0.96795925; edge TP/FP/FN 3914/98/112; division TP/FP/FN 1/3/4; 84,831 nodes.
- **HOCT voxel-unit correction + baseline:** 0.96751333; edge TP/FP/FN 3915/101/111; division TP/FP/FN 1/3/4; 84,831 nodes.
- **Cellpose + ultrack, no divisions:** 0.82987357; edge TP/FP/FN 3645/339/381; division TP/FP/FN 0/0/5; 96,405 nodes.
- **Cellpose + HOCT general_v1:** 0.75124007; edge TP/FP/FN 3602/657/424; division TP/FP/FN 2/189/3; 111,082 nodes.
- **Cellpose + HOCT general_v1, no divisions:** 0.75710496; edge TP/FP/FN 3544/540/482; division TP/FP/FN 0/0/5; 111,082 nodes.
- **Cellpose + HOCT ctc_v0:** 0.72692542; edge TP/FP/FN 3566/769/460; division TP/FP/FN 3/214/2; 111,082 nodes.
- **Cellpose + HOCT ctc_v0, no divisions:** 0.73802300; edge TP/FP/FN 3512/617/514; division TP/FP/FN 0/0/5; 111,082 nodes.
- **Cellpose + HOCT, source-only adaptation:** 0.72187748; edge TP/FP/FN 3590/831/436; division TP/FP/FN 3/333/2; 111,082 nodes.
- **Cellpose + HOCT, source-only adaptation, no divisions:** 0.74425025; edge TP/FP/FN 3521/589/505; division TP/FP/FN 0/0/5; 111,082 nodes.

## Source-only adaptation

The frozen general_v1 backbone supplies 288-dimensional edge embeddings. Mean features across the same overlapping windows feed one strongly regularized convex residual on log parent probabilities; the orphan alternative stays in the normalization. The recipe uses L2=0.1, source-only standardization and at most 300 L-BFGS iterations. No baseline features or edge membership are used.

Each direction fits only its own three source clips: 1,072 supported incoming-parent targets for 44b6 and 2,785 for 6bba. Competing incoming parents are negatives; unlabeled targets, absent candidate parents and unrecorded daughters remain unknown. Both models are frozen before adapted target graphs are produced. Each fit has a file-access guard rejecting the opposite embryo. This is one exploratory recipe after the frozen-checkpoint screen, not untouched validation.

Source-only adaptation reaches 0.74425025 with divisions disabled, below the frozen checkpoint in both transfer directions: -0.02163037 on target 44b6 and -0.00957489 on target 6bba. Lower source training loss did not transfer to a higher competition score. This recipe is not promoted.

## Remaining errors

Of 482 missed true edges, 313 had the correct candidate available but unselected, 167 lacked a matched detection endpoint, and 2 were outside the candidate bank. Of 540 evaluable false edges, 537 touch an unmatched prediction. Sparse annotation does not establish that every such prediction is a false cell.

The 0.07276860 score gap to ultrack consists of 0.05887984 in pooled edge Jaccard and 0.01388876 in the aggregation's node-count adjustment; both no-division arms have zero division reward. This is an arithmetic decomposition of the scorer, not a causal detector-versus-tracker ablation. HOCT keeps 111,082 observations versus ultrack's 96,405.

With divisions enabled, general_v1 recovers two of five annotated divisions and produces 189 evaluable false divisions. The source-only residual recovers three but produces 333 false divisions. Parent-choice supervision alone has not supplied the evidence needed to distinguish mitosis or a true track birth from a competing continuation.

The evidence points to observation selection and calibrated parent-versus-birth decisions as the next HOCT targets. Simply expanding the distance gate is unlikely to repair much here: only two missed true edges fall outside the candidate bank. This round does not rule out HOCT with different detections, supported birth supervision or a different observation-selection objective.

## Validation and interpretation

All 22 packaged HOCT source files match the pinned upstream Git blobs. Real-mask features agree with the upstream extractor on 273 regions, with maximum absolute difference 7.49e-6. Eight unit contracts and six actual upstream SCIP comparisons pass. CPU/CUDA FP32 logit agreement was checked on 16 real contextual batches across two models and two clips; this is tolerance agreement, not bit identity. Both checkpoints produce identical CPU/GPU edge graphs on one complete 100-frame clip, with and without divisions.

The standalone decoder exactly solves the default HOCT adjacent-frame, fixed-observation objective. This is possible because node cost -10 makes every observation favorable. All raw Cellpose detections therefore remain in the graph; ultrack uses a different observation-selection stage. A shared GPU queue limits HOCT leases to ten elapsed seconds, checked between batches, while other experiments continue.

The unit control changes model positions and equivalent diameter by 1/1.625 and inertia by 1/1.625². Its output coordinates, candidates, image/intensity features, windows, calibration and baseline prior stay fixed. This control does not test recalibration, independent detection or full upstream normalization.

New graphs were frozen before fresh official evaluation. The scorer matches native integer centers within 7 µm and includes the node-count adjustment and division component. Both embryos are reported separately in the comparison artifact. Unknown annotations are not treated as biological negatives.

## Artifacts

- [Interactive comparison](/home/mpf/.cursor/projects/home-mpf-code-kaggle-cell-tracking/canvases/hoct-reassessment-20260914.canvas.tsx)
- [Comparison and per-embryo counts](../results/hoct-reassessment-20260914/comparison.json)
- [Error attribution](../results/hoct-reassessment-20260914/diagnostics.json)
- [Reproduction](../tools/hoct_reassessment/README.md)
- [Pinned official HOCT source](https://github.com/royerlab/hoct/tree/2ccc5040823bc944ab67790abd1f56eea7cd4f05)

No baseline was replaced and no submission was made.
