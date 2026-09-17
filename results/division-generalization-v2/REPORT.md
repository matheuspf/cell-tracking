P0 retained

Execution: **incomplete_resumable**. Requested replicated score ≥0.95: **not established**.

| Arm / seed | Scope | Score | Δ P0 | Δ C4_m6 | Edge TP / FP / FN | Division TP / FP / FN | Edge raw / adjusted | Selected / matched nodes | Node hash | Status |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| P0 | pooled | 0.934864986413 | +0.000000000000 | -0.000313383844 | 123172 / 4996 / 5711 | 29 / 92 / 122 | 0.920024799 / 0.922930830 | 4108943 / 130836 | 16b1adc6abdd | verified |
| C4_m6 | pooled | 0.935178370257 | +0.000313383844 | +0.000000000000 | 123133 / 4968 / 5750 | 30 / 92 / 121 | 0.919925888 / 0.922832691 | 4108943 / 130836 | 16b1adc6abdd | verified |
| P0 | 44b6 | 0.931727256413 | +0.000000000000 | -0.001977297067 | 18869 / 1087 / 957 | 7 / 22 / 19 | 0.902261751 / 0.917143923 | 1972750 / 19968 | b93a5b8da8ce | verified |
| P0 | 6bba | 0.935284191137 | +0.000000000000 | +0.000096471603 | 104303 / 3909 / 4754 | 22 / 70 / 103 | 0.923313209 / 0.924002140 | 2136193 / 110868 | e04e9b22cbe2 | verified |
| C4_m6 | 44b6 | 0.933704553481 | +0.001977297067 | +0.000000000000 | 18866 / 1086 / 960 | 8 / 22 / 18 | 0.902161438 / 0.917037887 | 1972750 / 19968 | b93a5b8da8ce | verified |
| C4_m6 | 6bba | 0.935187719534 | -0.000096471603 | +0.000000000000 | 104267 / 3882 / 4790 | 22 / 70 / 103 | 0.923215187 / 0.923905668 | 2136193 / 110868 | e04e9b22cbe2 | verified |
| G30 | pooled + both embryos | null | null | null | null | null | null | null | null | source qualification pending |
| J_uniform | pooled + both embryos | null | null | null | null | null | null | null | null | source qualification pending |
| J_mined | pooled + both embryos | null | null | null | null | null | null | null | null | source qualification pending |

6/10 directional fits have completed their required updates. Each main arm requires 4,096 joint updates; uniform and mined arms share their first 2,048 updates per direction/seed. The shared prefix is counted once in compute and does not make independent experiments.

The fixed diagnostic panels contain 32 hashed groups each. The 6bba held panel has no positive-utility anchors, so its loss measures supported rejection and cannot establish division recovery. The 44b6 held panel has two positive-utility anchors. Full source graph screens govern qualification. The panels and extension rule remain as originally locked. [Panel composition](diagnostic_panel_composition.json).

Completed checkpoint calibrations: 9 direct held-source fits, 1 grouped small-head fallbacks, and 0 unestablished. The fallback fits three heads for 4,096 updates each on frozen features, holding complete overlap groups out. Those updates do not count toward the joint-training floor. Only the heads are out of fit: the frozen source encoder retains its original label exposure. [Calibration audit](calibration_audit.json).

The new module keeps native evidence inside learned relative complete-action scores. It uses fixed-seed probability sampling, separate positive exposure, masked unknown alternatives, negative-only groups, complete lost-link utility, and one two-scale raw scene shared across candidate pairs. The 490,804-parameter encoder/head uses ordered attention.

Both inherited baselines retain all 4,108,943 observations. All graph hashes and baseline metric receipts were verified, with fresh ordinary/crowded replays in both embryos and independent official aggregation. P0 and C4_m6 remain unchanged.

This is exploratory source-only direct fitting on exposed P0 observations. Reused whole-clip source splits union exact frame overlaps, but missing global acquisition offsets prevent independent inner-validation certification. Neither a local score nor a new raw-scene encoder establishes clean OOF or hidden leaderboard performance.

[Training receipts](training_receipts.json), [fixed source curves](training_curves.csv), [source curve plot](training_event_curves.png), [complete source screens](source_scores.csv), [per-embryo scores](per_embryo_scores.csv), [sampling audit](sampling_audit.json), [validation](validation.json), [resources](resource.json).

Accounted exclusive GPU leases: 4.018 h; waits: 2064.0 s. This includes 10.066 s estimated for an interrupted lease; host downtime is excluded. Full lease intervals include preprocessing. [Operation wall timings](runtime_breakdown.json) separate loading, transfers, augmentation, encoder/head, backward, optimizer and checkpoint work; pure CUDA kernel time is not measured.

An early image implementation sampled CNN feature maps at shifted coordinates. Those image fits were archived as implementation-invalid and restarted from scratch; their updates do not count toward training adequacy, and their GPU leases remain in the cost ledger. The cached-feature controls are unaffected, verified by exact output parity. [Corrected image proof](corrected_image_validation.json), [profile provenance](profile_provenance.json), [control compatibility](model_code_compatibility.json).

No production promotion, merge, Kaggle submission, weight publication or leaderboard claim has been made.
