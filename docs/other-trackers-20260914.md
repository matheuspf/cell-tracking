# OrganoidTracker2 and CELLECT — 14 September 2026

Cellpose + OrganoidTracker2 scores 0.83126292, a +0.00138935 difference from the ultrack control.

Cellpose + CELLECT association scores 0.72738658, a -0.10248698 difference from the ultrack control.

All comparisons use the same six complete 100-frame public clips. Each new family has a division-enabled arm and a no-division control; neither is fitted on Biohub labels.

OrganoidTracker2 uses its released image-patch link and division probabilities with its pruned graph costs, solved exactly by HiGHS. It can omit Cellpose observations. Organoid-specific track cleanup and depth cutoffs are excluded.

CELLECT's own image backbone supplies features at Cellpose centers. Its released between-frame matcher and gates are tested with those external observations. This is not an end-to-end run of CELLECT's native detector and grouping pipeline.

These are reused development clips, not unseen-embryo validation. The v3 reference has known training overlap. Exact training lists for the CELLECT checkpoint are unresolved; OrganoidTracker's release describes mouse intestinal organoid training.

OrganoidTracker2's pooled gain is driven by division detection: its adjusted edge score is 0.79792959, below ultrack's 0.82987357, but its division reward adds 0.03333333. It improves on embryo 44b6 and regresses on 6bba. The small pooled gain is not a reliable new-embryo ranking.

## Full competition metric

Scores are exactly aggregated by the official metric, including node-count adjustment and division reward. They are not averages of per-clip scores.

- **v3 reference — training exposed:** 0.97178168; edges TP/FP/FN 3923/91/103; divisions 1/3/4; 84,831 observations.
- **Cellpose + ultrack, no divisions:** 0.82987357; edges TP/FP/FN 3645/339/381; divisions 0/0/5; 96,405 observations.
- **Cellpose + HOCT general_v1, no divisions:** 0.75710496; edges TP/FP/FN 3544/540/482; divisions 0/0/5; 111,082 observations.
- **Cellpose + OrganoidTracker2:** 0.83126292; edges TP/FP/FN 3599/402/427; divisions 2/1/3; 106,331 observations.
- **Cellpose + OrganoidTracker2, no divisions:** 0.79785513; edges TP/FP/FN 3597/400/429; divisions 0/0/5; 106,320 observations.
- **Cellpose + CELLECT association:** 0.72738658; edges TP/FP/FN 3380/559/646; divisions 1/7/4; 111,082 observations.
- **Cellpose + CELLECT association, no divisions:** 0.71888108; edges TP/FP/FN 3377/556/649; divisions 0/0/5; 111,082 observations.

## Both embryos

This breakdown uses the fixed pooled configuration for each family; it does not select a different model on each embryo.

- Cellpose + OrganoidTracker2, embryo 44b6: 0.86722214; ultrack difference +0.03317144.
- Cellpose + OrganoidTracker2, embryo 6bba: 0.81581490; ultrack difference -0.01250390.
- Cellpose + CELLECT association, embryo 44b6: 0.73692937; ultrack difference -0.09712134.
- Cellpose + CELLECT association, embryo 6bba: 0.72361067; ultrack difference -0.10470812.

## Error attribution

An unmatched prediction is not necessarily a false cell: annotations are sparse. False edges below are only those the official scorer considers evaluable.

- **Cellpose + OrganoidTracker2:** 185 missed links had the correct candidate available; 176 lacked a matched endpoint; 66 were outside the candidate bank. Among evaluable false links, 400 touch an unmatched prediction and 2 connect two matched cells with the wrong identities.
- **Cellpose + OrganoidTracker2, no divisions:** 187 missed links had the correct candidate available; 176 lacked a matched endpoint; 66 were outside the candidate bank. Among evaluable false links, 398 touch an unmatched prediction and 2 connect two matched cells with the wrong identities.
- **Cellpose + CELLECT association:** 475 missed links had the correct candidate available; 167 lacked a matched endpoint; 4 were outside the candidate bank. Among evaluable false links, 554 touch an unmatched prediction and 5 connect two matched cells with the wrong identities.
- **Cellpose + CELLECT association, no divisions:** 478 missed links had the correct candidate available; 167 lacked a matched endpoint; 4 were outside the candidate bank. Among evaluable false links, 551 touch an unmatched prediction and 5 connect two matched cells with the wrong identities.

## Division controls

- Cellpose + OrganoidTracker2: enabling divisions changes the full score by +0.03340779; 2 true positives, 1 evaluable false divisions, and 3 missed annotated divisions.
- Cellpose + CELLECT association: enabling divisions changes the full score by +0.00850550; 1 true positives, 7 evaluable false divisions, and 4 missed annotated divisions.

Only five divisions are annotated on this panel. These controls measure branching behavior here; they do not establish reliable mitosis generalization.

The most useful next component to test is OrganoidTracker2's division probability with ultrack's stronger ordinary links. OrganoidTracker2 also loses 66 true links at its candidate-generation stage on this panel, so testing a less restrictive candidate bank is justified. Neither follow-up has been measured in this study.

## Verification and reproduction

OrganoidTracker's actual patch inputs and published calibration are compared to the released predictor. CPU/CUDA numerical checks use real image patches. The exact flow formulation is independently checked against every legal graph in 16 small cases. CELLECT feature-channel lookup is checked against the released expression, with border tile coverage and real-patch numerical checks. Every exported graph passes structural validation and a CSV round trip.

The original sources and weights are pinned. No shared environment, raw input, incumbent graph or other study output was modified. This was a local runtime experiment; Kaggle notebook integration was not tested.

- [Reproduction](../tools/other_trackers/README.md)
- [Complete counts](../results/other-trackers-20260914/comparison.json)
- [Interactive comparison](/home/mpf/.cursor/projects/home-mpf-code-kaggle-cell-tracking/canvases/other-trackers-20260914.canvas.tsx)
- [OrganoidTracker source](https://github.com/jvzonlab/OrganoidTracker/tree/db28ff26584ac6d1230ea90750a3324a2778fb31)
- [OrganoidTracker checkpoint release](https://zenodo.org/records/18479952)
- [CELLECT source](https://github.com/zzz333za/CELLECT/tree/3586070926f7f1fd5d8df37456861d22bdc63236)
