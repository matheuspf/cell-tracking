# Evidence review: change the training distribution, not just the search

Reviewed source: `292ecc2569f3a16676de92d80e382be02024d904`, 2026-09-09.
All numbers below are repository-reported completed experiments or prepared-data
audits. The authoring session has not accessed raw microscopy or trained v4.

## Current result

V3 selected A_residual_m3.0: **0.934802374260586**, only +0.000596059290246
above v2. Scores are 0.931664468721842 on 44b6 and 0.935221784097327 on 6bba.
Its 4,108,943 nodes yield edge TP/FP/FN 123,135/4,965/5,748 and division
TP/FP/FN 29/92/122. The association change recovered 112 GT edges and lost none,
but introduced 35 FP. Preserve that small gain rather than reverting to v2.
Source: `results/strong-tracker-v3/final_report.md` and `winning_config.json`.

V3 expanded event search to 190,139,632 alternatives; 98.40% had unknown labels.
Only 20 positive source groups in 44b6 and 92 in 6bba trained its image event
models. Forty thousand optimizer steps did not make these independent biological
events. All learned event policies regressed. The best event policy had division
TP/FP/FN 69/1,738/82 and score 0.915381141505. Its candidate oracle was stronger,
but that does not make learned transfer achievable. More candidates were not
more labels. Sources: `event_candidate_coverage.json`, `family_outcomes.json`.

The raw solver has division cost 1.2, free appearance, and second-edge reward
at most 1. Removing a fork edge and declaring a daughter birth always lowers
cost by at least 0.2. This was tested on the actual solver, not just conjectured.
Thus improving edge probabilities alone cannot create raw divisions under that
unchanged objective. Source: `results/strong-tracker-v3/raw_decoder_trace.json`.

## Newly available supervision

The downloaded release provides 1,539 static volumes / 423,853 centers and
2,174 six-frame movies / 4,056,226 nodes / 3,460,295 edges / 165,267 simulated
forks. These are many supervised examples, not independent real embryos.
The six Zoo graphs contain experimental tracking outputs, including 159,053
zebrafish fork observations, but have no paired downloaded microscopy and are
not blanket hand-validated truth. RIKEN animal C has millions of measurements
but no established temporal identities. Sources: `docs/external-data-guide/`.

The direct opportunity is representation pretraining and richer hard negatives:
image centers and candidate pair decisions on synthetic movies; motion/branch
representations on real Zoo graphs; source-only sparse adaptation on Biohub.
External graphs can also expose clipping, missing detections and longer context
unavailable in the six-frame release, provided corruption targets remain honest.
This is the hypothesis v4 tests, not a forecast of improvement.

## Failure modes v4 must remove

1. **Pseudo-dense real targets:** an unlabelled Biohub candidate is not background.
   Zoo tracking output is weak supervision; RIKEN positions are not track IDs.
2. **Coordinate mismatch:** synthetic sequence Y/X coordinates require /4, but
   sequence images must NOT be downsampled again. Static images remain native.
3. **Context/intensity mismatch:** six observed frames cannot become nine valid
   frames; normalized [0,1] pixels cannot enter raw-intensity crop thresholds.
4. **Generator shortcuts:** release has no internal births/deaths, clips positions,
   resamples radii per frame and mixes voxel/physical motion conventions. Add
   observation corruptions and source-derived image randomization; audit their
   actual distributions, not the generator's metadata claims.
5. **Class-balance mistaken for posterior calibration:** no 15.7x folklore prior
   correction, no inherited p=.2 threshold, no count-estimate inference feature.
6. **More postprocessing instead of learning:** all v4 primary models must update
   parameters on external training examples and have real-only controls.

## Retained uncertainty

Both Biohub embryos and prior outcomes have been reused; public checkpoints and
v2 teachers contain target-label exposure. Synthetic generator calibration used
44b6. Unknown clip overlap prevents claims of independent source inner folds.
External training does not erase any of this. New synthetic held-out scores are
engineering transfer diagnostics; Zoo weak-label tests are not biological truth.
The actual decision still uses fresh official graph scoring on all 199 clips,
separate embryo results, and a unchanged-incumbent fallback.
