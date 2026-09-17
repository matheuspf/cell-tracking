# Review of the completed study, not the old proposal

Inspected results and source at `ab861b5d8c1a37bd920f610b88f552dfc5b5dbb6`.
All paths below are repository-relative. This dated plan is allowed to learn from
these already-inspected outcomes; it cannot call them a fresh holdout later.

## What actually happened

Source: results/pipeline-error-training-20260915/REPORT.md, CONTINUATION.md,
scores.csv, training_fits.csv and source_feasibility_scores.csv.

| Pipeline | All-199 score | Division TP / FP / FN | Conclusion |
|---|---:|---|---|
| P0 | 0.934864986413134 | 29 / 92 / 122 | Retained production |
| C4_m6 | 0.935178370257 | 30 / 92 / 121 | Strongest complete pooled reference; not adopted |
| A10 | 0.934939378710 | 29 / 92 / 122 | Only +0.000074392296 vs P0; primary 6bba regressed |
| A10 replication | 0.934938660898 | 29 / 92 / 122 | Small gain; cannot rescue primary's failed gate |
| D10_adapted | 0.918152447085 | 38 / 850 / 113 | +9 recovered divisions, +758 false divisions |
| D20_compact | 0.918100978703 | 48 / 1245 / 103 | +19 recovered divisions, +1153 false divisions |
| D20_temporal | 0.919226643837 | 45 / 1052 / 106 | +16 recovered divisions, +960 false divisions |
| O10_swap | 0.934415338644 | 29 / 92 / 122 | Worse; only 3 edge TP gained, 30 lost |
| O10_restore | 0.934864986413134 | 29 / 92 / 122 | Exact no-op on all 199 clips |

These experiments were executed and scored, but not sufficient tests of converged
neural training. Sixteen directional fits received only **178 optimizer updates**.
D20_temporal used 89 identity-pretraining updates then 89 joint updates. No new
candidate passed every recommendation gate. The 0.95 target was not approached.

## Concrete source-level limitations

1. `tools/pipeline_error_training/dataset.py`: `event_keys` includes only groups
   with a supported biological positive. Event minibatches sample that pool.
   Negative alternatives inside positive groups are present, but ordinary
   negative-only event groups have no dedicated event-minibatch path. The report
   records only 15/74 positive groups for the two source directions. More epochs
   on this unchanged pool cannot by itself establish whole-field FP rejection.
2. `train.py`: the budget was divided among fourteen proposed first-seed fits,
   using a slow source pilot; the common budget fell from 16000 to 178. Every
   optimizer step loops through 32 groups serially, moves model/optimizer to GPU
   and back, clears caches, and writes resume state. These are optimization targets,
   not proven dominant costs until profiled. Full precision and recomputation
   were used. Later cache optimizations did not increase the frozen update budget.
3. `train.py` and executed_training_schedule.json: warmup and cosine were multiplied
   concurrently. With total=178, warmup lasts the whole fit; peak LR is approximately
   0.265799 of nominal, at update 75. Joint event training starts after this peak.
   This arithmetic is not proof that fixing the schedule alone will improve score.
4. `calibration.py` and calibration_sampling_audit.json: ordinary anchors use one
   deterministic time residue out of nine and weight nine. This is not a randomized
   row propensity or a whole-source census. Reported expanded supported positive
   fractions are about 0.432% and 0.698%. They are NOT biological mitosis prevalence.
   The binary fit also regularizes its intercept and temperature. Tail rejection
   after maximizing over many alternatives was not independently established.
5. `scoring.py`: complete-action values sum learned link residuals plus native
   link logits, birth/death costs and event/risk terms. A new extra edge can
   therefore receive a native-logit reward before learned evidence distinguishes
   division from an independent birth. The no-op-versus-edit comparison needs
   explicit zero-gain and extra-edge fixtures, not an assumption that all terms
   are on a well-calibrated decision scale.
6. `infer.py`: `zero=True` returns P0 immediately, before model/bank/scoring/solver.
   That establishes a disable switch, not literal zero-head identity through the
   active scoring path. New tests must distinguish these two claims.
7. `crops.py`: each token follows the incumbent's unique predecessor/successor
   path, then freezes its position when history is ambiguous. That is valid
   prediction-only input, but not independent evidence against a wrong incumbent
   identity. New shared raw-scene evidence tests this specific limitation.

These are observed implementation/sampling facts and plausible contributors.
Their separate causal effect has not been measured. The new controls isolate
representative training, hard-negative mining and image evidence; do not claim
that a particular issue explains all prior errors.

## Why divisions still offer the most useful route

The CURRENT source-label repaired-graph witnesses, on P0's fixed node universe:

| Embryo | Division witness score | Division TP / FP / FN | Continuation witness score |
|---|---:|---|---:|
| 44b6 | 0.956893956443 | 19 / 22 / 7 | 0.937499751206 |
| 6bba | 0.968346931854 | 86 / 70 / 39 | 0.937775381837 |

These are actually rescored graphs, not perfect-count arithmetic. They are
label-informed feasible repairs, not deployable models, independent validation,
upper bounds or a forecast. Do NOT average the embryo scores to invent a pooled
score. The observation-swap witness itself regressed (0.93134611 / 0.93513149);
that weak heuristic is not an upper bound, but does not justify repeating O10 now.

For scale only, P0's adjusted edge contribution is 0.9229308300345331. If this
term stayed fixed, reaching 0.95 would require total division TP 66 with FP=92
(+37 TP), or TP 72 with FP=112 (+43 TP and +20 FP). The helper reproduces these
count-only illustrations. The real result must rescore complete graphs and can
change edge terms too. Removing false forks alone is not enough to assume 0.95.

## External methodological references

Checked September 16, 2026; these motivate methods, not cell-tracking gain claims.

- Online hard example mining: https://arxiv.org/abs/1604.03540 . Use model-generated
  difficult SOURCE cases, but only with supported sparse labels. Do not import
  dense-detection background assumptions into microscopy.
- Focal loss: https://arxiv.org/abs/1708.02002 . Class imbalance matters, but focal
  loss does not supply missing negative labels or calibrated deployment tails.
  It is not another mandatory experiment in this plan.
- AMP recipe: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html .
  Test numerical behavior locally before enabling mixed precision; no blanket
  speedup claim or dependency upgrade is implied.
- Scheduler composition: https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.SequentialLR.html .
  This handover supplies its own explicit warmup/plateau/decay arithmetic.
- Pinned official metric: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md .
  Sparse edge evaluability, independent local division matching, one-to-one fork
  assignment and exact weighted aggregation remain authoritative.
