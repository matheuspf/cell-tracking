# Annotation selection v1 — measured local results

**Execution complete, 2026-09-08. Decision: `promising_but_uncertain`.** The
primary filter helped the clean classical baseline but harmed the stronger
public notebook. No improvement over the best public baseline is demonstrated.
FOCUS-3D has been assessed, but has not been run or measured.

- [Final report](report.md) — complete methods, results, limitations and failures.
- [Offline HTML dashboard](dashboard.html) — download/open locally in a browser;
  all chart data and images are embedded. No server or internet required.
- [Next-agent handover](../../handover/annotation-selection-v1/NEXT_AGENT.md).
- [FOCUS-3D assessment](../../handover/annotation-selection-v1/FOCUS3D.md).
- [Reproduction instructions](../../docs/annotation-selection-v1.md).

## How much does it improve over our best public baseline?

The [downloaded notebook snapshot](../../configs/notebooks.json), verified on
2026-09-08, records **0.946** for both
[Harmonic Fusion v29](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion?scriptVersionId=347965685)
and [942 TTA v1](https://www.kaggle.com/code/redoctopusk/biohub-942tta?scriptVersionId=347821442).
The third downloaded reproduction scores 0.945. Only Harmonic Fusion was fully
run locally on all 199 supplied training clips. The public-checkpoint lane is
contaminated by training/selection on supplied embryos and is diagnostic.

| Measured lane | Baseline local score | Primary filtered score | Delta |
| --- | ---: | ---: | ---: |
| Clean classical, 44b6 | 0.703072 | 0.715329 | +0.012257 |
| Clean classical, 6bba | 0.668908 | 0.669964 | +0.001056 |
| Clean classical, pooled | 0.674116 | 0.676907 | +0.002791 |
| Public Harmonic Fusion, 44b6 | 0.912521 | 0.877651 | −0.034870 |
| Public Harmonic Fusion, 6bba | 0.911597 | 0.872418 | −0.039179 |
| Public Harmonic Fusion, pooled | 0.911774 | 0.873322 | −0.038452 |

Across **all 72 learned/confidence filter settings**, **zero** improved the
public lane's pooled local score. Even the best hindsight setting—confidence
ranking of coherent tracklets, requested keep 90%—scored **0.911222**, a delta
of **−0.000552**. It improved 44b6 but harmed 6bba. The model/policy/budget
combination was identified after inspecting outcomes and is descriptive only.
Exact rows and the inclusion rule are in [baseline_comparison.json](baseline_comparison.json),
recomputed from [public_retention.csv](public_retention.csv).

The local score populations differ from the public leaderboard. **Do not add
the classical +0.002791 to 0.946 or claim a hidden-test uplift.** FOCUS-3D may
be useful for cell separation/localization or as a pseudo-label teacher, but
its score gain, local memory use and runtime remain unknown. See the linked
assessment for verified source pins, licenses, access conditions and integration.

## What the completed protocol established

Both embryo directions covered all 199 clips. The fixed seven-leaf boosted
tree used source-trained models and requested keep 90% with coherent tracklets;
realized pooled retention was 90.0250%, retaining 96.5800% of baseline candidate
annotation matches. All 199 prediction files reproduced byte-for-byte with
annotation reads blocked before outer evaluation.

The clean pooled count contribution was +0.007121 and graph contribution
−0.004331. The selector exceeded the exact-budget confidence control by
+0.005636 pooled, but quality/selection confounding remains unresolved.
Only two embryos and one conservative overlap group per embryo were available;
independent source inner tuning and informative bootstrap intervals could not
be established. Fixed preregistered settings replaced tuning. All-cell
prevalence is unknown; the two blinded 48-item audit packs have no manual labels.

## Portable evidence

- [Full sanitized summary](full_summary.json), [compact summary](summary.json),
  [public summary](public_summary.json), [stage status](status.json).
- [Clean sweep: 2,385 aggregate rows](retention.csv) and
  [public sweep: 975 aggregate rows](public_retention.csv).
- [Coverage](coverage.csv), [matching summary](matching_summary.csv),
  [classifier metrics](classifier_metrics.csv), [calibration](calibration.csv),
  [feature profiles](feature_profiles.csv).
- [Exact-cost controls](matched_budget_summary.csv),
  [random-seed results](random_exact_seed_results.csv),
  [both image-probe seeds](image_seed_results.csv), [uncertainty](uncertainty.csv).
- [Preregistration](preregistration.json),
  [public-selector preregistration](public_selector_preregistration.json),
  [measured environment](environment.json), [original validation receipt](validation_receipt.json),
  [portable dashboard validation](publication_validation.json).
- [Artifact hashes and export provenance](bundle_manifest.json),
  [local-only evidence inventory](LOCAL_ARTIFACTS.md). Static PNG/SVG figures
  are in `plots/`.

The report and dashboard retain measured values. Publication removes six
individual predicted-coordinate rows and per-clip graph-parity details from
their embedded JSON, preserving aggregate integrity findings. All copied
source files are checked against the sealed local manifest. Rebuild the export
on the original machine with
`PYTHONNOUSERSITE=1 python tools/package_annotation_findings.py`; this does not
rerun experiments or mutate the sealed artifact store.
Validate a checkout without competition data using
`PYTHONNOUSERSITE=1 python tools/validate_annotation_findings.py`.
Add `--browser` in the existing inspection environment to repeat offline Chromium
checks and refresh the separate publication receipt.

Raw inputs, notebook originals, existing environments and unrelated user work
were preserved. The complete 30.72 GiB store—including predictions, raw patches,
weights and detailed GT matching tables—remains outside public Git. A fresh
checkout contains code and aggregate evidence; raw-data reproduction requires
the local inputs described in the handover. No Kaggle submission or hosted
inference was performed. Findings were prepared for this branch at the user's
explicit request.
