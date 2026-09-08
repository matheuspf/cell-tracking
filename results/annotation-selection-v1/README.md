# Annotation selection v1 — measured local result

Status: **promising_but_uncertain**. Both embryo directions were evaluated on all 199
training clips. The primary coherent keep-90% rule used fixed source-trained models;
unresolved crop overlap prevented independent source inner tuning and informative
within-embryo bootstrap intervals. All 199 prediction files reproduced byte-for-byte
with annotation reads blocked before outer evaluation.

| Outer embryo | Baseline score | Filtered score | Delta | Realized retention |
| --- | ---: | ---: | ---: | ---: |
| 44b6 | 0.703072 | 0.715329 | +0.012257 | 90.0134% |
| 6bba | 0.668908 | 0.669964 | +0.001056 | 90.0398% |
| pooled | 0.674116 | 0.676907 | +0.002791 | 90.0250% |

The public Harmonic Fusion lane is separately marked contaminated and diagnostic.
Its transferred primary rule changes pooled score from 0.911774
to 0.873322 (-0.038452); this rule harms that tracker.
Exact sparse annotations, supplied-count coverage, candidate-conditional membership,
all fixed budgets, random/confidence controls, exact-cost coherent controls, fresh
official graph/division counts, count/graph attribution, and both image-probe seeds
are in the local artifacts. Exact all-cell prevalence remains unidentified; the
48-ROI blinded census pack has no manual labels. No hidden-test gain is claimed.

- [Final report](/kaggle/working/cell-tracking/annotation-selection-v1/final_report.md)
- [Offline dashboard](/kaggle/working/cell-tracking/annotation-selection-v1/dashboard.html)
- [Artifact manifest](/kaggle/working/cell-tracking/annotation-selection-v1/artifact_manifest.json)
- [Reproduction instructions](../../docs/annotation-selection-v1.md)
- [Sanitized aggregate JSON](summary.json)

Raw inputs, notebook originals and existing environments were preserved. Predictions,
raw patches, weights and detailed GT matching tables remain outside Git. No submission,
push, notebook publication, forum post, paid API call or hardware rental was performed.
