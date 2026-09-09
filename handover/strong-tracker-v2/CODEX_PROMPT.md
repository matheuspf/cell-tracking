# Execute strong-tracker v2

Read root AGENTS.md and relevant competition skills, then
`handover/annotation-selection-v1/NEXT_AGENT.md`,
`results/annotation-selection-v1/README.md`, this directory's REVIEW.md and
EXPERIMENTS.md. The v1 experiment is COMPLETE. Do not rerun its initial prompt.

Implement and execute V200-V270 locally using the existing data, caches and RTX
4090. Focus on significant improvement of the strong Harmonic Fusion graph.
The main priorities are recovering missed divisions, correcting ambiguous links,
and testing annotation selection trained on the strong pipeline's own candidates.
Do not equate the classical +0.002791 with a gain over the 0.946 public notebook.

Create a new output namespace:
`/kaggle/working/cell-tracking/strong-tracker-v2/` and ignored `work/strong-tracker-v2/`.
Create implementation under a new module such as `tools/strong_tracker_v2/`.
Existing v1 modules use a hard-coded OUT path: reuse their pure functions, not
stage commands that would mutate v1. Keep raw data, notebooks, models, prediction
locks, v1 result files and unrelated user work untouched. No reset, clean,
broad install, cloud rental, paid API, Kaggle submission or forum posting.

Start with existing cached graphs and pre-ILP scores. Verify raw/post-repair
baseline identity before trying changes. Instrument the true graph-transforming
stages in the notebook call graph; do not assume all 27 preserved function bodies
are successive graph transformations. Diagnose every annotated division window
using the actual local division matcher, not only the global edge matches.
Run oracle experiments only in a clearly separate evaluation-only diagnostic path.
No oracle/GT field may enter deployed repair, filtering, inference or feature code.

Use all available positive training events/units, with cluster-aware sampling of
negatives. A 30,000-candidate natural-prevalence probe is not an adequate final
image study. Record unique positives, unique lineage/event groups, steps and
learning curves; do not silently use a smoke test as the trained result.
Avoid large unconstrained models for only 151 annotated division observations.
Begin with native graph features and a small regularized division/link reranker.
Only train the larger temporal image probe when cheap models leave plausible
headroom and genuine image-dependent failure modes remain.

Keep both source/target directions separate. Train repair/selector labels on the
source embryo only and freeze both directional configurations before comparative
outer scoring. These embryos have already been inspected, so call v2 exploratory;
public checkpoint contamination is not fixed merely by retraining the selector.
Use clean pretrained/source-trained predictor provenance when available. Do not
block useful operational diagnostics solely because that clean lane is unavailable.

Recompute official matching and divisions for every graph variant, require the
full expected sample set, and compare exact run-level aggregation. Do not optimize
only center recall, AUROC, a detection metric or a count-based approximation.
Check forbidden merges, duplicate edges, degree limits, coordinates and consecutive
frames. Validate annotation-unavailable inference for every candidate pipeline.

Produce measured results, an offline dashboard, all per-sample score rows, failure
breakdowns, reproducible commands/configs and a final decision even when an arm
fails. Preserve the original weak-tracker and failed transfer results as historical
evidence. Do not promise a +0.02 gain: it is the proposed local optimization target.
Do not report a hypothetical division or link-repair calculation as an experiment.
