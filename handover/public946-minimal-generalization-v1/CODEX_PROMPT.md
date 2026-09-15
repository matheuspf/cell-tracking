# Execute the expanded public946 study

Work in the current `handover/public946-minimal-generalization-v1` branch. This is
revision 2 and supersedes the old exactly-one-change experiment limit. The user
explicitly requested more than disabling one parameter. **Do not stop at B1.**

Read AGENTS.md, relevant competition skills, then this directory's START_HERE.md,
BASELINE_AND_EVIDENCE.md, EXPERIMENTS.md, PLAN.md, VALIDATION.md and experiment_lock.json.
No prior conversation or installation/activation script is required.

Your task is to implement and execute the plan locally using the existing public
Harmonic Fusion v29 artifacts and GPU runtime. B0 is the source documented at
0.946 LB; B1 is the already-known no-motion control. Implement E01-E08 independently
against B0, test applicable arms over the full supplied population, run the three
eligible fixed combinations, lock up to two finalists and test their exact B1
transfers. No threshold/fusion-weight sweeps, architecture changes, training,
residual models, external-data campaign, arbitrary stacking or per-embryo routing.

First preserve the workspace and inspect actual archived source, model/input hashes,
effective settings, free storage, runtime and metric revision. The planner could
read source adapters and reports but not ignored original notebooks, images or weights.
Source-audit whether each proposed mechanism already exists or is a mathematical
no-op. Document such cases with code/tensor evidence; do not invent substitutions.

Implement the runner/modules/tests described in PLAN.md. The handover's
baseline_contract.py applies only to the B1 flag ablation. study_contract.py validates
and freezes the expanded registration; neither helper executes or certifies microscopy.
Run all synthetic tests, freeze the plan and source-resolved recipes before comparative
scoring, and keep status/results separate from the immutable registry.

Reproduce B0/B1 faithfully, avoiding old adapters' hard-coded no-motion defaults.
Use fresh official per-arm matching and exact run-level aggregation. Existing caches
are read-only and usable only with matching stage/source/model/input fingerprints.
Detector/feature experiments require actual neural evidence, not only repaired graphs.
Workers see images and public models, never annotations or old selected predictions.

Run the metadata-only pilots for correctness and compute planning, not scientific
selection. Continue all applicable independent arms even if an earlier arm fails.
Freeze and use the ranking/combination/transfer rules in VALIDATION.md. Do not retune
a failed method, substitute a new method, promote on a pilot only, or claim repeated
local embryos form a clean validation set. Zero improvement is an acceptable result.

Deliver implementation, tests, complete arm matrix and concise measured reports,
plus fresh-tested B0/B1 control notebooks and up to two eligible minimal candidate
notebooks with public input dependencies, scientific/shared-runtime diffs, hashes
and manual Kaggle instructions. No automatic uploads or invented LB claims. The
target is 0.95+ LB, not a local >=0.95 stop rule.

Write `results/public946-minimal-generalization-v1/START_HERE.md`, REPORT.md,
experiment_matrix.csv, per_embryo_scores.csv, status.json and NEXT_AGENT.md. Show
known versus new measurements, no-op proofs, regressions, cost, source identity,
contamination caveats and exact reproduction/resume commands. Raw artifacts and
weights stay ignored. Commit code and sanitized receipts to this same branch;
leave earlier studies unchanged and never force-push or discard user work.

A missing optional cache or clean holdout does not stop the entire study. Respect
actual safety/resource limits, finish independent feasible work and report named
blockers honestly. A baseline+B1-only result is incomplete under this revision;
do not label the expanded study complete without accounting for every registered arm.
