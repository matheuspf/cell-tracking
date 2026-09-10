# Execute strong-tracker v3 locally

You are the implementation/experiment agent, using the user's selected GPT 6 Pro
model and one RTX 4090. Implement and execute this handover; do not return another
plan. Read root AGENTS.md, relevant competition skills, docs/strong-tracker-v2.md,
results/strong-tracker-v2/{v2_report.md,summary.json,winning_config.json}, and every
file in this directory. Inspect actual local notebooks, arrays and scorer code.

The current champion is `bypass_motion_bounds` at 0.9342063149703403. Preserve its
selected_predictions and selected_prediction_lock. V3 targets another +0.02; a
smaller real gain or a well-supported negative result is preferable to an invented
improvement. Never promote a variant merely because it beats the obsolete 0.911774
baseline. Do not re-run the v1/v2 workflows into their sealed directories.

Follow V300–V370 in EXPERIMENTS.md. In particular:
- Fingerprint and rescore the actual incumbent on all 199 expected clips.
- Rebuild graph-dependent features on it. Existing v2 fits/caches point to the old
  V1/baseline/public graph and are not native to the new incumbent.
- Preserve agreeing good links; learn local arbitration over complete compatible
  alternatives where pipelines disagree. Account separately for coordinates and
  topology before constructing source labels.
- Train actual division-event classifiers, not another sparse-membership image
  classifier. Expand hypotheses beyond the current continuation and avoid using
  selected daughter tracks as the sole evidence of daughter persistence.
- Check the concrete owner-guard issue in REVIEW.md with synthetic tests. A fix
  is not a measured gain until rescored.
- Use fresh official matching and division assignment for every result. Graph
  input hashes and exact expected sample coverage are mandatory.

Use existing data, CUDA environment, local weights and cached neural outputs.
Resolve the current checkout and paths dynamically: the completed v2 run used a
/root checkout on a rented host, which is not a mandatory v3 path. Preserve working
local environments; source root_remote_env.sh only when applicable. Prefer an
explicit supplied interpreter or the existing study interpreter; never silently
use CPU Python for training. No bootstrap, whole-data download, reset, clean,
credential disclosure, paid API calls, rented hardware or automatic submission.

The source-only directions remain 44b6->6bba and 6bba->44b6. Both embryos and public
checkpoints are already contaminated/reused evaluation evidence. Preserve opposite-
embryo fitting but label v3 operational exploratory, not clean OOF. Freeze both
source directions and each bounded round before its comparative scoring. Do not
choose different policies using a held-out embryo ID. Do not invent independent
inner folds or confidence intervals when overlap cannot be certified. Do not
repeat the entire registration study; use its documented limitations.

Create `tools/strong_tracker_v3/`, `tests/strong_tracker_v3/`, a portable wrapper,
and isolated output/work roots. No module import or helper may mutate v1/v2 OUT
globals or caches. Reuse pure code or refactor it explicitly, with parity tests.
Record stage/model/graph/feature/code hashes; resumptions require exact fingerprints.
Complete cheap stages even when an optional GPU/teacher lane is blocked. Preserve
negative results. Deliver the final report, offline dashboard, complete score rows,
selected config and annotation-unavailable export, plus fresh-image pipeline smoke
checks. Cache-only results must remain labelled cache-only when that last test fails.

Write sanitized findings back to `results/strong-tracker-v3/` and update this
handover's STATUS.json after execution. Commit and push code/configs/sanitized
results to the current v3 branch after reviewing the staged file list; keep raw
artifacts local. Do not merge PRs, upload Kaggle submissions, publish notebooks,
post on forums or accept gated-model terms. Record artifacts needed to resume on
another host, without uploading them or promising external persistence.
