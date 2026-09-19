# V11: division reliability with a clean full-pipeline comparison

Created September 19, 2026. Branch: `handover/division-reliability-v11`.
Base: `08061f0a224387594dd3e915462bafb5d5928fb2` (v10 planning commit, including the completed September 16 study).

**This is a new execution plan, not new measured performance.** The remote v10 branch still has only its planning commit: neither `results/clean-validation-division-v10/REPORT_BACK.md` nor `tools/clean_validation_v10/` was available at review. Local Codex must reconcile any unpushed execution before doing more work. Do not assume v10 ran, failed, or reached any score.

## Give local Codex this instruction

> Read AGENTS.md and handover/division-reliability-v11/CODEX_PROMPT.md, then execute v11 locally. First reconcile local v10 code, processes, checkpoints and receipts without changing them. Reuse only provenance-qualified clean upstream artifacts; otherwise implement and execute the inherited clean upstream prerequisite. Build the lightweight learned fork baseline and the compact factorized fork model, run both source directions and both seeds, freeze predictions before target scoring, and validate cold image-to-CSV inference. Commit and push code and sanitized results only to this branch. Return results/division-reliability-v11/REPORT_BACK.md. Preserve P0 and all older experiments. Do not merge, submit to Kaggle, publish weights, or launch another handover queue.

Read [review](handover/division-reliability-v11/REVIEW.md), [plan](handover/division-reliability-v11/PLAN.md), [implementation](handover/division-reliability-v11/IMPLEMENTATION.md), [validation/reporting](handover/division-reliability-v11/VALIDATION_REPORTING.md), and [registry](handover/division-reliability-v11/study.json).

V11 takes priority for this execution; inherited v10 documents supply only the explicitly retained upstream recipe and isolation contract. Do not run both complete matrices. The intended training CLI does not exist at plan creation. Implement it; a planning checker or synthetic fixture is not execution.

The checked-in helpers use only the Python standard library:

```sh
python handover/division-reliability-v11/check_plan.py
python -m unittest discover -s handover/division-reliability-v11 -p 'test_*.py' -v
```

Raw data, model weights, optimizer states, complete predictions and caches remain under ignored local storage, normally `work/division-reliability-v11/`. Existing absolute paths are discovery hints, not required installation locations. No production default changes automatically.
