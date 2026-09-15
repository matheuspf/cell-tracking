# Error-driven training: start here

This is the active handover for branch `handover/pipeline-error-training-20260915`.
It starts from `main` commit `bd844731c0e93b401cf94e67c92350d4109e12df`.
The implementation and experiments are to be performed locally by Codex, on one
RTX 4090 (24 GB), 16 CPU cores and 64 GB RAM. This commit is a plan, not a measured improvement.

## Give Codex this instruction

> Read `AGENTS.md`, then `handover/pipeline-error-training-20260915/CODEX_PROMPT.md`.
> Implement and execute this handover. Start with the retained P0 pipeline, keep
> C4_m6 as the strongest complete local challenger, and improve divisions first
> through training or a narrow division-module swap. Then run the independent
> continuation-identity and observation-selection lanes. Preserve the baseline,
> do not tune thresholds on target results, and write measured results and a
> continuation report back to this branch.

## Handover

- [Plan and execution order](handover/pipeline-error-training-20260915/PLAN.md)
- [Implementation and learning contracts](handover/pipeline-error-training-20260915/IMPLEMENTATION.md)
- [Validation, exposure and adoption](handover/pipeline-error-training-20260915/VALIDATION.md)
- [Evidence and previous failures](handover/pipeline-error-training-20260915/EVIDENCE.md)
- [Machine-readable study](handover/pipeline-error-training-20260915/study.json)

The only ready-to-run new utility is a read-only plan check:

```sh
python handover/pipeline-error-training-20260915/check_plan.py
```

No activation script, file copying, old-study restart or new environment is
required just to begin reading this branch. Use existing local runtimes and
artifacts after verifying them. New training/inference commands described in
the handover are interfaces for Codex to implement, not already-working tools.
