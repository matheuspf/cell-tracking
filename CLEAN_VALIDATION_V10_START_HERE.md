# V10: clean validation and division learning

**Created:** September 18, 2026.  
**Branch:** `handover/clean-validation-division-v10`  
**Base:** `ab861b5d8c1a37bd920f610b88f552dfc5b5dbb6`, the completed September 16 pipeline-error study, not the older default branch.  
**State:** execution handover only. No v10 model has been trained or scored. P0 remains unchanged.

## Give Codex this instruction

> Read AGENTS.md, CLEAN_VALIDATION_V10_START_HERE.md and every document referenced by handover/clean-validation-division-v10/CODEX_PROMPT.md. Execute v10 locally: implement the isolated package, train the registered models, perform the complete embryo-isolated evaluation, validate cold image-to-CSV inference, and commit code plus sanitized results on this branch. Use the frozen decision rules and resource fallback, not target scores, to choose the executable schedule. Preserve all historical studies and P0. Do not stop at another plan, a synthetic test suite, a training-loss report, or an exposed local score. Return results/clean-validation-division-v10/REPORT_BACK.md with exact completion status and the next decision. Do not merge, submit to Kaggle, publish weights, or launch unrelated handovers.

## Selected work

The main deliverable is a new source-isolated detector/encoder/association pipeline and a compact division improvement, each evaluated in both embryo directions and both registered seeds. A separate P0-based operational lane tests event sampling and calibration; it never supplies weights, teachers or feature caches to the clean lane.

This is deliberately not a Cellpose/ultrack/HOCT survey, native-resolution sweep, observation-selection campaign, or resumption of the old 400-epoch experiment. The old checkpoints and histories remain intact. The clean reference is a newly registered sparse-supervision training recipe, not a claim to reproduce the upstream 400-epoch model.

## Read in order

1. [Execution instructions](handover/clean-validation-division-v10/CODEX_PROMPT.md).
2. [Decisions, stages and budget](handover/clean-validation-division-v10/PLAN.md).
3. [Isolation, scoring and adoption](handover/clean-validation-division-v10/VALIDATION.md).
4. [Implementation and training specification](handover/clean-validation-division-v10/IMPLEMENTATION.md).
5. [Required return package](handover/clean-validation-division-v10/REPORTING.md).
6. [Machine-readable registry](handover/clean-validation-division-v10/study.json).

The registry and these documents together define the study. Inconsistency is an implementation blocker to resolve explicitly before fitting, not permission to invent a target-selected recipe.

## Local prerequisites

Use the existing repository, competition data and suitable local runtime. Resolve paths from arguments and local inventory; archived absolute paths are artifact identities, not mandatory installation locations. Raw data, checkpoints, caches and submissions remain outside Git, normally under ignored `work/clean-validation-division-v10/`.

The planning checker needs only Python's standard library:

```sh
python handover/clean-validation-division-v10/check_plan.py --self-test
python handover/clean-validation-division-v10/check_plan.py
```

The planned training CLI does not exist at handover creation. Codex must implement it as specified, then execute it. Passing this checker verifies the plan's structure, not any model, dependency, data isolation or score.
