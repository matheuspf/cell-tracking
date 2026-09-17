# Division generalization v2 — start here

Active handover for `handover/division-generalization-v2`, based on completed
study commit `ab861b5d8c1a37bd920f610b88f552dfc5b5dbb6` (September 16, 2026).
This is a NEW plan, not another measured improvement. P0 remains production.

Goal: **at least 0.95 combined score**, without target threshold tuning or replacing
the strong detector. Concentrate training on a division verifier that distinguishes
one cell becoming two from two already-existing cells, wrong daughter ownership,
and spurious forks. Do not repeat the previous broad, 178-update architecture matrix.

Give local Codex this instruction:

> Read AGENTS.md, then handover/division-generalization-v2/CODEX_PROMPT.md.
> Implement and execute that handover on this branch. Preserve the completed
> pipeline-error-training study. Train the focused division module adequately,
> evaluate against P0 and C4_m6, and write measured results and continuation notes.

Read [the plan](handover/division-generalization-v2/PLAN.md),
[learning/implementation specification](handover/division-generalization-v2/SPEC.md),
[validation protocol](handover/division-generalization-v2/VALIDATION.md), and
[evidence review](handover/division-generalization-v2/EVIDENCE.md).

Ready now, requiring only Python:

```sh
python handover/division-generalization-v2/test_contracts.py
python handover/division-generalization-v2/contracts.py
```

These check planning contracts and score arithmetic, NOT training or pipeline
correctness. Codex must implement the new training/inference namespace. No
activation/copying script is needed. Do not execute an older handover or resume
an unrelated training process merely because its files are inherited here.
