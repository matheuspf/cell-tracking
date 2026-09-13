# Public946 generalization study — expanded revision 2

**Plan only. Microscopy experiments and LB validation have not been executed by the planner.**

Same branch: `handover/public946-minimal-generalization-v1`.
Original main base: `fb5521629eb41c8c485b291a5bcf344944c113ae`.
Supersedes the one-ablation plan at `dd7810dcc7bb4b0779860b572cd487f987a1df19`.

## What changed

The user rejected a study consisting only of disabling motion relinking. This
revision replaces that stopping rule with **eight mechanism-level experiments,
three fixed two-module combinations, and two finalist transfer slots**. B0 is
the original public Harmonic Fusion v29; B1 is the known no-motion ablation,
now a control rather than the whole research program. Maximum: 15 configurations
including both anchors; no Cartesian search or numerical threshold sweeps.

| ID | Single change to B0 |
|---|---|
| E01 | Native-evidence veto on motion-relink proposals |
| E02 | Analytic sub-voxel localization after topology, before smoothing |
| E03 | Trilinear rather than truncated node-feature sampling |
| E04 | Two-phase, half-downsampling-cell detector consensus |
| E05 | Association-only spatial-reflection test-time averaging |
| E06 | Forward-time overlapping-context association consensus |
| E07 | Fork-anchored trajectory smoothing |
| E08 | Median aggregation across existing detection augmentation views |

All public checkpoints, network architectures, harmonic **inter-model** fusion,
ILP costs, detection thresholds and unmentioned repairs remain fixed. E04/E08
explicitly change **within-model detection aggregation**; E05/E06 explicitly
change association evidence aggregation. Do not describe these as pure replay
ablations or silently substitute the simplified v5 model for the public pipeline.

## Read and execute

Read `BASELINE_AND_EVIDENCE.md`, `EXPERIMENTS.md`, `PLAN.md`, `VALIDATION.md`,
`experiment_lock.json`, then execute `CODEX_PROMPT.md`.

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s handover/public946-minimal-generalization-v1 -p 'test_*.py' -v
python handover/public946-minimal-generalization-v1/study_contract.py validate
python handover/public946-minimal-generalization-v1/study_contract.py matrix
```

These implemented helpers need only Python's standard library. They validate
source/plan structure, not inference correctness or scores. No installation or
activation script is required for this handover. The local executor implements
the experiment adapters and runner using the existing GPU environment.

## Required outcome

Do not stop after reproducing B1 or after writing another plan. Execute every
applicable single-mechanism arm; evaluate eligible registered combinations;
resolve at most two transfers to B1; deliver measured reports, code, test receipts
and standalone notebook packages. An already-implemented mechanism may be
`not_applicable` only with source and numerical identity evidence, not because a
pilot score was disappointing. Missing optional caches must not stop independent
arms. Real resource or asset blockers remain explicit; never fabricate execution.

Target **0.95+ LB**, not a local score threshold. Reused local embryos cannot
establish unseen-embryo generalization. No submission is authorized here; the
result is a locked manual test package, not an asserted leaderboard improvement.
