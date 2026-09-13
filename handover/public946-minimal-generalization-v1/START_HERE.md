# Public946 minimal generalization v1

**Status: plan written; local experiments and Kaggle validation not executed.**

Branch: `handover/public946-minimal-generalization-v1`

Base: main at `fb5521629eb41c8c485b291a5bcf344944c113ae` (2026-09-12).

## The decision

Do not start from the locally selected 0.9348/0.9349 residual model. Start from
**Biohub Harmonic Fusion v29**, script version **347965685**, documented at
**0.946 public LB**. Preserve its checkpoints, detection TTA, fusion, neural
association, ILP, gap repair, safe divisions, pruning and smoothing.

Test exactly one scientific change:

```python
OUTPUT_MOTION_RELINK = False
```

This retains the learned associations instead of allowing the later motion
heuristic to replace them. It is not a new method, threshold search, or claim
that the historical local gain transfers to the leaderboard. Disabling an
entire stage is a small code change but can alter many edges; measure its
behavioral footprint rather than calling it low-risk because it is one line.

The useful next step is to turn the already observed ablation into a faithful,
portable, source-audited public-notebook candidate and test the transfer
hypothesis once. Do not spend another study fitting local residuals first.

## Read and run

Read `BASELINE_AND_EVIDENCE.md`, `PLAN.md`, `experiment_lock.json`, then
`CODEX_PROMPT.md`. The prompt is self-contained; no prior chat is needed.

The only helper already implemented here inspects an original Python export
and builds an isolated one-assignment candidate. It does not run microscopy,
certify runtime behavior, package a Kaggle notebook, or submit anything.

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s handover/public946-minimal-generalization-v1 \
  -p 'test_baseline_contract.py' -v

python handover/public946-minimal-generalization-v1/baseline_contract.py \
  --source /kaggle/working/biohub-harmonic-fusion.py
```

Use explicit path overrides after discovering actual local mounts. No install
or activation script is required to read this plan or run its standard-library
source tests. The measured inference study still requires the existing local
GPU runtime, original data and public model artifacts described in the repo.

## Completion means

A measured B0/B1 report, immutable source/config/model hashes, a baseline-control
notebook and one no-motion candidate notebook, offline image-to-CSV verification,
and a manual LB trial protocol. These are **outputs for Codex to produce**, not
artifacts already measured by the planner. A scientifically failed or blocked
run must still leave a report and next-agent entry point; never fabricate a score.

The aspirational target is **0.95+ LB**, not 0.95 local validation. A valid package
without a submitted Kaggle run is `ready_for_manual_lb_test`, not an LB improvement.
