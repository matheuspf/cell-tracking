# Image-native tracking v5

**Execution handover, not measured v5 results.** Begin with [CODEX_PROMPT.md](CODEX_PROMPT.md).

This is a new iteration from completed `handover/multidata-training-v4` at
`cec119434f7001ea79beeb137a8f1063e5b0deeb`. V4 did not promote an external-data
model. Preserve **C0 / v3 A_residual_m3.0 = 0.934802374260586**. The goal is a
complete local score **at least 0.95**, a gain of 0.015197625739414, with replication
and both embryos reported. This is a research goal, not a forecast or a leaderboard claim.

## The change of direction

Stop scaling the small synthetic-to-real fork classifiers, global annotation
filters, and six-next-frame-candidate threshold sweeps. Test new *representations*
and a genuinely different predictor:

1. **H — pretrained HOCT:** link-centric multi-frame transformer, image-derived
   object features, then sparse real-edge adaptation. It is not the v4 geometry MLP.
2. **N — native 3D network adaptation:** actually fine-tune the installed strong
   U-Net/transformer on real images and supported transitions. Fixed-node controls
   isolate learned association quality before any detector changes.
3. **P — new observation hypotheses:** full-frame new peaks and competing one-cell/
   two-cell regions, with joint temporal selection. V4 only refined existing centers.

Use one explicit temporal graph objective across representation comparisons.
Do not preserve the old free-birth/fork-dominated objective, and do not force new
models through v4's geometry gate. The incumbent remains a valid no-op explanation.

## Start

```bash
git fetch origin
git switch --track origin/handover/image-native-tracking-v5
```

Give local Codex, using the user's selected GPT 6 Pro model:

```text
Read handover/image-native-tracking-v5/CODEX_PROMPT.md and execute it.
Implement and run X500-X580 using the existing data, checkpoints, and one RTX 4090.
Target >=0.95 against C0=0.934802374260586. Do not rerun v4 or optimize its small heads.
Run the HOCT/native-network/new-observation tests, preserve C0, and report failures honestly.
Commit and push sanitized results and CONTINUATION.md to this v5 branch.
```

No activation script, ZIP, PR merge or full data download is needed. New experiment
modules are work for Codex, not already-implemented pipeline commands.

Read [REVIEW.md](REVIEW.md), [EXPERIMENTS.md](EXPERIMENTS.md),
[IMPLEMENTATION.md](IMPLEMENTATION.md), [SOURCES.md](SOURCES.md) and `config.json`.
The supplied `contracts.py`, `test_contracts.py`, and `preflight.py` run without
microscopy, pretrained weights, or GPU access:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s handover/image-native-tracking-v5 -p 'test_*.py' -v
python handover/image-native-tracking-v5/preflight.py --repo .
```

Preflight is read-only and its local-cache findings are not a new data audit. No
user images, GT tables, predictions, weights or credentials belong in public Git.
