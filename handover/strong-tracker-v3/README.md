# Strong tracker v3 — preserve the winner, repair the disagreements

**Handover only. No v3 data experiment or score gain is claimed.**

Branch: `handover/strong-tracker-v3`, based on completed
`handover/strong-tracker-v2@a58c80b041db0a05262a58b3d1b5f34045b8d2ff`.
Start local Codex with the user's selected GPT 6 Pro model and give it
[CODEX_PROMPT.md](CODEX_PROMPT.md). No installation/activation overlay is needed.

The incumbent is **bypass_motion_bounds**, local score **0.9342063149703403**,
not the original 0.911774 Harmonic Fusion pipeline. The new goal is an additional
**+0.02** local improvement, with smaller verified improvements retained. This
is a research target, not a promise or a stop condition that justifies fabricating
success. Historical scores and division scenarios are in
[measured_baseline.json](measured_baseline.json) and [REVIEW.md](REVIEW.md).

## What changes in this iteration

1. Rebuild all graph-dependent features, candidates and labels on the actual v2
   winner. Old division/edge/selector models used the original final graph.
2. Arbitrate only where cached neural, old-final and winner graphs disagree;
   preserve stable agreeing associations instead of globally relinking again.
3. Expand temporally coherent division hypotheses and train on **division events**,
   including supported wrong-daughter alternatives, not annotation membership.
4. Add image-based localization/daughter rescue only in image-triggered uncertain
   regions. Do not spend another cycle on aggressive global annotation filtering.
5. Prove the selected changes work from raw images through the real notebook entry
   point, not only on cached graphs, before calling them deployment-ready.

Read [EXPERIMENTS.md](EXPERIMENTS.md) for V300–V370,
[IMPLEMENTATION.md](IMPLEMENTATION.md) for precise local contracts, and
[SOURCES.md](SOURCES.md) for evidence pins. [experiments.json](experiments.json)
fixes the initial small grid. Results must be compared to both v2 and original v1,
with v2 used for every promotion decision.

## Included executable helpers

```bash
PYTHONNOUSERSITE=1 python -m unittest discover -s handover/strong-tracker-v3 -p 'test_*.py' -v
python handover/strong-tracker-v3/preflight.py --help
python handover/strong-tracker-v3/scorecard.py scenarios
```

The helpers provide read-only cache checks, exact count arithmetic, and atomic
legal graph edits on synthetic/reference inputs. They do **not** implement a
trained v3 system or replace official matching/division evaluation. Local Codex
must build and run the experimental adapters specified in IMPLEMENTATION.md.

Preserve v1/v2 outputs and unrelated work. The historical v2 handover STATUS.json
still says planned; completed evidence is under `results/strong-tracker-v2/`.
Never restart the historical handover because of that stale authoring status.
The repository is public: raw images, detailed labels, predictions and weights
stay in ignored local storage. Only sanitized results and source belong in Git.
