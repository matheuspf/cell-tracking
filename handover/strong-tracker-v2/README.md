# Strong-tracker v2: repair divisions and associations before aggressive pruning

**Status: reviewed and planned, not executed.** Based on the completed v1 results
at `matheuspf/cell-tracking@f14c9eb368a80ce4cc7aaebae016fb182b1b6978`.
Branch: `handover/strong-tracker-v2`, based on `handover/annotation-selection-v1`.
This adds only `handover/strong-tracker-v2/`; it does not replace any v1 file.

From your existing repository checkout:

```sh
git fetch origin
git switch --track origin/handover/strong-tracker-v2
```

In local Codex, read and execute `handover/strong-tracker-v2/CODEX_PROMPT.md`.

The central change is to optimize the actual strong Harmonic Fusion graph, not
the classical feasibility tracker. Keep v1's sealed outputs immutable. Reuse its
cached neural graphs, candidate scores, final graphs and official metric adapter.

Read [REVIEW.md](REVIEW.md) for the diagnosis and [EXPERIMENTS.md](EXPERIMENTS.md)
for the local execution plan. Give Codex [CODEX_PROMPT.md](CODEX_PROMPT.md), with
the user's selected model, to implement and execute the next experiments.
No activation script, full bootstrap, model download or full inference rerun is
required to begin the cached-graph analysis.

```sh
PYTHONNOUSERSITE=1 python -m unittest discover \
  -s handover/strong-tracker-v2 -p 'test_*.py' -v
python handover/strong-tracker-v2/headroom.py
```

The 16 tests check arithmetic and the recorded baseline consistency. They do not
run graph evaluation or microscopy experiments. `calculated_scenarios.json`
contains calculated requirements, never measured v2 gains.

## Deliverables for local Codex

First produce a per-stage and per-division error census. Then execute frozen-node
division repair, local association repair, and a native-candidate filtering study.
Finish with a side-by-side official-score report against the exact strong baseline,
including per-embryo results, joint ablations and an explicit provenance limitation.
Do not restart v1, switch back to the weak classical tracker as the main target,
or stop after writing another plan.

## Important measured baseline

Harmonic Fusion local diagnostic: score **0.911774**, raw edge Jaccard **0.900072**,
adjusted edge **0.902500**, division Jaccard **0.092742**. Division counts are
**23 TP, 97 FP, 128 FN**. GT center coverage is **130,959 / 133,318 = 98.2305%**.
The classifier transferred from classical detections reduced this baseline to
**0.873322**. None of 72 learned/confidence settings improved it in aggregate.

The public checkpoint lane is contaminated, and both embryos have already been
examined. New local gains are exploratory, not proof of a hidden-test uplift.
The quoted public leaderboard value 0.946 is not this local baseline's score.
