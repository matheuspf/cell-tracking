# Lineage reconstruction v5 — replace the tracking bottleneck

Status: **plan, not executed**. Continue from completed v4, not from its original prompt.

Current incumbent: v3 `A_residual_m3.0` / v4 `C0`, official local score
**0.934802374260586** over all 199 supplied clips. Target: **at least 0.95 on that
same local population**, not an asserted leaderboard score. Preserve the incumbent.

This is a change of approach: benchmark and adapt pretrained 3D association
backbones, construct actual image-supported instance regions and new detections,
and solve complete lineage alternatives. Do not spend another iteration enlarging
the old D/G/I event heads, global annotation filters, or fixed-margin sweeps.

## Run with local Codex

Read [CODEX_PROMPT.md](CODEX_PROMPT.md) and execute [EXPERIMENTS.md](EXPERIMENTS.md).
The new code is to live in `tools/lineage_reconstruction_v5/` with explicit paths;
it does not exist yet. Reference helpers in this directory already run:

```bash
PYTHONNOUSERSITE=1 python -m unittest discover -s handover/lineage-reconstruction-v5 -p 'test_*.py' -v
python handover/lineage-reconstruction-v5/preflight.py
python handover/lineage-reconstruction-v5/target_budget.py
```

A checkout is sufficient to hand the work to Codex. There is no activation patch,
ZIP overlay, old queue restart or environment reinstall step. New optional
third-party dependencies go in isolated environments during implementation.

## Read order

1. [REVIEW.md](REVIEW.md): what failed in v4 and what was not tested.
2. [EXPERIMENTS.md](EXPERIMENTS.md): P500–P580, budgets and continuation logic.
3. [IMPLEMENTATION.md](IMPLEMENTATION.md): adapters, supervision, decoder and tests.
4. [VALIDATION.md](VALIDATION.md): source isolation, promotion and claim limits.
5. [SOURCES.md](SOURCES.md): inspected primary sources and exact versions.

`experiments.json`, `model_registry.json` and `baseline.json` are machine-readable
contracts. `reference_contracts.py` implements tiny, tested reference operations,
not a complete tracker or official scorer. `preflight.py` is read-only.

Required local outputs include a working image-to-graph runner, trained/adapted
weights where applicable, all complete score rows, a report, offline dashboard,
package manifest and `NEXT_AGENT.md`. Commit only sanitized code/configs/results.

The previous seven-hour execution window expired on September 10 at 06:09:09 UTC.
Do not copy that deadline, reset old queues, or assume another seven-hour mandate.
The v5 compute caps are new engineering bounds, not predicted completion times.
