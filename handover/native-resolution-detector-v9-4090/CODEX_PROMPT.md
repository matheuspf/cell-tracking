# Execute native-resolution-detector-v9-4090

You are the local implementation and experiment agent. The user authorized a new model-training study on their own PC: one RTX 4090 and approximately 65 GB RAM. The planning agent created this branch; YOU implement and execute locally. No model was trained during handover authoring.

Read, in order:
1. Root `AGENTS.md` and `NATIVE_DETECTOR_START_HERE.md`.
2. All `.agents/skills/competition-*/SKILL.md`, their relevant local official references, `docs/competition.md`, and `docs/notebooks.md`.
3. This directory's `PLAN.md`, `study.json`, `STATUS.json`, and helper scripts/tests.
4. The precise source and evidence paths in PLAN section 14; inspect actual data and installed upstream assets before interpreting results.

V9 is the sole active plan. V8 is inherited context and reusable code, not an instruction to switch branches or run a second study. Stay on `handover/native-resolution-detector-v9-4090`; do not merge other branches. Check the worktree before writing; preserve unrelated modifications and never reset/clean/stash the user's files automatically.

Start with these existing, dependency-light commands from the repository root:

```bash
PYTHONNOUSERSITE=1 python -m unittest discover -s handover/native-resolution-detector-v9-4090 -p 'test_*.py' -v
PYTHONNOUSERSITE=1 python handover/native-resolution-detector-v9-4090/preflight.py
```

The preflight prints JSON and exits nonzero when local inputs are absent. It does not install, train, read image chunks/GEFF labels, authenticate, kill processes, alter symlinks or write outputs. Discover an existing suitable Python environment and data mount yourself; pass `--data-root PATH --output-root PATH` when necessary. Do not interpret absent `/kaggle` aliases as absent data without inspecting repository configuration. All further CLI commands in PLAN are interfaces YOU must implement, not claimed pre-existing trainers.

Implement `tools/native_detector_v9/` and its tests. Train a randomly initialized whole-field center detector using acquired ZYX=(64,256,256) samples. Full-volume first; OOM fallback means native crops with halos, never resizing the primary input. The fixed primary is N_ema; the mandatory supervised anchor is N_base. Use source-only teachers, masked unknown labels, two embryo directions, fixed schedules and both prescribed seeds. Do not use exposed incumbent predictions, old synthetic teachers or target-image adaptation in primary training.

Do real source-image learning after geometry/loss tests. Prioritize paired native fits and matched coarse/lowpass controls over optional architecture breadth. Finish full inference coverage before calling a comparison complete. Freeze BOTH directions, all selected checkpoints, decoding and predictions before new outer scores. Never adapt later stages to an outer leaderboard, choose a lucky seed, or call random clip splits embryo-disjoint.

Measure actual GPU/RAM/disk consumption. Respect the default 48 active GPU-hour study budget and reservation policy in study.json; these are experiment caps, not wall-clock deadlines. Use bounded caches, durable atomic checkpoints, exact resume and sequential GPU work. Resume existing identical-run checkpoints rather than restarting after interruptions. Defer optional work before starving mandatory finals. A negative early score is not a stop condition.

Preserve the inherited tracker. Evaluate detector-only transfer first, then fixed-linker integration with newly sampled features and regenerated edges. Report exposed-linker results separately. Do not claim new LB performance, silently replace C0/P0, accept competition terms, upload data, submit to Kaggle, rent hardware or launch paid services.

Execute D900-D980 and maintain an atomic progress ledger. Technical blockers close only dependent stages where possible. Spend at most three documented repairs on the same technical failure; retain diagnosis and run independent feasible arms. If the resource budget ends, save usable checkpoints, complete reserved mandatory evaluation where feasible, mark missing comparisons honestly, and write exact resume commands.

Deliver actual trained checkpoint paths/hashes, a usable raw-Zarr inference CLI, reproducible configs, all-clip detector metrics, separate graph results, a visual report using real evidence, and CONTINUATION.md. Commit/push code/configs/tests/sanitized summaries on THIS branch with explicit path allowlists; never `git add .`. Large weights/predictions/raw images/references/credentials remain ignored locally. No more planning-only handoff as the terminal result when real training is feasible.
