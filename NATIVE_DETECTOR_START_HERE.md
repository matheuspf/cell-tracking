# Native-resolution detector — local Codex entry point

**Branch:** `handover/native-resolution-detector-v9-4090`  
**Base:** unexecuted v8 handover `e654c50c7543b3486556e2a1cd472f669f07a625`, itself based on `main@fb5521629eb41c8c485b291a5bcf344944c113ae`.  
**Machine:** one RTX 4090; approximately 65 GB RAM as reported by the user. Measure actual GiB and availability locally.  
**State:** plan and tested CPU contract helpers only; no trained detector or performance claim.

## Start

Open Codex at the repository root and give it this instruction:

> Read `AGENTS.md` and `handover/native-resolution-detector-v9-4090/CODEX_PROMPT.md`, then execute the v9 plan locally. Implement, train, evaluate and package the new native-resolution detector on this PC. Preserve embryo isolation and existing studies. Continue through the registered stages; do not stop after writing another plan or passing synthetic tests.

No activation patch, installation script, downloaded handover ZIP, or private conversation is required. The local agent handles environment/path discovery using the existing repository. A fresh clone still needs the competition data and suitable local training dependencies; this branch does not contain those large assets.

## Files

- [Execution plan](handover/native-resolution-detector-v9-4090/PLAN.md)
- [Codex instructions](handover/native-resolution-detector-v9-4090/CODEX_PROMPT.md)
- [Machine-readable study](handover/native-resolution-detector-v9-4090/study.json)
- [Read-only preflight](handover/native-resolution-detector-v9-4090/preflight.py)
- [State](handover/native-resolution-detector-v9-4090/STATUS.json)

The v8 directory is inherited historical design and helper material, NOT a second active execution plan. V9 overrides its run ordering, hardware limits, selection policy, localization-head default and resource fallback. Do not run both studies. Existing source, checkpoints and results are unchanged by this handover.
