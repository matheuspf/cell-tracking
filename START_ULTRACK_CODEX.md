# Ultrack integration: local Codex entry point

Execute [handover/ultrack-integration-v7/CODEX_PROMPT.md](handover/ultrack-integration-v7/CODEX_PROMPT.md).
The single authoritative study is [PLAN.md](handover/ultrack-integration-v7/PLAN.md).

This is a **plan and diagnostic tooling**, not an executed tracking result.
It starts from merged `main` at `fb5521629eb41c8c485b291a5bcf344944c113ae`.
Earlier v5/v6 handovers remain historical inputs; do not launch their queues.
No overlay, activation patch, ZIP extraction, or changes to historical environments are required.

Read-only inventory, from any working directory:

```sh
python /path/to/cell-tracking/handover/ultrack-integration-v7/preflight.py \
  --repo /path/to/cell-tracking \
  --scratch /kaggle/working/cell-tracking/ultrack-integration-v7
```

The inventory can report missing dependencies or inadequate scratch space. It
installs nothing and does not establish that Ultrack is runnable. Codex must do
V700 provisioning and run the real synthetic pipeline before microscopy work.
