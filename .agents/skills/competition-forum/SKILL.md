---
name: competition-forum
description: "Local forum index and on-demand topic retrieval for Biohub - Cell Tracking During Development. Use for organizer announcements, community discussion, data issues, or metric clarifications."
---
# Competition forum — Biohub - Cell Tracking During Development

Competition: `biohub-cell-tracking-during-development`. Reference snapshot: 2026-09-08T14:53:57Z.

99 topic metadata records are indexed in
[reference/forum/INDEX.md](../../../reference/forum/INDEX.md); search titles there to select relevant threads.
Metadata is not the thread body. Read cached text before drawing conclusions.

## Cached topic bodies

- [reference/forum/topics/716062-welcome-to-the-biohub-cell-tracking-during-development-.md](../../../reference/forum/topics/716062-welcome-to-the-biohub-cell-tracking-during-development-.md) — Welcome to the Biohub - Cell Tracking During Development Challenge
- [reference/forum/topics/716793-only-2-groups-of-embryo-id.md](../../../reference/forum/topics/716793-only-2-groups-of-embryo-id.md) — Only 2 groups of embryo_id？
- [reference/forum/topics/727154-division-metric-exploit-and-patch.md](../../../reference/forum/topics/727154-division-metric-exploit-and-patch.md) — Division Metric exploit and patch.
- [reference/forum/topics/728324-completed-rescore-underway.md](../../../reference/forum/topics/728324-completed-rescore-underway.md) — COMPLETED: Rescore Underway
- [reference/forum/topics/739018-question-about-the-node-count-adjustment-in-the-metric-.md](../../../reference/forum/topics/739018-question-about-the-node-count-adjustment-in-the-metric-.md) — Question about the node-count adjustment in the metric (adj_edge_jaccard can exceed 1)
- [reference/forum/topics/739686-metric-problem.md](../../../reference/forum/topics/739686-metric-problem.md) — Metric problem

## Refresh and fetch

Run from the repository root in the `cell-tracking` Conda environment:

```sh
PYTHONPATH=tools python -m kaggle_extract forum
PYTHONPATH=tools python -m kaggle_extract topic <id>
PYTHONPATH=tools python -m kaggle_extract skills
```

The topic command uses Playwright and writes Markdown plus a JSON sidecar under
`reference/forum/topics/`. Preserve the request throttles; fetch relevant topics
on demand. Distinguish organizer statements from participant suggestions. Optional
summary generation is available through `distill` after installing
`requirements-distill.txt`; use it only when requested.
