---
name: competition-status
description: "Local sync state, deadlines, submissions, and leaderboard lookup for Biohub - Cell Tracking During Development. Use when asked about competition progress or refreshing the workspace."
---
# Competition status — Biohub - Cell Tracking During Development

Competition: `biohub-cell-tracking-during-development`. Reference snapshot: 2026-09-08T14:53:57Z.

- Deadline at snapshot: 2026-09-29T23:59:00Z
- Pages tracked: 8
- Forum topics tracked: 99

Read [reference/manifest.json](../../../reference/manifest.json) for sync state; do not hand-edit it.
Run from the repository root in the `cell-tracking` environment:

```sh
PYTHONPATH=tools python -m kaggle_extract status
PYTHONPATH=tools python -m kaggle_extract sync
kaggle competitions submissions biohub-cell-tracking-during-development
kaggle competitions leaderboard biohub-cell-tracking-during-development --show
```

Sync refreshes competition metadata, official pages, forum metadata, and skills.
Topic bodies are fetched on demand or through an explicitly bounded `bodies`
command. Local files do not establish current rank; use the live read-only CLI
commands when asked. This skill does not submit or publish anything.
