---
name: competition-data
description: "Input paths, Zarr image volumes, cell annotations, and submission schema for Biohub - Cell Tracking During Development. Use when inspecting the competition data or planning data loading."
---
# Competition data — Biohub - Cell Tracking During Development

Competition: `biohub-cell-tracking-during-development`. Reference snapshot: 2026-09-08T14:53:57Z.

- Official schema: [reference/overview/data-description.md](../../../reference/overview/data-description.md)
- Local inventory and observations: [docs/competition.md](../../../docs/competition.md)
- Data: [data/](../../../data/) → `/kaggle/input/competitions/biohub-cell-tracking-during-development`
- Notebook alias: `/kaggle/input/biohub-cell-tracking-during-development`
- Output root: `/kaggle/working/cell-tracking`

Inspect actual Zarr metadata and CSV headers before assuming dimensions, axis
order, coordinate units, tracking IDs, or output columns. Use
`python tools/inspect_data.py` for a lightweight inventory. Keep raw inputs
unchanged; place derived arrays and caches in the output root.

Read [reference/overview/evaluation.md](../../../reference/overview/evaluation.md) for the current output contract.
Download or resume the main data with `python tools/download_data.py`; regenerate
aliases with `python tools/setup_paths.py`. Run from the repository root in the
`cell-tracking` environment. Verify the completed download with
`python tools/download_data.py --verify-only`.
