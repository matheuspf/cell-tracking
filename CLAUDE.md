# Biohub — Cell Tracking During Development

Read `AGENTS.md` for workspace conventions and the relevant
`.claude/skills/competition-*` skill for competition knowledge. The five skills
are generated identically for Codex and Claude by `tools/kaggle_extract/skills.py`.

Use Conda `cell-tracking` and `PYTHONNOUSERSITE=1`. Inputs live under
`/kaggle/input/competitions/biohub-cell-tracking-during-development`, exposed as
repo `data/`. Outputs belong in `/kaggle/working/cell-tracking` or ignored `work/`.

Refresh the local reference mirror from the repo root with
`PYTHONPATH=tools python -m kaggle_extract sync`. Fetch individual forum topics
with its `topic <id>` command. Preserve request throttles and keep credentials
and downloaded artifacts out of Git.
