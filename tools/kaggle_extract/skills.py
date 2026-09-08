"""Render the five competition skills for Codex and Claude from the local mirror."""

from dataclasses import dataclass
from pathlib import Path
import json

from .manifest import Manifest


@dataclass
class SkillSpec:
    name: str
    description: str
    body: str

    def render(self) -> str:
        description = json.dumps(self.description, ensure_ascii=False)
        return f"---\nname: {self.name}\ndescription: {description}\n---\n{self.body.rstrip()}\n"


def link(path: str) -> str:
    return f"[{path}](../../../{path})"


def write_all_skills(reference_dir: Path, manifest: Manifest, skills_dir: Path) -> list[Path]:
    comp = manifest.competition
    slug = comp["slug"]
    title = comp["title"]
    refreshed = manifest.last_sync or "never"
    pages = "\n".join(
        f"- {link('reference/overview/' + p.name)}"
        for p in sorted((reference_dir / "overview").glob("*.md"))
    )
    downloaded = []
    for tid, entry in manifest.forum_topics.items():
        local = entry.get("local_file")
        if local and entry.get("body_sha256") and (reference_dir / "forum" / local).exists():
            downloaded.append(f"- {link('reference/forum/' + local)} — {entry['title'].strip()}")
    topics = "\n".join(downloaded) or "No topic bodies cached yet; use the topic command below."
    command = "PYTHONPATH=tools python -m kaggle_extract"
    common = f"Competition: `{slug}`. Reference snapshot: {refreshed}."
    specs = [
        SkillSpec(
            "competition-overview",
            f"Goal, evaluation, timeline, and submission format for {title}. Use in the cell-tracking workspace for competition orientation and scoring questions.",
            f"""# Competition overview — {title}

{common}

Read the relevant official page before answering, and cite its local path or
source URL. Refresh time-sensitive facts when the snapshot may be stale.

{pages}

The local setup and observed input schema are in {link('docs/competition.md')}.
Refresh official pages with `{command} pages`, then regenerate these skills with
`{command} skills`. Run commands from the repository root in `conda activate cell-tracking`.
""",
        ),
        SkillSpec(
            "competition-data",
            f"Input paths, Zarr image volumes, cell annotations, and submission schema for {title}. Use when inspecting the competition data or planning data loading.",
            f"""# Competition data — {title}

{common}

- Official schema: {link('reference/overview/data-description.md')}
- Local inventory and observations: {link('docs/competition.md')}
- Data: {link('data/')} → `/kaggle/input/competitions/{slug}`
- Notebook alias: `/kaggle/input/{slug}`
- Output root: `/kaggle/working/cell-tracking`

Inspect actual Zarr metadata and CSV headers before assuming dimensions, axis
order, coordinate units, tracking IDs, or output columns. Use
`python tools/inspect_data.py` for a lightweight inventory. Keep raw inputs
unchanged; place derived arrays and caches in the output root.

Read {link('reference/overview/evaluation.md')} for the current output contract.
Download or resume the main data with `python tools/download_data.py`; regenerate
aliases with `python tools/setup_paths.py`. Run from the repository root in the
`cell-tracking` environment. Verify the completed download with
`python tools/download_data.py --verify-only`.
""",
        ),
        SkillSpec(
            "competition-rules",
            f"Official rules, external data allowances, team limits, and submission requirements for {title}. Use for questions about what is allowed in this competition.",
            f"""# Competition rules — {title}

{common}

Read {link('reference/overview/rules.md')} and the relevant overview page before
answering a rules question. Check the source date and refresh current rules when
needed with `{command} pages`. The official Kaggle rules take precedence over
local summaries.

Check external data and pretrained model allowances, team and entry deadlines,
daily submission limits, code runtime limits, and winner requirements as relevant.
Use {link('reference/forum/INDEX.md')} to find organizer clarifications; a
community post is not an organizer ruling. These skills do not authorize posting
to the forum, publishing notebooks, or submitting competition entries.
""",
        ),
        SkillSpec(
            "competition-forum",
            f"Local forum index and on-demand topic retrieval for {title}. Use for organizer announcements, community discussion, data issues, or metric clarifications.",
            f"""# Competition forum — {title}

{common}

{len(manifest.forum_topics)} topic metadata records are indexed in
{link('reference/forum/INDEX.md')}; search titles there to select relevant threads.
Metadata is not the thread body. Read cached text before drawing conclusions.

## Cached topic bodies

{topics}

## Refresh and fetch

Run from the repository root in the `cell-tracking` Conda environment:

```sh
{command} forum
{command} topic <id>
{command} skills
```

The topic command uses Playwright and writes Markdown plus a JSON sidecar under
`reference/forum/topics/`. Preserve the request throttles; fetch relevant topics
on demand. Distinguish organizer statements from participant suggestions. Optional
summary generation is available through `distill` after installing
`requirements-distill.txt`; use it only when requested.
""",
        ),
        SkillSpec(
            "competition-status",
            f"Local sync state, deadlines, submissions, and leaderboard lookup for {title}. Use when asked about competition progress or refreshing the workspace.",
            f"""# Competition status — {title}

{common}

- Deadline at snapshot: {comp.get('deadline', 'unknown')}
- Pages tracked: {len(manifest.pages)}
- Forum topics tracked: {len(manifest.forum_topics)}

Read {link('reference/manifest.json')} for sync state; do not hand-edit it.
Run from the repository root in the `cell-tracking` environment:

```sh
{command} status
{command} sync
kaggle competitions submissions {slug}
kaggle competitions leaderboard {slug} --show
```

Sync refreshes competition metadata, official pages, forum metadata, and skills.
Topic bodies are fetched on demand or through an explicitly bounded `bodies`
command. Local files do not establish current rank; use the live read-only CLI
commands when asked. This skill does not submit or publish anything.
""",
        ),
    ]
    written = []
    for spec in specs:
        target = skills_dir / spec.name / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(spec.render())
        written.append(target)
    return written
