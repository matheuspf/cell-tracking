"""LLM-powered distillation of raw pages / forum topics.

For each raw markdown file we ask Claude to produce a small structured
summary: type, relevance, 1-2 sentence summary, key takeaways, cross-refs.

Design notes
- Model: `claude-sonnet-4-6` by default — used for the many small per-topic
  forum distillation steps in this repo. Overridable via the `model`
  argument (e.g. `claude-opus-4-8` for higher-fidelity overview distillation).
- Prompt caching: the shared system prompt (competition metadata + overview
  + evaluation + distillation instructions) is sent as a single cached
  system block. When distilling many forum topics in one run this reduces
  per-call cost dramatically.
- Structured output: `client.messages.parse(... output_format=Distillation)`
  with a Pydantic schema, so we never have to handle malformed JSON.
- Incremental: sha256 of the raw input is recorded in the manifest as
  `distill_sha256`; re-runs skip inputs whose hash matches.
- API key: env var `ANTHROPIC_API_KEY`, or fallback parse of `~/.zshrc`
  for a commented `#?export ANTHROPIC_API_KEY=...` line (the user keeps
  it commented out there on purpose).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

_ZSHRC = Path.home() / ".zshrc"
_ZSHRC_KEY_RE = re.compile(
    r"^\s*#?\s*export\s+ANTHROPIC_API_KEY\s*=\s*[\"']?([A-Za-z0-9_\-]+)[\"']?\s*$",
    re.MULTILINE,
)


def load_anthropic_api_key() -> str:
    """Return the Anthropic API key from env, falling back to ~/.zshrc.

    The user keeps the key as a *commented* export in ~/.zshrc so it isn't
    automatically loaded into their shell. We honor that by parsing the
    commented line rather than asking them to uncomment it.
    """
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env
    if _ZSHRC.exists():
        m = _ZSHRC_KEY_RE.search(_ZSHRC.read_text())
        if m:
            return m.group(1)
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found in environment and no commented "
        "`export ANTHROPIC_API_KEY=...` line in ~/.zshrc"
    )


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

RelevanceLevel = Literal["high", "medium", "low"]
ContentType = Literal[
    "announcement",
    "question",
    "code-share",
    "discussion",
    "off-topic",
    "overview",  # reserved for competition overview pages
]


class CrossRefs(BaseModel):
    notebooks: list[str] = Field(
        default_factory=list,
        description="Kaggle notebook / code titles or URLs this content references.",
    )
    datasets: list[str] = Field(
        default_factory=list,
        description="Kaggle datasets or external data sources referenced.",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Other forum topics referenced (title or id).",
    )
    models: list[str] = Field(
        default_factory=list,
        description="Named models / checkpoints / architectures referenced (e.g. XGBoost, a pretrained CNN backbone, a particle filter).",
    )


class Distillation(BaseModel):
    type: ContentType = Field(
        description=(
            "Content classification. Use 'announcement' for organizer posts "
            "or meta-updates; 'question' for help requests; 'code-share' "
            "when the post is primarily a notebook/script share; 'discussion' "
            "for general back-and-forth; 'off-topic' if it's not about this "
            "competition; 'overview' for static competition overview pages."
        ),
    )
    relevance: RelevanceLevel = Field(
        description=(
            "Practical relevance for a participant trying to build a "
            "winning LoRA adapter for this competition. 'high' = directly "
            "useful (metric clarifications, known pitfalls, working "
            "techniques, organizer announcements); 'medium' = tangential "
            "but useful; 'low' = noise, greetings, duplicate questions."
        ),
    )
    summary: str = Field(
        description="1-2 sentence plain-English summary. No fluff."
    )
    takeaways: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 5 concrete bullets a competitor should remember from "
            "this content. Prefer facts/decisions/warnings over opinions."
        ),
    )
    cross_refs: CrossRefs = Field(default_factory=CrossRefs)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTIONS = """\
You are an assistant distilling raw Kaggle competition material into a tiny
structured summary that will be indexed and referenced by other Claude Code
sessions working on this competition.

Rules:
- Be terse. Prefer facts over opinions.
- If the content is a competition overview/data/rules/evaluation page,
  classify it as `overview` and set relevance to `high`.
- For forum posts: `announcement` is for organizer/staff/host posts with
  competition-wide implications (rescoring, deadline changes, clarifications,
  rule updates). `code-share` is when the post is primarily sharing a
  notebook or script. `question` is for help requests. `discussion` is
  general back-and-forth. `off-topic` is anything not about this competition
  (generic career talk, greetings, model promos unrelated to the task).
- Relevance is for *a participant trying to build a winning LoRA adapter*.
  Don't just judge novelty — judge actionability.
- Takeaways should be concrete: "Metric now treats binary answers as
  strings, not floats (rescored 2026-04-08)" > "there was a metric change".
- Cross-refs: only include identifiers you actually saw in the content.
  Empty lists are fine.
- Keep `summary` to 1-2 sentences. Keep at most 5 takeaways.
"""


def _build_system_blocks(competition_context: str) -> list[dict]:
    """Build the system prompt as two blocks so we can cache the big one.

    Block 0: static instructions + competition context. Cached ephemeral.
    """
    return [
        {
            "type": "text",
            "text": (
                _SYSTEM_INSTRUCTIONS
                + "\n---\nCOMPETITION CONTEXT:\n---\n"
                + competition_context
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _load_competition_context(reference_dir: Path) -> str:
    """Assemble a compact competition context string from overview pages.

    Reads the small pages (abstract, description, evaluation, data-description,
    timeline, prizes) and joins them. Skipped: rules (too long) and compute.
    If a file is missing we silently skip it — first-run distillation may
    happen before every page has been fetched.
    """
    wanted = [
        "abstract.md",
        "description.md",
        "evaluation.md",
        "data-description.md",
        "timeline.md",
        "prizes.md",
    ]
    parts: list[str] = []
    overview_dir = reference_dir / "overview"
    for name in wanted:
        p = overview_dir / name
        if p.exists():
            parts.append(p.read_text())
    if not parts:
        return "(no overview pages fetched yet)"
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Distiller
# ---------------------------------------------------------------------------


@dataclass
class DistillResult:
    input_sha256: str
    distillation: Distillation
    model: str

    def to_json(self) -> dict:
        return {
            "input_sha256": self.input_sha256,
            "model": self.model,
            "distillation": self.distillation.model_dump(),
        }


class Distiller:
    """Thin wrapper around `anthropic.Anthropic` for batch distillation.

    Builds the competition context once from `reference_dir`, caches it,
    then exposes `distill(text, label)` which runs one API call per input.
    """

    def __init__(
        self,
        reference_dir: Path,
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
    ):
        # Lazy import so `python -m kaggle_extract pages` doesn't need
        # the anthropic package installed.
        import anthropic  # noqa: F401

        self.reference_dir = reference_dir
        self.model = model
        self.max_tokens = max_tokens
        self._context_text = _load_competition_context(reference_dir)
        self._system = _build_system_blocks(self._context_text)
        self._client = anthropic.Anthropic(api_key=load_anthropic_api_key())

    @staticmethod
    def input_hash(text: str, *, model: str) -> str:
        """Hash raw input + model + prompt version so a model or prompt
        change invalidates cached distillations.
        """
        h = hashlib.sha256()
        h.update(b"distill-v1\0")
        h.update(model.encode())
        h.update(b"\0")
        h.update(text.encode())
        return h.hexdigest()

    def distill(self, text: str, *, label: str = "") -> DistillResult:
        """Distill one raw markdown document into a structured summary."""
        label_line = f"(label: {label})\n" if label else ""
        user_content = (
            f"{label_line}Distill the following content into the required "
            f"structured schema.\n\n---\n\n{text.strip()}\n"
        )
        response = self._client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system,
            messages=[{"role": "user", "content": user_content}],
            output_format=Distillation,
        )
        return DistillResult(
            input_sha256=self.input_hash(text, model=self.model),
            distillation=response.parsed_output,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Writing distillations alongside source files
# ---------------------------------------------------------------------------


def write_distill_sidecar(source_path: Path, result: DistillResult) -> Path:
    """Write `<source>.distill.json` next to the source markdown."""
    out = source_path.with_suffix(source_path.suffix + ".distill.json")
    out.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True))
    return out


def render_distilled_line(result: DistillResult) -> str:
    """One-line rendering of a distillation for an index page."""
    d = result.distillation
    return f"[{d.type}/{d.relevance}] {d.summary}"


__all__ = [
    "Distillation",
    "CrossRefs",
    "DistillResult",
    "Distiller",
    "load_anthropic_api_key",
    "write_distill_sidecar",
    "render_distilled_line",
]
