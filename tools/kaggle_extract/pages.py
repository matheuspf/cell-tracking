"""Fetch and write competition overview/data/rules/etc pages.

Kaggle returns its competition pages in a slightly odd format:
- `content` is mostly markdown, but with inline HTML tags (<h3>, <h4>, <br>, etc.)
  and some unicode-escaped entities like \u0022 (").
- Link references use "rules#18.-terms" style relative links.

For our purposes — feeding to an LLM + humans — keeping the content mostly
as-is is fine; we just clean up the obvious escapes and record enough metadata
to do incremental sync.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from slugify import slugify

from .client import KaggleInternalClient

# Canonical mapping of Kaggle page `name` -> our local filename.
# Unknown names get slugified automatically.
PAGE_NAME_ALIASES = {
    "rules": "rules",
    "Description": "description",
    "Evaluation": "evaluation",
    "Timeline": "timeline",
    "Prizes": "prizes",
    "abstract": "abstract",
    "data-description": "data-description",
}


@dataclass
class FetchedPage:
    kaggle_id: int
    kaggle_name: str
    filename: str           # relative to reference/overview/
    markdown: str
    sha256: str

    def to_manifest(self) -> dict:
        return {
            "id": self.kaggle_id,
            "name": self.kaggle_name,
            "filename": self.filename,
            "sha256": self.sha256,
        }


def _local_filename(kaggle_name: str) -> str:
    if kaggle_name in PAGE_NAME_ALIASES:
        return PAGE_NAME_ALIASES[kaggle_name] + ".md"
    return slugify(kaggle_name) + ".md"


_HTML_BLOCK_TAGS = re.compile(r"</?(h[1-6]|div|p|span|br|hr)\b[^>]*>", re.I)


def _clean_content(raw: str) -> str:
    """Light normalization.

    We deliberately keep inline HTML tags around; markdown renderers and LLMs
    handle them fine. We only fix the escaped Unicode sequences Kaggle ships.
    """
    # Kaggle over-escapes quotes and angle brackets in the JSON payload.
    out = raw.replace(r"\u0022", '"').replace(r"\u003E", ">").replace(r"\u003C", "<")
    # Collapse CRLF to LF.
    out = out.replace("\r\n", "\n")
    return out.strip() + "\n"


def _render_page_markdown(page: dict) -> str:
    """Wrap the page content with a small header block."""
    body = _clean_content(page["content"])
    header = (
        f"# {page['name']}\n\n"
        f"> Kaggle page id `{page['id']}` (name `{page['name']}`). "
        f"Auto-synced from `competitions.PageService/ListPages`.\n\n"
        f"---\n\n"
    )
    return header + body


def fetch_all(
    client: KaggleInternalClient,
    competition_id: int,
    out_dir: Path,
) -> list[FetchedPage]:
    """Fetch every page and write it to out_dir. Returns metadata for manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = client.list_pages(competition_id)
    results: list[FetchedPage] = []
    for p in pages:
        md = _render_page_markdown(p)
        sha = hashlib.sha256(md.encode()).hexdigest()
        filename = _local_filename(p["name"])
        (out_dir / filename).write_text(md)
        results.append(
            FetchedPage(
                kaggle_id=p["id"],
                kaggle_name=p["name"],
                filename=filename,
                markdown=md,
                sha256=sha,
            )
        )
    return results


__all__ = ["FetchedPage", "fetch_all"]
