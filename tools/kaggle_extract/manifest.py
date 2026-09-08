"""Sync state for the extractor.

`reference/manifest.json` records just enough to answer "what changed since
last sync?" so we never redownload a forum topic that hasn't had a new
comment, and never burn Claude tokens re-distilling unchanged content.

The manifest contains competition metadata, last_sync, per-page content hashes,
and forum topic metadata with optional body/distillation hashes.

All paths in the manifest are relative to the `reference/` directory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .forum import TopicMeta


SCHEMA_VERSION = 1


@dataclass
class Manifest:
    path: Path
    schema: int = SCHEMA_VERSION
    competition: dict[str, Any] = field(default_factory=dict)
    last_sync: str | None = None
    pages: dict[str, dict[str, Any]] = field(default_factory=dict)
    forum_topics: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ---- load / save ---------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if not path.exists():
            return cls(path=path)
        data = json.loads(path.read_text())
        return cls(
            path=path,
            schema=int(data.get("schema") or SCHEMA_VERSION),
            competition=data.get("competition") or {},
            last_sync=data.get("last_sync"),
            pages=data.get("pages") or {},
            forum_topics={str(k): v for k, v in (data.get("forum_topics") or {}).items()},
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.schema,
            "competition": self.competition,
            "last_sync": self.last_sync,
            "pages": self.pages,
            "forum_topics": self.forum_topics,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def touch_last_sync(self) -> None:
        self.last_sync = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ---- competition metadata ------------------------------------------

    def set_competition(
        self, slug: str, comp: dict[str, Any]
    ) -> None:
        self.competition = {
            "slug": slug,
            "id": comp.get("id"),
            "forum_id": comp.get("forumId"),
            "title": comp.get("title"),
            "deadline": comp.get("deadline"),
        }

    # ---- pages ---------------------------------------------------------

    def page_needs_refresh(self, filename: str, sha256: str) -> bool:
        entry = self.pages.get(filename)
        if entry is None:
            return True
        return entry.get("sha256") != sha256

    def record_page(self, filename: str, kaggle_id: int, sha256: str) -> None:
        self.pages[filename] = {"kaggle_id": kaggle_id, "sha256": sha256}

    # ---- forum topics --------------------------------------------------

    def topic_needs_refresh(self, meta: TopicMeta, local_file: Path | None = None) -> bool:
        """Refresh if:
        - we have no record of it, OR
        - the last comment date advanced, OR
        - the comment count changed, OR
        - the on-disk file was deleted.
        """
        entry = self.forum_topics.get(str(meta.id))
        if entry is None:
            return True
        if local_file is not None and not local_file.exists():
            return True
        if entry.get("last_comment_date") != meta.last_comment_date:
            return True
        if entry.get("comment_count") != meta.comment_count:
            return True
        return False

    def record_topic(
        self,
        meta: TopicMeta,
        local_file: str,
        body_sha256: str,
        distill_sha256: str | None = None,
    ) -> None:
        key = str(meta.id)
        prev = self.forum_topics.get(key, {})
        self.forum_topics[key] = {
            "title": meta.title,
            "author": meta.author,
            # Header metadata the renderer needs. cmd_bodies/cmd_topic build a
            # stub TopicMeta from the manifest when re-scraping, so carry the
            # prior value forward whenever the incoming meta leaves one blank —
            # otherwise a re-fetch would silently wipe the post date / tier.
            "author_type": meta.author_type or prev.get("author_type", ""),
            "author_tier": meta.author_tier or prev.get("author_tier", ""),
            "post_date": meta.post_date or prev.get("post_date", ""),
            "last_comment_date": meta.last_comment_date,
            "last_commenter": meta.last_commenter or prev.get("last_commenter", ""),
            "comment_count": meta.comment_count,
            "votes": meta.votes,
            "local_file": local_file,
            "body_sha256": body_sha256,
            "distill_sha256": distill_sha256 if distill_sha256 is not None
                              else prev.get("distill_sha256"),
        }

    def set_topic_distill(self, topic_id: int, distill_sha256: str) -> None:
        key = str(topic_id)
        if key in self.forum_topics:
            self.forum_topics[key]["distill_sha256"] = distill_sha256

    def topic_distill_needs_refresh(self, topic_id: int, current_input_sha: str) -> bool:
        entry = self.forum_topics.get(str(topic_id))
        if entry is None:
            return True
        return entry.get("distill_sha256") != current_input_sha

    # ---- reporting -----------------------------------------------------

    def status_summary(self) -> str:
        lines = [
            f"competition : {self.competition.get('slug')} "
            f"({self.competition.get('id')})",
            f"last_sync   : {self.last_sync or 'never'}",
            f"pages       : {len(self.pages)}",
            f"forum topics: {len(self.forum_topics)}",
        ]
        if self.forum_topics:
            distilled = sum(
                1 for v in self.forum_topics.values()
                if v.get("distill_sha256")
            )
            lines.append(f"  distilled : {distilled}")
        return "\n".join(lines)


__all__ = ["Manifest", "SCHEMA_VERSION"]
