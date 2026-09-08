"""Fetch forum topic list + per-topic bodies.

Two-layer design:

1. `list_all_topics(client, forum_id, max_pages)` — uses Kaggle's internal RPC
   to paginate through topic metadata. Fast and reliable.

2. `fetch_topic_body(slug, topic_id)` — uses Playwright to render the topic
   page in headless Chromium and scrape the original post plus all comments.
   This is the fallback for the fact that we haven't yet found an RPC for
   message bodies. We load with Kaggle basic auth via the HTTP-auth dialog
   workaround: pre-set an Authorization header on every request.

Topic markdown files are written to `reference/forum/topics/<id>-<slug>.md`
and a per-topic JSON sidecar stores the structured data (for incremental
sync / distillation).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_md
from slugify import slugify

from .client import KaggleInternalClient, KAGGLE_BASE, _load_credentials


@dataclass
class TopicMeta:
    """Metadata returned by the list endpoint (no message bodies)."""
    id: int
    title: str
    author: str
    author_type: str
    author_tier: str
    comment_count: int
    votes: int
    is_sticky: bool
    post_date: str
    last_comment_date: str
    last_commenter: str
    topic_url: str
    first_message_id: int | None = None

    @classmethod
    def from_rpc(cls, t: dict[str, Any]) -> "TopicMeta":
        author_user = t.get("authorUser") or {}
        return cls(
            id=t["id"],
            title=t.get("title") or "",
            author=author_user.get("displayName") or "",
            author_type=t.get("authorType") or "",
            author_tier=author_user.get("tier") or "",
            comment_count=t.get("commentCount") or 0,
            votes=t.get("votes") or 0,
            is_sticky=bool(t.get("isSticky")),
            post_date=t.get("postDate") or "",
            last_comment_date=t.get("lastCommentPostDate") or "",
            last_commenter=t.get("lastCommenterName") or "",
            topic_url=t.get("topicUrl") or "",
            first_message_id=t.get("firstForumMessageId"),
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "author_type": self.author_type,
            "author_tier": self.author_tier,
            "comment_count": self.comment_count,
            "votes": self.votes,
            "is_sticky": self.is_sticky,
            "post_date": self.post_date,
            "last_comment_date": self.last_comment_date,
            "last_commenter": self.last_commenter,
            "topic_url": self.topic_url,
        }

    @property
    def local_filename(self) -> str:
        # 55-char title cap keeps filenames readable.
        slug = slugify(self.title or f"topic-{self.id}")[:55] or f"topic-{self.id}"
        return f"{self.id}-{slug}.md"


@dataclass
class Message:
    message_id: int | None
    author: str
    author_tier: str
    post_date: str
    body_markdown: str
    votes: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "author": self.author,
            "author_tier": self.author_tier,
            "post_date": self.post_date,
            "votes": self.votes,
            "body_markdown": self.body_markdown,
        }


@dataclass
class Topic:
    meta: TopicMeta
    messages: list[Message] = field(default_factory=list)

    @property
    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(self.meta.title.encode())
        h.update(self.meta.last_comment_date.encode())
        for m in self.messages:
            h.update((m.body_markdown or "").encode())
            h.update((m.post_date or "").encode())
        return h.hexdigest()

    def to_markdown(self) -> str:
        meta = self.meta
        lines = [
            f"# {meta.title}",
            "",
            f"> Topic id `{meta.id}` — [{meta.topic_url}]"
            f"({KAGGLE_BASE}{meta.topic_url})",
            f"> Started by **{meta.author}** ({meta.author_type or 'USER'}"
            f"{', ' + meta.author_tier if meta.author_tier else ''}) "
            f"on {meta.post_date}.",
            f"> {meta.comment_count} comments · {meta.votes} votes"
            f"{' · STICKY' if meta.is_sticky else ''}"
            f" · last activity {meta.last_comment_date} by {meta.last_commenter}.",
            "",
            "---",
            "",
        ]
        if not self.messages:
            lines.append("_(Message bodies unavailable — scraper failed to load this topic.)_")
            lines.append("")
        for i, m in enumerate(self.messages):
            role = "Original post" if i == 0 else f"Reply {i}"
            header = (
                f"## {role} — {m.author}"
                f"{' (' + m.author_tier + ')' if m.author_tier else ''}"
                f"  ·  {m.post_date}"
            )
            lines.append(header)
            if m.votes:
                lines.append(f"_{m.votes} votes_")
            lines.append("")
            lines.append(m.body_markdown.strip() or "_(empty)_")
            lines.append("")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 1. Topic list via RPC
# --------------------------------------------------------------------------

def list_all_topics(
    client: KaggleInternalClient,
    forum_id: int,
    max_pages: int = 20,
    page_size: int = 20,
) -> list[TopicMeta]:
    """Walk the paginated topic list, stopping when the server returns fewer
    than `page_size` entries or we've already seen every id.
    """
    seen: set[int] = set()
    out: list[TopicMeta] = []
    for page in range(1, max_pages + 1):
        resp = client.list_forum_topics(forum_id, page=page, limit=page_size)
        topics = resp.get("topics") or []
        if not topics:
            break
        new_count = 0
        for t in topics:
            if t["id"] in seen:
                continue
            seen.add(t["id"])
            out.append(TopicMeta.from_rpc(t))
            new_count += 1
        if len(topics) < page_size or new_count == 0:
            break
    return out


# --------------------------------------------------------------------------
# 2. Topic body via Playwright
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_DATE_TITLE_RE = re.compile(
    r"^(?:Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+\s+\d{4}"
)
_RELATIVE_TIME_RE = re.compile(
    r"^Posted\s+\S+\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+ago$",
    re.I,
)
# Leaderboard-rank byline, e.g. "· 934th in this Competition" or
# "· 1st in this Competition". These appear as standalone <p> elements
# next to the body <p>, and are metadata, not content.
_RANK_META_RE = re.compile(
    r"^\s*[·•]?\s*\d+(?:st|nd|rd|th)\s+in\s+this\s+Competition\s*$",
    re.I,
)
# Material icon glyph text names Kaggle renders — these are links/buttons
# whose .text() returns the glyph name, not a real author name.
_ICON_TEXTS = {
    "push_pin",
    "arrow_drop_up",
    "arrow_drop_down",
    "more_vert",
    "thumb_up",
    "thumb_down",
    "reply",
    "edit",
    "delete",
    "share",
}
_KNOWN_TIERS = {
    "Kaggle Staff",
    "Grandmaster",
    "Master",
    "Expert",
    "Contributor",
    "Novice",
    "Admin",
    "Host",
}


def _clean_text(s: str) -> str:
    return _WS_RE.sub(" ", s or "").strip()


def _in_nested_comment(el: Any, root: Any) -> bool:
    """True if `el` lives inside a `discussions-comment` that is itself a
    descendant of `root` — i.e. a threaded reply nested under this comment.

    Kaggle renders reply chains by NESTING child comment nodes inside their
    parent's DOM subtree. `find_all("[data-testid=discussions-comment]")`
    therefore returns both the parent and its nested children. Without this
    guard, `_extract_container(parent)` would pick the *smallest* descendant
    body div — which belongs to the deepest nested child — and every ancestor
    in the chain would emit that same child's text (the duplicate-body bug).
    Each comment node is still emitted separately (find_all visits it too); we
    only restrict body/author extraction to a node's OWN, non-nested content.
    """
    p = el.parent
    while p is not None and p is not root:
        get = getattr(p, "get", None)
        if get is not None and get("data-testid") == "discussions-comment":
            return True
        p = p.parent
    return False


def _extract_container(node: Any) -> Message | None:
    """Pull author/date/tier/body out of a topic-header or comment container.

    Kaggle's class names are emotion/styled-components hashes and change each
    deploy, so we use structural heuristics:
    - Author: first `<a href="/<slug>">` inside the container (skipping link
      wrappers around the avatar image).
    - Date: first element with a `title="Sun Apr 04 …"` attribute.
    - Tier: any of a fixed set of tier strings appearing in the metadata.
    - Body: among all descendant `<div>`s that contain at least one `<p>`,
      the one with the smallest total text length — this is the innermost
      wrapper around the markdown body, excluding the metadata row.
    """
    # --- author --------------------------------------------------------
    # Walk all user-link anchors; skip ones whose text is a material-icon
    # glyph (push_pin, more_vert, …). Prefer the first real text; otherwise
    # fall back to the href slug.
    author = ""
    author_slug = ""
    for a in node.find_all("a", href=True):
        if _in_nested_comment(a, node):
            continue
        href = a["href"]
        if not re.match(r"^/[^/]+$", href) or href == "/":
            continue
        if not author_slug:
            author_slug = href.lstrip("/")
        text = _clean_text(a.get_text())
        if text and text not in _ICON_TEXTS:
            author = text
            break
    if not author:
        author = author_slug

    # --- date ----------------------------------------------------------
    post_date = ""
    for el in node.find_all(True, title=True):
        t = el.get("title") or ""
        if _DATE_TITLE_RE.match(t):
            post_date = t
            break

    # --- tier ----------------------------------------------------------
    tier = ""
    head_text = _clean_text(node.get_text(" ", strip=True))[:400]
    for k in _KNOWN_TIERS:
        if k in head_text:
            tier = k
            break

    # --- body ----------------------------------------------------------
    # Among all `<div>`s that contain at least one `<p>`, reject candidates
    # whose subtree contains a metadata <p> (either "Posted 3 days ago" OR
    # a standalone leaderboard-rank byline "· 934th in this Competition").
    # The rank byline is what tripped us up before — ranked users get a
    # separate rank <p> sibling to their body <p>, and without filtering
    # it out the rank line wins the "smallest text" tiebreak over the body.
    #
    # With both metadata patterns filtered, the remaining candidates are
    # pure-body wrappers nested at various depths; the innermost (smallest
    # total text) is the most specific body element. For OPs/replies where
    # the "Posted ago" text lives in a non-<p> element (so the outer
    # wrapper isn't auto-rejected), smallest-tlen still picks the tightest
    # body-only subtree among the nested wrappers.
    body_el = None
    candidates: list[tuple[int, int, Any]] = []
    for d in node.find_all("div"):
        if _in_nested_comment(d, node):
            continue
        ps = d.find_all("p")
        if not ps:
            continue
        has_metadata_p = False
        for p in ps:
            pt = _clean_text(p.get_text(" ", strip=True))
            if _RELATIVE_TIME_RE.match(pt) or _RANK_META_RE.match(pt):
                has_metadata_p = True
                break
        if has_metadata_p:
            continue
        text = _clean_text(d.get_text(" ", strip=True))
        if not text:
            continue
        candidates.append((len(text), len(ps), d))
    if candidates:
        # Innermost pure-body wrapper wins: smallest tlen, ties broken by
        # most <p>s so multi-paragraph bodies stay intact.
        candidates.sort(key=lambda x: (x[0], -x[1]))
        body_el = candidates[0][2]
    body_md = ""
    if body_el is not None:
        body_md = html_to_md(
            str(body_el),
            heading_style="ATX",
            bullets="-",
            strip=["style", "script"],
        ).strip()

    if not body_md and not author:
        return None
    return Message(
        message_id=None,
        author=author,
        author_tier=tier,
        post_date=post_date,
        body_markdown=body_md,
    )


def _extract_messages_from_html(html: str) -> list[Message]:
    """Parse a rendered discussion page and return OP + comments in order."""
    soup = BeautifulSoup(html, "html.parser")
    messages: list[Message] = []
    hdr = soup.find(attrs={"data-testid": "discussions-topic-header"})
    if hdr is not None:
        msg = _extract_container(hdr)
        if msg is not None:
            messages.append(msg)
    for node in soup.find_all(attrs={"data-testid": "discussions-comment"}):
        msg = _extract_container(node)
        if msg is not None:
            messages.append(msg)
    return messages


class TopicBodyScraper:
    """Reusable Playwright context for scraping many topic pages politely.

    Design goals:
    - ONE browser + ONE context for the whole run (avoids cold starts).
    - A minimum delay between page loads (default 2 seconds) so we never
      burst Kaggle — opening a discussion page renders a lot of content and
      we don't want to look like an abuser.
    - Basic Auth via http_credentials so we see the logged-in version of
      each page (anonymous pageviews can truncate long threads).
    """

    def __init__(
        self,
        competition_slug: str,
        *,
        headless: bool = True,
        wait_ms: int = 8000,
        min_interval_s: float = 2.0,
    ) -> None:
        self.competition_slug = competition_slug
        self.headless = headless
        self.wait_ms = wait_ms
        self.min_interval_s = min_interval_s
        self._pw = None
        self._browser = None
        self._username: str = ""
        self._key: str = ""
        self._last_fetch_t: float = 0.0
        self._fetch_count: int = 0

    def __enter__(self) -> "TopicBodyScraper":
        from playwright.sync_api import sync_playwright  # lazy import

        self._username, self._key = _load_credentials()
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        # We create a fresh context per fetch (see _new_context) instead of
        # sharing one across the whole run. Reused contexts hit the HTTP
        # cache aggressively on Kaggle's SPA — the XHR that loads the
        # discussion content is served from cache, networkidle fires before
        # content is injected, and we end up parsing an empty shell.
        return self

    def _new_context(self):
        assert self._browser is not None
        # IMPORTANT: do NOT override user_agent. Kaggle serves a degraded
        # SPA shell (no forum XHR, no comments) when the UA is anything
        # other than a default browser string. We authenticate with Basic
        # auth + are rate-limited via min_interval_s — that's sufficient
        # politeness without a custom UA.
        return self._browser.new_context(
            http_credentials={"username": self._username, "password": self._key},
        )

    def __exit__(self, exc_type, exc, tb) -> None:
        with contextlib.suppress(Exception):
            if self._browser is not None:
                self._browser.close()
            if self._pw is not None:
                self._pw.stop()

    @property
    def fetch_count(self) -> int:
        return self._fetch_count

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self.min_interval_s - (now - self._last_fetch_t)
        if wait > 0:
            time.sleep(wait)
        self._last_fetch_t = time.monotonic()
        self._fetch_count += 1

    def fetch(self, topic_id: int) -> list[Message]:
        if self._browser is None:
            raise RuntimeError("TopicBodyScraper must be used as a context manager")
        self._throttle()
        url = f"{KAGGLE_BASE}/competitions/{self.competition_slug}/discussion/{topic_id}"
        context = self._new_context()
        try:
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded")
            # networkidle alone is unreliable: on the 2nd+ fetch in a
            # session, the browser-wide HTTP cache serves static assets
            # quickly and networkidle fires *before* the discussion XHR
            # finishes. So we also explicitly wait for the topic-header
            # element to appear in the DOM, which is only injected once
            # the SPA's discussion content has rendered.
            with contextlib.suppress(Exception):
                page.wait_for_load_state("networkidle", timeout=self.wait_ms)
            with contextlib.suppress(Exception):
                page.wait_for_selector(
                    '[data-testid="discussions-topic-header"]',
                    timeout=self.wait_ms,
                )
            # Scroll-and-poll until the rendered discussions-comment count
            # stops growing AND is non-zero (otherwise two consecutive
            # zeros would be treated as "done" during a slow load). Kaggle
            # lazy-loads comment bodies on scroll, so we keep scrolling
            # until the count stabilizes, with a hard ceiling.
            last_count = -1
            steady = 0
            saw_nonzero = False
            for _ in range(15):
                page.mouse.wheel(0, 25000)
                page.wait_for_timeout(400)
                count = page.evaluate(
                    "() => document.querySelectorAll("
                    "'[data-testid=\"discussions-comment\"]').length"
                )
                if count > 0:
                    saw_nonzero = True
                if saw_nonzero and count == last_count:
                    steady += 1
                    if steady >= 2:
                        break
                else:
                    steady = 0
                last_count = count
            # Kaggle collapses threaded reply chains behind a "N more replies"
            # button that the scroll loop above never reveals. Click every such
            # button — re-querying after each click, since expanding one thread
            # can surface nested ones — until none remain, re-scrolling between
            # clicks so the newly injected reply bodies lazy-render. Capped so a
            # button that never disappears can't spin forever.
            for _ in range(40):
                more = page.locator(
                    "button", has_text=re.compile(r"more repl", re.I)
                ).filter(visible=True).first
                try:
                    if more.count() == 0:
                        break
                    more.scroll_into_view_if_needed(timeout=2000)
                    more.click(timeout=2000)
                    page.wait_for_timeout(500)
                    page.mouse.wheel(0, 20000)
                    page.wait_for_timeout(300)
                except Exception:
                    break
            # One scroll-to-top then scroll-back-down so the first comment
            # definitely enters the viewport at least once. Kaggle sometimes
            # lazy-renders comment body divs only on visibility, and the
            # first comment can end up with an empty body div if the page
            # loaded below it and was never scrolled back up.
            page.mouse.wheel(0, -100000)
            page.wait_for_timeout(300)
            page.mouse.wheel(0, 50000)
            page.wait_for_timeout(400)
            html = page.content()
        finally:
            with contextlib.suppress(Exception):
                context.close()
        return _extract_messages_from_html(html)


def fetch_topic_body(
    competition_slug: str,
    topic_id: int,
    *,
    headless: bool = True,
    wait_ms: int = 4500,
    min_interval_s: float = 2.0,
) -> list[Message]:
    """Single-topic convenience wrapper. Prefer TopicBodyScraper for batches."""
    with TopicBodyScraper(
        competition_slug,
        headless=headless,
        wait_ms=wait_ms,
        min_interval_s=min_interval_s,
    ) as scraper:
        return scraper.fetch(topic_id)


def fetch_topic(
    scraper: TopicBodyScraper,
    meta: TopicMeta,
) -> Topic:
    messages = scraper.fetch(meta.id)
    return Topic(meta=meta, messages=messages)


# --------------------------------------------------------------------------
# 3. Writing to disk
# --------------------------------------------------------------------------

def write_topic(topic: Topic, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / topic.meta.local_filename
    path.write_text(topic.to_markdown())
    # Also a .json sidecar for downstream distillation.
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "meta": topic.meta.to_manifest(),
                "messages": [m.to_json() for m in topic.messages],
                "sha256": topic.sha256,
            },
            indent=2,
        )
    )
    return path


def write_forum_index(
    topics: Iterable[TopicMeta],
    out_dir: Path,
) -> Path:
    """Write a bare index at reference/forum/INDEX.md.

    This is the pre-distillation index — the distiller rewrites it later
    with relevance scores and summaries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "INDEX.md"
    topics_list = list(topics)
    lines = [
        "# Forum index (raw)",
        "",
        f"{len(topics_list)} topics — sorted by most recent activity.",
        "",
        "| id | title | author | last activity | comments | votes |",
        "|---:|-------|--------|---------------|---------:|------:|",
    ]
    for t in sorted(topics_list, key=lambda t: t.last_comment_date, reverse=True):
        title = (t.title or "").replace("|", "\\|")
        author = (t.author or "").replace("|", "\\|")
        lines.append(
            f"| [{t.id}](topics/{t.local_filename}) "
            f"| {title} | {author} | {t.last_comment_date} "
            f"| {t.comment_count} | {t.votes} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))
    return path


__all__ = [
    "TopicMeta",
    "Message",
    "Topic",
    "list_all_topics",
    "fetch_topic_body",
    "fetch_topic",
    "write_topic",
    "write_forum_index",
]
