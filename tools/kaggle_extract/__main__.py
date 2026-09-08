"""CLI entry point.

Run from the repo root (with the extractor deps on the path):

    PYTHONPATH=tools python -m kaggle_extract <subcommand>

Subcommands:
    init     — first-time bootstrap: metadata + pages + data download + small
               forum sample + skills + data symlink. Distillation is opt-in.
    sync     — refresh metadata, pages, forum list, and skills; mark changed
               topics stale for subsequent on-demand retrieval.
    pages    — refetch all overview/data/rules/etc pages.
    forum    — refetch forum topic list (not bodies).
    topic    — refetch one forum topic body (Playwright).
    bodies   — fetch bodies for any tracked topics missing a local file or
               marked stale.
    distill  — re-run Claude distillation for inputs whose sha changed.
    skills   — regenerate .agents/skills/ and .claude/skills/ from reference/.
    status   — print manifest summary.

Every subcommand accepts `--competition <slug>` and `--reference-dir <path>`
(default: `./reference`). Destructive / expensive commands honor `--dry-run`
and `--limit`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .client import KaggleInternalClient
from .forum import (
    Topic,
    TopicBodyScraper,
    TopicMeta,
    list_all_topics,
    write_forum_index,
    write_topic,
)
from .manifest import Manifest
from .pages import fetch_all as fetch_all_pages
from .skills import write_all_skills

DEFAULT_COMPETITION = "biohub-cell-tracking-during-development"
DEFAULT_REFERENCE_DIR = Path("reference")
DEFAULT_SKILLS_DIR = Path(".claude/skills")
DEFAULT_CODEX_SKILLS_DIR = Path(".agents/skills")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[kaggle_extract] {msg}", flush=True)


def _ensure_manifest(reference_dir: Path) -> Manifest:
    return Manifest.load(reference_dir / "manifest.json")


def _ensure_competition(client: KaggleInternalClient, manifest: Manifest, slug: str) -> dict:
    """Fetch competition metadata and update manifest."""
    comp = client.get_competition(slug)
    manifest.set_competition(slug, comp)
    return comp


def _forum_id(manifest: Manifest) -> int:
    fid = manifest.competition.get("forum_id")
    if not fid:
        raise RuntimeError("manifest has no forum_id — run `init` first")
    return int(fid)


def _competition_id(manifest: Manifest) -> int:
    cid = manifest.competition.get("id")
    if not cid:
        raise RuntimeError("manifest has no competition id — run `init` first")
    return int(cid)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_pages(args: argparse.Namespace) -> int:
    reference_dir = Path(args.reference_dir)
    manifest = _ensure_manifest(reference_dir)
    with KaggleInternalClient() as kg:
        _ensure_competition(kg, manifest, args.competition)
        comp_id = _competition_id(manifest)
        _log(f"fetching pages for competition {comp_id}...")
        results = fetch_all_pages(kg, comp_id, reference_dir / "overview")
    changed = 0
    for r in results:
        if manifest.page_needs_refresh(r.filename, r.sha256):
            changed += 1
        manifest.record_page(r.filename, r.kaggle_id, r.sha256)
    manifest.touch_last_sync()
    manifest.save()
    _log(f"pages: {len(results)} total, {changed} changed")
    return 0


def cmd_forum(args: argparse.Namespace) -> int:
    """Refetch the forum topic list (metadata only) and update the manifest."""
    reference_dir = Path(args.reference_dir)
    manifest = _ensure_manifest(reference_dir)
    with KaggleInternalClient() as kg:
        if not manifest.competition:
            _ensure_competition(kg, manifest, args.competition)
        forum_id = _forum_id(manifest)
        _log(f"listing forum topics for forum {forum_id}...")
        topics = list_all_topics(
            kg,
            forum_id,
            max_pages=args.max_pages,
            page_size=args.page_size,
        )
    changed = 0
    for meta in topics:
        # Clear body_sha256 on change so cmd_bodies re-scrapes grown topics.
        stale = manifest.topic_needs_refresh(meta)
        if stale:
            changed += 1
        local_file = f"topics/{meta.local_filename}"
        entry = manifest.forum_topics.get(str(meta.id), {})
        manifest.forum_topics[str(meta.id)] = {
            **entry,
            "title": meta.title,
            "author": meta.author,
            "author_type": meta.author_type,
            "author_tier": meta.author_tier,
            "post_date": meta.post_date,
            "last_comment_date": meta.last_comment_date,
            "last_commenter": meta.last_commenter,
            "comment_count": meta.comment_count,
            "votes": meta.votes,
            "local_file": local_file,
            "body_sha256": None if stale else entry.get("body_sha256"),
            "distill_sha256": entry.get("distill_sha256"),
        }
    # Write/refresh raw INDEX.md
    write_forum_index(topics, reference_dir / "forum")
    manifest.touch_last_sync()
    manifest.save()
    _log(f"forum list: {len(topics)} topics, {changed} new/updated since last sync")
    return 0


def cmd_bodies(args: argparse.Namespace) -> int:
    """Scrape bodies for topics that are stale or missing a local file.

    Honors --limit to cap the per-run fetch count so we never surprise-blast
    the Kaggle servers. Default limit is 25.
    """
    reference_dir = Path(args.reference_dir)
    manifest = _ensure_manifest(reference_dir)
    slug = manifest.competition.get("slug") or args.competition
    topics_dir = reference_dir / "forum" / "topics"

    # Build the work queue from the manifest (not from a fresh API call).
    stale: list[TopicMeta] = []
    for tid_str, entry in manifest.forum_topics.items():
        local = entry.get("local_file") or f"topics/{tid_str}.md"
        local_path = reference_dir / "forum" / local
        meta = TopicMeta(
            id=int(tid_str),
            title=entry.get("title") or "",
            author=entry.get("author") or "",
            author_type=entry.get("author_type") or "",
            author_tier=entry.get("author_tier") or "",
            comment_count=int(entry.get("comment_count") or 0),
            votes=int(entry.get("votes") or 0),
            is_sticky=False,
            post_date=entry.get("post_date") or "",
            last_comment_date=entry.get("last_comment_date") or "",
            last_commenter=entry.get("last_commenter") or "",
            topic_url=f"/competitions/{slug}/discussion/{tid_str}",
        )
        if not local_path.exists() or not entry.get("body_sha256"):
            stale.append(meta)
            continue
        # We can't compare last_comment_date without a fresh list; that
        # comparison already happened inside cmd_forum. If the manifest was
        # updated there and body_sha256 is still present but the file is
        # missing, we caught it above. Otherwise treat as fresh.

    # Add topics flagged stale during the last `forum` run: detected by
    # lack of body_sha256 or absence of local file.
    stale.sort(key=lambda m: m.last_comment_date, reverse=True)

    if args.limit:
        stale = stale[: args.limit]

    _log(f"bodies: {len(stale)} topics queued (limit={args.limit}, dry_run={args.dry_run})")
    for m in stale:
        _log(f"  - {m.id} {m.title[:70]}")
    if args.dry_run or not stale:
        return 0

    with TopicBodyScraper(
        slug,
        headless=True,
        min_interval_s=args.min_interval,
    ) as scraper:
        for meta in stale:
            try:
                messages = scraper.fetch(meta.id)
            except Exception as e:  # noqa: BLE001
                _log(f"  ! topic {meta.id} failed: {e}")
                continue
            topic = Topic(meta=meta, messages=messages)
            write_topic(topic, topics_dir)
            manifest.record_topic(
                meta=meta,
                local_file=f"topics/{meta.local_filename}",
                body_sha256=topic.sha256,
            )
            _log(f"  + {meta.id} ({len(messages)} msgs)")
            # Persist incrementally so a crash doesn't lose work.
            manifest.save()
    manifest.touch_last_sync()
    manifest.save()
    _log(f"bodies: scraped {scraper.fetch_count} topic page(s)")
    return 0


def cmd_topic(args: argparse.Namespace) -> int:
    """Fetch one topic body by id."""
    reference_dir = Path(args.reference_dir)
    manifest = _ensure_manifest(reference_dir)
    slug = manifest.competition.get("slug") or args.competition
    entry = manifest.forum_topics.get(str(args.topic_id))
    if not entry:
        _log(
            f"topic {args.topic_id} not in manifest — running `forum` first "
            f"to refresh the topic list"
        )
        # Do a partial forum list refresh to find it
        with KaggleInternalClient() as kg:
            if not manifest.competition:
                _ensure_competition(kg, manifest, args.competition)
            forum_id = _forum_id(manifest)
            topics = list_all_topics(kg, forum_id, max_pages=10, page_size=20)
        found = next((t for t in topics if t.id == args.topic_id), None)
        if not found:
            _log(f"topic {args.topic_id} not found in forum listing")
            return 1
        manifest.forum_topics[str(found.id)] = {
            "title": found.title,
            "author": found.author,
            "author_type": found.author_type,
            "author_tier": found.author_tier,
            "post_date": found.post_date,
            "last_comment_date": found.last_comment_date,
            "last_commenter": found.last_commenter,
            "comment_count": found.comment_count,
            "votes": found.votes,
            "local_file": f"topics/{found.local_filename}",
            "body_sha256": None,
            "distill_sha256": None,
        }
        meta = found
    else:
        meta = TopicMeta(
            id=args.topic_id,
            title=entry.get("title") or "",
            author=entry.get("author") or "",
            author_type=entry.get("author_type") or "",
            author_tier=entry.get("author_tier") or "",
            comment_count=int(entry.get("comment_count") or 0),
            votes=int(entry.get("votes") or 0),
            is_sticky=False,
            post_date=entry.get("post_date") or "",
            last_comment_date=entry.get("last_comment_date") or "",
            last_commenter=entry.get("last_commenter") or "",
            topic_url=f"/competitions/{slug}/discussion/{args.topic_id}",
        )
    _log(f"scraping topic {meta.id}: {meta.title!r}")
    with TopicBodyScraper(slug, min_interval_s=args.min_interval) as scraper:
        messages = scraper.fetch(meta.id)
    topic = Topic(meta=meta, messages=messages)
    write_topic(topic, reference_dir / "forum" / "topics")
    manifest.record_topic(
        meta=meta,
        local_file=f"topics/{meta.local_filename}",
        body_sha256=topic.sha256,
    )
    manifest.touch_last_sync()
    manifest.save()
    _log(f"topic {meta.id}: {len(messages)} messages written")
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    """Run Claude distillation on changed inputs."""
    # Lazy import so a plain `pages`/`forum` run doesn't require anthropic.
    from .distill import Distiller, write_distill_sidecar

    reference_dir = Path(args.reference_dir)
    manifest = _ensure_manifest(reference_dir)

    distiller = Distiller(
        reference_dir,
        model=args.model,
    )

    # 1. Distill overview pages
    overview_dir = reference_dir / "overview"
    overview_changed = 0
    overview_skipped = 0
    for page_file in sorted(overview_dir.glob("*.md")):
        text = page_file.read_text()
        input_sha = Distiller.input_hash(text, model=args.model)
        sidecar = page_file.with_suffix(page_file.suffix + ".distill.json")
        if not args.force and sidecar.exists():
            try:
                existing = sidecar.read_text()
                if input_sha in existing:
                    overview_skipped += 1
                    continue
            except Exception:
                pass
        _log(f"distilling overview: {page_file.name}")
        if args.dry_run:
            continue
        result = distiller.distill(text, label=f"overview-page:{page_file.name}")
        write_distill_sidecar(page_file, result)
        overview_changed += 1

    # 2. Distill forum topics
    topics_dir = reference_dir / "forum" / "topics"
    forum_changed = 0
    forum_skipped = 0
    forum_failed = 0
    seen_ids: list[int] = []
    for topic_md in sorted(topics_dir.glob("*.md")):
        text = topic_md.read_text()
        input_sha = Distiller.input_hash(text, model=args.model)
        # Extract the topic id from the filename ("687798-slug.md" -> 687798)
        stem = topic_md.stem
        tid_str = stem.split("-", 1)[0]
        try:
            topic_id = int(tid_str)
        except ValueError:
            continue
        # Self-heal from a matching sidecar even if the manifest wasn't saved
        # (e.g. a prior run crashed mid-batch): reconcile the manifest and skip.
        sidecar = topic_md.with_suffix(topic_md.suffix + ".distill.json")
        if not args.force and sidecar.exists() and input_sha in sidecar.read_text():
            manifest.set_topic_distill(topic_id, input_sha)
            forum_skipped += 1
            continue
        if not args.force and not manifest.topic_distill_needs_refresh(topic_id, input_sha):
            forum_skipped += 1
            continue
        _log(f"distilling topic: {topic_md.name}")
        if args.dry_run:
            continue
        try:
            result = distiller.distill(text, label=f"forum-topic:{topic_id}")
        except Exception as e:  # noqa: BLE001
            _log(f"  ! topic {topic_id} distill failed: {e}")
            forum_failed += 1
            continue
        # Override the stored input_sha so the manifest matches what we hashed
        result.input_sha256 = input_sha
        write_distill_sidecar(topic_md, result)
        manifest.set_topic_distill(topic_id, input_sha)
        seen_ids.append(topic_id)
        forum_changed += 1
        manifest.save()
        if args.limit and forum_changed >= args.limit:
            _log(f"hit --limit {args.limit}, stopping early")
            break

    manifest.touch_last_sync()
    manifest.save()
    _log(
        f"distill: overview {overview_changed} changed / {overview_skipped} skipped; "
        f"forum {forum_changed} changed / {forum_skipped} skipped"
        + (f" / {forum_failed} failed" if forum_failed else "")
    )
    return forum_failed and 1 or 0


def cmd_skills(args: argparse.Namespace) -> int:
    reference_dir = Path(args.reference_dir)
    manifest = _ensure_manifest(reference_dir)
    skills_dirs = [Path(args.skills_dir), Path(args.codex_skills_dir)]
    skills_dirs = list(dict.fromkeys(skills_dirs))
    written: list[Path] = []
    for skills_dir in skills_dirs:
        written.extend(write_all_skills(reference_dir, manifest, skills_dir))
    _log(f"skills: wrote {len(written)} files")
    for p in written:
        _log(f"  - {p}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    reference_dir = Path(args.reference_dir)
    manifest = _ensure_manifest(reference_dir)
    print(manifest.status_summary())
    return 0


def cmd_download_data(args: argparse.Namespace) -> int:
    """Use the official `kaggle` CLI to download competition data.

    Uses the kaggle CLI rather than the Python API so the progress output
    is visible. Downloads into /kaggle/input/competitions/<slug>/, provides
    Kaggle's flat /kaggle/input/<slug> compatibility alias, and symlinks
    `data/` at the repo root.
    """
    from competition_paths import COMPETITION
    if args.competition != COMPETITION:
        raise ValueError("This workspace downloader is scoped to " + COMPETITION)
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "download_data.py")]
    if args.dry_run:
        _log("(dry-run) " + " ".join(command))
        return 0
    return subprocess.call(command)


def cmd_init(args: argparse.Namespace) -> int:
    """First-time bootstrap. Safe to re-run."""
    reference_dir = Path(args.reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    (reference_dir / "overview").mkdir(exist_ok=True)
    (reference_dir / "forum" / "topics").mkdir(parents=True, exist_ok=True)

    rc = cmd_pages(args)
    if rc:
        return rc
    rc = cmd_download_data(args)
    if rc:
        return rc
    rc = cmd_forum(args)
    if rc:
        return rc
    # Scrape a conservative first batch of topic bodies.
    args.limit = args.limit or 10
    rc = cmd_bodies(args)
    if rc:
        return rc
    if args.distill:
        rc = cmd_distill(args)
        if rc:
            return rc
    rc = cmd_skills(args)
    if rc:
        return rc
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    """Incremental refresh."""
    rc = cmd_pages(args)
    if rc:
        return rc
    rc = cmd_forum(args)
    if rc:
        return rc
    if args.distill:
        rc = cmd_distill(args)
        if rc:
            return rc
    rc = cmd_skills(args)
    return rc


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--competition", default=DEFAULT_COMPETITION)
    p.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    p.add_argument("--skills-dir", default=str(DEFAULT_SKILLS_DIR))
    p.add_argument("--codex-skills-dir", default=str(DEFAULT_CODEX_SKILLS_DIR))
    p.add_argument("--dry-run", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kaggle_extract",
        description="Extract a Kaggle competition into a local skills index.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="First-time bootstrap")
    _add_common_args(p_init)
    p_init.add_argument("--limit", type=int, default=10,
                        help="Max forum topic bodies to scrape on first run")
    p_init.add_argument("--min-interval", type=float, default=2.5)
    p_init.add_argument("--max-pages", type=int, default=20)
    p_init.add_argument("--page-size", type=int, default=20)
    p_init.add_argument("--model", default="claude-sonnet-4-6")
    p_init.add_argument("--distill", action="store_true", help="Opt in to Anthropic API summaries")
    p_init.add_argument("--force", action="store_true",
                        help="Force re-distill even if sidecar exists")
    p_init.set_defaults(func=cmd_init)

    p_sync = sub.add_parser("sync", help="Incremental refresh")
    _add_common_args(p_sync)
    p_sync.add_argument("--limit", type=int, default=25)
    p_sync.add_argument("--min-interval", type=float, default=2.5)
    p_sync.add_argument("--max-pages", type=int, default=20)
    p_sync.add_argument("--page-size", type=int, default=20)
    p_sync.add_argument("--model", default="claude-sonnet-4-6")
    p_sync.add_argument("--distill", action="store_true", help="Opt in to Anthropic API summaries")
    p_sync.add_argument("--force", action="store_true",
                        help="Force re-distill even if sidecar exists")
    p_sync.set_defaults(func=cmd_sync)

    p_pages = sub.add_parser("pages", help="Refetch all overview pages")
    _add_common_args(p_pages)
    p_pages.set_defaults(func=cmd_pages)

    p_forum = sub.add_parser("forum", help="Refetch forum topic list (metadata)")
    _add_common_args(p_forum)
    p_forum.add_argument("--max-pages", type=int, default=20)
    p_forum.add_argument("--page-size", type=int, default=20)
    p_forum.set_defaults(func=cmd_forum)

    p_bodies = sub.add_parser("bodies", help="Scrape stale forum topic bodies")
    _add_common_args(p_bodies)
    p_bodies.add_argument("--limit", type=int, default=25)
    p_bodies.add_argument("--min-interval", type=float, default=2.5)
    p_bodies.set_defaults(func=cmd_bodies)

    p_topic = sub.add_parser("topic", help="Fetch one topic body")
    _add_common_args(p_topic)
    p_topic.add_argument("topic_id", type=int)
    p_topic.add_argument("--min-interval", type=float, default=2.5)
    p_topic.set_defaults(func=cmd_topic)

    p_distill = sub.add_parser("distill", help="Run Claude distillation on stale inputs")
    _add_common_args(p_distill)
    p_distill.add_argument("--model", default="claude-sonnet-4-6")
    p_distill.add_argument("--limit", type=int, default=0,
                           help="Max forum topic distillations per run (0 = no cap)")
    p_distill.add_argument("--force", action="store_true",
                           help="Re-distill even if hash matches")
    p_distill.set_defaults(func=cmd_distill)

    p_skills = sub.add_parser(
        "skills", help="Regenerate .agents/skills/* and .claude/skills/*"
    )
    _add_common_args(p_skills)
    p_skills.set_defaults(func=cmd_skills)

    p_status = sub.add_parser("status", help="Show manifest summary")
    _add_common_args(p_status)
    p_status.set_defaults(func=cmd_status)

    p_data = sub.add_parser("download-data", help="Run `kaggle competitions download`")
    _add_common_args(p_data)
    p_data.set_defaults(func=cmd_download_data)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
