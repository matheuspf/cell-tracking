"""Kaggle competition knowledge extractor.

Pulls competition pages and forum discussions from Kaggle's internal RPC API
(plus a Playwright fallback for forum message bodies), distills them with the
Anthropic API, and writes Codex/Claude Code skills + raw reference markdown.
"""

__version__ = "0.1.0"
