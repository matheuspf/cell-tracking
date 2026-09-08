"""HTTP client for Kaggle's internal JSON-RPC API.

The official `kaggle` package only exposes competition data downloads, submissions
and leaderboards. For overview pages and forum metadata we hit Kaggle's
undocumented endpoints at https://www.kaggle.com/api/i/<Service>/<Method>.

Auth scheme:
- HTTP Basic with username + key from ~/.kaggle/kaggle.json
- Plus an XSRF cookie obtained by hitting https://www.kaggle.com/ once;
  the cookie value must also be sent back in the X-XSRF-TOKEN header on
  every POST or Kaggle returns the SPA HTML shell instead of JSON.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

KAGGLE_BASE = "https://www.kaggle.com"
KAGGLE_RPC_BASE = f"{KAGGLE_BASE}/api/i"
DEFAULT_KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


class KaggleAuthError(RuntimeError):
    pass


class KaggleRPCError(RuntimeError):
    def __init__(self, service: str, method: str, status: int, body: str):
        self.service = service
        self.method = method
        self.status = status
        self.body = body
        super().__init__(
            f"{service}/{method} -> HTTP {status}: {body[:200]}"
        )


def _load_credentials(path: Path = DEFAULT_KAGGLE_JSON) -> tuple[str, str]:
    if not path.exists():
        raise KaggleAuthError(
            f"{path} not found — set up Kaggle API credentials first"
        )
    data = json.loads(path.read_text())
    try:
        return data["username"], data["key"]
    except KeyError as e:
        raise KaggleAuthError(f"{path} missing key: {e}") from None


class KaggleInternalClient:
    """Authenticated client for Kaggle's internal RPC endpoints.

    Usage:
        with KaggleInternalClient() as kg:
            comp = kg.rpc("competitions.CompetitionService", "GetCompetition",
                          {"competitionName": "..."})
    """

    def __init__(
        self,
        credentials_path: Path = DEFAULT_KAGGLE_JSON,
        timeout: float = 30.0,
        min_interval_s: float = 1.0,
    ):
        """
        min_interval_s: minimum wall time between HTTP requests. Defaults to
            1 second so we stay firmly in "polite scraper" territory and
            never risk the user's account over this tool.
        """
        self._username, self._key = _load_credentials(credentials_path)
        self._client = httpx.Client(
            base_url=KAGGLE_BASE,
            timeout=timeout,
            auth=(self._username, self._key),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "kaggle-extract/0.1 (+https://kaggle.com)",
            },
            follow_redirects=True,
        )
        self._xsrf: str | None = None
        self._min_interval_s = min_interval_s
        self._last_request_t: float = 0.0
        self._request_count: int = 0

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._min_interval_s - (now - self._last_request_t)
        if wait > 0:
            time.sleep(wait)
        self._last_request_t = time.monotonic()
        self._request_count += 1

    @property
    def request_count(self) -> int:
        return self._request_count

    # ---- context manager ------------------------------------------------

    def __enter__(self) -> "KaggleInternalClient":
        self._bootstrap_session()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._client.close()

    def close(self) -> None:
        self._client.close()

    # ---- session ---------------------------------------------------------

    def _bootstrap_session(self) -> None:
        """Hit the homepage once to get an XSRF-TOKEN cookie."""
        self._throttle()
        r = self._client.get("/")
        r.raise_for_status()
        xsrf = self._client.cookies.get("XSRF-TOKEN")
        if not xsrf:
            raise KaggleAuthError(
                "Failed to obtain XSRF-TOKEN cookie from kaggle.com — "
                "check network and credentials"
            )
        self._xsrf = xsrf

    # ---- core RPC --------------------------------------------------------

    def rpc(
        self,
        service: str,
        method: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._xsrf is None:
            self._bootstrap_session()
        url = f"/api/i/{service}/{method}"
        body = payload if payload is not None else {}
        self._throttle()
        r = self._client.post(
            url,
            content=json.dumps(body),
            headers={"X-XSRF-TOKEN": self._xsrf or ""},
        )
        # Kaggle returns 404 -> the SPA HTML shell rather than a JSON 404,
        # so detect both real HTTP errors and the "wrong endpoint name" case.
        ct = r.headers.get("content-type", "")
        if r.status_code >= 400 or "application/json" not in ct:
            raise KaggleRPCError(service, method, r.status_code, r.text)
        return r.json()

    # ---- convenience helpers --------------------------------------------

    def get_competition(self, slug: str) -> dict[str, Any]:
        return self.rpc(
            "competitions.CompetitionService",
            "GetCompetition",
            {"competitionName": slug},
        )

    def list_pages(self, competition_id: int) -> list[dict[str, Any]]:
        resp = self.rpc(
            "competitions.PageService",
            "ListPages",
            {"competitionId": competition_id},
        )
        return resp.get("pages", [])

    def list_forum_topics(
        self,
        forum_id: int,
        page: int = 1,
        limit: int = 20,
    ) -> dict[str, Any]:
        return self.rpc(
            "discussions.DiscussionsService",
            "GetTopicListByForumId",
            {"forumId": forum_id, "page": page, "limit": limit},
        )

    def get_forum_topic(self, topic_id: int) -> dict[str, Any]:
        return self.rpc(
            "discussions.DiscussionsService",
            "GetForumTopicById",
            {"forumTopicId": topic_id},
        )


__all__ = [
    "KaggleInternalClient",
    "KaggleAuthError",
    "KaggleRPCError",
    "KAGGLE_BASE",
]
