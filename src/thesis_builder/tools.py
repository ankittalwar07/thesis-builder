"""Tools the researcher can call: web search + source fetching + EDGAR.

Both web_search and fetch_source are wrapped in ``diskcache`` with TTLs
matching the project brief (24h for searches, 7d for source extractions).
Cache keys include the active prompt version so prompt edits invalidate
downstream caches.
"""

from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import diskcache
import httpx

from . import prompts
from .llm import TraceWriter
from .schemas import SearchResult

DEFAULT_CACHE_DIR = os.environ.get("THESIS_CACHE_DIR", ".diskcache")
SEARCH_TTL_S = 24 * 60 * 60       # 24h
EXTRACT_TTL_S = 7 * 24 * 60 * 60  # 7d

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; thesis-builder/0.1; "
        "+https://github.com/ankittalwar07/thesis-builder)"
    )
}


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


# --- Caches -----------------------------------------------------------------


class CacheBundle:
    """Bundle of diskcaches used by the project, one sub-cache per concern."""

    def __init__(self, root: str | Path = DEFAULT_CACHE_DIR) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.search = diskcache.Cache(str(root / "search"))
        self.extract = diskcache.Cache(str(root / "extract"))
        self.hits = 0
        self.misses = 0

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# --- Web search -------------------------------------------------------------


class WebSearch:
    """Pluggable web-search front-end.

    Provider selection (first that has credentials wins):
      1. Tavily            (if TAVILY_API_KEY is set; tavily-python installed)
      2. DuckDuckGo HTML   (no key; via duckduckgo-search)

    Both providers normalise to ``SearchResult``.
    """

    def __init__(self, cache: CacheBundle, trace: TraceWriter | None = None) -> None:
        self.cache = cache
        self.trace = trace
        self._tavily = None
        if os.environ.get("TAVILY_API_KEY"):
            try:
                from tavily import TavilyClient

                self._tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            except ImportError:
                self._tavily = None

    def search(self, query: str, k: int = 8) -> list[SearchResult]:
        key = _hash("search", query, str(k), prompts.cache_suffix())
        cached = self.cache.search.get(key)
        if cached is not None:
            self.cache.hits += 1
            self._emit("cache_hit", note=f"search:{query}", key=key)
            return [SearchResult.model_validate(r) for r in cached]
        self.cache.misses += 1
        self._emit("cache_miss", note=f"search:{query}", key=key)

        t0 = time.perf_counter()
        try:
            if self._tavily is not None:
                results = self._search_tavily(query, k)
            else:
                results = self._search_ddg(query, k)
        except Exception as e:  # noqa: BLE001
            self._emit("error", note=f"search:{query}: {type(e).__name__}: {e}")
            return []
        latency_ms = int((time.perf_counter() - t0) * 1000)
        self._emit(
            "search",
            note=f"q={query!r} n={len(results)} provider={'tavily' if self._tavily else 'ddg'}",
            latency_ms=latency_ms,
        )
        self.cache.search.set(key, [r.model_dump(mode="json") for r in results], expire=SEARCH_TTL_S)
        return results

    def _search_tavily(self, query: str, k: int) -> list[SearchResult]:
        resp = self._tavily.search(query=query, max_results=k, search_depth="basic")
        out: list[SearchResult] = []
        for r in resp.get("results", []):
            try:
                out.append(
                    SearchResult(
                        title=r.get("title") or "(untitled)",
                        url=r["url"],
                        snippet=r.get("content", "")[:400],
                        provider="tavily",
                    )
                )
            except Exception:
                continue
        return out

    def _search_ddg(self, query: str, k: int) -> list[SearchResult]:
        # duckduckgo-search is a hard dependency in pyproject.toml so this
        # import is safe to do lazily — it just keeps llm.py import-time light.
        from duckduckgo_search import DDGS

        out: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=k):
                try:
                    out.append(
                        SearchResult(
                            title=r.get("title") or "(untitled)",
                            url=r.get("href") or r.get("link"),
                            snippet=(r.get("body") or "")[:400],
                            provider="ddg",
                        )
                    )
                except Exception:
                    continue
        return out

    def _emit(self, kind: str, note: str = "", key: str | None = None, latency_ms: int | None = None) -> None:
        if self.trace is None:
            return
        self.trace.write(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                "note": note,
                "cache_key": key,
                "latency_ms": latency_ms,
            }
        )


# --- Source fetch + clean ---------------------------------------------------


def fetch_source(
    url: str,
    cache: CacheBundle,
    trace: TraceWriter | None = None,
    timeout_s: float = 15.0,
    max_chars: int = 12000,
) -> str | None:
    """Fetch a URL and return cleaned main-content text.

    Never returns raw HTML. Cached for 7 days. The raw page is discarded —
    only the trafilatura-extracted text is persisted.
    """
    key = _hash("extract", url, prompts.cache_suffix())
    cached = cache.extract.get(key)
    if cached is not None:
        cache.hits += 1
        _emit_simple(trace, "cache_hit", note=f"extract:{url}", key=key)
        return cached
    cache.misses += 1
    _emit_simple(trace, "cache_miss", note=f"extract:{url}", key=key)

    t0 = time.perf_counter()
    try:
        with httpx.Client(
            timeout=timeout_s, follow_redirects=True, headers=_HTTP_HEADERS
        ) as c:
            r = c.get(url)
            r.raise_for_status()
            html = r.text
    except Exception as e:  # noqa: BLE001
        _emit_simple(trace, "error", note=f"fetch:{url}: {type(e).__name__}: {e}")
        return None

    text = _extract_main(html)
    if text is None:
        _emit_simple(trace, "error", note=f"extract:{url}: empty")
        return None
    text = text[:max_chars]
    cache.extract.set(key, text, expire=EXTRACT_TTL_S)
    _emit_simple(
        trace,
        "fetch",
        note=f"url={url} chars={len(text)}",
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
    return text


def _extract_main(html: str) -> str | None:
    try:
        import trafilatura

        return trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_recall=False,
        )
    except Exception:
        return None


def _emit_simple(
    trace: TraceWriter | None,
    kind: str,
    note: str = "",
    key: str | None = None,
    latency_ms: int | None = None,
) -> None:
    if trace is None:
        return
    trace.write(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "note": note,
            "cache_key": key,
            "latency_ms": latency_ms,
        }
    )


# --- EDGAR (optional) -------------------------------------------------------


def fetch_edgar_filings(
    ticker: str,
    cache: CacheBundle,
    forms: tuple[str, ...] = ("10-K", "S-1"),
    limit: int = 1,
    trace: TraceWriter | None = None,
) -> list[dict[str, Any]]:
    """Best-effort EDGAR fetch. Optional; returns [] if dependency is missing
    or the user has not set ``EDGAR_USER_AGENT``.
    """
    ua = os.environ.get("EDGAR_USER_AGENT")
    if not ua:
        _emit_simple(trace, "error", note="edgar: EDGAR_USER_AGENT not set; skipping")
        return []
    try:
        from sec_edgar_downloader import Downloader  # type: ignore
    except ImportError:
        _emit_simple(trace, "error", note="edgar: sec-edgar-downloader not installed; skipping")
        return []

    key = _hash("edgar", ticker, ",".join(forms), str(limit))
    cached = cache.extract.get(key)
    if cached is not None:
        cache.hits += 1
        return cached
    cache.misses += 1

    try:
        # The downloader writes to disk; we point it at our cache root and
        # then just record the directory so the researcher can decide whether
        # to feed any of it through the per-source summarizer.
        out_dir = Path(cache.extract.directory).parent / "edgar"
        out_dir.mkdir(parents=True, exist_ok=True)
        # sec-edgar-downloader's UA format: "Name email"
        dl = Downloader("thesis-builder", ua, str(out_dir))
        results: list[dict[str, Any]] = []
        for form in forms:
            dl.get(form, ticker, limit=limit)
            results.append({"ticker": ticker, "form": form, "dir": str(out_dir)})
        cache.extract.set(key, results, expire=EXTRACT_TTL_S)
        return results
    except Exception as e:  # noqa: BLE001
        _emit_simple(trace, "error", note=f"edgar:{ticker}: {type(e).__name__}: {e}")
        return []
