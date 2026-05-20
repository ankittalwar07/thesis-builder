"""Researcher agent.

Pipeline per round:
  1. Plan N queries spanning the four chain layers (tier-small).
  2. Run all queries against the web-search tool.
  3. Map: for each unique URL, fetch + extract main content, then summarize
     to a ≤200-token ``SourceSummary`` with tier-small.
  4. Reduce: drop low-relevance summaries; dedup by entity overlap.
  5. Audit coverage (tier-reasoning); if gaps remain and round budget left,
     queue follow-up queries and loop.

Final step builds a draft Node tree with tier-reasoning. Raw HTML is never
carried forward in working context — only the compressed ``SourceSummary``
list is.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from rich.console import Console

from .. import prompts
from ..llm import LLMClient
from ..schemas import (
    CoverageReport,
    Node,
    QueryPlan,
    ResearchBundle,
    SearchQuery,
    SearchResult,
    SourceSummary,
)
from ..tools import CacheBundle, WebSearch, fetch_source

LAYERS = ("raw_inputs", "enabling", "integrators", "applications")


class Researcher:
    def __init__(
        self,
        llm: LLMClient,
        search: WebSearch,
        cache: CacheBundle,
        console: Console | None = None,
        max_rounds: int = 3,
        queries_per_round: int = 8,
        results_per_query: int = 5,
        min_searches: int = 15,
        max_searches: int = 25,
        summarize_workers: int = 8,
    ) -> None:
        self.llm = llm
        self.search = search
        self.cache = cache
        self.console = console or Console(quiet=True)
        self.max_rounds = max_rounds
        self.queries_per_round = queries_per_round
        self.results_per_query = results_per_query
        self.min_searches = min_searches
        self.max_searches = max_searches
        self.summarize_workers = summarize_workers

    # ---- public ------------------------------------------------------------

    def run(self, theme: str) -> ResearchBundle:
        summaries: list[SourceSummary] = []
        queries_run: list[SearchQuery] = []
        seen_urls: set[str] = set()

        for round_idx in range(1, self.max_rounds + 1):
            remaining = self.max_searches - len(queries_run)
            if remaining <= 0:
                break
            n_queries = min(self.queries_per_round, remaining)
            plan = self._plan_queries(
                theme=theme,
                round_idx=round_idx,
                total_rounds=self.max_rounds,
                n_queries=n_queries,
                summaries=summaries,
            )
            self.console.log(
                f"[round {round_idx}] planned {len(plan.queries)} queries"
            )
            new_summaries = self._execute_round(
                theme=theme, queries=plan.queries, seen_urls=seen_urls
            )
            summaries.extend(new_summaries)
            queries_run.extend(plan.queries)

            if len(queries_run) >= self.min_searches:
                coverage = self._audit_coverage(
                    theme=theme,
                    summaries=summaries,
                    rounds=round_idx,
                )
                self.console.log(
                    f"[round {round_idx}] coverage: {coverage.per_layer_counts} "
                    f"done={coverage.done}"
                )
                if coverage.done and len(queries_run) >= self.min_searches:
                    break

        draft_nodes = self._draft_tree(theme=theme, summaries=summaries)
        return ResearchBundle(
            theme=theme,
            rounds=min(self.max_rounds, len(queries_run) // max(self.queries_per_round, 1) + 1),
            summaries=summaries,
            queries_run=queries_run,
            draft_nodes=draft_nodes,
        )

    # ---- planning ----------------------------------------------------------

    def _plan_queries(
        self,
        theme: str,
        round_idx: int,
        total_rounds: int,
        n_queries: int,
        summaries: list[SourceSummary],
    ) -> QueryPlan:
        prior = _coverage_block(summaries) if summaries else "(none yet — this is the first round)"
        msgs = [
            {"role": "system", "content": prompts.get("planner_system")},
            {
                "role": "user",
                "content": prompts.get("planner_user").format(
                    theme=theme,
                    round_idx=round_idx,
                    total_rounds=total_rounds,
                    n_queries=n_queries,
                    prior_coverage=f"Prior coverage:\n{prior}",
                ),
            },
        ]
        return self.llm.complete(tier="small", messages=msgs, response_model=QueryPlan)

    # ---- map: search + summarize ------------------------------------------

    def _execute_round(
        self,
        theme: str,
        queries: list[SearchQuery],
        seen_urls: set[str],
    ) -> list[SourceSummary]:
        # 1. Run searches sequentially (search providers rate-limit hard;
        # parallelism here costs more than it saves).
        hits: list[tuple[SearchQuery, SearchResult]] = []
        for q in queries:
            results = self.search.search(q.query, k=self.results_per_query)
            for r in results:
                url = str(r.url)
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                hits.append((q, r))

        if not hits:
            return []

        # 2. Map: fetch + summarize in parallel. The fetch+summarize unit is
        # bounded by ``max_chars`` in tools.fetch_source so each tier-small
        # call sees a known-bounded prompt size.
        summaries: list[SourceSummary] = []
        with ThreadPoolExecutor(max_workers=self.summarize_workers) as ex:
            futures = {
                ex.submit(self._summarize_one, theme, q, r): (q, r) for q, r in hits
            }
            for fut in as_completed(futures):
                try:
                    s = fut.result()
                except Exception as e:  # noqa: BLE001
                    self.console.log(f"[red]summarize failed: {e}[/red]")
                    continue
                if s is None:
                    continue
                if s.relevance < 0.3:
                    continue
                summaries.append(s)

        # 3. Reduce: dedup by entity overlap (cheap: token-set Jaccard).
        return _dedup_by_entities(summaries)

    def _summarize_one(
        self, theme: str, query: SearchQuery, result: SearchResult
    ) -> SourceSummary | None:
        content = fetch_source(str(result.url), self.cache, trace=self.llm.trace)
        if not content:
            # Fall back to the snippet — useful when a site blocks scrapers.
            content = result.snippet
            if not content:
                return None
        msgs = [
            {"role": "system", "content": prompts.get("summarizer_system")},
            {
                "role": "user",
                "content": prompts.get("summarizer_user").format(
                    theme=theme,
                    url=str(result.url),
                    title=result.title,
                    content=content[:8000],
                ),
            },
        ]
        try:
            summary: SourceSummary = self.llm.complete(
                tier="small",
                messages=msgs,
                response_model=SourceSummary,
            )
        except Exception:
            return None
        # The model may echo the URL/title from the prompt; trust our copies.
        return summary.model_copy(update={"url": result.url, "title": result.title})

    # ---- coverage audit ----------------------------------------------------

    def _audit_coverage(
        self,
        theme: str,
        summaries: list[SourceSummary],
        rounds: int,
    ) -> CoverageReport:
        block = _coverage_block(summaries)
        msgs = [
            {"role": "system", "content": prompts.get("coverage_system")},
            {
                "role": "user",
                "content": prompts.get("coverage_user").format(
                    theme=theme,
                    summary_index=block,
                    n_urls=len({str(s.url) for s in summaries}),
                    rounds=rounds,
                ),
            },
        ]
        try:
            return self.llm.complete(
                tier="reasoning", messages=msgs, response_model=CoverageReport
            )
        except Exception:
            # Conservative default: assume coverage isn't done, no gap queries.
            return CoverageReport(
                per_layer_counts=_counts_by_layer(summaries),
                gaps=[],
                done=False,
            )

    # ---- draft tree --------------------------------------------------------

    def _draft_tree(self, theme: str, summaries: list[SourceSummary]) -> list[Node]:
        if not summaries:
            return []
        summaries_block = _summaries_block(summaries)
        msgs = [
            {"role": "system", "content": prompts.get("draft_tree_system")},
            {
                "role": "user",
                "content": prompts.get("draft_tree_user").format(
                    theme=theme, summaries_block=summaries_block
                ),
            },
        ]
        from pydantic import BaseModel, Field

        class _NodeList(BaseModel):
            nodes: list[Node] = Field(default_factory=list)

        try:
            wrapped: _NodeList = self.llm.complete(
                tier="reasoning", messages=msgs, response_model=_NodeList
            )
            return wrapped.nodes
        except Exception as e:
            self.console.log(f"[yellow]draft_tree failed: {e}[/yellow]")
            return []


# --- helpers ----------------------------------------------------------------


def _counts_by_layer(summaries: Iterable[SourceSummary]) -> dict[str, int]:
    counts = {layer: 0 for layer in LAYERS}
    for s in summaries:
        for layer in s.layer_hints:
            if layer in counts:
                counts[layer] += 1
    return counts


def _coverage_block(summaries: list[SourceSummary]) -> str:
    counts = _counts_by_layer(summaries)
    lines = [f"  {layer}: {n} sources" for layer, n in counts.items()]
    return "\n".join(lines)


def _summaries_block(summaries: list[SourceSummary]) -> str:
    lines: list[str] = []
    for s in summaries:
        layer_tag = ",".join(s.layer_hints) or "?"
        entities = ", ".join(s.entities[:8])
        lines.append(
            f"- [{layer_tag}] {s.title} ({s.url})\n"
            f"  entities: {entities}\n"
            f"  summary: {s.summary}"
        )
    return "\n".join(lines)


def _dedup_by_entities(summaries: list[SourceSummary], threshold: float = 0.8) -> list[SourceSummary]:
    """Drop a summary if a stronger (higher-relevance) one already covers
    nearly the same entity set."""
    kept: list[SourceSummary] = []
    # Sort highest-relevance first so kept[] always holds the strongest.
    ordered = sorted(summaries, key=lambda s: s.relevance, reverse=True)
    for s in ordered:
        s_ents = {e.lower() for e in s.entities}
        is_dup = False
        for k in kept:
            k_ents = {e.lower() for e in k.entities}
            if not s_ents or not k_ents:
                continue
            jaccard = len(s_ents & k_ents) / len(s_ents | k_ents)
            if jaccard >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(s)
    return kept
