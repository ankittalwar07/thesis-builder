"""Pydantic schemas shared across agents.

The public contract (Node, Thesis) is defined in the project brief. Internal
schemas below it are intermediates used by individual agents — they are kept
in this file so structured-output retries via Instructor can import them
without circular dependencies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


Layer = Literal["raw_inputs", "enabling", "integrators", "applications"]


# --- Public schemas ---------------------------------------------------------


class Node(BaseModel):
    name: str
    layer: Layer
    description: str
    public_names: list[str] = Field(default_factory=list, description="Tickers, e.g. NVDA, TSM")
    private_names: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    sources: list[HttpUrl] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class Thesis(BaseModel):
    theme: str
    summary: str
    nodes: list[Node]
    investable_picks: list[str] = Field(
        default_factory=list,
        description="Top 5 picks with rationale, one per string.",
    )
    skeptic_notes: list[str] = Field(default_factory=list)
    generated_at: datetime


# --- Internal: search + extraction ------------------------------------------


class SearchResult(BaseModel):
    """One raw hit from the search provider. Never carried forward verbatim."""

    title: str
    url: HttpUrl
    snippet: str = ""
    provider: str = "unknown"


class SourceSummary(BaseModel):
    """≤200-token summary of a single source. This is what agents see."""

    url: HttpUrl
    title: str
    summary: str
    layer_hints: list[Layer] = Field(default_factory=list)
    entities: list[str] = Field(
        default_factory=list,
        description="Companies, tickers, or technologies named in the source.",
    )
    relevance: float = Field(ge=0.0, le=1.0, default=0.5)

    @field_validator("summary")
    @classmethod
    def _cap_summary(cls, v: str) -> str:
        # Soft cap; tier-small is prompted to stay under 200 tokens but rough
        # char-cap here defends against runaway returns.
        if len(v) > 1600:
            return v[:1600].rstrip() + "…"
        return v


# --- Internal: researcher planning ------------------------------------------


class SearchQuery(BaseModel):
    query: str
    target_layer: Layer
    rationale: str = ""


class QueryPlan(BaseModel):
    """A round of search queries produced by the researcher planner."""

    queries: list[SearchQuery]


class CoverageGap(BaseModel):
    layer: Layer
    missing_aspect: str
    suggested_query: str


class CoverageReport(BaseModel):
    """Self-critique after a research round; drives deepening loops."""

    per_layer_counts: dict[str, int]
    gaps: list[CoverageGap] = Field(default_factory=list)
    done: bool = False


class ResearchBundle(BaseModel):
    """Everything the researcher hands to the skeptic / synthesizer."""

    theme: str
    rounds: int
    summaries: list[SourceSummary]
    queries_run: list[SearchQuery]
    draft_nodes: list[Node] = Field(default_factory=list)


class NodeList(BaseModel):
    """Wrapper used by Instructor when asking for a flat list of nodes."""

    nodes: list[Node] = Field(default_factory=list)


# --- Internal: skeptic ------------------------------------------------------


CritiqueAngle = Literal["bottleneck_vs_commodity", "substitution_risk", "priced_in"]


class Critique(BaseModel):
    """One angle of attack on one node."""

    node_name: str
    angle: CritiqueAngle
    verdict: Literal["weak", "mixed", "strong"]
    argument: str
    suggested_change: str = Field(
        default="",
        description="Concrete edit to the node: drop, demote confidence, add risk, etc.",
    )


class SkepticReport(BaseModel):
    """Aggregate of per-node critiques. Feeds the synthesizer."""

    theme: str
    critiques: list[Critique]
    headline_concerns: list[str] = Field(
        default_factory=list,
        description="Top-level concerns that should appear in the final memo.",
    )


# --- Internal: synthesizer output ------------------------------------------


class FinalOutput(BaseModel):
    """What the synthesizer produces: structured thesis + two text artifacts."""

    thesis: Thesis
    memo_md: str = Field(description="Full markdown memo, ready to write to memo.md.")
    diagram_mmd: str = Field(
        description="Mermaid graph source, ready to write to diagram.mmd."
    )


# --- Internal: trace --------------------------------------------------------


class TraceEvent(BaseModel):
    """One line in trace.jsonl. Written by llm.py and tools.py."""

    ts: datetime
    kind: Literal["llm", "search", "fetch", "cache_hit", "cache_miss", "error"]
    tier: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    cache_key: str | None = None
    note: str | None = None
