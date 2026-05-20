"""LangGraph wiring + token-budget enforcement.

Linear three-node graph: researcher → skeptic → synthesizer. State is a
plain ``TypedDict`` (LangGraph's native style) carrying the bundle / report
/ final output between agents.

Before the synthesizer (the most expensive call), we measure the user-prompt
context size with tiktoken; if it exceeds the configured budget we trim the
researcher's source summaries by recency × relevance until it fits. Token
budgets for researcher / skeptic are advisory — those agents already chunk
their inputs internally.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph
from rich.console import Console

from . import prompts
from .agents import Researcher, Skeptic, Synthesizer
from .llm import LLMClient
from .schemas import FinalOutput, ResearchBundle, SkepticReport, SourceSummary
from .tools import CacheBundle, WebSearch


class ThesisState(TypedDict, total=False):
    theme: str
    research: ResearchBundle
    skeptic: SkepticReport
    final: FinalOutput


def build_graph(
    llm: LLMClient,
    search: WebSearch,
    cache: CacheBundle,
    console: Console | None = None,
):
    researcher = Researcher(llm=llm, search=search, cache=cache, console=console)
    skeptic = Skeptic(llm=llm, console=console)
    synthesizer = Synthesizer(llm=llm, console=console)
    budgets = llm.budgets

    def research_node(state: ThesisState) -> ThesisState:
        bundle = researcher.run(state["theme"])
        # Even though the researcher chunks its inputs, cap how much we hand
        # to downstream agents at the synthesizer budget — skeptic / synthesizer
        # would otherwise burn cost on duplicative low-relevance summaries.
        trimmed = _trim_summaries(
            bundle.summaries,
            target_tokens=budgets.get("synthesizer_context_tokens", 80_000),
            overhead=_estimate_overhead(state["theme"]),
        )
        if len(trimmed) != len(bundle.summaries):
            (console or Console(quiet=True)).log(
                f"[graph] trimmed summaries {len(bundle.summaries)} -> {len(trimmed)} "
                f"to fit synthesizer budget"
            )
            bundle = bundle.model_copy(update={"summaries": trimmed})
        return {"research": bundle}

    def skeptic_node(state: ThesisState) -> ThesisState:
        bundle = state["research"]
        skeptic_budget = budgets.get("skeptic_context_tokens", 30_000)
        # The skeptic only sees per-critique evidence packs (4 summaries each),
        # so the budget guard here is a defensive trim of the input bundle
        # rather than a per-call cap.
        if _token_estimate(_dump_summaries(bundle.summaries)) > skeptic_budget:
            bundle = bundle.model_copy(
                update={
                    "summaries": _trim_summaries(
                        bundle.summaries,
                        target_tokens=skeptic_budget,
                        overhead=2000,
                    )
                }
            )
        report = skeptic.run(bundle)
        return {"skeptic": report, "research": bundle}

    def synth_node(state: ThesisState) -> ThesisState:
        bundle = state["research"]
        skeptic_report = state["skeptic"]
        # Re-check budget against the actual synthesizer user prompt.
        user_msg = prompts.get("synthesizer_user").format(
            theme=bundle.theme,
            research_block=_dump_summaries(bundle.summaries),
            draft_tree_block="(elided for sizing)",
            skeptic_block="(elided for sizing)",
        )
        budget = budgets.get("synthesizer_context_tokens", 80_000)
        if _token_estimate(user_msg) > budget:
            bundle = bundle.model_copy(
                update={
                    "summaries": _trim_summaries(
                        bundle.summaries,
                        target_tokens=budget,
                        overhead=_estimate_overhead(bundle.theme),
                    )
                }
            )
        final = synthesizer.run(bundle, skeptic_report)
        return {"final": final}

    g = StateGraph(ThesisState)
    g.add_node("researcher", research_node)
    g.add_node("skeptic", skeptic_node)
    g.add_node("synthesizer", synth_node)
    g.set_entry_point("researcher")
    g.add_edge("researcher", "skeptic")
    g.add_edge("skeptic", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


# --- token accounting -------------------------------------------------------


def _token_estimate(text: str) -> int:
    """Best-effort token count via tiktoken, with a 4-char/token fallback."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _dump_summaries(summaries: list[SourceSummary]) -> str:
    return "\n".join(
        f"- [{','.join(s.layer_hints) or '?'}] {s.title} ({s.url}) :: "
        f"entities={','.join(s.entities[:6])} :: {s.summary}"
        for s in summaries
    )


def _estimate_overhead(theme: str) -> int:
    """Token overhead from system prompt + skeptic block + draft tree block.

    Rough constant — within 20% is fine for the trim heuristic.
    """
    return 4000 + _token_estimate(theme)


def _trim_summaries(
    summaries: list[SourceSummary],
    target_tokens: int,
    overhead: int,
) -> list[SourceSummary]:
    """Drop the lowest-relevance summaries until the dump fits.

    "Recency + relevance" per the brief — researcher emits summaries in
    discovery order, so original ordering already encodes recency. We keep
    that order for ties but score by relevance first.
    """
    if not summaries:
        return summaries
    budget = max(0, target_tokens - overhead)
    if _token_estimate(_dump_summaries(summaries)) <= budget:
        return summaries
    # Sort by relevance ASC so we pop the weakest first; preserve original
    # order for the kept set so the synthesizer sees recency-ordered evidence.
    indexed = list(enumerate(summaries))
    indexed.sort(key=lambda x: x[1].relevance)  # ascending
    drop_set: set[int] = set()
    keep_text = _dump_summaries(summaries)
    while _token_estimate(keep_text) > budget and len(drop_set) < len(indexed):
        idx, _ = indexed[len(drop_set)]
        drop_set.add(idx)
        kept = [s for i, s in enumerate(summaries) if i not in drop_set]
        keep_text = _dump_summaries(kept)
    return [s for i, s in enumerate(summaries) if i not in drop_set]
