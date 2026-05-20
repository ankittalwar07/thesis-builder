"""Synthesizer agent.

Merges the research bundle and skeptic report into a final ``FinalOutput``
(thesis JSON + memo markdown + Mermaid diagram). Uses tier-quality only —
the brief reserves the expensive tier for this single pass.

The system prompt is stable across runs, so Anthropic prompt caching
(applied automatically in ``llm.py`` for tier-quality) hits on every
call after the first.
"""

from __future__ import annotations

from datetime import datetime, timezone

from rich.console import Console

from .. import prompts
from ..llm import LLMClient
from ..schemas import FinalOutput, ResearchBundle, SkepticReport, SourceSummary


class Synthesizer:
    def __init__(self, llm: LLMClient, console: Console | None = None) -> None:
        self.llm = llm
        self.console = console or Console(quiet=True)

    def run(self, bundle: ResearchBundle, skeptic: SkepticReport) -> FinalOutput:
        msgs = [
            {"role": "system", "content": prompts.get("synthesizer_system")},
            {
                "role": "user",
                "content": prompts.get("synthesizer_user").format(
                    theme=bundle.theme,
                    research_block=_research_block(bundle.summaries),
                    draft_tree_block=_draft_tree_block(bundle),
                    skeptic_block=_skeptic_block(skeptic),
                ),
            },
        ]
        out: FinalOutput = self.llm.complete(
            tier="quality",
            messages=msgs,
            response_model=FinalOutput,
        )
        # Stamp the timestamp + theme ourselves rather than trust the model.
        out.thesis.generated_at = datetime.now(timezone.utc)
        out.thesis.theme = bundle.theme
        return out


# --- helpers ----------------------------------------------------------------


def _research_block(summaries: list[SourceSummary]) -> str:
    if not summaries:
        return "(no summaries)"
    lines = []
    for s in summaries:
        layer_tag = ",".join(s.layer_hints) or "?"
        ents = ", ".join(s.entities[:8])
        lines.append(
            f"- [{layer_tag}] {s.title} ({s.url})\n"
            f"  entities: {ents}\n"
            f"  summary: {s.summary}"
        )
    return "\n".join(lines)


def _draft_tree_block(bundle: ResearchBundle) -> str:
    if not bundle.draft_nodes:
        return "(researcher returned no draft nodes; build the tree from summaries alone)"
    lines = []
    for n in bundle.draft_nodes:
        tickers = ", ".join(n.public_names) or "—"
        priv = ", ".join(n.private_names) or "—"
        lines.append(
            f"- [{n.layer}] {n.name} (conf={n.confidence:.2f})\n"
            f"  tickers: {tickers}; private: {priv}\n"
            f"  desc: {n.description}"
        )
    return "\n".join(lines)


def _skeptic_block(report: SkepticReport) -> str:
    if not report.critiques:
        return "(skeptic produced no critiques)"
    lines = ["### Headlines"]
    for h in report.headline_concerns:
        lines.append(f"- {h}")
    lines.append("\n### Per-node critiques")
    for c in report.critiques:
        lines.append(
            f"- [{c.verdict}] {c.node_name} / {c.angle}: {c.argument}"
            + (f"  (suggest: {c.suggested_change})" if c.suggested_change else "")
        )
    return "\n".join(lines)
