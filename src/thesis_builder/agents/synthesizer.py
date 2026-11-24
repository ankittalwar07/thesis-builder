"""Synthesizer agent.

Merges the research bundle and skeptic report into a final ``FinalOutput``
(thesis JSON + memo markdown + Mermaid diagram). Uses tier-quality only;
the brief reserves the expensive tier for this single pass.

The system prompt is stable across runs, so Anthropic prompt caching
(applied automatically in ``llm.py`` for tier-quality) hits on every
call after the first.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from rich.console import Console

from .. import prompts, protocol
from ..llm import LLMClient
from ..protocol import ProtocolResult
from ..schemas import FinalOutput, ResearchBundle, SkepticReport, SourceSummary

# Called after every attempt with (attempt number, check result).
ProtocolHook = Callable[[int, ProtocolResult], None]


class Synthesizer:
    """Writes the final thesis and must answer every critique (obligated rebuttal).

    After each attempt the protocol checker compares the draft tree with the
    final tree. If the answers fail, the Synthesizer gets the errors back and
    tries again, up to ``max_attempts``. The last result is returned either
    way; callers must not publish when ``result.passed`` is False.
    """

    def __init__(
        self,
        llm: LLMClient,
        console: Console | None = None,
        max_attempts: int = 2,
        on_protocol: ProtocolHook | None = None,
    ) -> None:
        self.llm = llm
        self.console = console or Console(quiet=True)
        self.max_attempts = max(1, max_attempts)
        self.on_protocol = on_protocol

    def run(self, bundle: ResearchBundle, skeptic: SkepticReport) -> FinalOutput:
        """Backward-compatible entry point: returns only the final output."""
        final, _ = self.run_checked(bundle, skeptic)
        return final

    def run_checked(
        self, bundle: ResearchBundle, skeptic: SkepticReport
    ) -> tuple[FinalOutput, ProtocolResult]:
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
        result = ProtocolResult()
        out: FinalOutput | None = None
        for attempt in range(1, self.max_attempts + 1):
            out = self.llm.complete(
                tier="quality",
                messages=msgs,
                response_model=FinalOutput,
            )
            # Stamp the timestamp + theme ourselves rather than trust the model.
            out.thesis.generated_at = datetime.now(timezone.utc)
            out.thesis.theme = bundle.theme
            result = protocol.check(
                bundle.draft_nodes, out.thesis.nodes, skeptic.critiques, out.answers
            )
            result.attempts = attempt
            self.console.log(
                f"[synthesizer] protocol attempt {attempt}: "
                + ("passed" if result.passed else f"{len(result.errors)} errors")
            )
            if self.on_protocol is not None:
                self.on_protocol(attempt, result)
            if result.passed:
                break
            msgs = msgs + [{"role": "user", "content": protocol.feedback(result)}]
        assert out is not None
        return out, result


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
        tickers = ", ".join(n.public_names) or "n/a"
        priv = ", ".join(n.private_names) or "n/a"
        lines.append(
            f"- [{n.layer}] {n.name} (conf={n.confidence:.2f})\n"
            f"  tickers: {tickers}; private: {priv}\n"
            f"  desc: {n.description}\n"
            f"  risks: {'; '.join(n.risks) or 'none'}"
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
            f"- {c.id} [{c.verdict}] {c.node_name} / {c.angle}: {c.argument}"
            + (f"  (suggest: {c.suggested_change})" if c.suggested_change else "")
        )
    return "\n".join(lines)
