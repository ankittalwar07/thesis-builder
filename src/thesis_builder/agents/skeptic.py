"""Skeptic agent.

For each draft node, runs three critique angles in parallel against
tier-reasoning:
  - bottleneck_vs_commodity
  - substitution_risk
  - priced_in

Each critique gets a small evidence pack drawn from the source summaries
most relevant to that node (entity-overlap + relevance). After the per-node
fan-out, a single rollup call (also tier-reasoning) condenses the critiques
into 3–6 headline concerns for the memo.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import get_args

from rich.console import Console

from .. import prompts
from ..llm import LLMClient
from ..schemas import (
    Critique,
    CritiqueAngle,
    Node,
    ResearchBundle,
    SkepticReport,
    SourceSummary,
)

ANGLES: tuple[CritiqueAngle, ...] = get_args(CritiqueAngle)


class Skeptic:
    def __init__(
        self,
        llm: LLMClient,
        console: Console | None = None,
        max_workers: int = 8,
        max_evidence_per_critique: int = 4,
    ) -> None:
        self.llm = llm
        self.console = console or Console(quiet=True)
        self.max_workers = max_workers
        self.max_evidence_per_critique = max_evidence_per_critique

    def run(self, bundle: ResearchBundle) -> SkepticReport:
        if not bundle.draft_nodes:
            return SkepticReport(theme=bundle.theme, critiques=[], headline_concerns=[])

        jobs: list[tuple[Node, CritiqueAngle]] = [
            (n, a) for n in bundle.draft_nodes for a in ANGLES
        ]
        critiques: list[Critique] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(self._critique_one, bundle.theme, node, angle, bundle.summaries): (
                    node.name,
                    angle,
                )
                for node, angle in jobs
            }
            for fut in as_completed(futures):
                node_name, angle = futures[fut]
                try:
                    c = fut.result()
                except Exception as e:  # noqa: BLE001
                    self.console.log(
                        f"[yellow]critique failed ({node_name}/{angle}): {e}[/yellow]"
                    )
                    continue
                if c is not None:
                    critiques.append(c)

        self.console.log(
            f"[skeptic] {len(critiques)} critiques across {len(bundle.draft_nodes)} nodes"
        )

        headlines = self._rollup(bundle.theme, critiques)
        return SkepticReport(
            theme=bundle.theme, critiques=critiques, headline_concerns=headlines
        )

    # ---- internals ---------------------------------------------------------

    def _critique_one(
        self,
        theme: str,
        node: Node,
        angle: CritiqueAngle,
        all_summaries: list[SourceSummary],
    ) -> Critique | None:
        evidence = _evidence_for_node(
            node, all_summaries, k=self.max_evidence_per_critique
        )
        msgs = [
            {"role": "system", "content": prompts.get("skeptic_system")},
            {
                "role": "user",
                "content": prompts.get("skeptic_user").format(
                    theme=theme,
                    node_name=node.name,
                    node_layer=node.layer,
                    node_description=node.description,
                    public_names=", ".join(node.public_names) or "(none)",
                    private_names=", ".join(node.private_names) or "(none)",
                    catalysts="; ".join(node.catalysts) or "(none)",
                    risks="; ".join(node.risks) or "(none)",
                    confidence=f"{node.confidence:.2f}",
                    angle=angle,
                    evidence_block=evidence,
                ),
            },
        ]
        try:
            crit: Critique = self.llm.complete(
                tier="reasoning", messages=msgs, response_model=Critique
            )
        except Exception:
            return None
        # Trust our copies of identifiers; the model may rename things.
        return crit.model_copy(update={"node_name": node.name, "angle": angle})

    def _rollup(self, theme: str, critiques: list[Critique]) -> list[str]:
        if not critiques:
            return []
        block = "\n".join(
            f"- [{c.verdict}] {c.node_name} / {c.angle}: {c.argument}"
            for c in critiques
        )
        msgs = [
            {"role": "system", "content": prompts.get("skeptic_rollup_system")},
            {
                "role": "user",
                "content": prompts.get("skeptic_rollup_user").format(
                    theme=theme, critiques_block=block
                ),
            },
        ]

        from pydantic import BaseModel, Field

        class _Headlines(BaseModel):
            headlines: list[str] = Field(default_factory=list)

        try:
            out: _Headlines = self.llm.complete(
                tier="reasoning", messages=msgs, response_model=_Headlines
            )
            return out.headlines
        except Exception:
            # Fallback: take strongest critiques verbatim.
            return [
                f"{c.node_name}: {c.argument}"
                for c in critiques
                if c.verdict == "weak"
            ][:6]


# --- helpers ----------------------------------------------------------------


def _evidence_for_node(node: Node, summaries: list[SourceSummary], k: int) -> str:
    """Pick the top-k summaries most relevant to a node.

    Score = entity overlap (weight 2) + layer match (weight 1) + relevance.
    Keeps tier-reasoning context tight so we can fan out cheaply.
    """
    if not summaries:
        return "(no source summaries available)"
    node_terms = {t.lower() for t in (node.public_names + node.private_names + [node.name])}
    scored: list[tuple[float, SourceSummary]] = []
    for s in summaries:
        ents = {e.lower() for e in s.entities}
        overlap = len(node_terms & ents)
        layer_match = 1 if node.layer in s.layer_hints else 0
        score = 2 * overlap + layer_match + s.relevance
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored[:k]]
    return "\n".join(
        f"- ({s.relevance:.2f}) {s.title} [{s.url}]: {s.summary}" for s in top
    )
