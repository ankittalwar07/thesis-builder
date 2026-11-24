"""Obligated rebuttal: the rule that makes the Skeptic's critiques binding.

Every critique targets one node of the Researcher's draft tree. Before anything
is published, the Synthesizer must give exactly one answer per critique, and
every change it claims must be visible in the diff between the draft tree and
the final tree. This module checks that with plain Python, no LLM.

Rules (numbered as in the README):

  R1 coverage      every critique has exactly one answer, and every answer
                   points at a real critique.
  R2 grounding     every critique targets a node in the draft tree.
  R3 consistency   ``conceded`` claims at least one change; ``held`` claims
                   none and keeps the node; ``partial`` keeps the node; every
                   claimed change is real:
                     confidence_lowered  node kept, final confidence < draft
                     risk_added          node kept, a risk not in the draft
                     dropped             node not in the final tree
  R4 disclosure    the published file carries every critique with its answer
                   (enforced by the publish contract, not here).

Warnings (do not block publishing): a ``weak`` verdict answered ``held``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .schemas import Critique, CritiqueAnswer, Node

_DASHES = re.compile(r"\s*[\u2014\u2013]\s*")


@dataclass
class ProtocolResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempts: int = 1  # Synthesizer attempts it took (set by the Synthesizer)

    @property
    def passed(self) -> bool:
        return not self.errors


def assign_ids(critiques: Iterable[Critique], prefix: str = "C", start: int = 1) -> list[Critique]:
    """Give every critique without an id a stable one (C1, C2, ...). Existing ids are kept."""
    out: list[Critique] = []
    used = {c.id for c in critiques if c.id}
    n = start
    for c in critiques:
        if c.id:
            out.append(c)
            continue
        while f"{prefix}{n}" in used:
            n += 1
        out.append(c.model_copy(update={"id": f"{prefix}{n}"}))
        used.add(f"{prefix}{n}")
        n += 1
    return out


def _norm(text: str) -> str:
    return " ".join(_DASHES.sub(", ", text).lower().split())


def check(
    draft_nodes: Iterable[Node],
    final_nodes: Iterable[Node],
    critiques: Iterable[Critique],
    answers: Iterable[CritiqueAnswer],
) -> ProtocolResult:
    """Check R1 to R3. Returns every problem found, not just the first."""
    res = ProtocolResult()
    draft = {n.name: n for n in draft_nodes}
    final = {n.name: n for n in final_nodes}
    crits = list(critiques)
    by_id: dict[str, Critique] = {}

    for c in crits:
        if not c.id:
            res.errors.append(f"critique on {c.node_name!r} ({c.angle}) has no id")
            continue
        if c.id in by_id:
            res.errors.append(f"duplicate critique id {c.id}")
            continue
        by_id[c.id] = c
        if c.node_name not in draft:  # R2
            res.errors.append(f"{c.id} targets {c.node_name!r}, which is not in the draft tree")

    answered: dict[str, CritiqueAnswer] = {}
    for a in answers:
        if a.critique_id not in by_id:  # R1
            res.errors.append(f"answer for unknown critique {a.critique_id!r}")
            continue
        if a.critique_id in answered:  # R1
            res.errors.append(f"{a.critique_id} has more than one answer")
            continue
        answered[a.critique_id] = a

    for cid, c in by_id.items():
        a = answered.get(cid)
        if a is None:  # R1
            res.errors.append(f"{cid} ({c.node_name}, {c.angle}) has no answer")
            continue
        if not a.rebuttal.strip():
            res.errors.append(f"{cid} has an empty rebuttal")
        d = draft.get(c.node_name)
        f = final.get(c.node_name)
        if d is None:
            continue  # already reported under R2
        changes = list(dict.fromkeys(a.changes))

        # R3: stance against claimed changes and node presence.
        if a.stance == "conceded" and not changes:
            res.errors.append(f"{cid} is conceded but claims no change to {c.node_name!r}")
        if a.stance == "held":
            if changes:
                res.errors.append(f"{cid} is held but claims changes {changes}")
            if f is None:
                res.errors.append(f"{cid} is held but {c.node_name!r} was dropped")
        if a.stance == "partial" and f is None:
            res.errors.append(f"{cid} is partial but {c.node_name!r} was dropped (that is a concession)")

        # R3: every claimed change must be real.
        for ch in changes:
            if ch == "dropped":
                if f is not None:
                    res.errors.append(f"{cid} claims {c.node_name!r} was dropped, but it is still in the final tree")
            elif f is None:
                res.errors.append(f"{cid} claims {ch} on {c.node_name!r}, but the node was dropped")
            elif ch == "confidence_lowered":
                if not f.confidence < d.confidence - 1e-9:
                    res.errors.append(
                        f"{cid} claims confidence was lowered on {c.node_name!r}, "
                        f"but it went {d.confidence:.2f} to {f.confidence:.2f}"
                    )
            elif ch == "risk_added":
                if not {_norm(r) for r in f.risks} - {_norm(r) for r in d.risks}:
                    res.errors.append(f"{cid} claims a risk was added to {c.node_name!r}, but its risks are unchanged")

        if c.verdict == "weak" and a.stance == "held":
            res.warnings.append(f"{cid} has a weak verdict but was held; check the rebuttal")
    return res


def feedback(result: ProtocolResult) -> str:
    """Message sent back to the Synthesizer when its answers fail the check."""
    lines = "\n".join(f"- {e}" for e in result.errors)
    return (
        "Protocol check failed on your previous output. Nothing was published.\n"
        f"{lines}\n\n"
        "Return a corrected FinalOutput. Give exactly one answer per critique id. "
        "Only claim a change you actually made to the final tree, and make every "
        "change you claim."
    )
