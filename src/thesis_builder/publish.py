"""Published-thesis contract for the static UI in ``docs/``.

The GitHub Pages UI reads one JSON file per theme from
``docs/data/themes/<slug>.json`` plus an index at ``docs/data/index.json``.
Those files can be written three ways, all producing the same shape:

  * ``claude-routine``: a scheduled Claude Code session plays the three agents
    itself (see ``routines/weekly-theses.md``). No paid API calls.
  * ``litellm-pipeline``: ``thesis "<theme>" --publish-dir docs/data`` after a
    live LiteLLM run.
  * ``demo``: ``scripts/export_demo_data.py`` from the canned demo packs.

This module imports only pydantic and ``schemas`` so the no-LLM validator
(``scripts/validate_thesis.py``) runs without LiteLLM installed.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from . import protocol
from .schemas import ChangeKind, Critique, JudgeScores, Layer, Node, Stance, Thesis

SCHEMA_VERSION = 1
LAYERS: tuple[Layer, ...] = ("raw_inputs", "enabling", "integrators", "applications")
GeneratedBy = Literal["claude-routine", "litellm-pipeline", "demo"]


def slugify(theme: str) -> str:
    """Same slug rule as the CLI's output directories."""
    s = re.sub(r"[^a-z0-9]+", "_", theme.lower()).strip("_")
    return s or "theme"


# --- Published schema -------------------------------------------------------


class PublishedCritique(Critique):
    """A skeptic critique plus the synthesizer's answer to it."""

    stance: Stance = Field(
        description="held = bull case defended; partial = concern carried into the memo; "
        "conceded = node changed (demoted, risk added, or dropped)."
    )
    rebuttal: str = Field(min_length=1, description="One or two sentences answering the critique.")
    changes: list[ChangeKind] = Field(
        default_factory=list,
        description="Changes the answer claims on the target node; the validator checks "
        "each one against the draft_nodes to thesis.nodes diff.",
    )


class ProtocolBlock(BaseModel):
    """Outcome of the obligated-rebuttal check (protocol.py) for this file."""

    passed: bool
    attempts: int = Field(default=1, ge=1, description="Synthesizer attempts before it passed.")
    refused: list[str] = Field(
        default_factory=list,
        description="Errors from attempts that were refused before the published one.",
    )
    warnings: list[str] = Field(default_factory=list)


class ResearchStats(BaseModel):
    queries_run: int = Field(ge=0)
    unique_urls: int = Field(ge=0)
    summaries: int = Field(ge=0)
    search_queries: list[str] = Field(default_factory=list)


class HardChecks(BaseModel):
    golden: str | None = None
    passed: bool
    failures: list[str] = Field(default_factory=list)


class EvalBlock(BaseModel):
    judge: str | None = Field(
        default=None, description="Who scored it, e.g. 'claude-subagent (rubric: evals/judges.py)'."
    )
    scores: JudgeScores | None = None
    total: int | None = Field(default=None, ge=0, le=20)
    scored_at: date | None = None
    hard_checks: HardChecks | None = None


class PublishedThesis(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    slug: str
    theme: str
    run_date: date
    sample: bool = False
    generated_by: GeneratedBy
    thesis: Thesis
    draft_nodes: list[Node] = Field(
        default_factory=list,
        description="The Researcher's draft tree the critiques targeted. Needed to check "
        "that every claimed change is real.",
    )
    protocol: ProtocolBlock | None = None
    memo_md: str
    diagram_mmd: str
    critiques: list[PublishedCritique] = Field(default_factory=list)
    headline_concerns: list[str] = Field(default_factory=list)
    research: ResearchStats
    eval: EvalBlock | None = None


class IndexEntry(BaseModel):
    slug: str
    theme: str
    file: str
    run_date: date
    sample: bool
    generated_by: GeneratedBy
    nodes: int
    critiques: int
    eval_total: int | None = None


class PublishedIndex(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    updated_at: datetime
    themes: list[IndexEntry]


# --- Builders ---------------------------------------------------------------


class ProtocolRefused(RuntimeError):
    """Raised when the Synthesizer's answers fail the obligated-rebuttal check."""

    def __init__(self, result: protocol.ProtocolResult) -> None:
        super().__init__("; ".join(result.errors) or "protocol check failed")
        self.result = result


_DASHES = re.compile(r"\s*[\u2014\u2013]\s*")


def undash(text: str) -> str:
    """Replace em and en dashes with commas (house style for published copy)."""
    return _DASHES.sub(", ", text)


def _undash_pick(p: str) -> str:
    return _DASHES.sub(": ", p, count=1) if _DASHES.search(p) else p


def _undash_md(md: str) -> str:
    out = []
    for line in md.splitlines():
        if line.startswith("# "):
            out.append(_DASHES.sub(": ", line, count=1))
        else:
            out.append(undash(line))
    return "\n".join(out)


def from_pipeline(
    result: dict[str, Any],
    *,
    generated_by: GeneratedBy,
    sample: bool,
    run_date: date | None = None,
) -> PublishedThesis:
    """Convert a LangGraph result dict (``research``, ``skeptic``, ``final``)."""
    final = result["final"]
    research = result.get("research")
    skeptic = result.get("skeptic")
    thesis: Thesis = final.thesis.model_copy(deep=True)

    draft_nodes = [n.model_copy(deep=True) for n in (research.draft_nodes if research else [])]
    raw_critiques = list(skeptic.critiques) if skeptic else []
    answers = list(getattr(final, "answers", []) or [])

    # Obligated rebuttal: never publish answers that fail the check.
    check = result.get("protocol") or protocol.check(draft_nodes, thesis.nodes, raw_critiques, answers)
    if not check.passed:
        raise ProtocolRefused(check)
    refused = list(result.get("protocol_refused", []))

    by_id = {a.critique_id: a for a in answers}
    critiques: list[PublishedCritique] = []
    for c in sorted(raw_critiques, key=lambda c: (c.node_name, c.angle, c.id)):
        a = by_id[c.id]
        critiques.append(
            PublishedCritique(
                id=c.id,
                node_name=c.node_name,
                angle=c.angle,
                verdict=c.verdict,
                argument=undash(c.argument),
                suggested_change=undash(c.suggested_change),
                stance=a.stance,
                rebuttal=undash(a.rebuttal),
                changes=list(dict.fromkeys(a.changes)),
            )
        )

    # House style: no em or en dashes in published copy.
    thesis.summary = undash(thesis.summary)
    thesis.investable_picks = [_undash_pick(p) for p in thesis.investable_picks]
    thesis.skeptic_notes = [undash(s) for s in thesis.skeptic_notes]
    for n in [*thesis.nodes, *draft_nodes]:
        n.description = undash(n.description)
        n.catalysts = [undash(x) for x in n.catalysts]
        n.risks = [undash(x) for x in n.risks]

    stats = ResearchStats(
        queries_run=len(research.queries_run) if research else 0,
        unique_urls=len({str(s.url) for s in research.summaries}) if research else 0,
        summaries=len(research.summaries) if research else 0,
        search_queries=[q.query for q in research.queries_run] if research else [],
    )
    return PublishedThesis(
        slug=slugify(thesis.theme),
        theme=thesis.theme,
        run_date=run_date or datetime.now(timezone.utc).date(),
        sample=sample,
        generated_by=generated_by,
        thesis=thesis,
        draft_nodes=draft_nodes,
        protocol=ProtocolBlock(
            passed=True,
            attempts=getattr(check, "attempts", 1),
            refused=[undash(e) for e in refused],
            warnings=[undash(w) for w in check.warnings],
        ),
        memo_md=_undash_md(final.memo_md),
        diagram_mmd=undash(final.diagram_mmd),
        critiques=critiques,
        headline_concerns=[undash(h) for h in (skeptic.headline_concerns if skeptic else [])],
        research=stats,
    )


def write_published(pub: PublishedThesis, data_dir: str | Path) -> Path:
    path = Path(data_dir) / "themes" / f"{pub.slug}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pub.model_dump_json(indent=2) + "\n")
    return path


def build_index(data_dir: str | Path) -> PublishedIndex:
    data_dir = Path(data_dir)
    entries: list[IndexEntry] = []
    for p in sorted((data_dir / "themes").glob("*.json")):
        pub = PublishedThesis.model_validate_json(p.read_text())
        entries.append(
            IndexEntry(
                slug=pub.slug,
                theme=pub.theme,
                file=f"themes/{p.name}",
                run_date=pub.run_date,
                sample=pub.sample,
                generated_by=pub.generated_by,
                nodes=len(pub.thesis.nodes),
                critiques=len(pub.critiques),
                eval_total=pub.eval.total if pub.eval else None,
            )
        )
    return PublishedIndex(updated_at=datetime.now(timezone.utc).replace(microsecond=0), themes=entries)


# --- Golden-spec hard checks (no LLM) ---------------------------------------


def check_requirements(
    spec: dict[str, Any], thesis: dict[str, Any], memo_md: str, meta: dict[str, Any] | None
) -> tuple[bool, list[str]]:
    """Deterministic checks from an ``evals/golden/*.yaml`` spec.

    Shared by ``evals/run_eval.py`` and ``scripts/validate_thesis.py``.
    """
    failures: list[str] = []
    nodes = thesis.get("nodes", [])
    if len(nodes) < spec.get("min_nodes", 0):
        failures.append(f"only {len(nodes)} nodes (need >= {spec['min_nodes']})")

    layers_seen = {n.get("layer") for n in nodes}
    for layer in spec.get("expected_layers", []):
        if layer not in layers_seen:
            failures.append(f"missing layer: {layer}")

    blob = json.dumps(thesis).lower() + " " + memo_md.lower()
    for cluster in spec.get("must_mention_any", []):
        hit = any(re.search(rf"\b{re.escape(opt.lower())}\b", blob) for opt in cluster["options"])
        if not hit:
            failures.append(f"no mention of any of {cluster['cluster']}: {cluster['options']}")

    if meta:
        if meta.get("unique_urls", 0) < spec.get("min_unique_urls", 0):
            failures.append(
                f"only {meta['unique_urls']} unique URLs (need >= {spec['min_unique_urls']})"
            )
        if meta.get("critiques", 0) < spec.get("min_skeptic_critiques", 0):
            failures.append(
                f"only {meta['critiques']} skeptic critiques "
                f"(need >= {spec['min_skeptic_critiques']})"
            )
    return len(failures) == 0, failures
