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

from .schemas import Critique, JudgeScores, Layer, Thesis

SCHEMA_VERSION = 1
LAYERS: tuple[Layer, ...] = ("raw_inputs", "enabling", "integrators", "applications")
Stance = Literal["held", "partial", "conceded"]
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


def stance_for(verdict: str) -> Stance:
    return {"strong": "held", "mixed": "partial"}.get(verdict, "conceded")  # type: ignore[return-value]


def derived_rebuttal(c: Critique, final_conf: float | None, draft_conf: float | None) -> str:
    """Rebuttal text for pipeline runs, derived from the verdict and the
    confidence change the synthesizer actually made. The Claude routine
    writes its own rebuttals instead."""
    conf = f"{final_conf:.2f}" if final_conf is not None else "n/a"
    if c.verdict == "strong":
        return f"Defense holds under this angle; the node stays in the tree at confidence {conf}."
    if c.verdict == "mixed":
        return (
            "Partly conceded. The concern is carried into the memo's "
            f"'What we'd have to be wrong about' section; confidence {conf}."
        )
    change = c.suggested_change.strip() or "flag the failure mode as a key risk"
    moved = (
        f" (draft {draft_conf:.2f})"
        if draft_conf is not None and final_conf is not None and abs(draft_conf - final_conf) > 1e-9
        else ""
    )
    return f"Conceded. Skeptic's suggested change: {change}. Final confidence {conf}{moved}."


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

    final_conf = {n.name: n.confidence for n in thesis.nodes}
    draft_conf = {n.name: n.confidence for n in (research.draft_nodes if research else [])}

    critiques: list[PublishedCritique] = []
    for c in sorted(
        skeptic.critiques if skeptic else [],
        key=lambda c: (c.node_name, c.angle),
    ):
        critiques.append(
            PublishedCritique(
                node_name=c.node_name,
                angle=c.angle,
                verdict=c.verdict,
                argument=undash(c.argument),
                suggested_change=undash(c.suggested_change),
                stance=stance_for(c.verdict),
                rebuttal=derived_rebuttal(
                    c, final_conf.get(c.node_name), draft_conf.get(c.node_name)
                ),
            )
        )

    # House style: no em or en dashes in published copy.
    thesis.summary = undash(thesis.summary)
    thesis.investable_picks = [_undash_pick(p) for p in thesis.investable_picks]
    thesis.skeptic_notes = [undash(s) for s in thesis.skeptic_notes]
    for n in thesis.nodes:
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
