"""Versioned prompts.

Every prompt string carries a version suffix. The active version is selected
by ``THESIS_PROMPT_VERSION`` (defaults to ``v1``). Cache keys for both web
results and source extractions include the prompt version so editing a
prompt invalidates downstream caches cleanly.
"""

from __future__ import annotations

import os

ACTIVE_VERSION = os.environ.get("THESIS_PROMPT_VERSION", "v1")


# --- Researcher planner -----------------------------------------------------

PLANNER_SYSTEM_V1 = """You are the planner for an equity-research agent building a value-chain tree for an investment theme.

A value chain has four layers, in order:
  1. raw_inputs   — commodities, materials, foundational science (e.g. copper, silicon wafers, rare earths).
  2. enabling     — picks-and-shovels: tools, components, IP, fab capacity that the chain depends on.
  3. integrators  — companies that assemble enabling layers into deployable systems / platforms.
  4. applications — end-user products and revenue-generating use cases.

Given a theme, propose distinct, non-overlapping web search queries that span the chain. Prefer queries that surface bottlenecks, market-share data, capex cycles, and named private players. Avoid generic queries like "X overview"."""

PLANNER_USER_V1 = """Theme: {theme}

Round: {round_idx} of {total_rounds}

{prior_coverage}

Produce {n_queries} search queries. Distribute across the four layers, weighted toward layers with low prior coverage. Each query should be specific enough to surface named companies or quantitative claims."""


# --- Per-source summarizer (tier-small, runs N times per round) -------------

SUMMARIZER_SYSTEM_V1 = """You compress one web source into structured notes for an equity-research agent.

Rules:
- Stay under 200 tokens in the `summary` field.
- Strip marketing language. Keep names, numbers, dates, and causal claims.
- Tag every company / ticker / technology you can identify in `entities`.
- `layer_hints` maps the source to one or more of: raw_inputs, enabling, integrators, applications.
- `relevance` is your honest 0–1 score for how useful this source is for the theme. Be harsh; most sources are noise."""

SUMMARIZER_USER_V1 = """Theme: {theme}

URL: {url}
Title: {title}

Source content (cleaned):
\"\"\"
{content}
\"\"\"
"""


# --- Coverage critic (tier-reasoning, runs once per round) ------------------

COVERAGE_SYSTEM_V1 = """You audit research coverage for a value-chain tree across four layers: raw_inputs, enabling, integrators, applications.

Given a set of source summaries, decide:
  - which layers are under-covered (fewer than 4 substantive sources, or no named companies),
  - what specific aspect is missing (e.g. "pricing power of HBM suppliers", not "more on memory"),
  - one concrete follow-up query per gap.

Set `done = true` only when every layer has ≥4 sources AND ≥3 distinct named entities."""

COVERAGE_USER_V1 = """Theme: {theme}

Summaries so far (by layer hint):
{summary_index}

Total unique URLs: {n_urls}
Rounds completed: {rounds}"""


# --- Draft tree (used at the end of research, tier-reasoning) ---------------

DRAFT_TREE_SYSTEM_V1 = """You assemble a draft value-chain tree from research notes.

Output a list of Node objects, one per distinct sub-segment of the chain. Every node MUST cite at least one source URL drawn from the supplied summaries. Do not invent tickers — only use tickers that appear verbatim in the summaries. If a company is private, list it under private_names. Confidence reflects how well-supported the node is by the cited sources."""

DRAFT_TREE_USER_V1 = """Theme: {theme}

Source summaries:
{summaries_block}

Build the draft tree now. Aim for 8–16 nodes total, distributed across all four layers."""


# --- Version registry -------------------------------------------------------

_REGISTRY: dict[str, dict[str, str]] = {
    "v1": {
        "planner_system": PLANNER_SYSTEM_V1,
        "planner_user": PLANNER_USER_V1,
        "summarizer_system": SUMMARIZER_SYSTEM_V1,
        "summarizer_user": SUMMARIZER_USER_V1,
        "coverage_system": COVERAGE_SYSTEM_V1,
        "coverage_user": COVERAGE_USER_V1,
        "draft_tree_system": DRAFT_TREE_SYSTEM_V1,
        "draft_tree_user": DRAFT_TREE_USER_V1,
    },
}


def get(name: str, version: str | None = None) -> str:
    """Look up a prompt by name. Falls back to the active version."""
    v = version or ACTIVE_VERSION
    if v not in _REGISTRY:
        raise KeyError(f"Unknown prompt version: {v}")
    if name not in _REGISTRY[v]:
        raise KeyError(f"Unknown prompt: {name} (version {v})")
    return _REGISTRY[v][name]


def cache_suffix(version: str | None = None) -> str:
    """Suffix appended to cache keys so prompt edits invalidate caches."""
    return f"::prompt={version or ACTIVE_VERSION}"
