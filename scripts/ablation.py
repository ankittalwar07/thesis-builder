#!/usr/bin/env python3
"""Does the Skeptic change what gets published? A controlled comparison.

    uv run python scripts/ablation.py --demo                       # offline check of the script
    uv run python scripts/ablation.py --max-cost 0.60              # real runs (paid API calls)
    uv run python scripts/ablation.py --themes "AI infrastructure" "Robotics"

For each theme it runs the Researcher and the Skeptic once, then the
Synthesizer twice on the same research: once with the critiques (the normal
pipeline, under the rebuttal check) and once with an empty critique list.
Holding the research fixed isolates the critic's effect from search noise.

Writes outputs/ablation/<slug>/{with_skeptic,without_skeptic}.json plus
outputs/ablation/summary.json and prints a table with LaTeX rows for the paper.
Counts only: with three themes there is no claim of statistical significance.
Judge scores are separate (evals/run_eval.py on the two theses).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from thesis_builder.agents.synthesizer import Synthesizer  # noqa: E402
from thesis_builder.graph import build_graph  # noqa: E402
from thesis_builder.llm import LLMClient, TraceWriter  # noqa: E402
from thesis_builder.protocol import _norm  # noqa: E402
from thesis_builder.publish import slugify  # noqa: E402
from thesis_builder.schemas import Node, SkepticReport  # noqa: E402
from thesis_builder.tools import CacheBundle, WebSearch  # noqa: E402


def diff_counts(draft: list[Node], final: list[Node]) -> dict[str, Any]:
    d = {n.name: n for n in draft}
    f = {n.name: n for n in final}
    kept = [name for name in d if name in f]
    return {
        "draft_nodes": len(d),
        "final_nodes": len(f),
        "dropped": sorted(set(d) - set(f)),
        "added": sorted(set(f) - set(d)),
        "demoted": sorted(n for n in kept if f[n].confidence < d[n].confidence - 1e-9),
        "risks_added": sorted(
            n for n in kept if {_norm(r) for r in f[n].risks} - {_norm(r) for r in d[n].risks}
        ),
        "mean_confidence": round(sum(n.confidence for n in f.values()) / max(1, len(f)), 3),
    }


def run_theme(theme: str, args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    slug = slugify(theme)
    out = out_root / slug
    out.mkdir(parents=True, exist_ok=True)
    trace = TraceWriter(out / "trace.jsonl")
    if args.demo:
        from thesis_builder.demo import DemoLLM, DemoSearch, offline_fetch

        llm: Any = DemoLLM(theme=theme, trace=trace)
        search: Any = DemoSearch(theme=theme, trace=trace)
        cache = CacheBundle(str(ROOT / ".diskcache_demo"))
        ctx: Any = offline_fetch()
    else:
        llm = LLMClient(config_path=args.config, trace=trace, max_cost_usd=args.max_cost)
        cache = CacheBundle(args.cache_dir)
        search = WebSearch(cache=cache, trace=trace)
        ctx = contextlib.nullcontext()

    with ctx:
        result = build_graph(llm=llm, search=search, cache=cache).invoke({"theme": theme})
        research, skeptic, with_final = result["research"], result["skeptic"], result["final"]
        check = result["protocol"]
        without_final, _ = Synthesizer(llm=llm).run_checked(
            research, SkepticReport(theme=theme, critiques=[])
        )

    for name, final in (("with_skeptic", with_final), ("without_skeptic", without_final)):
        (out / f"{name}.json").write_text(final.model_dump_json(indent=2) + "\n")
    stances: dict[str, int] = {}
    for a in with_final.answers:
        stances[a.stance] = stances.get(a.stance, 0) + 1
    row = {
        "theme": theme,
        "mode": "demo" if args.demo else "live",
        "critiques": len(skeptic.critiques),
        "stances": stances,
        "rebuttal_check": {"passed": check.passed, "attempts": check.attempts},
        "with_skeptic": diff_counts(research.draft_nodes, with_final.thesis.nodes),
        "without_skeptic": diff_counts(research.draft_nodes, without_final.thesis.nodes),
        "cost_usd": round(trace.total_cost_usd, 4),
    }
    (out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--themes", nargs="*", help="Default: every theme in routines/themes.yaml")
    ap.add_argument("--themes-file", default=str(ROOT / "routines" / "themes.yaml"))
    ap.add_argument("--demo", action="store_true", help="Stubbed LLM and search (no keys, no network).")
    ap.add_argument("--config", default=str(ROOT / "config" / "models.yaml"))
    ap.add_argument("--cache-dir", default=str(ROOT / ".diskcache"))
    ap.add_argument("--max-cost", type=float, default=None, help="Per-theme cost cap in USD.")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "ablation"))
    args = ap.parse_args(argv)

    themes = args.themes or [
        t["theme"] for t in yaml.safe_load(Path(args.themes_file).read_text())["themes"]
    ]
    out_root = Path(args.out)
    rows = [run_theme(t, args, out_root) for t in themes]
    (out_root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")

    print(f"{'theme':24} {'crit':>5} {'dropped w/wo':>13} {'demoted w/wo':>13} {'risks+ w/wo':>12}")
    for r in rows:
        w, wo = r["with_skeptic"], r["without_skeptic"]
        print(f"{r['theme'][:24]:24} {r['critiques']:>5} "
              f"{len(w['dropped']):>6}/{len(wo['dropped']):<6} "
              f"{len(w['demoted']):>6}/{len(wo['demoted']):<6} "
              f"{len(w['risks_added']):>5}/{len(wo['risks_added']):<6}")
    print("\nLaTeX rows (Theme & Critiques & Nodes dropped & Nodes demoted & Judge):")
    for r in rows:
        w = r["with_skeptic"]
        print(f"{r['theme']} & {r['critiques']} & {len(w['dropped'])} & {len(w['demoted'])} & TODO \\\\")
    if any(r["mode"] == "demo" for r in rows):
        print("\n(demo mode: stubbed models; these numbers are NOT results for the paper)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
