"""Run the thesis pipeline against the golden themes and score the output.

Two modes:
  - default: build a thesis from scratch for each theme, then judge.
  - --use-existing: judge whatever artifacts are already in outputs/<theme>/
    (cheap CI mode after a prior live run).

Exits non-zero if any golden theme falls below the per-dimension threshold
or fails its hard requirements (min nodes, min URLs, min critiques, named
entities present).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from thesis_builder.cli import _slug
from thesis_builder.llm import LLMClient, TraceWriter
from thesis_builder.graph import build_graph
from thesis_builder.publish import check_requirements
from thesis_builder.tools import CacheBundle, WebSearch

from .judges import JudgeScores, judge

GOLDEN_DIR = Path(__file__).parent / "golden"
DEFAULT_MIN_PER_DIM = 3


def _load_golden(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_one(
    theme: str, output_dir: Path, console: Console, demo: bool = False
) -> dict[str, Path]:
    slug = _slug(theme)
    out = output_dir / slug
    out.mkdir(parents=True, exist_ok=True)
    trace = TraceWriter(out / "trace.jsonl")
    if demo:
        from thesis_builder.demo import run_demo

        result = run_demo(theme, trace=trace, console=console)
    else:
        llm = LLMClient(trace=trace)
        cache = CacheBundle()
        search = WebSearch(cache=cache, trace=trace)
        graph = build_graph(llm=llm, search=search, cache=cache, console=console)
        result = graph.invoke({"theme": theme})
    final = result["final"]
    paths = {
        "thesis": out / "thesis.json",
        "memo": out / "memo.md",
        "diagram": out / "diagram.mmd",
    }
    paths["thesis"].write_text(final.thesis.model_dump_json(indent=2))
    paths["memo"].write_text(final.memo_md)
    paths["diagram"].write_text(final.diagram_mmd)
    # Also persist the raw research counts so the hard-requirement check has
    # something to read back in --use-existing mode.
    (out / "_eval_meta.json").write_text(
        json.dumps(
            {
                "summaries": len(result["research"].summaries),
                "queries": len(result["research"].queries_run),
                "unique_urls": len(
                    {str(s.url) for s in result["research"].summaries}
                ),
                "critiques": len(result["skeptic"].critiques),
            }
        )
    )
    return paths


def _hard_checks(spec: dict[str, Any], out_dir: Path) -> tuple[bool, list[str]]:
    thesis = json.loads((out_dir / "thesis.json").read_text())
    memo = (out_dir / "memo.md").read_text()
    meta_path = out_dir / "_eval_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return check_requirements(spec, thesis, memo, meta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Score whatever artifacts already exist in outputs/, do not rebuild.",
    )
    parser.add_argument("--min-per-dim", type=int, default=DEFAULT_MIN_PER_DIM)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Offline plumbing check: demo LLM + search for both build and judge "
        "(no API keys; scores are canned, so this tests wiring, not quality).",
    )
    parser.add_argument(
        "--themes", nargs="*", help="Subset of golden themes to run (by filename stem)."
    )
    args = parser.parse_args(argv)

    console = Console()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    golden_files = sorted(GOLDEN_DIR.glob("*.yaml"))
    if args.themes:
        wanted = set(args.themes)
        golden_files = [p for p in golden_files if p.stem in wanted]
    if not golden_files:
        console.print("[red]no golden themes selected[/red]")
        return 2

    if args.demo:
        from thesis_builder.demo import DemoLLM

        judge_llm: LLMClient = DemoLLM(theme="judge")
    else:
        judge_llm = LLMClient(trace=TraceWriter())

    table = Table(title="Eval results")
    table.add_column("theme")
    table.add_column("cov", justify="right")
    table.add_column("spec", justify="right")
    table.add_column("nov", justify="right")
    table.add_column("act", justify="right")
    table.add_column("total", justify="right")
    table.add_column("hard checks")

    any_fail = False
    for path in golden_files:
        spec = _load_golden(path)
        theme = spec["theme"]
        slug = _slug(theme)
        out_dir = output_dir / slug
        if not args.use_existing:
            console.print(f"[bold]Building {theme}…[/bold]")
            _build_one(theme, output_dir, console, demo=args.demo)
        elif not (out_dir / "thesis.json").exists():
            console.print(f"[red]no prior artifacts for {theme}, skipping[/red]")
            any_fail = True
            continue

        ok, failures = _hard_checks(spec, out_dir)
        thesis_json = (out_dir / "thesis.json").read_text()
        memo_md = (out_dir / "memo.md").read_text()
        scores: JudgeScores = judge(judge_llm, theme, thesis_json, memo_md)

        below = []
        for name in ("coverage", "specificity", "novelty", "actionability"):
            dim: Any = getattr(scores, name)
            if dim.score < args.min_per_dim:
                below.append(f"{name}={dim.score}")

        status = "ok" if ok and not below else "; ".join(failures + below) or "below threshold"
        if not ok or below:
            any_fail = True

        table.add_row(
            theme,
            str(scores.coverage.score),
            str(scores.specificity.score),
            str(scores.novelty.score),
            str(scores.actionability.score),
            str(scores.total),
            status,
        )

    console.print(table)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
