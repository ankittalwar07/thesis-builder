"""CLI entry point: `uv run thesis "<theme>"`.

Sets up the LLM gateway, web search, and cache; binds trace.jsonl to the
per-theme output directory; runs the graph; writes thesis.json, memo.md,
diagram.mmd; prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from .graph import build_graph
from .llm import BudgetExceeded, LLMClient, TraceWriter
from .schemas import FinalOutput
from .tools import CacheBundle, WebSearch


def _slug(theme: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", theme.lower()).strip("_")
    return s or "theme"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="thesis",
        description="Build an investable value-chain thesis for any theme.",
    )
    parser.add_argument("theme", help='Investment theme, e.g. "AI infrastructure"')
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root directory for per-theme outputs (default: outputs)",
    )
    parser.add_argument(
        "--config",
        default="config/models.yaml",
        help="Model tier config (default: config/models.yaml)",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("THESIS_CACHE_DIR", ".diskcache"),
        help="diskcache root directory (default: .diskcache)",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Hard cost cap in USD (default: THESIS_MAX_COST_USD env or unlimited)",
    )
    args = parser.parse_args(argv)

    console = Console()
    slug = _slug(args.theme)
    out_dir = Path(args.output_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    # Truncate any prior trace for this theme so token / cost counters are
    # per-run rather than cumulative across runs of the same theme.
    trace_path.write_text("")

    trace = TraceWriter(trace_path)
    try:
        llm = LLMClient(
            config_path=args.config, trace=trace, max_cost_usd=args.max_cost
        )
    except FileNotFoundError as e:
        console.print(f"[red]config not found: {e}[/red]")
        return 2

    cache = CacheBundle(args.cache_dir)
    search = WebSearch(cache=cache, trace=trace)

    console.print(
        Panel.fit(
            f"[bold]Building thesis for:[/bold] {args.theme}\n"
            f"[dim]output: {out_dir}\n"
            f"cache: {args.cache_dir}\n"
            f"cost cap: ${llm.max_cost_usd if llm.max_cost_usd is not None else 'unlimited'}[/dim]",
            title="thesis-builder",
        )
    )

    graph = build_graph(llm=llm, search=search, cache=cache, console=console)
    t0 = time.perf_counter()
    try:
        result = graph.invoke({"theme": args.theme})
    except BudgetExceeded as e:
        console.print(f"[red]cost cap hit before completion: {e}[/red]")
        return 3
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]graph failed: {type(e).__name__}: {e}[/red]")
        return 1
    elapsed = time.perf_counter() - t0

    final: FinalOutput | None = result.get("final")
    if final is None:
        console.print("[red]no final output produced[/red]")
        return 1

    # --- write artifacts ----------------------------------------------------

    thesis_path = out_dir / "thesis.json"
    thesis_path.write_text(final.thesis.model_dump_json(indent=2))

    memo_path = out_dir / "memo.md"
    memo_path.write_text(final.memo_md)

    diagram_path = out_dir / "diagram.mmd"
    diagram_path.write_text(final.diagram_mmd)

    # --- summary ------------------------------------------------------------

    bundle = result.get("research")
    skeptic = result.get("skeptic")
    table = Table(title="Run summary", show_lines=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("elapsed", f"{elapsed:.1f}s")
    table.add_row("nodes in final tree", str(len(final.thesis.nodes)))
    if bundle is not None:
        table.add_row("research summaries", str(len(bundle.summaries)))
        table.add_row("queries run", str(len(bundle.queries_run)))
        table.add_row(
            "unique source URLs", str(len({str(s.url) for s in bundle.summaries}))
        )
    if skeptic is not None:
        table.add_row("skeptic critiques", str(len(skeptic.critiques)))
    table.add_row("cache hit rate", f"{cache.hit_rate():.0%}")
    table.add_row("total cost (USD)", f"${trace.total_cost_usd:.4f}")
    total_tokens = sum(trace.tokens_by_tier.values())
    cheap = trace.tokens_by_tier["small"] + trace.tokens_by_tier["reasoning"]
    cheap_ratio = (cheap / total_tokens) if total_tokens else 0.0
    table.add_row(
        "tokens (small/reasoning/quality)",
        f"{trace.tokens_by_tier['small']} / "
        f"{trace.tokens_by_tier['reasoning']} / "
        f"{trace.tokens_by_tier['quality']}",
    )
    table.add_row("cheap-tier token share", f"{cheap_ratio:.0%}")
    console.print(table)
    console.print("\n[green]artifacts written:[/green]")
    console.print(f"  {thesis_path}")
    console.print(f"  {memo_path}")
    console.print(f"  {diagram_path}")
    console.print(f"  {trace_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
