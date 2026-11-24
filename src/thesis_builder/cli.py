"""CLI entry point: `uv run thesis "<theme>"`.

Sets up the LLM gateway, web search, and cache; binds trace.jsonl to the
per-theme output directory; runs the graph; writes thesis.json, memo.md,
diagram.mmd; prints a summary table.
"""

from __future__ import annotations

import argparse
import contextlib
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
from .llm import BudgetExceeded, LLMClient, TraceWriter, default_config_path
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
        default=default_config_path(),
        help="Model tier config (default: THESIS_MODELS_CONFIG or config/models.yaml; "
        "config/models.local.yaml runs on a local model server)",
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
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the real graph with stubbed LLM + search (no API keys, no network).",
    )
    parser.add_argument(
        "--publish-dir",
        default=None,
        help="Also write the GitHub Pages JSON (themes/<slug>.json + index.json) here, "
        "e.g. docs/data",
    )
    parser.add_argument(
        "--no-skeptic",
        action="store_true",
        help="Ablation: skip the Skeptic (writes to <slug>__no_skeptic, cannot be published). "
        "scripts/ablation.py is the controlled version that reuses one research run.",
    )
    parser.add_argument(
        "--demo-fault",
        action="store_true",
        help="Demo only: the first Synthesizer attempt claims a concession it did not make, "
        "to show the rebuttal check refusing it and the retry.",
    )
    args = parser.parse_args(argv)

    console = Console()
    if args.no_skeptic and args.publish_dir:
        console.print("[red]--no-skeptic runs are for comparison only and cannot be published[/red]")
        return 2
    slug = _slug(args.theme) + ("__no_skeptic" if args.no_skeptic else "")
    out_dir = Path(args.output_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    # Truncate any prior trace for this theme so token / cost counters are
    # per-run rather than cumulative across runs of the same theme.
    trace_path.write_text("")

    trace = TraceWriter(trace_path)
    if args.demo:
        from .demo import DemoLLM, DemoSearch, offline_fetch

        llm: LLMClient = DemoLLM(theme=args.theme, trace=trace, fault_once=args.demo_fault)
        search = DemoSearch(theme=args.theme, trace=trace)
        cache_dir = ".diskcache_demo"
        cache = CacheBundle(cache_dir)
        fetch_ctx = offline_fetch()
    else:
        try:
            llm = LLMClient(
                config_path=args.config, trace=trace, max_cost_usd=args.max_cost
            )
        except FileNotFoundError as e:
            console.print(f"[red]config not found: {e}[/red]")
            return 2
        cache_dir = args.cache_dir
        cache = CacheBundle(cache_dir)
        search = WebSearch(cache=cache, trace=trace)
        fetch_ctx = contextlib.nullcontext()

    mode = "demo (stubbed LLM + search, no API calls)" if args.demo else "live (LiteLLM)"
    cap = llm.max_cost_usd if llm.max_cost_usd is not None else "unlimited"
    console.print(
        Panel.fit(
            f"[bold]Building thesis for:[/bold] {args.theme}\n"
            f"[dim]mode: {mode}\n"
            f"output: {out_dir}\n"
            f"cache: {cache_dir}\n"
            f"cost cap: ${cap}[/dim]",
            title="thesis-builder",
        )
    )

    t0 = time.perf_counter()
    try:
        with fetch_ctx:
            graph = build_graph(
                llm=llm, search=search, cache=cache, console=console,
                skip_skeptic=args.no_skeptic,
                on_protocol=lambda attempt, res: console.print(
                    f"[{'green' if res.passed else 'red'}]rebuttal check, attempt {attempt}: "
                    + ("passed" if res.passed else "refused") + "[/]"
                    + "".join(f"\n  - {e}" for e in res.errors)
                ),
            )
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

    # --- obligated rebuttal ------------------------------------------------

    import json

    check = result.get("protocol")
    bundle = result.get("research")
    skeptic = result.get("skeptic")
    answers = {a.critique_id: a for a in final.answers}
    audit = {
        "protocol": {
            "passed": bool(check is None or check.passed),
            "attempts": getattr(check, "attempts", 1),
            "errors": list(getattr(check, "errors", [])),
            "warnings": list(getattr(check, "warnings", [])),
            "refused": list(result.get("protocol_refused", [])),
        },
        "draft_nodes": [json.loads(n.model_dump_json()) for n in (bundle.draft_nodes if bundle else [])],
        "critiques": [
            {**json.loads(c.model_dump_json()),
             "answer": json.loads(answers[c.id].model_dump_json()) if c.id in answers else None}
            for c in (skeptic.critiques if skeptic else [])
        ],
    }
    audit_path = out_dir / "rebuttals.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    if check is not None and not check.passed:
        console.print(
            "[red]rebuttal check refused the Synthesizer's answers; nothing published.[/red]\n"
            f"  details: {audit_path}"
        )
        return 4

    # --- write artifacts ----------------------------------------------------

    thesis_path = out_dir / "thesis.json"
    thesis_path.write_text(final.thesis.model_dump_json(indent=2))

    memo_path = out_dir / "memo.md"
    memo_path.write_text(final.memo_md)

    diagram_path = out_dir / "diagram.mmd"
    diagram_path.write_text(final.diagram_mmd)

    # --- summary ------------------------------------------------------------

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
        table.add_row(
            "rebuttal check",
            f"passed on attempt {getattr(check, 'attempts', 1)}" if check is not None else "n/a",
        )
    table.add_row("cache hit rate", f"{cache.hit_rate():.0%}")
    table.add_row(
        "total cost (USD, simulated)" if args.demo else "total cost (USD)",
        f"${trace.total_cost_usd:.4f}",
    )
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
    console.print(f"  {audit_path}  [dim](every critique beside its answer)[/dim]")
    console.print(f"  {trace_path}")

    if args.publish_dir:
        from . import publish

        pub = publish.from_pipeline(
            result,
            generated_by="demo" if args.demo else "litellm-pipeline",
            sample=args.demo,
        )
        path = publish.write_published(pub, args.publish_dir)
        index = publish.build_index(args.publish_dir)
        (Path(args.publish_dir) / "index.json").write_text(index.model_dump_json(indent=2) + "\n")
        console.print(f"  {path}  [dim](published for the Pages UI)[/dim]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
