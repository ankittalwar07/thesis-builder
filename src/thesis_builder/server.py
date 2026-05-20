"""FastAPI server + static frontend.

Run with:
    uv run thesis-serve --demo            # canned data, no API keys needed
    uv run thesis-serve                   # real API calls (uses your .env)

Endpoints:
    GET  /                       static UI
    POST /api/run                start a run, returns {"run_id": "..."}
    GET  /api/events/{run_id}    SSE: live trace events + lifecycle markers
    GET  /api/result/{run_id}    final FinalOutput JSON (after completion)
    GET  /api/runs               list known run ids
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from .graph import build_graph
from .llm import LLMClient, TraceWriter
from .tools import CacheBundle, WebSearch

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


# --- Run tracking -----------------------------------------------------------


class Run:
    """One in-flight or completed pipeline run."""

    def __init__(self, run_id: str, theme: str, demo: bool) -> None:
        self.id = run_id
        self.theme = theme
        self.demo = demo
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.completed = threading.Event()
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.metrics: dict[str, Any] = {
            "cost_usd": 0.0,
            "tokens_by_tier": {"small": 0, "reasoning": 0, "quality": 0},
            "cache_hit_rate": 0.0,
        }


_RUNS: dict[str, Run] = {}
_RUNS_LOCK = threading.Lock()


# --- Pipeline executor ------------------------------------------------------


def _run_pipeline(run: Run) -> None:
    """Build and execute the graph in a worker thread."""
    def emit(event: dict[str, Any]) -> None:
        # Lightweight enrichment for the UI.
        run.events.put({"type": "trace", **event})
        run.metrics["cost_usd"] = float(trace.total_cost_usd)
        run.metrics["tokens_by_tier"] = dict(trace.tokens_by_tier)

    trace = TraceWriter(on_event=emit)
    cache = CacheBundle(os.environ.get("THESIS_CACHE_DIR", ".diskcache_demo" if run.demo else ".diskcache"))

    if run.demo:
        from .demo import DemoLLM, DemoSearch, demo_fetch_source
        from . import tools as _tools_mod

        # Patch the module-level fetch_source the researcher uses. The patch
        # is process-wide, but since the demo only runs in this server it's
        # fine — and we restore the original on completion.
        _orig_fetch = _tools_mod.fetch_source
        _tools_mod.fetch_source = demo_fetch_source  # type: ignore[assignment]

        llm: LLMClient = DemoLLM(theme=run.theme, trace=trace)
        search: Any = DemoSearch(theme=run.theme, trace=trace)
    else:
        try:
            llm = LLMClient(trace=trace)
        except FileNotFoundError as e:
            run.error = f"config not found: {e}"
            run.events.put({"type": "error", "note": run.error})
            run.completed.set()
            return
        search = WebSearch(cache=cache, trace=trace)
        _tools_mod = None
        _orig_fetch = None

    try:
        run.events.put({"type": "stage", "stage": "researcher", "status": "started"})
        graph = build_graph(llm=llm, search=search, cache=cache)
        # Emit stage markers as nodes complete via LangGraph's stream.
        final_state: dict[str, Any] = {}
        for chunk in graph.stream({"theme": run.theme}):
            for node_name, state_delta in chunk.items():
                final_state.update(state_delta or {})
                next_stages = {
                    "researcher": ("researcher", "completed", "skeptic", "started"),
                    "skeptic": ("skeptic", "completed", "synthesizer", "started"),
                    "synthesizer": ("synthesizer", "completed", None, None),
                }.get(node_name)
                if not next_stages:
                    continue
                done_name, done_status, next_name, next_status = next_stages
                run.events.put(
                    {"type": "stage", "stage": done_name, "status": done_status}
                )
                if next_name:
                    run.events.put(
                        {"type": "stage", "stage": next_name, "status": next_status}
                    )

        final = final_state.get("final")
        bundle = final_state.get("research")
        skeptic = final_state.get("skeptic")
        run.metrics["cache_hit_rate"] = cache.hit_rate()
        run.metrics["unique_urls"] = (
            len({str(s.url) for s in bundle.summaries}) if bundle else 0
        )
        run.metrics["queries_run"] = len(bundle.queries_run) if bundle else 0
        run.metrics["summaries"] = len(bundle.summaries) if bundle else 0
        run.metrics["critiques"] = len(skeptic.critiques) if skeptic else 0

        if final is not None:
            run.result = {
                "thesis": json.loads(final.thesis.model_dump_json()),
                "memo_md": final.memo_md,
                "diagram_mmd": final.diagram_mmd,
                "metrics": run.metrics,
            }
            run.events.put({"type": "done", "metrics": run.metrics})
        else:
            run.error = "no final output produced"
            run.events.put({"type": "error", "note": run.error})
    except Exception as e:  # noqa: BLE001
        run.error = f"{type(e).__name__}: {e}"
        run.events.put({"type": "error", "note": run.error})
    finally:
        if _orig_fetch is not None and _tools_mod is not None:
            _tools_mod.fetch_source = _orig_fetch  # type: ignore[assignment]
        run.completed.set()


# --- Schemas ----------------------------------------------------------------


class RunRequest(BaseModel):
    theme: str
    demo: bool = True


# --- App --------------------------------------------------------------------


def build_app(default_demo: bool = True) -> FastAPI:
    app = FastAPI(title="thesis-builder", version="0.1.0")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        return {
            "default_demo": default_demo,
            "has_anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "has_groq": bool(os.environ.get("GROQ_API_KEY")),
            "has_openai": bool(os.environ.get("OPENAI_API_KEY")),
            "has_openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
            "has_tavily": bool(os.environ.get("TAVILY_API_KEY")),
        }

    @app.post("/api/run")
    async def start_run(req: RunRequest) -> dict[str, Any]:
        theme = req.theme.strip()
        if not theme:
            raise HTTPException(400, "theme is required")
        run_id = uuid.uuid4().hex[:12]
        run = Run(run_id=run_id, theme=theme, demo=req.demo or default_demo)
        with _RUNS_LOCK:
            _RUNS[run_id] = run
        threading.Thread(target=_run_pipeline, args=(run,), daemon=True).start()
        return {"run_id": run_id, "theme": run.theme, "demo": run.demo}

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        with _RUNS_LOCK:
            return {
                "runs": [
                    {
                        "id": r.id,
                        "theme": r.theme,
                        "demo": r.demo,
                        "completed": r.completed.is_set(),
                        "created_at": r.created_at,
                    }
                    for r in _RUNS.values()
                ]
            }

    @app.get("/api/result/{run_id}")
    async def get_result(run_id: str) -> JSONResponse:
        run = _RUNS.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        if not run.completed.is_set():
            raise HTTPException(409, "still running")
        if run.error:
            raise HTTPException(500, run.error)
        return JSONResponse(run.result or {})

    @app.get("/api/events/{run_id}")
    async def events(run_id: str) -> EventSourceResponse:
        run = _RUNS.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")

        async def gen():
            yield {"event": "start", "data": json.dumps({"theme": run.theme, "demo": run.demo})}
            while True:
                try:
                    ev = run.events.get(timeout=0.5)
                except queue.Empty:
                    if run.completed.is_set() and run.events.empty():
                        yield {
                            "event": "end",
                            "data": json.dumps(
                                {"ok": run.error is None, "error": run.error}
                            ),
                        }
                        return
                    await asyncio.sleep(0)
                    continue
                yield {"event": ev.get("type", "trace"), "data": json.dumps(ev, default=str)}

        return EventSourceResponse(gen())

    # Static assets (CSS / JS) under /static/*. The index page is served from /.
    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    return app


# --- entrypoint -------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="thesis-serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Default new runs to demo mode (no API keys required).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Default new runs to live mode (uses .env API keys).",
    )
    args = parser.parse_args(argv)

    default_demo = True
    if args.live:
        default_demo = False
    elif args.demo:
        default_demo = True

    import uvicorn

    app = build_app(default_demo=default_demo)
    print(
        f"\n  thesis-builder UI:  http://{args.host}:{args.port}/  "
        f"(default mode: {'demo' if default_demo else 'live'})\n"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
