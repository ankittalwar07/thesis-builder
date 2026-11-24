"""FastAPI server + static frontend.

Run with:
    uv run thesis-serve --demo                     # canned data, no API keys needed
    uv run thesis-serve --live                     # real API calls (uses your .env)
    uv run thesis-serve --live --record runs/      # also save each run for replay
    uv run thesis-serve --replay runs/             # booth fallback: no network at all

Endpoints:
    GET  /                          static UI
    POST /api/run                   start a run, returns {"run_id": "..."}
    POST /api/challenge/{run_id}    a visitor challenges one node of a finished run;
                                    starts a child run in which the Synthesizer must
                                    answer it under the same protocol
    GET  /api/events/{run_id}       SSE: live trace events, protocol checks, lifecycle
    GET  /api/result/{run_id}       final thesis, critiques beside answers, protocol result
    GET  /api/runs                  list known run ids
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from . import protocol
from .graph import build_graph
from .llm import LLMClient, TraceWriter
from .protocol import ProtocolResult
from .publish import slugify
from .schemas import Critique, FinalOutput, ResearchBundle, SkepticReport
from .tools import CacheBundle, WebSearch

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


# --- Run tracking -----------------------------------------------------------


class Run:
    """One in-flight or completed pipeline run."""

    def __init__(
        self,
        run_id: str,
        theme: str,
        demo: bool,
        *,
        kind: str = "run",
        parent_id: str | None = None,
        show_refusal: bool = False,
        replay: dict[str, Any] | None = None,
    ) -> None:
        self.id = run_id
        self.theme = theme
        self.demo = demo
        self.kind = kind  # "run" or "challenge"
        self.parent_id = parent_id
        self.show_refusal = show_refusal
        self.replay = replay  # a recorded run, when the server runs with --replay
        self.created_at = datetime.now(timezone.utc).isoformat()
        self._t0 = time.monotonic()
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.log: list[dict[str, Any]] = []  # (time offset, event) for --record
        self.completed = threading.Event()
        self.result: dict[str, Any] | None = None
        self.state: dict[str, Any] = {}  # research / skeptic / final, for challenges
        self.error: str | None = None
        self.metrics: dict[str, Any] = {
            "cost_usd": 0.0,
            "tokens_by_tier": {"small": 0, "reasoning": 0, "quality": 0},
            "cache_hit_rate": 0.0,
        }

    def push(self, event: dict[str, Any]) -> None:
        self.log.append({"t": round(time.monotonic() - self._t0, 3), "ev": event})
        self.events.put(event)


_RUNS: dict[str, Run] = {}
_RUNS_LOCK = threading.Lock()


# --- Pipeline executor ------------------------------------------------------


def _run_pipeline(run: Run) -> None:
    """Build and execute the graph in a worker thread."""
    def emit(event: dict[str, Any]) -> None:
        # Lightweight enrichment for the UI.
        run.push({"type": "trace", **event})
        run.metrics["cost_usd"] = float(trace.total_cost_usd)
        run.metrics["tokens_by_tier"] = dict(trace.tokens_by_tier)

    trace = TraceWriter(on_event=emit)
    cache = CacheBundle(os.environ.get("THESIS_CACHE_DIR", ".diskcache_demo" if run.demo else ".diskcache"))

    if run.demo:
        from .demo import DemoLLM, DemoSearch, offline_fetch

        # Swap fetch_source for canned snippets (process-wide while the run
        # lasts; restored in the finally block below).
        fetch_patch = offline_fetch()
        fetch_patch.__enter__()

        llm: LLMClient = DemoLLM(theme=run.theme, trace=trace, fault_once=run.show_refusal)
        search: Any = DemoSearch(theme=run.theme, trace=trace)
    else:
        try:
            llm = LLMClient(trace=trace)
        except FileNotFoundError as e:
            run.error = f"config not found: {e}"
            run.push({"type": "error", "note": run.error})
            run.completed.set()
            return
        search = WebSearch(cache=cache, trace=trace)
        fetch_patch = None

    try:
        run.push({"type": "stage", "stage": "researcher", "status": "started"})
        graph = build_graph(
            llm=llm, search=search, cache=cache,
            on_protocol=lambda attempt, res: run.push(_protocol_event(attempt, res)),
        )
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
                run.push(
                    {"type": "stage", "stage": done_name, "status": done_status}
                )
                if next_name:
                    run.push(
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
            run.state = {
                "research": bundle,
                "skeptic": skeptic,
                "final": final,
                "protocol": final_state.get("protocol"),
                "protocol_refused": final_state.get("protocol_refused", []),
            }
            run.result = _result_payload(run)
            run.push({"type": "done", "metrics": run.metrics,
                      "published": run.result["published"]})
        else:
            run.error = "no final output produced"
            run.push({"type": "error", "note": run.error})
    except Exception as e:  # noqa: BLE001
        run.error = f"{type(e).__name__}: {e}"
        run.push({"type": "error", "note": run.error})
    finally:
        if fetch_patch is not None:
            fetch_patch.__exit__(None, None, None)
        run.completed.set()


def _protocol_event(attempt: int, res: ProtocolResult) -> dict[str, Any]:
    return {
        "type": "protocol",
        "attempt": attempt,
        "passed": res.passed,
        "errors": list(res.errors),
        "warnings": list(res.warnings),
    }


def _result_payload(run: Run) -> dict[str, Any]:
    """What the UI shows: the tree, and every critique beside its answer."""
    st = run.state
    final: FinalOutput = st["final"]
    research: ResearchBundle | None = st.get("research")
    skeptic: SkepticReport | None = st.get("skeptic")
    check: ProtocolResult | None = st.get("protocol")
    answers = {a.critique_id: a for a in final.answers}
    critiques = []
    for c in skeptic.critiques if skeptic else []:
        a = answers.get(c.id)
        critiques.append({
            **json.loads(c.model_dump_json()),
            "stance": a.stance if a else None,
            "rebuttal": a.rebuttal if a else None,
            "changes": list(a.changes) if a else [],
        })
    passed = bool(check is None or check.passed)
    return {
        "published": passed,
        "kind": run.kind,
        "parent_id": run.parent_id,
        "replay": bool(run.replay),
        "recorded_at": (run.replay or {}).get("recorded_at"),
        "recorded_mode": (run.replay or {}).get("mode"),
        # Nothing is published when the check fails: the UI shows the errors instead.
        "thesis": json.loads(final.thesis.model_dump_json()) if passed else None,
        "memo_md": final.memo_md if passed else "",
        "diagram_mmd": final.diagram_mmd if passed else "",
        "draft_nodes": [json.loads(n.model_dump_json()) for n in research.draft_nodes] if research else [],
        "critiques": critiques,
        "protocol": {
            "passed": passed,
            "attempts": check.attempts if check else 1,
            "errors": list(check.errors) if check else [],
            "warnings": list(check.warnings) if check else [],
            "refused": list(st.get("protocol_refused", [])),
        },
        "metrics": run.metrics,
    }


# --- Visitor challenge ------------------------------------------------------


def _run_challenge(run: Run, parent: Run, node_name: str, argument: str) -> None:
    """Re-run only the Synthesizer, answering one new critique from a visitor.

    The 'draft' for this step is the parent's published tree; the visitor's
    critique must get exactly one answer, and any change it claims must be
    visible in the new tree. Answers to the earlier critiques are carried over
    and re-checked against the original draft, so a revision cannot quietly
    undo an earlier concession.
    """
    try:
        pst = parent.state
        research: ResearchBundle = pst["research"]
        skeptic: SkepticReport = pst["skeptic"]
        pfinal: FinalOutput = pst["final"]
        n_visitor = sum(1 for c in skeptic.critiques if c.angle == "visitor")
        visitor = Critique(
            id=f"V{n_visitor + 1}", node_name=node_name, angle="visitor", verdict="weak",
            argument=argument, suggested_change="",
        )
        run.push({"type": "stage", "stage": "synthesizer", "status": "started"})
        run.push({"type": "challenge", "critique": json.loads(visitor.model_dump_json())})

        if run.demo or run.replay:
            # No model available: deterministic answer, clearly labeled.
            from .demo import rule_based_answers

            nodes, ans = rule_based_answers(pfinal.thesis.nodes, [visitor])
            new_final = pfinal.model_copy(deep=True)
            new_final.thesis.nodes = nodes
            new_final.answers = ans
            res = protocol.check(pfinal.thesis.nodes, nodes, [visitor], ans)
            run.push({"type": "trace", "kind": "synthesizer", "tier": "offline",
                      "note": "offline rule-based answer to the visitor challenge (no model call)"})
            run.push(_protocol_event(1, res))
        else:
            from .agents.synthesizer import Synthesizer

            trace = TraceWriter(on_event=lambda ev: run.push({"type": "trace", **ev}))
            synth = Synthesizer(
                llm=LLMClient(trace=trace),
                on_protocol=lambda attempt, r: run.push(_protocol_event(attempt, r)),
            )
            # Only the challenged node is rewritten, so this takes one short call.
            nodes, answer, res = synth.answer_challenge(parent.theme, pfinal.thesis.nodes, visitor)
            new_final = pfinal.model_copy(deep=True)
            new_final.thesis.nodes = nodes
            ans = [answer]

        # Carry the earlier answers over and re-check them against the original draft.
        earlier = protocol.check(research.draft_nodes, new_final.thesis.nodes,
                                 skeptic.critiques, pfinal.answers)
        merged = ProtocolResult(errors=res.errors + [f"earlier answer: {e}" for e in earlier.errors],
                                warnings=res.warnings, attempts=res.attempts)
        new_final = new_final.model_copy(update={"answers": [*pfinal.answers, *ans]})
        run.state = {
            "research": research,
            "skeptic": skeptic.model_copy(update={"critiques": [*skeptic.critiques, visitor]}),
            "final": new_final,
            "protocol": merged,
            "protocol_refused": pst.get("protocol_refused", []),
        }
        if earlier.errors:
            run.push(_protocol_event(res.attempts, merged))
        run.push({"type": "stage", "stage": "synthesizer", "status": "completed"})
        run.result = _result_payload(run)
        run.push({"type": "done", "metrics": run.metrics, "published": run.result["published"]})
    except Exception as e:  # noqa: BLE001
        run.error = f"{type(e).__name__}: {e}"
        run.push({"type": "error", "note": run.error})
    finally:
        run.completed.set()


# --- Record and replay ------------------------------------------------------


def _record(run: Run, record_dir: Path) -> Path | None:
    if not run.state or run.replay or run.kind != "run":
        return None
    record_dir.mkdir(parents=True, exist_ok=True)
    path = record_dir / f"{slugify(run.theme)}-{run.id}.json"
    st = run.state
    blob = {
        "theme": run.theme,
        "recorded_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mode": "demo" if run.demo else "live",
        "events": run.log,
        "result": run.result,
        "state": {
            "research": json.loads(st["research"].model_dump_json()),
            "skeptic": json.loads(st["skeptic"].model_dump_json()),
            "final": json.loads(st["final"].model_dump_json()),
            "protocol": {"errors": st["protocol"].errors, "warnings": st["protocol"].warnings,
                         "attempts": st["protocol"].attempts} if st.get("protocol") else None,
            "protocol_refused": st.get("protocol_refused", []),
        },
    }
    path.write_text(json.dumps(blob, indent=1, default=str) + "\n")
    return path


def load_recordings(replay_dir: Path) -> dict[str, dict[str, Any]]:
    """Newest recording per theme slug."""
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(replay_dir.glob("*.json")):
        try:
            blob = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        slug = slugify(blob.get("theme", ""))
        if slug not in out or blob.get("recorded_at", "") >= out[slug].get("recorded_at", ""):
            out[slug] = blob
    return out


def _replay(run: Run, speed: float) -> None:
    """Stream a recorded run's events with their original pacing (gaps capped)."""
    rec = run.replay or {}
    try:
        last = 0.0
        for item in rec.get("events", []):
            gap = max(0.0, float(item.get("t", 0)) - last)
            last = float(item.get("t", 0))
            if speed > 0:
                time.sleep(min(gap, 2.0) / speed)
            run.events.put(item["ev"])
        st = rec["state"]
        pr = st.get("protocol") or {}
        run.state = {
            "research": ResearchBundle.model_validate(st["research"]),
            "skeptic": SkepticReport.model_validate(st["skeptic"]),
            "final": FinalOutput.model_validate(st["final"]),
            "protocol": ProtocolResult(errors=pr.get("errors", []), warnings=pr.get("warnings", []),
                                       attempts=pr.get("attempts", 1)) if pr else None,
            "protocol_refused": st.get("protocol_refused", []),
        }
        run.metrics = (rec.get("result") or {}).get("metrics", run.metrics)
        run.result = _result_payload(run)
    except Exception as e:  # noqa: BLE001
        run.error = f"replay failed: {type(e).__name__}: {e}"
        run.events.put({"type": "error", "note": run.error})
    finally:
        run.completed.set()


# --- Schemas ----------------------------------------------------------------


class RunRequest(BaseModel):
    theme: str
    demo: bool = True
    # Demo mode only: the first Synthesizer attempt claims a concession it did
    # not make, so visitors can watch the checker refuse it.
    show_refusal: bool = False


class ChallengeRequest(BaseModel):
    node_name: str
    argument: str = Field(min_length=3, max_length=400)


# --- App --------------------------------------------------------------------


def build_app(
    default_demo: bool = True,
    *,
    record_dir: str | Path | None = None,
    replay_dir: str | Path | None = None,
    replay_speed: float = 1.0,
) -> FastAPI:
    app = FastAPI(title="thesis-builder", version="0.2.0")
    record_path = Path(record_dir) if record_dir else None
    recordings = load_recordings(Path(replay_dir)) if replay_dir else {}
    replay_mode = replay_dir is not None

    def _start(run: Run, target: Any, *args: Any) -> None:
        def go() -> None:
            target(run, *args)
            if record_path is not None and run.error is None:
                _record(run, record_path)

        with _RUNS_LOCK:
            _RUNS[run.id] = run
        threading.Thread(target=go, daemon=True).start()

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
            "replay": replay_mode,
            "replay_themes": sorted(r.get("theme", "") for r in recordings.values()),
            "recording": record_path is not None,
        }

    @app.post("/api/run")
    async def start_run(req: RunRequest) -> dict[str, Any]:
        theme = req.theme.strip()
        if not theme:
            raise HTTPException(400, "theme is required")
        run_id = uuid.uuid4().hex[:12]
        if replay_mode:
            rec = recordings.get(slugify(theme))
            if rec is None:
                raise HTTPException(
                    404, f"no recorded run for {theme!r}; recorded: "
                    + ", ".join(sorted(r.get("theme", "") for r in recordings.values()))
                )
            run = Run(run_id=run_id, theme=rec["theme"], demo=rec.get("mode") == "demo", replay=rec)
            _start(run, _replay, replay_speed)
        else:
            demo = req.demo or default_demo
            run = Run(run_id=run_id, theme=theme, demo=demo, show_refusal=demo and req.show_refusal)
            _start(run, _run_pipeline)
        return {"run_id": run_id, "theme": run.theme, "demo": run.demo, "replay": bool(run.replay)}

    @app.post("/api/challenge/{run_id}")
    async def challenge(run_id: str, req: ChallengeRequest) -> dict[str, Any]:
        parent = _RUNS.get(run_id)
        if parent is None:
            raise HTTPException(404, "no such run")
        if not parent.completed.is_set() or not parent.state:
            raise HTTPException(409, "the run has not finished yet")
        if parent.result is None or not parent.result.get("published"):
            raise HTTPException(409, "nothing was published for this run, so there is nothing to challenge")
        names = {n.name for n in parent.state["final"].thesis.nodes}
        if req.node_name not in names:
            raise HTTPException(400, f"{req.node_name!r} is not a node in this run's tree")
        argument = " ".join(req.argument.split())
        child = Run(run_id=uuid.uuid4().hex[:12], theme=parent.theme, demo=parent.demo,
                    kind="challenge", parent_id=parent.id, replay=parent.replay)
        _start(child, _run_challenge, parent, req.node_name, argument)
        return {"run_id": child.id, "parent_id": parent.id, "theme": child.theme}

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        with _RUNS_LOCK:
            return {
                "runs": [
                    {
                        "id": r.id,
                        "theme": r.theme,
                        "demo": r.demo,
                        "kind": r.kind,
                        "parent_id": r.parent_id,
                        "replay": bool(r.replay),
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
            yield {"event": "start", "data": json.dumps(
                {"theme": run.theme, "demo": run.demo, "kind": run.kind, "replay": bool(run.replay)}
            )}
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
    parser.add_argument(
        "--record", metavar="DIR",
        help="Save every finished run (events with timing, result, state) to DIR for replay.",
    )
    parser.add_argument(
        "--replay", metavar="DIR",
        help="Booth fallback: serve recorded runs from DIR. No network or API keys used; "
        "visitor challenges are answered by an offline rule-based Synthesizer.",
    )
    parser.add_argument(
        "--config", metavar="YAML",
        help="Model tier config for live runs (e.g. config/models.local.yaml for a local model "
        "server). Same as setting THESIS_MODELS_CONFIG.",
    )
    parser.add_argument("--replay-speed", type=float, default=1.0,
                        help="Replay pacing multiplier (2 = twice as fast, 0 = instant).")
    args = parser.parse_args(argv)

    if args.config:
        os.environ["THESIS_MODELS_CONFIG"] = args.config
    default_demo = True
    if args.live:
        default_demo = False
    elif args.demo:
        default_demo = True

    import uvicorn

    app = build_app(default_demo=default_demo, record_dir=args.record,
                    replay_dir=args.replay, replay_speed=args.replay_speed)
    mode = "replay" if args.replay else ("demo" if default_demo else "live")
    print(
        f"\n  thesis-builder UI:  http://{args.host}:{args.port}/  "
        f"(default mode: {mode}{', recording' if args.record else ''})\n"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
