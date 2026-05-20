# thesis-builder

Multi-agent CLI that turns any investment theme (e.g. "AI infrastructure",
"robotics", "energy transition") into an investable value-chain tree —
raw inputs → enabling layers → integrators → applications — with public
tickers, private names, catalysts, risks, and a skeptical pass.

> Status: **scaffolding in progress**. Researcher pipeline complete;
> Skeptic, Synthesizer, LangGraph wiring, CLI, and evals still to come.

## Architecture (target)

```
       theme
         │
         ▼
   ┌───────────┐    map-reduce summaries (tier-small)
   │ Researcher├──────────────► SourceSummary[]  ─┐
   └─────┬─────┘                                  │
         │ draft Node[]                           │
         ▼                                        │
   ┌───────────┐    parallel critiques            │
   │  Skeptic  │    (tier-reasoning)              │
   └─────┬─────┘                                  │
         │ rebuttals                              │
         ▼                                        │
   ┌───────────┐    final synthesis               │
   │Synthesizer│◄───────────────────────────────  ┘
   └─────┬─────┘    (tier-quality, Anthropic prompt-cached)
         │
         ▼
  thesis.json · memo.md · diagram.mmd · trace.jsonl
```

Three model tiers, all routed through LiteLLM:

| Tier         | Used for                                  | Default model                  |
|--------------|-------------------------------------------|--------------------------------|
| `small`      | per-source summarization, dedup, planning | `groq/llama-3.1-8b-instant`    |
| `reasoning`  | coverage audit, skeptic critiques, ranking| `deepseek/deepseek-chat`       |
| `quality`    | **final synthesis only**                  | `anthropic/claude-sonnet-4-5`  |

See [`config/models.yaml`](config/models.yaml) for the full fallback chain.

## Setup

```bash
uv sync
cp .env.example .env
# fill in at least one tier-small key (Groq free tier works out of the box)
```

## Free-provider setup (cheapest path)

For a $0 tier-small + tier-reasoning run, set these two:

- `GROQ_API_KEY` — Groq's free tier covers Llama-3.1-8B for summarization.
- `OPENROUTER_API_KEY` — OpenRouter offers free DeepSeek for reasoning.

You still need an `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` fallback) for the
final synthesizer pass. That's typically <$0.10 per run thanks to prompt
caching.

`TAVILY_API_KEY` is optional — without it, search falls back to DuckDuckGo
(no key required).

## Usage

```bash
uv run thesis "AI infrastructure"
```

Outputs land under `outputs/<theme>/`:

```
outputs/ai_infrastructure/
├── thesis.json     # structured tree + picks
├── memo.md         # written-up investment memo
├── diagram.mmd     # Mermaid value-chain diagram
└── trace.jsonl     # every tool + model call with tier, tokens, cost
```

## Project layout

```
thesis-builder/
├── config/models.yaml
├── src/thesis_builder/
│   ├── agents/{researcher,skeptic,synthesizer}.py
│   ├── llm.py        # LiteLLM + Instructor wrapper, cost tracking, prompt cache
│   ├── tools.py      # web_search, fetch_source, EDGAR (all diskcache-backed)
│   ├── schemas.py    # Pydantic v2 models (Node, Thesis, internals)
│   ├── prompts.py    # versioned PROMPT_V1 … (cache keys include the version)
│   ├── graph.py      # LangGraph wiring + token-budget enforcement
│   └── cli.py
├── evals/            # LLM-as-judge across 3 golden themes
└── .github/workflows/eval.yml
```

## Acceptance targets (v1)

- `uv run thesis "AI infrastructure"` finishes in <8 min
- Researcher: ≥15 distinct searches, ≥30 unique source URLs cited
- Skeptic: ≥3 substantive critiques that change synthesizer output
- Cost <$0.40/run (hard cap configurable via `THESIS_MAX_COST_USD`)
- ≥80% of tokens routed to tier-small or tier-reasoning
- Cache hit rate ≥60% on re-runs of the same theme
