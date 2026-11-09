# Routine: weekly-theses

A standalone prompt for a **scheduled Claude Code session** (suggested: weekly,
Sunday 7:00 AM America/Chicago). In this mode Claude itself plays the three
agents of the pipeline, so the run makes **zero paid LLM API calls**. The output
is committed to `docs/data/` and rendered by the GitHub Pages UI.

One-line prompt for the trigger:

> Follow routines/weekly-theses.md in this repo exactly: research every theme in routines/themes.yaml, publish to docs/data, validate, and push only docs/data to main. Never call paid LLM APIs.

Everything below is the prompt the session follows.

---

## Your job

You are running the `thesis-builder` multi-agent pipeline by hand, as the model.
For every theme in `routines/themes.yaml` you will act as the **Researcher**, the
**Skeptic** (three critique angles) and the **Synthesizer**, following the repo's
real prompts and schemas, then publish one JSON file per theme for the static UI
in `docs/`, validate it with a no-LLM script, and push the data to `main`.

## Hard rules (read twice)

1. **Never call paid LLM APIs.** You are the model. Do not run `uv run thesis`
   without `--demo`, do not run `python -m evals.run_eval` (it calls LiteLLM), do
   not import `litellm` or any provider SDK, do not read or set API keys, and do
   not create a `.env`. Your own web search and web fetch tools are allowed and
   expected.
2. **Never commit secrets.** Commit only files under `docs/data/`. If anything
   that looks like a key, token, cookie or password appears in your output, remove
   it before committing.
3. **Cite sources with URLs.** Every node cites at least one real `https://` URL
   that you actually opened during this run. Never cite a URL you did not open,
   and never use example.com or other placeholders.
4. **Do not invent facts.** Only use tickers, private company names, numbers and
   dates that appear in sources you read. If the evidence does not support a
   claim, leave it out.
5. **Not investment advice.** This is research automation for a portfolio demo.
   Write like an analyst; never tell a reader to buy or sell. The UI and README
   already carry the disclaimer; do not remove it anywhere.
6. **House style:** no em dashes or en dashes anywhere in the output (the
   validator rejects them). Use commas, colons or parentheses.
7. Touch nothing outside `docs/data/` and the scratch folder `work/` (which is
   git-ignored). Do not change code, prompts, workflows, settings or other
   branches.

## 0. Setup

```bash
git checkout main && git pull --ff-only origin main
uv sync || pip install "pydantic>=2.9" "pyyaml>=6.0"   # validator needs only these two
mkdir -p work
```

If `uv` is unavailable, run every `uv run python X` below as
`PYTHONPATH=src python3 X` instead.

Read these files before starting; they are the contract you follow:

| File | What you take from it |
|---|---|
| `routines/themes.yaml` | The themes to research and their golden specs |
| `src/thesis_builder/prompts.py` | The exact agent prompts (`*_SYSTEM_V1`, `*_USER_V1`) |
| `src/thesis_builder/schemas.py` | `SourceSummary`, `Node`, `Critique`, `Thesis`, `JudgeScores` |
| `src/thesis_builder/publish.py` | `PublishedThesis`, the JSON shape the UI reads |
| `src/thesis_builder/agents/skeptic.py` | `_evidence_for_node`: how evidence packs are picked |
| `evals/judges.py` | `JUDGE_SYSTEM`, the scoring rubric |
| `evals/golden/*.yaml` | Hard requirements per theme (min nodes, URLs, named clusters) |
| `docs/data/themes/*.json` | Last week's output (or the seed sample) for shape reference |

Use today's date in America/Chicago as `run_date` (format `YYYY-MM-DD`).

## 1. Researcher (per theme)

Follow `PLANNER_SYSTEM_V1`, `SUMMARIZER_SYSTEM_V1`, `COVERAGE_SYSTEM_V1` and
`DRAFT_TREE_SYSTEM_V1` from `prompts.py`.

1. **Plan** a round of 6 to 8 specific search queries spread across the four
   layers (`raw_inputs`, `enabling`, `integrators`, `applications`), weighted to
   the layers with the least coverage so far.
2. **Search** each query with your web search tool. Collect unique result URLs.
3. **Map:** open each promising URL and write one `SourceSummary` per source
   (summary under 200 tokens; names, numbers, dates and causal claims only;
   `entities`, `layer_hints`, an honest `relevance` from 0 to 1). Drop anything
   under 0.3 relevance. You may hand batches of URLs to parallel subagents for
   this map step; each returns `SourceSummary` JSON objects.
4. **Reduce:** de-duplicate summaries whose entity sets overlap heavily, keeping
   the more relevant one.
5. **Audit coverage** per `COVERAGE_SYSTEM_V1`. If a layer has fewer than 4
   substantive sources or fewer than 3 named entities, plan another round aimed at
   the gaps. Stop after at most 3 rounds.
6. Targets: at least 15 distinct searches, and at least the golden spec's
   `min_unique_urls` (25 for the seeded themes) unique summarized sources.
7. **Draft tree:** 8 to 16 `Node` objects across all four layers. Every node
   cites at least one summarized URL; tickers only if they appear verbatim in
   the summaries; private companies under `private_names`.

Save your working notes to `work/<slug>/research.json` (summaries, queries,
draft nodes). This folder is never committed.

`<slug>` is the theme lowercased with every run of non-alphanumeric characters
replaced by `_` (for example `AI infrastructure` becomes `ai_infrastructure`).

## 2. Skeptic (per theme): three angles, in parallel

Follow `SKEPTIC_SYSTEM_V1` / `SKEPTIC_USER_V1`. For **every** draft node, write
one `Critique` for **each** of the three angles:

- `bottleneck_vs_commodity`: true chokepoint with pricing power, or commoditizing?
- `substitution_risk`: what could displace it in 3 to 7 years?
- `priced_in`: is the bull case already in valuations and consensus?

Preferred: launch **three subagents in parallel, one per angle**. Give each one
only the draft tree, its angle, and per node an evidence pack of the 4 most
relevant summaries (score = 2 x entity overlap + 1 if the layer matches +
relevance, as in `skeptic._evidence_for_node`). Do not give them your own
reasoning; the point is an independent adversary. Each returns a JSON list of
`Critique` objects. If subagents are unavailable, do the three angles yourself
one at a time and argue against your own draft honestly.

`verdict` is `weak` only when you can state the failure mode, `strong` when the
bull case survives the angle. Then roll the critiques up into 3 to 6 specific,
non-overlapping `headline_concerns` per `SKEPTIC_ROLLUP_SYSTEM_V1`.

## 3. Synthesizer (per theme)

Follow `SYNTHESIZER_SYSTEM_V1` to produce the `Thesis`, `memo_md` and
`diagram_mmd`, with these overrides:

- Investable picks: at most 5, formatted `TICKER: one-sentence why` (a colon, not
  a dash), mixed across layers.
- Memo title line: `# <theme>: Value Chain Thesis`.
- Demote `confidence` on nodes the skeptic marked `weak`; drop a node only if the
  critique shows the segment does not belong in the chain.
- `skeptic_notes` = the headline concerns, verbatim.
- `diagram_mmd` starts with `graph LR`, four `subgraph` blocks (one per layer),
  labels `name<br/>up to 3 tickers`, edges from raw inputs to applications.
- **Rebuttals:** for every critique, add `stance` and `rebuttal`:
  - `held`: the bull case survives; say why in one or two sentences.
  - `partial`: the concern is real but bounded; say where the memo carries it.
  - `conceded`: say exactly what changed (confidence from X to Y, risk added, node
    dropped).

  The rebuttal must match what you actually did to the tree.
  Critiques must name a node that is in the final tree. If you dropped a node,
  re-point its critiques to the node that absorbed it or leave them out.

## 4. Publish (per theme)

Write `docs/data/themes/<slug>.json` matching `PublishedThesis` in
`src/thesis_builder/publish.py`:

```json
{
  "schema_version": 1,
  "slug": "<slug>",
  "theme": "<theme exactly as in themes.yaml>",
  "run_date": "YYYY-MM-DD",
  "sample": false,
  "generated_by": "claude-routine",
  "thesis": { "theme": "...", "summary": "...", "nodes": [ ...Node... ],
              "investable_picks": [ "TICKER: why" ], "skeptic_notes": [ ... ],
              "generated_at": "<ISO 8601 UTC timestamp>" },
  "memo_md": "# <theme>: Value Chain Thesis ...",
  "diagram_mmd": "graph LR ...",
  "critiques": [ { "node_name": "...", "angle": "priced_in", "verdict": "mixed",
                   "argument": "...", "suggested_change": "...",
                   "stance": "partial", "rebuttal": "..." } ],
  "headline_concerns": [ "..." ],
  "research": { "queries_run": 0, "unique_urls": 0, "summaries": 0,
                "search_queries": [ "..." ] },
  "eval": null
}
```

`research` holds the real counts from this run. Overwrite the previous file for
the same slug (this also replaces the seed `sample: true` files). If a theme
could not be researched properly (for example, search kept failing), leave its
previous file untouched and say so in your final summary; never publish a
partial thesis.

## 5. Score against the eval rubric (recommended)

Mirror `evals/judges.py`: the judge sees **only** the final thesis JSON and the
memo, never the research notes. Preferred: launch a **fresh subagent** with
`JUDGE_SYSTEM` as its instructions and the thesis JSON plus memo as input; it
returns `JudgeScores` JSON (coverage, specificity, novelty, actionability, each
an integer 0 to 5 with a one-sentence justification). Be harsh, as the rubric
says: a generic thesis scores 2 per dimension. If you cannot use a subagent,
score it yourself and label it as a self-score.

Store it in the file:

```json
"eval": { "judge": "claude-subagent (rubric: evals/judges.py)",
          "scores": { "coverage": { "score": 4, "justification": "..." }, "...": {} },
          "total": 14, "scored_at": "YYYY-MM-DD" }
```

Use `"judge": "claude-self-score (rubric: evals/judges.py)"` for a self-score.
`total` must equal the sum of the four scores.

## 6. Validate (no LLM)

```bash
uv run python scripts/validate_thesis.py --write
```

This checks every file against the pydantic schemas, adds the golden-spec hard
checks to each file's `eval.hard_checks`, and rebuilds `docs/data/index.json`.

- Fix every `error` and re-run until it reports `0 error(s)`.
- A `warn: golden hard check` means the thesis misses a golden requirement (too
  few URLs, a missing company cluster). Do **one** more research round for that
  theme, update the file, and validate again. If it still warns, publish anyway;
  the failure is recorded in the file and shown in the UI.

## 7. Commit and push only `docs/data/`

```bash
git add docs/data
git diff --cached --name-only            # every path must start with docs/data/
git status --porcelain                   # nothing else may be staged
git commit -m "data: weekly theses <run_date> (<N> themes)"
git pull --rebase origin main
git push origin main
```

If `git diff --cached --name-only` lists anything outside `docs/data/`, unstage
it (`git restore --staged <path>`) before committing. If the push fails on a
network error, retry up to 4 times, waiting 2, 4, 8 and 16 seconds. Do not force
push.

## 8. Final summary

End the session with a short summary: per theme, the node count, unique URLs,
critique count, eval total and hard-check result; any theme you skipped and why;
and the commit hash you pushed.
