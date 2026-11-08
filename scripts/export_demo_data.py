#!/usr/bin/env python3
"""Seed docs/data/ with sample theses from the offline demo pipeline.

    uv run python scripts/export_demo_data.py

Runs the real Researcher -> Skeptic -> Synthesizer LangGraph for every theme
in routines/themes.yaml with the stubbed demo LLM and search (no API keys, no
network), converts each result to the published JSON contract, marks it
``sample: true`` and rebuilds index.json. The first scheduled Claude run
overwrites these files with real research.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from thesis_builder import publish  # noqa: E402
from thesis_builder.demo import run_demo  # noqa: E402
from thesis_builder.publish import EvalBlock  # noqa: E402
from thesis_builder.schemas import DimensionScore, JudgeScores  # noqa: E402

CANNED = "Canned demo score. A real rubric score replaces it after the first Claude run."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "docs" / "data"))
    ap.add_argument("--themes-file", default=str(ROOT / "routines" / "themes.yaml"))
    ap.add_argument("--seed", type=int, default=7, help="Seed for the demo skeptic's verdicts.")
    args = ap.parse_args(argv)

    themes = [t["theme"] for t in yaml.safe_load(Path(args.themes_file).read_text())["themes"]]
    for theme in themes:
        result = run_demo(theme, seed=args.seed, cache_dir=str(ROOT / ".diskcache_demo"))
        pub = publish.from_pipeline(result, generated_by="demo", sample=True)
        scores = JudgeScores(
            coverage=DimensionScore(score=4, justification=CANNED),
            specificity=DimensionScore(score=4, justification=CANNED),
            novelty=DimensionScore(score=3, justification=CANNED),
            actionability=DimensionScore(score=4, justification=CANNED),
        )
        pub.eval = EvalBlock(
            judge="demo stub (canned scores, not a real evaluation)",
            scores=scores,
            total=scores.total,
            scored_at=pub.run_date,
        )
        path = publish.write_published(pub, args.data_dir)
        print(f"wrote {path.relative_to(ROOT)}: {len(pub.thesis.nodes)} nodes, "
              f"{len(pub.critiques)} critiques")

    index = publish.build_index(args.data_dir)
    (Path(args.data_dir) / "index.json").write_text(index.model_dump_json(indent=2) + "\n")
    print("wrote index.json; now run: uv run python scripts/validate_thesis.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
