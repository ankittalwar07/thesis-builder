"""LLM-as-judge scoring.

Four dimensions per the brief:
  - coverage:       did the tree span all four layers with named entities?
  - specificity:    are claims concrete (numbers, names, dates) vs. vague?
  - novelty:        is there at least one non-obvious thread vs. the consensus?
  - actionability:  are the investable picks ranked and defensible?

Each dimension is scored 0–5 by tier-reasoning. The judge sees only the
final thesis JSON + memo — never the research summaries — so it cannot just
re-summarize the inputs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from thesis_builder.llm import LLMClient


class DimensionScore(BaseModel):
    score: int = Field(ge=0, le=5)
    justification: str


class JudgeScores(BaseModel):
    coverage: DimensionScore
    specificity: DimensionScore
    novelty: DimensionScore
    actionability: DimensionScore

    @property
    def total(self) -> int:
        return sum(
            s.score for s in (self.coverage, self.specificity, self.novelty, self.actionability)
        )


JUDGE_SYSTEM = """You are an experienced equity-research editor scoring a value-chain investment thesis.

Score each of four dimensions on a 0–5 integer scale and justify in one sentence.

  - coverage:      0 = missing layers or entities; 5 = all 4 layers, ≥2 named companies/layer.
  - specificity:   0 = vague marketing prose; 5 = consistently names companies, numbers, catalysts.
  - novelty:       0 = obvious consensus list only; 5 = surfaces ≥1 non-obvious bottleneck or pick.
  - actionability: 0 = picks unranked or unjustified; 5 = top picks each have a defensible one-line case.

Be harsh. The default for a generic thesis should be 2/5 per dimension."""

JUDGE_USER = """Theme: {theme}

Final thesis (JSON):
{thesis_json}

Memo:
{memo_md}

Score the thesis now."""


def judge(llm: LLMClient, theme: str, thesis_json: str, memo_md: str) -> JudgeScores:
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_USER.format(
                theme=theme,
                thesis_json=thesis_json[:8000],
                memo_md=memo_md[:6000],
            ),
        },
    ]
    return llm.complete(tier="reasoning", messages=msgs, response_model=JudgeScores)
