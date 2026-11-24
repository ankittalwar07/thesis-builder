#!/usr/bin/env python3
"""Check that every model tier answers, and how fast. Run this first on a new machine.

    uv run python scripts/check_models.py --config config/models.local.yaml

For each tier it sends one short request that must come back as structured
JSON (the same path the agents use) and prints the model, the time taken and
the output speed. A tier that fails prints the error and the script exits 1.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pydantic import BaseModel, Field  # noqa: E402

from thesis_builder.llm import LLMClient, TraceWriter, default_config_path  # noqa: E402


class Probe(BaseModel):
    layer: str = Field(description="one of raw_inputs, enabling, integrators, applications")
    reason: str = Field(description="one sentence")


PROMPT = [
    {"role": "system", "content": "You classify companies into value-chain layers. Reply in JSON."},
    {"role": "user", "content": "Which layer is a maker of EUV lithography machines in, for the theme "
     "'AI infrastructure'? Give the layer and a one-sentence reason."},
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=default_config_path())
    args = ap.parse_args(argv)

    events: list[dict] = []
    llm = LLMClient(config_path=args.config, trace=TraceWriter(on_event=events.append))
    print(f"config: {args.config}\n")
    failed = 0
    for tier in ("small", "reasoning", "quality"):
        cfg = llm.tier(tier)  # type: ignore[arg-type]
        where = f" at {cfg.api_base}" if cfg.api_base else ""
        t0 = time.perf_counter()
        try:
            out = llm.complete(tier=tier, messages=PROMPT, response_model=Probe, max_tokens=200)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {tier:9} {cfg.primary}{where}\n       {type(e).__name__}: {str(e)[:300]}")
            continue
        secs = time.perf_counter() - t0
        last = next((e for e in reversed(events) if e.get("kind") == "llm"), {})
        toks = last.get("completion_tokens") or 0
        speed = f", {toks / secs:.1f} tokens/s" if toks and secs else ""
        print(f"[ok]   {tier:9} {last.get('model', cfg.primary)}{where}: {secs:.1f}s{speed}")
        print(f"       answer: {out.layer} ({out.reason[:90]})")
    if failed:
        print("\nAt least one tier failed. Check that the model server is running and the "
              "models exist (see LOCAL_MODELS.md).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
