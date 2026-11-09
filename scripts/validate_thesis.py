#!/usr/bin/env python3
"""Validate the published theses that the GitHub Pages UI reads. No LLM calls.

    uv run python scripts/validate_thesis.py            # check docs/data
    uv run python scripts/validate_thesis.py --write    # also refresh hard checks + index.json

Checks every ``docs/data/themes/*.json`` against the pydantic contract in
``thesis_builder.publish`` (which reuses ``Node`` / ``Thesis`` / ``Critique`` /
``JudgeScores`` from ``schemas.py``), plus integrity rules the schema cannot
express:

  errors (exit 1)
    * schema violations
    * slug does not match the file name or the theme
    * a node without a source URL, or a missing value-chain layer
    * a critique pointing at a node that is not in the tree, or a node that is
      missing one of the three skeptic angles
    * more than 5 (or zero) investable picks
    * a Mermaid diagram that is not a ``graph``/``flowchart``
    * eval total that disagrees with the per-dimension scores
    * em or en dashes anywhere (house style)
    * anything that looks like an API key or token
    * non-sample files that still carry placeholder (example.com) sources
    * index.json out of sync with the theme files (run with --write)

  warnings (exit 0)
    * golden-spec hard requirements (evals/golden/*.yaml) that are not met

Needs only pydantic + pyyaml, so it also runs as
``PYTHONPATH=src python3 scripts/validate_thesis.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, get_args

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from thesis_builder.publish import (  # noqa: E402
    LAYERS,
    EvalBlock,
    HardChecks,
    PublishedIndex,
    PublishedThesis,
    build_index,
    check_requirements,
    slugify,
)
from thesis_builder.schemas import CritiqueAngle  # noqa: E402

ANGLES = set(get_args(CritiqueAngle))
DASH_RE = re.compile(r"[\u2014\u2013]")
SECRET_RES = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),          # OpenAI / Anthropic style
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),       # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),               # AWS access key id
    re.compile(r"\bxox[abpr]-[A-Za-z0-9\-]{10,}"),     # Slack
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),             # Groq
    re.compile(r"\btvly-[A-Za-z0-9]{16,}"),            # Tavily
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
PLACEHOLDER_HOSTS = ("example.com", "example.org", "example.net")


def _strings(obj: Any, path: str = "$"):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}[{i}]")


def _golden_specs(themes_file: Path) -> dict[str, tuple[str, dict[str, Any]]]:
    if not themes_file.exists():
        return {}
    cfg = yaml.safe_load(themes_file.read_text()) or {}
    out: dict[str, tuple[str, dict[str, Any]]] = {}
    for entry in cfg.get("themes", []):
        golden = entry.get("golden")
        if golden and (ROOT / golden).exists():
            out[slugify(entry["theme"])] = (golden, yaml.safe_load((ROOT / golden).read_text()))
    return out


def validate_file(
    path: Path, specs: dict[str, tuple[str, dict[str, Any]]]
) -> tuple[PublishedThesis | None, list[str], list[str], HardChecks | None]:
    errors: list[str] = []
    warnings: list[str] = []
    raw_text = path.read_text()
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return None, [f"invalid JSON: {e}"], [], None

    for p, s in _strings(raw):
        if DASH_RE.search(s):
            errors.append(f"em/en dash at {p}: {s[:60]!r}")
        for rx in SECRET_RES:
            if rx.search(s):
                errors.append(f"possible secret at {p} (pattern {rx.pattern[:20]}...)")

    try:
        pub = PublishedThesis.model_validate(raw)
    except ValidationError as e:
        for err in e.errors()[:20]:
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"schema: {loc}: {err['msg']}")
        return None, errors, warnings, None

    if pub.slug != path.stem:
        errors.append(f"slug {pub.slug!r} does not match file name {path.stem!r}")
    if slugify(pub.theme) != pub.slug:
        errors.append(f"slug {pub.slug!r} does not match theme {pub.theme!r}")
    if pub.thesis.theme != pub.theme:
        errors.append("thesis.theme differs from top-level theme")

    names = {n.name for n in pub.thesis.nodes}
    for n in pub.thesis.nodes:
        if not n.sources:
            errors.append(f"node {n.name!r} cites no source")
        if not pub.sample:
            for u in n.sources:
                if (u.host or "").endswith(PLACEHOLDER_HOSTS):
                    errors.append(f"node {n.name!r} cites placeholder source {u}")
    missing_layers = set(LAYERS) - {n.layer for n in pub.thesis.nodes}
    if missing_layers:
        errors.append(f"tree is missing layers: {sorted(missing_layers)}")

    angles_by_node: dict[str, set[str]] = {n: set() for n in names}
    for c in pub.critiques:
        if c.node_name not in names:
            errors.append(f"critique targets unknown node {c.node_name!r}")
        else:
            angles_by_node[c.node_name].add(c.angle)
    for n, got in angles_by_node.items():
        if got != ANGLES:
            errors.append(f"node {n!r} is missing skeptic angles {sorted(ANGLES - got)}")

    picks = pub.thesis.investable_picks
    if not 1 <= len(picks) <= 5:
        errors.append(f"expected 1 to 5 investable picks, got {len(picks)}")

    if not re.match(r"^\s*(graph|flowchart)\s+(LR|RL|TB|TD|BT)\b", pub.diagram_mmd):
        errors.append("diagram_mmd must start with 'graph LR' (or another flowchart direction)")

    if pub.eval and pub.eval.scores and pub.eval.total is not None:
        if pub.eval.scores.total != pub.eval.total:
            errors.append(
                f"eval.total {pub.eval.total} != sum of scores {pub.eval.scores.total}"
            )

    hard: HardChecks | None = None
    if pub.slug in specs:
        golden_path, spec = specs[pub.slug]
        ok, failures = check_requirements(
            spec,
            json.loads(pub.thesis.model_dump_json()),
            pub.memo_md,
            {"unique_urls": pub.research.unique_urls, "critiques": len(pub.critiques)},
        )
        hard = HardChecks(golden=golden_path, passed=ok, failures=failures)
        for f in failures:
            warnings.append(f"golden hard check: {f}")
        stored = pub.eval.hard_checks if pub.eval else None
        if stored is not None and (stored.passed, stored.failures) != (ok, failures):
            warnings.append("stored eval.hard_checks is stale (run with --write)")

    return pub, errors, warnings, hard


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=str(ROOT / "docs" / "data"))
    ap.add_argument("--themes-file", default=str(ROOT / "routines" / "themes.yaml"))
    ap.add_argument(
        "--write",
        action="store_true",
        help="Store computed hard checks in each file and rebuild index.json.",
    )
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    files = sorted((data_dir / "themes").glob("*.json"))
    if not files:
        print(f"no theme files under {data_dir / 'themes'}")
        return 1

    specs = _golden_specs(Path(args.themes_file))
    total_errors = 0
    for path in files:
        pub, errors, warnings, hard = validate_file(path, specs)
        status = "FAIL" if errors else "ok"
        sample = " (sample)" if pub and pub.sample else ""
        print(f"[{status}] {path.relative_to(data_dir)}{sample}")
        for e in errors:
            print(f"    error: {e}")
        for w in warnings:
            print(f"    warn:  {w}")
        total_errors += len(errors)
        if args.write and pub is not None and not errors and hard is not None:
            pub.eval = pub.eval or EvalBlock()
            pub.eval.hard_checks = hard
            path.write_text(pub.model_dump_json(indent=2) + "\n")

    index_path = data_dir / "index.json"
    if total_errors == 0:
        fresh = build_index(data_dir)
        if args.write:
            index_path.write_text(fresh.model_dump_json(indent=2) + "\n")
            print(f"[ok] wrote {index_path.relative_to(data_dir)} ({len(fresh.themes)} themes)")
        elif not index_path.exists():
            print("[FAIL] index.json missing (run with --write)")
            total_errors += 1
        else:
            try:
                current = PublishedIndex.model_validate_json(index_path.read_text())
                same = [e.model_dump() for e in current.themes] == [
                    e.model_dump() for e in fresh.themes
                ]
            except ValidationError as e:
                same = False
                print(f"    error: index.json schema: {e.errors()[0]['msg']}")
            if same:
                print(f"[ok] index.json ({len(fresh.themes)} themes)")
            else:
                print("[FAIL] index.json is out of sync with themes/ (run with --write)")
                total_errors += 1

    print(f"\n{len(files)} theme file(s), {total_errors} error(s)")
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
