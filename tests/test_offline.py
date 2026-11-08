"""Offline tests: no API keys, no network. Run in CI on every PR."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from thesis_builder import publish
from thesis_builder.demo import run_demo

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ai_infra_result(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    return run_demo("AI infrastructure", seed=1, cache_dir=str(cache))


def test_demo_pipeline_runs_all_three_agents(ai_infra_result):
    research, skeptic, final = (
        ai_infra_result["research"],
        ai_infra_result["skeptic"],
        ai_infra_result["final"],
    )
    assert research.summaries and research.queries_run
    nodes = final.thesis.nodes
    assert {n.layer for n in nodes} == set(publish.LAYERS)
    # Skeptic attacks every draft node from all three angles.
    assert len(skeptic.critiques) == 3 * len(research.draft_nodes)
    assert skeptic.headline_concerns


def test_demo_makes_no_network_calls(monkeypatch, tmp_path):
    import httpx

    def _boom(*a, **k):
        raise AssertionError("demo mode attempted a network request")

    monkeypatch.setattr(httpx.Client, "get", _boom)
    run_demo("Robotics", seed=1, cache_dir=str(tmp_path))


def test_published_contract_round_trips(ai_infra_result, tmp_path):
    pub = publish.from_pipeline(ai_infra_result, generated_by="demo", sample=True)
    assert pub.slug == "ai_infrastructure"
    assert all(c.rebuttal and c.stance for c in pub.critiques)
    blob = pub.model_dump_json()
    assert "\u2014" not in blob and "\u2013" not in blob
    path = publish.write_published(pub, tmp_path)
    again = publish.PublishedThesis.model_validate_json(path.read_text())
    assert again == pub
    index = publish.build_index(tmp_path)
    assert [e.slug for e in index.themes] == ["ai_infrastructure"]


def test_stance_tracks_verdict():
    assert publish.stance_for("strong") == "held"
    assert publish.stance_for("mixed") == "partial"
    assert publish.stance_for("weak") == "conceded"


def test_committed_pages_data_is_valid():
    """docs/data must always pass the same validator the Claude routine runs."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_thesis.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_validator_rejects_bad_files(tmp_path):
    src = ROOT / "docs" / "data" / "themes" / "robotics.json"
    data = json.loads(src.read_text())
    data["thesis"]["summary"] += " \u2014 dash"
    data["critiques"][0]["node_name"] = "Not a node"
    (tmp_path / "themes").mkdir()
    (tmp_path / "themes" / "robotics.json").write_text(json.dumps(data))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_thesis.py"), "--data-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "em/en dash" in proc.stdout
    assert "unknown node" in proc.stdout


def test_routine_themes_have_golden_specs():
    import yaml

    cfg = yaml.safe_load((ROOT / "routines" / "themes.yaml").read_text())
    for entry in cfg["themes"]:
        assert (ROOT / entry["golden"]).exists(), entry
