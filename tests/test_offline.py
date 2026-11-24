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


def test_every_critique_gets_its_own_answer(ai_infra_result):
    """R1: the Synthesizer answers each critique id exactly once, and R3 passes."""
    crit_ids = [c.id for c in ai_infra_result["skeptic"].critiques]
    assert crit_ids and len(set(crit_ids)) == len(crit_ids)
    answered = [a.critique_id for a in ai_infra_result["final"].answers]
    assert sorted(answered) == sorted(crit_ids)
    assert ai_infra_result["protocol"].passed
    pub = publish.from_pipeline(ai_infra_result, generated_by="demo", sample=True)
    assert pub.protocol and pub.protocol.passed and pub.draft_nodes
    # A conceded weak verdict shows up as a real change in the published tree.
    draft = {n.name: n for n in pub.draft_nodes}
    final = {n.name: n for n in pub.thesis.nodes}
    for c in pub.critiques:
        if "confidence_lowered" in c.changes:
            assert final[c.node_name].confidence < draft[c.node_name].confidence


def test_checker_refuses_a_concession_the_tree_does_not_show(tmp_path):
    """R3: a 'conceded, confidence lowered' answer with no change is refused, then fixed."""
    seen = []
    result = run_demo(
        "AI infrastructure", seed=1, cache_dir=str(tmp_path), fault_once=True,
        on_protocol=lambda attempt, res: seen.append((attempt, res.passed, list(res.errors))),
    )
    assert seen[0][0] == 1 and seen[0][1] is False
    assert any("confidence was lowered" in e for e in seen[0][2])
    assert seen[-1][1] is True and result["protocol"].attempts == 2
    pub = publish.from_pipeline(result, generated_by="demo", sample=True)
    assert pub.protocol.attempts == 2 and pub.protocol.refused


def test_publish_refuses_when_the_check_fails(ai_infra_result):
    from thesis_builder.schemas import CritiqueAnswer

    bad = dict(ai_infra_result)
    final = ai_infra_result["final"].model_copy(deep=True)
    first = final.answers[0]
    final.answers = [CritiqueAnswer(critique_id=first.critique_id, stance="conceded",
                                    rebuttal="Conceded.", changes=["dropped"])]
    bad["final"], bad["protocol"] = final, None
    with pytest.raises(publish.ProtocolRefused) as exc:
        publish.from_pipeline(bad, generated_by="demo", sample=True)
    assert "has no answer" in str(exc.value)


def test_protocol_rules_unit():
    from thesis_builder import protocol
    from thesis_builder.schemas import Critique, CritiqueAnswer, Node

    node = Node(name="HBM", layer="enabling", description="d", confidence=0.8,
                sources=["https://a.test/x"], risks=["supply"])
    lowered = node.model_copy(update={"confidence": 0.7})
    c = Critique(id="C1", node_name="HBM", angle="priced_in", verdict="weak", argument="a")
    ok = protocol.check([node], [lowered], [c], [CritiqueAnswer(
        critique_id="C1", stance="conceded", rebuttal="r", changes=["confidence_lowered"])])
    assert ok.passed
    held_dropped = protocol.check([node], [], [c], [CritiqueAnswer(
        critique_id="C1", stance="held", rebuttal="r")])
    assert any("dropped" in e for e in held_dropped.errors)
    fake_risk = protocol.check([node], [node.model_copy(update={"risks": ["Supply"]})], [c],
                               [CritiqueAnswer(critique_id="C1", stance="conceded",
                                               rebuttal="r", changes=["risk_added"])])
    assert any("risks are unchanged" in e for e in fake_risk.errors)
    ghost = protocol.check([node], [node], [c], [
        CritiqueAnswer(critique_id="C1", stance="held", rebuttal="r"),
        CritiqueAnswer(critique_id="C9", stance="held", rebuttal="r")])
    assert any("unknown critique" in e for e in ghost.errors)


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


# --- Live server: visitor challenge, record and replay ----------------------


def _wait(client, run_id, timeout=60):
    import time

    for _ in range(int(timeout / 0.1)):
        r = client.get(f"/api/result/{run_id}")
        if r.status_code != 409:
            return r
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_server_visitor_challenge_and_refusal(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from thesis_builder.server import build_app

    monkeypatch.setenv("THESIS_CACHE_DIR", str(tmp_path / "cache"))
    client = TestClient(build_app(default_demo=True, record_dir=tmp_path / "runs"))
    run_id = client.post("/api/run", json={"theme": "Robotics", "show_refusal": True}).json()["run_id"]
    res = _wait(client, run_id).json()
    assert res["published"] and res["protocol"]["attempts"] == 2 and res["protocol"]["refused"]
    assert all(c["stance"] and c["rebuttal"] for c in res["critiques"])

    node = res["thesis"]["nodes"][0]["name"]
    bad = client.post(f"/api/challenge/{run_id}", json={"node_name": "Nope", "argument": "too cheap"})
    assert bad.status_code == 400
    child = client.post(f"/api/challenge/{run_id}",
                        json={"node_name": node, "argument": "Margins are already priced in."}).json()
    cres = _wait(client, child["run_id"]).json()
    assert cres["published"] and cres["kind"] == "challenge"
    visitor = [c for c in cres["critiques"] if c["angle"] == "visitor"]
    assert len(visitor) == 1 and visitor[0]["id"] == "V1" and visitor[0]["stance"] == "conceded"
    challenged = next(n for n in cres["thesis"]["nodes"] if n["name"] == node)
    assert any(r.startswith("Visitor challenge:") for r in challenged["risks"])
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1  # the challenge is not recorded


def test_server_replays_a_recorded_run_offline(tmp_path, monkeypatch):
    import socket

    from fastapi.testclient import TestClient

    from thesis_builder.server import build_app

    monkeypatch.setenv("THESIS_CACHE_DIR", str(tmp_path / "cache"))
    rec = TestClient(build_app(default_demo=True, record_dir=tmp_path / "runs"))
    _wait(rec, rec.post("/api/run", json={"theme": "Energy transition"}).json()["run_id"])

    def _boom(*a, **k):
        raise AssertionError("replay attempted a network request")

    # TestClient talks to the app in memory, so any real socket means the replay went online.
    monkeypatch.setattr(socket.socket, "connect", _boom)
    client = TestClient(build_app(default_demo=False, replay_dir=tmp_path / "runs", replay_speed=0))
    assert client.get("/api/config").json()["replay_themes"] == ["Energy transition"]
    assert client.post("/api/run", json={"theme": "Robotics"}).status_code == 404
    run_id = client.post("/api/run", json={"theme": "energy transition"}).json()["run_id"]
    res = _wait(client, run_id).json()
    assert res["replay"] and res["published"] and res["critiques"]
    node = res["thesis"]["nodes"][-1]["name"]
    child = client.post(f"/api/challenge/{run_id}",
                        json={"node_name": node, "argument": "A cheaper substitute is coming."}).json()
    cres = _wait(client, child["run_id"]).json()
    assert cres["replay"] and any(c["angle"] == "visitor" for c in cres["critiques"])
