(() => {
  const $ = (id) => document.getElementById(id);
  const STAGES = ["researcher", "skeptic", "synthesizer"];
  const LAYERS = [
    ["raw_inputs", "Raw inputs"],
    ["enabling", "Enabling layer"],
    ["integrators", "Integrators"],
    ["applications", "Applications"],
  ];

  let currentRunId = null;
  let lastResult = null;
  let challengeNode = null;
  let eventSource = null;
  let stageCounters = { researcher: 0, skeptic: 0, synthesizer: 0 };
  let llmCalls = 0;
  let searchCount = 0;

  // ---- tabs ----
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      $("tab-" + t.dataset.tab).classList.add("active");
    })
  );

  // ---- presets ----
  document.querySelectorAll(".preset").forEach((b) =>
    b.addEventListener("click", () => {
      $("theme-input").value = b.dataset.theme;
    })
  );

  // ---- config probe ----
  fetch("/api/config")
    .then((r) => r.json())
    .then((cfg) => {
      const badge = $("mode-badge");
      const mode = cfg.replay ? "replay" : cfg.default_demo ? "demo" : "live";
      badge.textContent = cfg.replay ? "offline replay of recorded runs" : mode;
      badge.classList.add(mode);
      if (!cfg.default_demo || cfg.replay) $("refusal-toggle").hidden = true;
      if (cfg.replay && cfg.replay_themes && cfg.replay_themes.length) {
        document.querySelector(".presets").innerHTML =
          "<span>recorded:</span>" +
          cfg.replay_themes.map((t) => `<button class="preset" data-theme="${escapeHtml(t)}">${escapeHtml(t)}</button>`).join("");
        document.querySelectorAll(".preset").forEach((b) =>
          b.addEventListener("click", () => { $("theme-input").value = b.dataset.theme; })
        );
      }
      const keys = [];
      if (cfg.has_anthropic) keys.push("anthropic");
      if (cfg.has_groq) keys.push("groq");
      if (cfg.has_openai) keys.push("openai");
      if (cfg.has_openrouter) keys.push("openrouter");
      if (cfg.has_tavily) keys.push("tavily");
      $("config-hint").textContent = keys.length
        ? "keys present: " + keys.join(", ")
        : "no API keys, demo only";
    });

  // ---- run ----
  $("run-btn").addEventListener("click", startRun);
  $("theme-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") startRun();
  });

  async function startRun() {
    const theme = $("theme-input").value.trim();
    if (!theme) return;
    resetUI();
    $("run-btn").disabled = true;
    const cfg = await fetch("/api/config").then((r) => r.json());
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ theme, demo: cfg.default_demo, show_refusal: $("show-refusal").checked }),
    });
    if (!resp.ok) {
      alert("failed to start run: " + (await resp.text()));
      $("run-btn").disabled = false;
      return;
    }
    const { run_id } = await resp.json();
    currentRunId = run_id;
    openStream(run_id);
  }

  function resetUI() {
    STAGES.forEach((s) => {
      const el = $("stage-" + s);
      el.classList.remove("active", "done");
      el.querySelector(".stage-status").textContent = "idle";
      $("meta-" + s).textContent = "";
    });
    stageCounters = { researcher: 0, skeptic: 0, synthesizer: 0 };
    llmCalls = 0;
    searchCount = 0;
    $("trace-list").innerHTML = "";
    lastResult = null;
    setBanner("pending", "Synthesizer has not answered yet. Nothing is published until every critique has an answer the checker accepts.");
    $("tab-critiques").innerHTML = '<div class="placeholder">Waiting for the Skeptic and the Synthesizer…</div>';
    $("tab-tree").innerHTML = '<div class="placeholder">Building tree…</div>';
    $("tab-memo").innerHTML = '<div class="placeholder">Synthesizing memo…</div>';
    $("tab-diagram").innerHTML = '<div class="placeholder">Diagram pending…</div>';
    $("tab-picks").innerHTML = '<div class="placeholder">Picks pending…</div>';
    $("json-out").textContent = "{}";
    setMetric("cost", "$0.0000");
    setMetric("small", "0");
    setMetric("reasoning", "0");
    setMetric("quality", "0");
    setMetric("cheap", "n/a");
    setMetric("searches", "0");
    setMetric("summaries", "0");
    setMetric("critiques", "0");
  }

  function openStream(runId) {
    if (eventSource) eventSource.close();
    eventSource = new EventSource(`/api/events/${runId}`);
    eventSource.addEventListener("start", () => {
      markStage("researcher", "active", "planning queries…");
    });
    eventSource.addEventListener("stage", (e) => onStage(JSON.parse(e.data)));
    eventSource.addEventListener("trace", (e) => onTrace(JSON.parse(e.data)));
    eventSource.addEventListener("done", (e) => onDone(JSON.parse(e.data)));
    eventSource.addEventListener("protocol", (e) => onProtocol(JSON.parse(e.data)));
    eventSource.addEventListener("challenge", (e) => {
      const c = JSON.parse(e.data).critique;
      setBanner("pending", `Visitor challenge ${c.id} on ${c.node_name}: the Synthesizer must answer it before anything is republished.`);
    });
    eventSource.addEventListener("error", (e) => {
      try { onError(JSON.parse(e.data)); } catch { /* network blip */ }
    });
    eventSource.addEventListener("end", async () => {
      eventSource.close();
      $("run-btn").disabled = false;
      // Make sure we render the final result even if we missed the "done" event.
      try {
        const r = await fetch(`/api/result/${runId}`);
        if (r.ok) renderResult(await r.json());
      } catch {}
    });
  }

  function onStage({ stage, status }) {
    if (status === "started") markStage(stage, "active", "running…");
    if (status === "completed") {
      markStage(stage, "done", "done");
    }
  }

  function markStage(stage, klass, status) {
    const el = $("stage-" + stage);
    el.classList.remove("active", "done");
    if (klass) el.classList.add(klass);
    el.querySelector(".stage-status").textContent = status;
  }

  function onTrace(ev) {
    // Update metrics.
    if (ev.kind === "llm") {
      llmCalls++;
      const tier = ev.tier || "small";
      const tokens =
        Number(ev.prompt_tokens || 0) + Number(ev.completion_tokens || 0);
      bumpMetric("m-" + tier, tokens);
      bumpCost(Number(ev.cost_usd || 0));
      updateCheapShare();
      stageCounters[currentStage()] = (stageCounters[currentStage()] || 0) + 1;
      $("meta-" + currentStage()).textContent =
        `${stageCounters[currentStage()]} model calls`;
    } else if (ev.kind === "search") {
      searchCount++;
      setMetric("searches", String(searchCount));
    }

    // Append to trace tab (capped to last 400 entries).
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="trace-kind ${ev.kind || ""}">${ev.kind || "?"}</span>
      <span class="trace-tier">${ev.tier || ""}</span>
      <span class="trace-note">${escapeHtml(ev.note || "")}</span>
    `;
    const list = $("trace-list");
    list.appendChild(li);
    while (list.childElementCount > 400) list.removeChild(list.firstChild);
    list.scrollTop = list.scrollHeight;
  }

  function currentStage() {
    for (const s of STAGES) {
      if ($("stage-" + s).classList.contains("active")) return s;
    }
    return "researcher";
  }

  function onDone({ metrics }) {
    if (metrics) {
      setMetric("summaries", String(metrics.summaries || 0));
      setMetric("critiques", String(metrics.critiques || 0));
      if (metrics.cache_hit_rate != null) {
        // append to footer-ish; reuse the m-cheap slot if we want, but keep it.
      }
    }
  }

  function onProtocol({ attempt, passed, errors }) {
    if (passed) {
      setBanner("ok", `Checker accepted attempt ${attempt}: every critique has exactly one answer, and every claimed change is in the tree.`);
    } else {
      setBanner(
        "refused",
        `Checker refused attempt ${attempt}. Nothing was published. The errors go back to the Synthesizer:`,
        errors
      );
    }
  }

  function setBanner(kind, text, items) {
    const b = $("protocol-banner");
    b.hidden = false;
    b.className = "protocol-banner " + kind;
    b.innerHTML = escapeHtml(text) +
      (items && items.length ? "<ul>" + items.map((x) => `<li>${escapeHtml(x)}</li>`).join("") + "</ul>" : "");
  }

  function onError({ note }) {
    alert("run failed: " + note);
  }

  function renderResult(res) {
    if (!res) return;
    lastResult = res;
    const p = res.protocol || {};
    if (!res.published) {
      setBanner("refused", `Refused after ${p.attempts || 1} attempt(s). Nothing was published.`, p.errors || []);
      renderCritiques(res);
      return;
    }
    const refusedNote = (p.refused || []).length
      ? ` An earlier attempt was refused (${p.refused.length} error${p.refused.length > 1 ? "s" : ""}) and corrected.`
      : "";
    const replayNote = res.replay
      ? ` Offline replay of a ${res.recorded_mode || ""} run recorded ${res.recorded_at || ""}; visitor challenges are answered by an offline rule-based Synthesizer.`
      : "";
    setBanner(
      "ok",
      `Published. ${res.critiques.length} critiques, each answered once; every claimed change checked against the draft tree.` +
        refusedNote + replayNote,
      (p.refused || []).length ? p.refused : null
    );
    if (!res.thesis) return;
    renderTree(res.thesis);
    renderCritiques(res);
    renderMemo(res.memo_md);
    renderDiagram(res.diagram_mmd);
    renderPicks(res.thesis.investable_picks || []);
    $("json-out").textContent = JSON.stringify(res, null, 2);
  }

  function renderCritiques(res) {
    const root = $("tab-critiques");
    const crits = (res.critiques || []).slice().sort((a, b) =>
      (a.angle === "visitor") === (b.angle === "visitor")
        ? a.node_name.localeCompare(b.node_name) || a.angle.localeCompare(b.angle)
        : a.angle === "visitor" ? -1 : 1
    );
    if (!crits.length) {
      root.innerHTML = '<div class="placeholder">No critiques in this run (the Skeptic was skipped).</div>';
      return;
    }
    root.innerHTML = crits.map((c) => `
      <div class="crit-row ${c.angle === "visitor" ? "visitor" : ""}">
        <div class="crit-side">
          <div class="crit-meta">${escapeHtml(c.id)} · ${escapeHtml(c.node_name)} · ${escapeHtml(c.angle.replaceAll("_", " "))} · verdict ${escapeHtml(c.verdict)}</div>
          ${escapeHtml(c.argument)}
        </div>
        <div class="crit-side">
          <div class="crit-meta">
            ${c.stance ? `<span class="stance ${c.stance}">${c.stance}</span>` : "<em>no answer</em>"}
            ${(c.changes || []).map((x) => `<span class="change">${escapeHtml(x.replaceAll("_", " "))}</span>`).join("")}
          </div>
          ${escapeHtml(c.rebuttal || "")}
        </div>
      </div>`).join("");
  }

  // ---- visitor challenge ----
  function openChallenge(nodeName) {
    if (!lastResult || !lastResult.published || !currentRunId) return;
    challengeNode = nodeName;
    $("challenge-node").textContent = nodeName;
    $("challenge-text").value = "";
    $("challenge-dialog").showModal();
  }

  $("challenge-dialog").addEventListener("close", async () => {
    if ($("challenge-dialog").returnValue !== "send") return;
    const argument = $("challenge-text").value.trim();
    if (!argument || !challengeNode) return;
    const resp = await fetch(`/api/challenge/${currentRunId}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ node_name: challengeNode, argument }),
    });
    if (!resp.ok) {
      alert("challenge failed: " + (await resp.text()));
      return;
    }
    const { run_id } = await resp.json();
    currentRunId = run_id;
    markStage("synthesizer", "active", "answering the visitor…");
    openStream(run_id);
    document.querySelector('.tab[data-tab="critiques"]').click();
  });

  function renderTree(thesis) {
    const root = $("tab-tree");
    root.innerHTML = "";
    const byLayer = {};
    (thesis.nodes || []).forEach((n) => {
      (byLayer[n.layer] ||= []).push(n);
    });
    for (const [layer, label] of LAYERS) {
      const nodes = byLayer[layer] || [];
      const group = document.createElement("div");
      group.className = "layer-group";
      group.innerHTML = `
        <div class="layer-title">
          <h3>${label}</h3>
          <span class="layer-count">${nodes.length}</span>
        </div>
      `;
      nodes.forEach((n) => group.appendChild(renderNode(n)));
      root.appendChild(group);
    }
  }

  function renderNode(n) {
    const el = document.createElement("div");
    el.className = "node-card";
    el.innerHTML = `
      <div class="node-head">
        <span class="node-name">${escapeHtml(n.name)}</span>
        <span class="node-conf">confidence ${Number(n.confidence || 0).toFixed(2)}</span>
      </div>
      <div class="node-desc">${escapeHtml(n.description || "")}</div>
      <div class="chips">
        ${(n.public_names || []).map((t) => `<span class="chip">${escapeHtml(t)}</span>`).join("")}
        ${(n.private_names || []).map((t) => `<span class="chip private">${escapeHtml(t)}</span>`).join("")}
      </div>
      ${
        (n.catalysts || []).length
          ? `<div class="node-row"><strong>catalysts:</strong> <span>${n.catalysts.map(escapeHtml).join("; ")}</span></div>`
          : ""
      }
      ${
        (n.risks || []).length
          ? `<div class="node-row"><strong>risks:</strong> <span>${n.risks.map(escapeHtml).join("; ")}</span></div>`
          : ""
      }
      ${
        (n.sources || []).length
          ? `<div class="node-row"><strong>sources:</strong> <span>${(n.sources || []).map((u) => `<a href="${escapeHtml(u)}" target="_blank">${shortUrl(u)}</a>`).join("  ")}</span></div>`
          : ""
      }
      <button class="challenge-btn">Challenge this node</button>
    `;
    el.querySelector(".challenge-btn").addEventListener("click", () => openChallenge(n.name));
    return el;
  }

  function renderMemo(md) {
    const el = $("tab-memo");
    el.innerHTML = `<article class="memo">${marked.parse(md || "")}</article>`;
  }

  function renderDiagram(mmd) {
    const el = $("tab-diagram");
    el.innerHTML = `<div class="diagram-wrap"><div class="mermaid">${escapeHtml(mmd || "")}</div></div>`;
    try {
      mermaid.initialize({ startOnLoad: false, theme: "default" });
      mermaid.run({ querySelector: ".mermaid" });
    } catch (e) {
      console.error(e);
    }
  }

  function renderPicks(picks) {
    const el = $("tab-picks");
    el.innerHTML = picks
      .map((p, i) => {
        const m = p.match(/^([^:\u2014-]+?)\s*[:\u2014-]\s*(.*)$/);
        const ticker = m ? m[1].trim() : "PICK";
        const rationale = m ? m[2].trim() : p;
        return `<div class="pick"><span class="rank">#${i + 1}</span><span class="ticker">${escapeHtml(ticker)}</span>: ${escapeHtml(rationale)}</div>`;
      })
      .join("");
  }

  // ---- metric helpers ----
  function setMetric(name, value) {
    const id = name.startsWith("m-") ? name : "m-" + name;
    const el = $(id);
    if (el) el.textContent = value;
  }
  function bumpMetric(id, by) {
    const el = $(id);
    if (!el) return;
    const cur = Number(el.textContent.replace(/[^0-9.]/g, "")) || 0;
    el.textContent = String(cur + by);
  }
  let _cost = 0;
  function bumpCost(by) {
    _cost += by;
    setMetric("cost", "$" + _cost.toFixed(4));
  }
  function updateCheapShare() {
    const s = Number($("m-small").textContent) || 0;
    const r = Number($("m-reasoning").textContent) || 0;
    const q = Number($("m-quality").textContent) || 0;
    const total = s + r + q;
    setMetric("cheap", total ? Math.round((100 * (s + r)) / total) + "%" : "n/a");
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  }
  function shortUrl(u) {
    try { return new URL(u).hostname; } catch { return u; }
  }
})();
