(function () {
  "use strict";
  var RUNS = JSON.parse(document.getElementById("data").textContent);
  var state = { run: 0, tab: "flow", round: null, sel: null, stEdge: null };

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function run() { return RUNS[state.run]; }
  function factById(id) {
    var f = run().facts, i;
    for (i = 0; i < f.length; i++) if (f[i].id === id) return f[i];
    return null;
  }
  function flaggedIds() {
    var s = {}, fl = run().flags, i;
    for (i = 0; i < fl.length; i++) if (fl[i].severity === "high") s[fl[i].fact_id] = 1;
    return s;
  }

  /* ---------- flow ---------- */
  function viewFlow() {
    var r = run(), cols = r.columns, flagged = flaggedIds();
    var h = '<div class="legend">' +
      '<span><i style="background:var(--gold)"></i>from a gold paragraph</span>' +
      '<span><i style="background:var(--distr)"></i>from a distractor paragraph</span>' +
      '<span><i style="background:var(--intro)"></i>introduced by an agent</span>' +
      '<span><i style="background:var(--surface-3)"></i>not expressed here</span>' +
      "</div>";
    h += '<div class="flow-wrap"><table class="flow"><thead><tr><th class="lab">' +
      r.facts.length + " canonical facts</th>";
    cols.forEach(function (c, i) {
      var gap = i > 1 && c[1] !== cols[i - 1][1] ? ' style="border-left:1px solid var(--rule-2)"' : "";
      h += "<th" + gap + ">" + esc(c) + "</th>";
    });
    h += "</tr></thead><tbody>";
    r.facts.forEach(function (f) {
      var sel = state.sel === f.id ? " sel" : "";
      h += '<tr class="' + sel.trim() + '" data-fid="' + esc(f.id) + '">';
      h += '<td class="lab" data-fid="' + esc(f.id) + '">' +
        '<span class="rowtag ' + f.origin + '">' + f.origin.slice(0, 4) + "</span>" +
        (flagged[f.id] ? '<span class="rowtag" style="color:var(--high);background:var(--high-bg)">flag</span>' : "") +
        esc(f.text) + "</td>";
      cols.forEach(function (c) {
        var on = f.slots.indexOf(c) >= 0;
        h += "<td><div class=\"cellwrap\"><div class=\"dot" + (on ? " on " + f.origin : "") +
          (c === "SRC" ? " src" : "") + '"></div></div></td>';
      });
      h += "</tr>";
    });
    return h + "</tbody></table></div>";
  }

  /* ---------- rounds ---------- */
  function viewRounds() {
    var r = run(), flagged = flaggedIds();
    if (state.round === null) state.round = r.rounds[0];
    var h = '<div class="roundpick">';
    r.rounds.forEach(function (n) {
      h += '<button class="rbtn" data-round="' + n + '" aria-pressed="' +
        (n === state.round) + '">round ' + n + "</button>";
    });
    h += "</div>";

    var turns = r.turns.filter(function (t) { return t.round === state.round; });
    if (!turns.length) return h + '<p class="empty">no turns in this round</p>';

    turns.forEach(function (t) {
      h += '<div class="turn"><div class="turn-head">' +
        '<span class="who">agent ' + esc(t.agent) + " &middot; round " + t.round + "</span>" +
        '<span class="cnt">' + t.facts.length + " atomic facts extracted</span></div>";
      h += '<div class="cols"><div class="col"><h4>what the agent said</h4>' +
        '<div class="said">' + esc(t.text).replace(/(FINAL ANSWER:.*)$/m, '<span class="fin">$1</span>') +
        "</div></div>";
      h += '<div class="col"><h4>atomic facts extracted from it</h4><div class="flist">';
      t.facts.forEach(function (f) {
        var meta = factById(f.id);
        var cls = (meta ? meta.origin : "introduced") + (flagged[f.id] ? " flagged" : "");
        h += '<div class="fitem ' + cls + '" data-fid="' + esc(f.id) + '">' + esc(f.text) + "</div>";
      });
      if (!t.facts.length) h += '<div class="empty">none</div>';
      h += "</div></div></div>";

      h += '<details class="ctx"><summary>context this agent was given (' +
        (t.peers.length ? t.peers.length + " peer message" + (t.peers.length > 1 ? "s" : "") + " + documents" : "documents only") +
        ')</summary><div class="ctxbody">';
      t.peers.forEach(function (p) {
        h += '<div class="blk"><b>panelist ' + esc(p.agent) + ", round " + (t.round - 1) + "</b>" + esc(p.text) + "</div>";
      });
      h += '<div class="blk"><b>source documents</b>' + esc(r.documents) + "</div>";
      h += "</div></details></div>";
    });
    return h;
  }

  /* ---------- spacetime ---------- */
  function viewSpacetime() {
    var r = run(), st = r.spacetime;
    if (!st || !st.nodes.length) return '<p class="empty">no spacetime graph for this run</p>';
    var agents = st.agents, rounds = st.rounds;
    var padL = 84, padT = 46, colW = 190, rowH = 92, rad = 24;
    var W = padL + colW * rounds.length + 60, H = padT + rowH * agents.length + 30;

    function pos(id) {
      if (id === "SRC") return { x: 34, y: padT + (agents.length - 1) * rowH / 2 };
      var a = id[0], rd = +id.slice(1);
      return { x: padL + colW * rounds.indexOf(rd) + 40, y: padT + rowH * agents.indexOf(a) };
    }
    var sel = state.stEdge;
    var h = '<p class="sub" style="margin:18px 0 0">Each column is a round; each node is an agent at ' +
      'that round. Edges carry <strong>facts</strong>, not messages. Click an edge to list what moved.</p>' +
      '<div class="st-legend">' +
      '<span><i style="background:var(--intro)"></i>origin — a source fact first surfaces</span>' +
      '<span><i style="background:var(--gold)"></i>transmission — a fact reaches an agent that had not said it</span>' +
      '<span><i style="background:var(--rule-2)"></i>persistence — an agent says it again</span>' +
      "</div>";

    h += '<div class="st-wrap"><svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Fact flow across rounds">';
    rounds.forEach(function (rd, i) {
      h += '<text class="st-col" x="' + (padL + colW * i + 40) + '" y="20">round ' + rd + "</text>";
    });
    h += '<text class="st-col" x="34" y="20">source</text>';

    st.edges.forEach(function (e, i) {
      var a = pos(e.src), b = pos(e.dst);
      var w = Math.min(9, 1 + Math.sqrt(e.facts.length) * 1.7);
      var mx = (a.x + b.x) / 2;
      var d = "M" + (a.x + rad) + "," + a.y + " C" + mx + "," + a.y + " " + mx + "," + b.y + " " + (b.x - rad) + "," + b.y;
      var dim = sel !== null && sel !== i ? " dim" : "";
      h += '<path class="st-edge ' + e.kind + dim + '" d="' + d + '" stroke-width="' + w +
        '" stroke-opacity="' + (e.kind === "persistence" ? 0.5 : 0.75) + '" data-edge="' + i + '"></path>';
      if (e.facts.length > 1 && e.src !== e.dst[0] + (e.dst.slice(1) - 1))
        h += '<text class="st-elab" x="' + mx + '" y="' + ((a.y + b.y) / 2 - w / 2 - 4) + '">' + e.facts.length + "</text>";
    });

    h += '<circle class="st-node src" cx="34" cy="' + pos("SRC").y + '" r="' + rad + '"></circle>';
    h += '<text class="st-lab" x="34" y="' + pos("SRC").y + '">SRC</text>';
    st.nodes.forEach(function (n) {
      var p = pos(n.id);
      h += '<circle class="st-node" cx="' + p.x + '" cy="' + p.y + '" r="' + rad + '"></circle>';
      h += '<text class="st-lab" x="' + p.x + '" y="' + (p.y - 4) + '">' + esc(n.agent) + "</text>";
      h += '<text class="st-cnt" x="' + p.x + '" y="' + (p.y + 11) + '">' + n.n + " facts</text>";
    });
    h += "</svg></div>";

    if (sel !== null && st.edges[sel]) {
      var e = st.edges[sel];
      h += '<div class="st-detail"><div class="h">' + esc(e.kind) + " &nbsp;" + esc(e.src) +
        " &rarr; " + esc(e.dst) + " &nbsp;&middot;&nbsp; " + e.facts.length + " facts</div><ul>";
      e.facts.forEach(function (fid) {
        var f = factById(fid);
        h += '<li data-fid="' + esc(fid) + '">' + esc(f ? f.text : fid) + "</li>";
      });
      h += "</ul></div>";
    }
    h += '<p class="sub" style="margin-top:16px;font-size:13.5px;color:var(--muted)">' +
      "Transmission is inferred from co-expression, not observed: every agent sees every document " +
      "under this topology, so B may have read the same paragraph rather than picked it up from A. " +
      "Treat these edges as an upper bound on real transfer.</p>";
    return h;
  }

  /* ---------- audit ---------- */
  function viewAudit() {
    var r = run();
    if (!r.flags.length) return '<p class="empty">no flags raised for this run</p>';
    var h = '<p class="sub" style="margin:18px 0 16px">' + r.flags.length +
      " flags. These are over-reporting heuristics, not a validator &mdash; " +
      "they put the rows worth reading first.</p>";
    r.flags.forEach(function (f) {
      var fact = factById(f.fact_id), partner = f.partner_id ? factById(f.partner_id) : null;
      h += '<div class="aud ' + f.severity + '" data-fid="' + esc(f.fact_id) + '">';
      h += '<div class="k">' + esc(f.kind.replace(/-/g, " ")) + "</div>";
      h += '<div class="t">' + esc(fact ? fact.text : f.fact_id) + "</div>";
      if (partner) h += '<div class="t" style="color:var(--muted)">&harr; ' + esc(partner.text) + "</div>";
      h += '<div class="d">' + esc(f.detail) + "</div></div>";
    });
    return h;
  }

  /* ---------- detail drawer ---------- */
  function openDetail(id) {
    var f = factById(id), d = document.getElementById("detail");
    if (!f) return;
    state.sel = id;
    var h = '<button class="close" aria-label="Close">&times;</button>';
    h += "<h3>" + esc(f.text) + "</h3>";
    h += '<div class="pills"><span class="pill ' + f.origin[0] + '">' + esc(f.origin) + "</span>";
    if (f.doc) h += '<span class="pill">' + esc(f.doc) + "</span>";
    h += '<span class="pill">' + f.n + " mention" + (f.n > 1 ? "s" : "") + "</span></div>";
    h += '<h4 style="font-family:\'IBM Plex Mono\',monospace;font-size:10.5px;letter-spacing:.1em;' +
      'text-transform:uppercase;color:var(--muted);margin:20px 0 4px">every phrasing in this cluster</h4>';
    f.mentions.slice().sort(function (a, b) { return a.slot < b.slot ? -1 : 1; }).forEach(function (m) {
      h += '<div class="mrow"><span class="s ' + esc(m.slot) + '">' + esc(m.slot) + "</span><span>" + esc(m.text) + "</span></div>";
    });
    d.innerHTML = h;
    d.classList.add("on");
    d.querySelector(".close").onclick = closeDetail;
  }
  function closeDetail() {
    document.getElementById("detail").classList.remove("on");
    state.sel = null;
    render();
  }

  /* ---------- shell ---------- */
  function render() {
    var r = run(), body;
    if (state.tab === "flow") body = viewFlow();
    else if (state.tab === "rounds") body = viewRounds();
    else if (state.tab === "spacetime") body = viewSpacetime();
    else body = viewAudit();

    var opts = RUNS.map(function (x, i) {
      return '<option value="' + i + '"' + (i === state.run ? " selected" : "") + ">" +
        esc(x.id + " — " + x.question.slice(0, 68) + (x.question.length > 68 ? "…" : "")) + "</option>";
    }).join("");

    var agreed = Object.keys(r.finals).map(function (k) { return r.finals[k]; });
    var unanimous = agreed.length > 1 && agreed.every(function (v) { return v === agreed[0]; });

    document.getElementById("app").innerHTML =
      "<header><div class=\"eyebrow\">factflow &middot; explorer</div>" +
      "<h1>Fact Flow Explorer</h1>" +
      '<p class="sub">Every atomic fact traced from the source paragraphs through a three-agent debate. ' +
      "Pick a run, then read across the flow grid, open a round to check extraction against what was " +
      "actually said, or work the audit list.</p></header>" +
      '<div class="controls"><select id="runsel">' + opts + "</select>" +
      '<div class="tabs" role="tablist">' +
      ['flow', 'spacetime', 'rounds', 'audit'].map(function (t) {
        var n = t === "audit" && r.flags.length ? " (" + r.flags.length + ")" : "";
        return '<button class="tab" role="tab" data-tab="' + t + '" aria-selected="' +
          (state.tab === t) + '">' + t + n + "</button>";
      }).join("") + "</div></div>" +
      '<div class="qbar"><div class="q">' + esc(r.question) + "</div>" +
      '<div class="a">gold answer <b>' + esc(r.gold_answer) + "</b> &middot; agents " +
      (unanimous ? "unanimous" : "split") + " &middot; gold paragraphs: " + esc(r.gold_titles.join(", ")) + "</div>" +
      '<div class="pills"><span class="pill">' + r.stats.mentions + " mentions</span>" +
      '<span class="pill">' + r.stats.facts + " facts</span>" +
      '<span class="pill g">' + r.stats.gold + " gold</span>" +
      '<span class="pill d">' + r.stats.distractor + " distractor</span>" +
      '<span class="pill i">' + r.stats.introduced + " agent-introduced</span></div></div>" +
      '<div class="panel on">' + body + "</div>";

    document.getElementById("runsel").onchange = function (e) {
      state.run = +e.target.value; state.round = null; state.sel = null; state.stEdge = null;
      closeDetailSilently(); render();
    };
    Array.prototype.forEach.call(document.querySelectorAll(".tab"), function (b) {
      b.onclick = function () { state.tab = b.dataset.tab; render(); };
    });
    Array.prototype.forEach.call(document.querySelectorAll(".rbtn"), function (b) {
      b.onclick = function () { state.round = +b.dataset.round; render(); };
    });
    document.getElementById("app").addEventListener("click", function (e) {
      var edge = e.target.closest("[data-edge]");
      if (edge) {
        var i = +edge.dataset.edge;
        state.stEdge = state.stEdge === i ? null : i;
        render();
        return;
      }
      var el = e.target.closest("[data-fid]");
      if (el) openDetail(el.dataset.fid);
    });
  }
  function closeDetailSilently() { document.getElementById("detail").classList.remove("on"); }

  var drawer = document.createElement("div");
  drawer.id = "detail";
  document.body.appendChild(drawer);
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeDetail(); });
  render();
})();
