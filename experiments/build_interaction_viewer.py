#!/usr/bin/env python3
"""Render the interaction digraph from findings/data/interaction-graph.json.

One relation type is drawn at a time.  Showing all three at once puts 81
curves over nine nodes, which was unreadable; the selector keeps it at 27 and
lets the shapes be compared directly against one another.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G = json.loads((ROOT / "findings/data/interaction-graph.json").read_text())
CELLS = ["full-generic", "full-specialist", "split-generic", "split-specialist"]
NAME = {"full-generic": "全量来源 · 通用角色", "full-specialist": "全量来源 · 专家角色",
        "split-generic": "分割来源 · 通用角色", "split-specialist": "分割来源 · 专家角色"}
SUB = {"full-generic": "三人都读 paper-b + paper-c", "full-specialist": "三人都读 paper-b + paper-c",
       "split-generic": "A 只有 paper-b，B 只有 paper-c，C 无来源",
       "split-specialist": "A 只有 paper-b，B 只有 paper-c，C 无来源"}

HTML = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>谁在改写谁</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--ground:#f7f7f5;--panel:#fffefc;--edge:#e0dfd9;--ink:#1b1f24;--dim:#6b7178;
 --faint:#9aa0a6;--eq:#1f7a6b;--weak:#c4703a;--ref:#3f5fa8;--node:#8a9099;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
 --ground:#15171a;--panel:#1c1f23;--edge:#2e3238;--ink:#e8e6e1;--dim:#9aa0a6;
 --faint:#6b7178;--eq:#4fbfa8;--weak:#e0955c;--ref:#7b96d8;--node:#767c85;}}
:root[data-theme=dark]{--ground:#15171a;--panel:#1c1f23;--edge:#2e3238;--ink:#e8e6e1;
 --dim:#9aa0a6;--faint:#6b7178;--eq:#4fbfa8;--weak:#e0955c;--ref:#7b96d8;--node:#767c85;}
body{background:var(--ground);color:var(--ink);margin:0;
 font:400 15px/1.6 "Noto Sans SC","PingFang SC",system-ui,sans-serif;}
main{max-width:1020px;margin:0 auto;padding:52px 24px 90px;
 display:flex;flex-direction:column;gap:30px;}
h1{font-size:29px;margin:0 0 6px;letter-spacing:-.01em;text-wrap:balance;}
h2{font-size:15px;margin:0;font-weight:500;}
.eyebrow{font:500 11px/1 "IBM Plex Mono",monospace;letter-spacing:.14em;
 text-transform:uppercase;color:var(--faint);}
.lede{color:var(--dim);max-width:64ch;margin:0;}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;
 position:sticky;top:0;background:var(--ground);padding:12px 0;z-index:5;
 border-bottom:1px solid var(--edge);}
button{font:500 13px/1 "Noto Sans SC",system-ui,sans-serif;cursor:pointer;
 background:var(--panel);color:var(--dim);border:1px solid var(--edge);
 border-radius:5px;padding:8px 14px;}
button[aria-pressed=true]{color:var(--panel);border-color:transparent;}
button.eq[aria-pressed=true]{background:var(--eq);}
button.weak[aria-pressed=true]{background:var(--weak);}
button.ref[aria-pressed=true]{background:var(--ref);}
button.ghost[aria-pressed=true]{background:var(--node);}
.hint{font-size:12px;color:var(--faint);margin-left:auto;
 font-family:"IBM Plex Mono",monospace;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:18px;}
.card{background:var(--panel);border:1px solid var(--edge);border-radius:7px;
 padding:16px 18px 10px;}
.card header{display:flex;justify-content:space-between;align-items:baseline;
 gap:10px;margin-bottom:2px;}
.card .sub{font:400 12px/1.4 "IBM Plex Mono",monospace;color:var(--faint);}
.tot{font:500 12px "IBM Plex Mono",monospace;color:var(--dim);white-space:nowrap;}
svg{display:block;width:100%;height:auto;}
.lbl{font:500 11px "IBM Plex Mono",monospace;fill:var(--faint);}
.seat{font:500 13px "IBM Plex Mono",monospace;fill:var(--ink);}
.cnt{font:400 9px "IBM Plex Mono",monospace;fill:var(--panel);}
.nd{fill:var(--node);cursor:pointer;}
.nd.sel{fill:var(--ink);}
path.e{fill:none;stroke-linecap:round;transition:opacity .12s;}
.dim path.e{opacity:.05;}
.dim path.e.on{opacity:.85;}
.dim .nd{opacity:.3;} .dim .nd.sel,.dim .nd.adj{opacity:1;}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
 background:var(--ink);color:var(--ground);font:500 12px/1.5 "IBM Plex Mono",monospace;
 padding:7px 10px;border-radius:5px;z-index:20;max-width:280px;}
.note{font-size:13px;color:var(--dim);max-width:66ch;}
.note b{color:var(--ink);font-weight:500;}
footer{font:400 12px/1.6 "IBM Plex Mono",monospace;color:var(--faint);
 border-top:1px solid var(--edge);padding-top:16px;}
</style>
<main>
<header>
 <p class="eyebrow">IDRBench · 40 场 · deepseek-v4-pro · 每格 10 场平均</p>
 <h1>谁在改写谁</h1>
 <p class="lede">节点是一个 agent 的一轮发言，边是被 NLI 直接判定过的命题关系，
 方向按时间从早指向晚，且要求那条命题当时确实对接收方可见。
 A→A 这样的自持边保留下来，作为交互边的比较基线。</p>
</header>

<div class="bar">
 <button class="ref" data-t="细化" aria-pressed="true">细化（后者更强）</button>
 <button class="weak" data-t="弱化" aria-pressed="false">弱化（后者更弱）</button>
 <button class="eq" data-t="等价" aria-pressed="false">等价</button>
 <button class="ghost" id="selfbtn" aria-pressed="true">含自持边</button>
 <span class="hint">点节点只看它的边，再点一次取消</span>
</div>

<div class="grid" id="grid"></div>

<p class="note">四格里的交互边和自持边一样密：<b>agent 复述自己上一轮的概率，
和复述同伴的概率没有差别</b>（全部 p&gt;0.09）。这不是 DelibTrace 能测的量——
沿单个 agent 的时间线追踪一条种子事实，永远看不到 A 的话被 B 改写成什么样。</p>

<footer>
 由 experiments/analyze_interaction_graph.py 与 build_interaction_viewer.py 生成。
 边数为每场平均。同轮内不连边；跨轮的 X→X 记为自持。
</footer>
</main>
<div id="tip"></div>
<script>
const G = __DATA__, CELLS = __CELLS__, NAME = __NAME__, SUB = __SUB__;
const COL = {"等价":"eq","弱化":"weak","细化":"ref"};
const W = 430, H = 250, X0 = 72, XG = 122, Y0 = 56, YG = 62;
const pos = s => { const [a,r] = s.split("|");
  return [X0 + (r-1)*XG, Y0 + "ABC".indexOf(a)*YG]; };

function draw(c, type, withSelf) {
  const g = G[c], es = g.edges.filter(e => e.type === type
    && (withSelf || e.src.split("|")[0] !== e.dst.split("|")[0]));
  const max = Math.max(1, ...es.map(e => e.per_case));
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${NAME[c]}的交互图">`;
  for (let r = 1; r <= 3; r++)
    s += `<text class="lbl" x="${X0 + (r-1)*XG}" y="26" text-anchor="middle">第 ${r} 轮</text>`;
  for (const e of es) {
    const [x1,y1] = pos(e.src), [x2,y2] = pos(e.dst);
    const mx = (x1+x2)/2, off = (y1===y2 ? -26 : 0);
    const w = 0.7 + 5.3*e.per_case/max;
    s += `<path class="e" data-s="${e.src}" data-d="${e.dst}"`
      + ` d="M${x1},${y1} Q${mx},${(y1+y2)/2+off} ${x2},${y2}"`
      + ` stroke="var(--${COL[type]})" stroke-width="${w.toFixed(2)}"`
      + ` opacity="${(0.2+0.55*e.per_case/max).toFixed(2)}"`
      + ` data-tip="${e.src} → ${e.dst}｜${type} ${e.per_case.toFixed(1)} 条/场"></path>`;
  }
  for (const [slot, n] of Object.entries(g.nodes)) {
    const [x,y] = pos(slot);
    s += `<g class="ndg"><circle class="nd" data-n="${slot}" cx="${x}" cy="${y}"`
      + ` r="${(7 + 0.28*Math.sqrt(n)*3).toFixed(1)}"`
      + ` data-tip="${slot}｜${n.toFixed(1)} 条命题/场"></circle>`
      + `<text class="cnt" x="${x}" y="${y+3}" text-anchor="middle">${Math.round(n)}</text></g>`;
  }
  for (const a of "ABC")
    s += `<text class="seat" x="${X0-40}" y="${Y0 + "ABC".indexOf(a)*YG + 4}">${a}</text>`;
  const tot = es.reduce((k,e) => k + e.per_case, 0);
  return {svg: s + "</svg>", tot};
}

let type = "细化", withSelf = true, sel = null;
function render() {
  document.getElementById("grid").innerHTML = CELLS.map(c => {
    const {svg, tot} = draw(c, type, withSelf);
    return `<div class="card" data-c="${c}"><header><div><h2>${NAME[c]}</h2>`
      + `<p class="sub">${SUB[c]}</p></div>`
      + `<span class="tot">${tot.toFixed(0)} 条/场</span></header>${svg}</div>`;
  }).join("");
  if (sel) apply();
}
function apply() {
  document.querySelectorAll(".card").forEach(card => {
    card.classList.toggle("dim", !!sel);
    card.querySelectorAll("path.e").forEach(p => {
      const on = sel && (p.dataset.s === sel || p.dataset.d === sel);
      p.classList.toggle("on", !!on);
    });
    card.querySelectorAll(".nd").forEach(n => {
      n.classList.toggle("sel", n.dataset.n === sel);
      const adj = sel && [...card.querySelectorAll("path.e.on")].some(
        p => p.dataset.s === n.dataset.n || p.dataset.d === n.dataset.n);
      n.classList.toggle("adj", !!adj);
    });
  });
}
document.querySelectorAll("button[data-t]").forEach(b => b.onclick = () => {
  type = b.dataset.t;
  document.querySelectorAll("button[data-t]").forEach(
    o => o.setAttribute("aria-pressed", o === b));
  render();
});
document.getElementById("selfbtn").onclick = e => {
  withSelf = !withSelf;
  e.target.setAttribute("aria-pressed", withSelf); render();
};
const tip = document.getElementById("tip");
addEventListener("mouseover", e => {
  const t = e.target.closest("[data-tip]");
  if (!t) return;
  tip.textContent = t.dataset.tip; tip.style.opacity = 1;
});
addEventListener("mousemove", e => {
  if (tip.style.opacity == 1) {
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 290) + "px";
    tip.style.top = (e.clientY + 18) + "px";
  }
});
addEventListener("mouseout", e => {
  if (e.target.closest("[data-tip]")) tip.style.opacity = 0;
});
addEventListener("click", e => {
  const n = e.target.closest(".nd");
  sel = (n && n.dataset.n !== sel) ? n.dataset.n : null;
  apply();
});
render();
</script>
"""

out = ROOT / "findings" / "2026-09-07-interaction-graph.html"
out.write_text(HTML
               .replace("__DATA__", json.dumps(G, ensure_ascii=False))
               .replace("__CELLS__", json.dumps(CELLS, ensure_ascii=False))
               .replace("__NAME__", json.dumps(NAME, ensure_ascii=False))
               .replace("__SUB__", json.dumps(SUB, ensure_ascii=False)))
print(f"写入 {out}  ({out.stat().st_size / 1024:.0f} KB)")
