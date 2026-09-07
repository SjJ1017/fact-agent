#!/usr/bin/env python3
"""Render the topic-conditioned findings page (artifact-ready: no doctype/head)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = json.loads((ROOT / "findings/data/topic-flow.json").read_text())
S = json.loads((ROOT / "findings/data/topic-specialisation.json").read_text())
CELLS = ["full-generic", "full-specialist", "split-generic", "split-specialist"]
NAME = {"full-generic": "全量来源 · 通用角色", "full-specialist": "全量来源 · 专家角色",
        "split-generic": "分割来源 · 通用角色", "split-specialist": "分割来源 · 专家角色"}
SER = {"full-generic": "fg", "full-specialist": "fs",
       "split-generic": "sg", "split-specialist": "ss"}

# ---- specialisation chart -------------------------------------------------
W, H, PL, PR, PT, PB = 640, 300, 62, 118, 24, 44
def px(r): return PL + (r - 1) * (W - PL - PR) / 2
def py(v): return H - PB - v / 0.50 * (H - PT - PB)

chart = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="专业化指数随轮次变化">']
for g in (0, .1, .2, .3, .4, .5):
    chart.append(f'<line x1="{PL}" y1="{py(g):.1f}" x2="{W-PR}" y2="{py(g):.1f}" '
                 f'stroke="var(--edge)" stroke-width="1"/>')
    chart.append(f'<text class="ax" x="{PL-10}" y="{py(g)+4:.1f}" text-anchor="end">'
                 f'{g*100:.0f}%</text>')
for r in (1, 2, 3):
    chart.append(f'<text class="ax" x="{px(r):.1f}" y="{H-PB+20}" '
                 f'text-anchor="middle">第 {r} 轮</text>')
for c in CELLS:
    pts = [(px(r), py(S[f"{c}|{r}"]["mean"])) for r in (1, 2, 3)]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    chart.append(f'<path d="{d}" fill="none" stroke="var(--{SER[c]})" '
                 f'stroke-width="2.4" stroke-linejoin="round"/>')
    for x, y in pts:
        chart.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" '
                     f'fill="var(--{SER[c]})"/>')
    lx, ly = pts[-1]
    chart.append(f'<text class="sl" x="{lx+10:.1f}" y="{ly+4:.1f}" '
                 f'fill="var(--{SER[c]})">{NAME[c]}</text>')
chart.append("</svg>")
CHART = "".join(chart)

# ---- manipulation check ---------------------------------------------------
def strip(cell, rnd):
    rows = []
    for seat, who in (("A", "A（持 paper-b）"), ("B", "B（持 paper-c）"),
                      ("C", "C（无来源）")):
        v = T["seat_topic"][cell].get(f"{seat}|{rnd}", {})
        n = sum(v.values()) or 1
        segs = "".join(
            f'<span class="{k}" style="--w:{v.get(k,0)/n*100:.2f}%" '
            f'title="{k} {v.get(k,0)/n:.1%}"></span>'
            for k in ("B", "C", "X", "G", "U", "未标注"))
        rows.append(f'<tr><th>{who if cell.startswith("split") else seat}</th>'
                    f'<td><div class="stack">{segs}</div></td>'
                    f'<td class="n">{v.get("B",0)/n:.0%} / {v.get("C",0)/n:.0%}</td></tr>')
    return "".join(rows)

MANIP = "".join(
    f'<div><h3>{NAME[c]} · 第 {r} 轮</h3><table>{strip(c, r)}</table></div>'
    for c in ("split-generic", "full-specialist") for r in (1, 2))

# ---- overlap composition --------------------------------------------------
def ov(cell, rnd, key):
    v = T["overlap_paper"][f"{cell}|{rnd}"][key]
    n = sum(v.values()) or 1
    return "".join(f'<td class="n{" hi" if k=="P" and cell.startswith("split") and rnd==1 else ""}">'
                   f'{v.get(k,0)/n:.1%}</td>' for k in "BCPU") + f'<td class="n">{n:,}</td>'

OVER = "".join(
    f'<tr><th>{NAME[c]} · 第 {r} 轮</th>{ov(c, r, "被他人等价覆盖")}</tr>'
    for c in CELLS for r in (1, 3))

TESTS = [("只加角色（专家 vs 通用，来源都全）", "+1.6pp", "p = 0.692", False),
         ("只分信息（分割 vs 全量，角色都通用）", "+40.6pp", "p < 0.001", True),
         ("分信息之后再加角色", "−4.1pp", "p = 0.556", False)]
TROWS = "".join(
    f'<tr><th>{a}</th><td class="n{" hi" if hi else ""}">{b}</td>'
    f'<td class="n{" hi" if hi else ""}">{c}</td></tr>' for a, b, c, hi in TESTS)

HTML = f"""<title>角色没用，分信息只管一轮</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--ground:#f7f7f5;--panel:#fffefc;--edge:#e0dfd9;--ink:#1b1f24;--dim:#6b7178;
 --faint:#9aa0a6;--hi:#1f7a6b;
 --B:#3f5fa8;--C:#c4703a;--X:#1f7a6b;--G:#9aa0a6;--U:#cfcdc6;
 --fg:#9aa0a6;--fs:#6b7178;--sg:#c4703a;--ss:#3f5fa8;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{
 --ground:#15171a;--panel:#1c1f23;--edge:#2e3238;--ink:#e8e6e1;--dim:#9aa0a6;
 --faint:#6b7178;--hi:#4fbfa8;--B:#7b96d8;--C:#e0955c;--X:#4fbfa8;--G:#6e6c67;
 --U:#3a3d42;--fg:#6b7178;--fs:#9aa0a6;--sg:#e0955c;--ss:#7b96d8;}}}}
:root[data-theme=dark]{{--ground:#15171a;--panel:#1c1f23;--edge:#2e3238;--ink:#e8e6e1;
 --dim:#9aa0a6;--faint:#6b7178;--hi:#4fbfa8;--B:#7b96d8;--C:#e0955c;--X:#4fbfa8;
 --G:#6e6c67;--U:#3a3d42;--fg:#6b7178;--fs:#9aa0a6;--sg:#e0955c;--ss:#7b96d8;}}
body{{background:var(--ground);color:var(--ink);margin:0;
 font:400 15px/1.65 "Noto Sans SC","PingFang SC",system-ui,sans-serif;}}
main{{max-width:900px;margin:0 auto;padding:54px 24px 92px;
 display:flex;flex-direction:column;gap:40px;}}
h1{{font-size:30px;line-height:1.25;margin:0 0 8px;letter-spacing:-.01em;text-wrap:balance;}}
h2{{font-size:19px;margin:0 0 4px;}}
h3{{font-size:13px;margin:0 0 7px;color:var(--dim);font-weight:500;
 font-family:"IBM Plex Mono",monospace;}}
.eyebrow{{font:500 11px/1 "IBM Plex Mono",monospace;letter-spacing:.14em;
 text-transform:uppercase;color:var(--faint);}}
.lede{{color:var(--dim);max-width:64ch;margin:0;}}
section{{display:flex;flex-direction:column;gap:10px;}}
.card{{background:var(--panel);border:1px solid var(--edge);border-radius:6px;
 padding:20px 22px;overflow-x:auto;}}
table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;}}
th{{text-align:left;font-weight:500;color:var(--ink);white-space:nowrap;
 padding:6px 14px 6px 0;}}
thead th{{font:500 11px/1 "IBM Plex Mono",monospace;letter-spacing:.06em;
 color:var(--faint);border-bottom:1px solid var(--edge);padding-bottom:8px;}}
td{{padding:5px 6px;}}
td.n{{text-align:right;font-family:"IBM Plex Mono",monospace;color:var(--dim);
 white-space:nowrap;}}
td.n.hi{{color:var(--hi);font-weight:500;}}
.stack{{display:flex;height:16px;border-radius:3px;overflow:hidden;min-width:210px;
 background:var(--edge);}}
.stack span{{width:var(--w);}}
.stack .B{{background:var(--B);}} .stack .C{{background:var(--C);}}
.stack .X{{background:var(--X);}} .stack .G{{background:var(--G);}}
.stack .U{{background:var(--U);}}
.stack .未标注{{background:repeating-linear-gradient(45deg,var(--U) 0 3px,transparent 3px 7px);}}
.pair{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:20px;}}
.key{{display:flex;flex-wrap:wrap;gap:13px;font:400 12px "IBM Plex Mono",monospace;
 color:var(--dim);margin-top:4px;}}
.key i{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;}}
svg{{display:block;width:100%;height:auto;}}
.ax{{font:400 11px "IBM Plex Mono",monospace;fill:var(--faint);}}
.sl{{font:500 11px "Noto Sans SC",system-ui,sans-serif;}}
.note{{font-size:13px;color:var(--dim);margin:10px 0 0;max-width:66ch;}}
.note b{{color:var(--ink);font-weight:500;}}
.big{{font:500 40px/1 "IBM Plex Mono",monospace;letter-spacing:-.02em;color:var(--hi);}}
footer{{font:400 12px/1.6 "IBM Plex Mono",monospace;color:var(--faint);
 border-top:1px solid var(--edge);padding-top:16px;}}
</style>
<main>
<header>
 <p class="eyebrow">IDRBench · 40 场 · 9,989 条原子事实带学科标签</p>
 <h1>角色没用，分信息只管一轮</h1>
 <p class="lede">每条原子事实带两根独立的标签轴：<b>主题</b>说它谈的是哪篇论文的内容，
 <b>归属</b>说它是在陈述某篇论文还是在提新方案。有了这两根轴，
 "专家角色让 agent 各说各的"这句话第一次变得可以证伪。</p>
</header>

<section>
 <p class="eyebrow">主结果</p>
 <h2>专业化指数：本侧主题占比减对侧</h2>
 <div class="card">{CHART}
  <p class="note">A 席位名义上属于 paper-b 一侧、B 席位属于 paper-c 一侧
  （通用角色条件下没有这个指派，按同一映射读作零假设）。
  两条灰线几乎贴在零上：<b>只给角色提示，内容分不开</b>。
  橙蓝两条在第 1 轮冲到 40% 以上，第 2 轮就掉回灰线里。</p>
 </div>
 <div class="card">
  <table><thead><tr><th>对照</th><th class="n">第 1 轮差值</th>
  <th class="n">置换检验</th></tr></thead><tbody>{TROWS}</tbody></table>
  <p class="note">三组对照只有一组显著。<b>角色提示对内容构成没有可测量的影响</b>
  （三轮 p 分别为 0.69、0.88、0.84）；<b>信息分割在第 1 轮制造 40.6 个百分点的差异，
  到第 2 轮只剩 0.8 个百分点</b>（p=0.84）。分信息之后再叠角色，也没有额外效应。</p>
 </div>
</section>

<section>
 <p class="eyebrow">操纵检查</p>
 <h2>分割确实分开了，也确实一轮就没了</h2>
 <div class="card"><div class="pair">{MANIP}</div>
  <div class="key">
   <span><i style="background:var(--B)"></i>B 侧主题</span>
   <span><i style="background:var(--C)"></i>C 侧主题</span>
   <span><i style="background:var(--X)"></i>两篇共有</span>
   <span><i style="background:var(--G)"></i>通用</span>
   <span><i style="background:var(--U)"></i>无法判定</span>
  </div>
  <p class="note">右列是该席位的 B 主题 / C 主题占比。分割条件第 1 轮，
  A 是 43%/0%、B 是 4%/40%、C 席位（没有任何来源）两者都是 0%，
  六成内容落在"通用"、三成半"无法判定"。到第 2 轮 A 就变成 30%/26%——
  <b>和全量条件下的任何一个席位都分不出来了</b>。</p>
 </div>
</section>

<section>
 <p class="eyebrow">机制</p>
 <h2>没有共同来源时，重合的全是各自编出来的</h2>
 <div class="card">
  <p class="big">0.0%</p>
  <p class="note">分割条件第 1 轮，被另一个 agent 等价复述的内容里，
  <b>陈述 paper-b 的占 0.0%，陈述 paper-c 的也占 0.0%</b>。
  全部是新提案和无法归属的表述。这在机制上是必然的——他们没读过对方的论文，
  不可能在论文内容上重合；<b>但他们仍然独立地想到了同样的东西</b>。</p>
  <table style="margin-top:14px"><thead><tr><th>被他人等价复述的内容</th>
   <th class="n">陈述 B</th><th class="n">陈述 C</th><th class="n">新提案</th>
   <th class="n">不确定</th><th class="n">n</th></tr></thead>
   <tbody>{OVER}</tbody></table>
  <p class="note">上一版报告说这里的重合"是共享文献造成的起点重合"，那是错的：
  分割条件下三个 agent 零共同文献。真实机制是<b>独立发明的趋同</b>，
  而角色分化把这种趋同的发生率减半（23.4% → 11.0%）。
  到第 3 轮，分割条件的重合里"陈述 B"回升到 30–49%，说明论文内容确实传过去了。</p>
 </div>
</section>

<section>
 <p class="eyebrow">范围</p>
 <h2>这批标签能支持到哪</h2>
 <div class="card">
  <p class="note">9,989 条事实拿到标签，四个条件各缺 2.0–2.2%——
  三个 case 里有几个批次，模型坚持给"归属"轴打 <code>G</code>，
  而 <code>G</code> 只存在于"主题"轴。那类事实是"两篇都归不上"的元陈述
  （"这个联系是因果的"、"任何整合都会凭空发明素材"）。
  <b>这是标签体系的缺口，不是模型不稳</b>：温度为零，重跑逐字复现。
  我没有替它把 G 映射成 P 或 U，缺的记作未标注，在图里画成斜纹。
  缺失在四个条件间均匀，不偏向任何一格。</p>
  <p class="note">另外：自持边和交互边携带的主题构成几乎完全一样
  （全量通用 31.7/28.1/32.9 对 32.0/27.1/32.9）——
  <b>agent 把自己的话往下带、和把同伴的话往下带，内容类型上没有区别</b>。</p>
 </div>
</section>

<footer>
 标注 qwen3.8-flash，enable_thinking=false，实测 reasoning_chars=0。
 分析脚本 experiments/analyze_topic_flow.py 与 test_topic_specialisation.py。
 置换检验以「场」为独立单位，每格 10 场，20000 次重排，双侧。
 无干预，以上均为描述性结果。
</footer>
</main>
"""
out = ROOT / "findings" / "2026-09-07-topic-flow.html"
out.write_text(HTML)
print(f"写入 {out}  ({out.stat().st_size/1024:.0f} KB)")
