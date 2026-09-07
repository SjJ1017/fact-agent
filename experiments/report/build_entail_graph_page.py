import json, statistics as st
from pathlib import Path

D = json.load(open('findings/data/entail-graph-fig.json'))
cells = D['cells']
S, P, C, M, A = D['summary'], D['points'], D['chains'], D['mix'], D['agents']
LAB = {'full-generic':'完整信息 · 同质角色','full-specialist':'完整信息 · 专家角色',
       'split-generic':'划分信息 · 同质角色','split-specialist':'划分信息 · 专家角色'}
SHORT = {'full-generic':'full/generic','full-specialist':'full/specialist',
         'split-generic':'split/generic','split-specialist':'split/specialist'}

# ---- 图 1：每场一个点的降级率 ------------------------------------------
W, H, PADL, PADR, PADT, PADB = 720, 320, 58, 18, 18, 54
lo, hi = 0.15, 1.0
def y(v): return PADT + (hi - v) / (hi - lo) * (H - PADT - PADB)
colw = (W - PADL - PADR) / 4
strip = []
for i, c in enumerate(cells):
    cx = PADL + colw * (i + .5)
    vals = sorted(P[c])
    # 同值错开，避免叠死
    for j, v in enumerate(vals):
        near = sum(1 for u in vals[:j] if abs(u - v) < .012)
        strip.append(f'<circle cx="{cx + (near % 3 - 1) * 7:.1f}" cy="{y(v):.1f}" r="4" '
                     f'class="dot {"sp" if c.startswith("split") else "fu"}"/>')
    m = st.mean(vals)
    strip.append(f'<line x1="{cx-34:.0f}" x2="{cx+34:.0f}" y1="{y(m):.1f}" y2="{y(m):.1f}" class="mean"/>')
    strip.append(f"<text x=\"{cx+40:.0f}\" y=\"{y(m)+4:.0f}\" class=\"mval\">{m:.2f}</text>")
    strip.append(f'<text x="{cx:.0f}" y="{H-PADB+20:.0f}" class="xl">{SHORT[c].split("/")[0]}</text>')
    strip.append(f'<text x="{cx:.0f}" y="{H-PADB+35:.0f}" class="xl2">{SHORT[c].split("/")[1]}</text>')
grid = []
for v in (0.2, 0.4, 0.6, 0.8, 1.0):
    grid.append(f'<line x1="{PADL}" x2="{W-PADR}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/>')
    grid.append(f'<text x="{PADL-10}" y="{y(v)+4:.1f}" class="yl">{v:.1f}</text>')
fig1 = f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="每场辩论的降级率">{"".join(grid)}{"".join(strip)}</svg>'

# ---- 图 2：边的构成 ------------------------------------------------------
W2, RH, GAP = 720, 30, 16
rows = []
top = 8
for c in cells:
    m = M[c]; tot = sum(m.values())
    x = 190
    for k, cls in (('degraded','deg'), ('refined','ref'), ('concurrent','con')):
        w = (W2 - 210) * m[k] / tot
        rows.append(f'<rect x="{x:.1f}" y="{top}" width="{w:.1f}" height="{RH}" class="bar {cls}"/>')
        if w > 46:
            rows.append(f'<text x="{x + w/2:.1f}" y="{top + RH/2 + 4:.0f}" class="bl">{m[k]:,}</text>')
        x += w
    rows.append(f'<text x="182" y="{top + RH/2 + 4:.0f}" class="rl">{SHORT[c]}</text>')
    top += RH + GAP
fig2 = f'<svg viewBox="0 0 {W2} {top}" role="img" aria-label="蕴含边的构成">{"".join(rows)}</svg>'

# ---- 图 3：agent 净收支 --------------------------------------------------
W3, BH = 720, 22
rr, ty = [], 10
mx = max(abs(v[0]-v[1]) for c in cells for v in A[c].values())
mid = 400
for c in cells:
    rr.append(f'<text x="8" y="{ty+14}" class="rl2">{SHORT[c]}</text>')
    for ag in sorted(A[c]):
        out, inn = A[c][ag]; net = out - inn
        w = abs(net) / mx * 230
        x0 = mid if net >= 0 else mid - w
        rr.append(f'<rect x="{x0:.1f}" y="{ty}" width="{max(w,1.5):.1f}" height="{BH-6}" '
                  f'class="bar {"pos" if net>=0 else "neg"}"/>')
        rr.append(f'<text x="{mid - 250}" y="{ty+13}" class="agl">{ag}</text>')
        rr.append(f'<text x="{(mid + w + 8) if net>=0 else (mid - w - 8):.0f}" y="{ty+13}" '
                  f'class="agv" text-anchor="{"start" if net>=0 else "end"}">{net:+d}</text>')
        ty += BH
    ty += 8
rr.insert(0, f'<line x1="{mid}" x2="{mid}" y1="4" y2="{ty-8}" class="zero"/>')
fig3 = f'<svg viewBox="0 0 {W3} {ty}" role="img" aria-label="每个 agent 的净弱化收支">{"".join(rr)}</svg>'

def cellrow(c):
    s = S[c]
    return (f'<tr><td>{SHORT[c]}</td><td>{s["facts"]:,}</td><td class="hi">{s["deg_per_fact"]:.3f}</td>'
            f'<td>{s["longest_chain_mean"]:.1f}</td><td>{s["round_span_mean"]:.2f}</td>'
            f'<td>{s["cross_agent_share"]:.0%}</td></tr>')

tot_deg = sum(S[c]['degraded'] for c in cells)
tot_ref = sum(S[c]['refined'] for c in cells)
tot_con = sum(S[c]['concurrent'] for c in cells)
full_mean = st.mean([S['full-generic']['deg_per_fact'], S['full-specialist']['deg_per_fact']])
split_mean = st.mean([S['split-generic']['deg_per_fact'], S['split-specialist']['deg_per_fact']])

html = f'''<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>信息划分与事实降级</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Noto+Sans+SC:wght@400;500&display=swap">
<style>
:root{{--bg:#f6f4f0;--card:#fffefb;--ink:#1b1a17;--body:#3a3630;--dim:#6f685e;
 --rule:#ddd7cd;--rule2:#ece7de;--deg:#a8501f;--ref:#2f6b58;--con:#9b9488;
 --pos:#a8501f;--neg:#2f6b58;--hi:#8c3f16;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{
 --bg:#171613;--card:#1f1e1a;--ink:#eeeae2;--body:#c9c3b8;--dim:#928b80;
 --rule:#35322b;--rule2:#292620;--deg:#d9814e;--ref:#63b394;--con:#7d766b;
 --pos:#d9814e;--neg:#63b394;--hi:#e39a68;}}}}
:root[data-theme=dark]{{--bg:#171613;--card:#1f1e1a;--ink:#eeeae2;--body:#c9c3b8;--dim:#928b80;
 --rule:#35322b;--rule2:#292620;--deg:#d9814e;--ref:#63b394;--con:#7d766b;
 --pos:#d9814e;--neg:#63b394;--hi:#e39a68;}}
*{{box-sizing:border-box}}
body{{background:var(--bg);color:var(--body);margin:0;
 font:15px/1.65 "IBM Plex Sans","Noto Sans SC",system-ui,sans-serif}}
.wrap{{max-width:830px;margin:0 auto;padding:46px 26px 90px}}
h1{{font-size:31px;line-height:1.22;margin:0 0 10px;color:var(--ink);font-weight:600;
 letter-spacing:-.015em;text-wrap:balance}}
.sub{{font-size:15.5px;color:var(--dim);margin:0 0 6px;max-width:62ch}}
.meta{{font:12px/1.5 "IBM Plex Mono",monospace;color:var(--dim);
 border-top:1px solid var(--rule);padding-top:12px;margin-top:26px}}
h2{{font-size:19px;margin:52px 0 6px;color:var(--ink);font-weight:600;letter-spacing:-.01em}}
h2+p{{margin-top:0}}
p{{max-width:64ch}}
figure{{margin:22px 0 0;background:var(--card);border:1px solid var(--rule);
 border-radius:9px;padding:20px 22px 16px}}
figcaption{{font-size:12.5px;color:var(--dim);margin-top:12px;
 border-top:1px solid var(--rule2);padding-top:10px;max-width:none}}
svg{{display:block;width:100%;height:auto;overflow:visible}}
.grid{{stroke:var(--rule2);stroke-width:1}}
.zero{{stroke:var(--rule);stroke-width:1}}
.dot{{opacity:.82}}
.dot.fu{{fill:var(--deg)}} .dot.sp{{fill:var(--ref)}}
.mean{{stroke:var(--ink);stroke-width:2.2}}
.mval{{fill:var(--ink);font:500 13px "IBM Plex Mono",monospace;text-anchor:start}}
.xl{{fill:var(--ink);font:500 12.5px "IBM Plex Mono",monospace;text-anchor:middle}}
.xl2{{fill:var(--dim);font:11.5px "IBM Plex Mono",monospace;text-anchor:middle}}
.yl{{fill:var(--dim);font:11.5px "IBM Plex Mono",monospace;text-anchor:end}}
.bar.deg{{fill:var(--deg)}} .bar.ref{{fill:var(--ref)}} .bar.con{{fill:var(--con)}}
.bar.pos{{fill:var(--pos)}} .bar.neg{{fill:var(--neg)}}
.bl{{fill:#fff;font:500 11.5px "IBM Plex Mono",monospace;text-anchor:middle}}
.rl{{fill:var(--ink);font:12px "IBM Plex Mono",monospace;text-anchor:end}}
.rl2{{fill:var(--ink);font:500 12px "IBM Plex Mono",monospace}}
.agl{{fill:var(--dim);font:11.5px "IBM Plex Mono",monospace}}
.agv{{fill:var(--dim);font:11.5px "IBM Plex Mono",monospace}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--dim);margin-top:12px}}
.legend i{{width:11px;height:11px;border-radius:2.5px;display:inline-block;
 margin-right:5px;vertical-align:-1px}}
table{{width:100%;border-collapse:collapse;margin-top:20px;
 font-variant-numeric:tabular-nums;font-size:13.5px}}
th{{text-align:right;font:500 11.5px "IBM Plex Mono",monospace;color:var(--dim);
 padding:0 0 8px;border-bottom:1px solid var(--rule);letter-spacing:.04em}}
th:first-child,td:first-child{{text-align:left;font-family:"IBM Plex Mono",monospace}}
td{{text-align:right;padding:8px 0;border-bottom:1px solid var(--rule2);color:var(--body)}}
td.hi{{color:var(--hi);font-weight:600}}
.note{{background:var(--card);border:1px solid var(--rule);border-left:3px solid var(--deg);
 border-radius:0 8px 8px 0;padding:15px 18px;margin-top:24px;font-size:14px}}
.note b{{color:var(--ink);font-weight:600}}
ul{{max-width:64ch;padding-left:20px}} li{{margin:7px 0}}
code{{font:12.5px "IBM Plex Mono",monospace;background:var(--rule2);
 padding:1px 5px;border-radius:3px}}
</style>

<div class="wrap">
<h1>划分信息把事实降级砍掉三分之一，分化角色没有影响</h1>
<p class="sub">IDRBench 的 40 场 idea-generation 辩论，用本地 NLI 判决器重做匹配后，
第一次能把「同一条事实」和「被削弱的同一条事实」分开来数。</p>
<p class="meta">40 场 · 9,542 条事实 · 94,843 组候选对 · 判决器 Qwen3-14B 双向蕴含，阈值 5.28<br>
生成 <code>experiments/analyze_entail_graph.py</code></p>

<h2>降级率</h2>
<p>降级边是指：一条事实的更弱版本在更晚的轮次出现，而说出弱版本的 agent 当时能看到强版本那一轮。
下图每个点是一场辩论，纵轴是降级边数除以该场的事实数。</p>
<figure>{fig1}
<div class="legend"><span><i style="background:var(--deg)"></i>完整信息</span>
<span><i style="background:var(--ref)"></i>划分信息</span>
<span><i style="background:var(--ink);height:2px;border-radius:0"></i>组均值</span></div>
<figcaption>两个完整信息条件（{S['full-generic']['deg_per_fact']:.2f}、{S['full-specialist']['deg_per_fact']:.2f}）和两个划分信息条件（{S['split-generic']['deg_per_fact']:.2f}、{S['split-specialist']['deg_per_fact']:.2f}）几乎不重叠，
而每一对内部的差异小于 0.01。角色分化在这个量上不起作用。</figcaption>
</figure>

<table>
<tr><th>条件</th><th>事实</th><th>降级/事实</th><th>最长链</th><th>轮跨度</th><th>跨 agent</th></tr>
{''.join(cellrow(c) for c in cells)}
</table>

<h2>边的构成</h2>
<p>把蕴含边按强弱两句的时间关系分成三类。只有第一类是关于「信息在传递中丢失」的断言。</p>
<figure>{fig2}
<div class="legend"><span><i style="background:var(--deg)"></i>降级 · 强在前 {tot_deg:,}</span>
<span><i style="background:var(--ref)"></i>细化 · 强在后 {tot_ref:,}</span>
<span><i style="background:var(--con)"></i>同轮 · 无先后 {tot_con:,}</span></div>
<figcaption>「细化」是先说笼统的、后补具体的，在 idea generation 里是正常动作，不是信息损失。
「同轮」两个 agent 并行生成，不可能传递。三类占比在四个条件下形状接近，
变化的是绝对量而不是构成。</figcaption>
</figure>

<h2>没有专门的「泛化者」</h2>
<p>如果某个座位系统性地把别人的事实说弱，它的净收支应该明显为正。实际不是。</p>
<figure>{fig3}
<figcaption>净收支 = 该 agent 说出的强版本被别人弱化的次数 − 它弱化别人的次数。
四个条件十二个座位，净值全部落在 ±54 以内，而各自的总量在 228–464 之间。
降级不是某个角色的行为，是系统的行为。</figcaption>
</figure>

<div class="note">
<b>一个必须先说的限制。</b>「能看到」这个条件在 cumulative memory 下几乎不筛任何东西——
9,202 条跨轮边里 0 条因为不可见被排除。所以降级边的含义是「弱版本更晚、且强版本当时在上下文里」，
不是「弱版本从强版本来」。要把它坐实成因果，需要能切断可见性的条件（peer-only 或 self-last 记忆），
现有语料没有。
</div>

<h2>怎么读这些数</h2>
<ul>
<li><b>划分信息降低降级，不是因为事实更少。</b>四个条件的事实数在 2,285–2,506 之间，差异小于 10%，
而降级率差了 36%。</li>
<li><b>降级链也变短</b>：完整信息下平均最长链 3.3–3.4 跳，划分信息下 2.6–2.8。一条事实被反复削弱的
次数变少了。</li>
<li><b>轮跨度几乎不变</b>（1.12–1.29），说明降级主要发生在相邻轮次，不是跨越整场慢慢流失。</li>
<li><b>跨 agent 比例 61–67%</b>，其余是同一 agent 把自己上一轮的说法说弱了——那一类同样是降级，
而且不需要传递。</li>
</ul>

<h2>与 DelibTrace 的关系</h2>
<p>DelibTrace 用 Jaccard 测事实存活，一条事实要么在要么不在。上面这 {tot_deg:,} 条降级边在那个框架里
只有两种记法：算作同一条（信息损失不可见），或算作两条不同（看起来像丢了一条又新造一条）。
两种都描述不了「还在但被削弱了」这个状态，而它占了非无关边的三分之一。</p>
</div>
'''
Path('findings/2026-09-04-idrbench-entail-graph.html').write_text(html)
print('写入 findings/2026-09-04-idrbench-entail-graph.html',
      f"{len(html)/1024:.0f} KB")
