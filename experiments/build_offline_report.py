#!/usr/bin/env python3
"""Render findings/2026-09-07-offline-flow.html from the cached analysis JSON.

Two display rules carry the brief's constraints into the page.  Multi-hot rows
(§1, §2) get bar-backed cells rather than a stacked bar, because an exposure
can be equivalent to one output and weaker than another and the shares do not
sum to 100%.  Unscored is drawn hatched everywhere, so absence of a judgement
never reads as a category.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "findings/data/offline-flow.json").read_text())
T = json.loads((ROOT / "findings/data/offline-flow-tests.json").read_text())
CELLS = ["full-generic", "full-specialist", "split-generic", "split-specialist"]
NAME = {"full-generic": "全量来源 · 通用角色", "full-specialist": "全量来源 · 专家角色",
        "split-generic": "分割来源 · 通用角色", "split-specialist": "分割来源 · 专家角色"}
KINDS = ["等价", "弱化", "细化", "无关"]
VAR = {"等价": "eq", "弱化": "weak", "细化": "ref", "无关": "un", "未评分": "no"}


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def bar_row(label: str, cnt: dict, keys: list[str], n: int) -> str:
    cells = "".join(
        f'<td class="v {VAR.get(k, "")}"><i style="--w:{cnt.get(k, 0) / n * 100:.1f}%"></i>'
        f'<b>{pct(cnt.get(k, 0) / n)}</b></td>' for k in keys)
    return f'<tr><th>{label}</th>{cells}<td class="n">{n:,}</td></tr>'


def stack(cnt: dict, keys: list[str], colors: list[str]) -> str:
    n = sum(cnt.get(k, 0) for k in keys) or 1
    segs = "".join(
        f'<span class="{c}" style="--w:{cnt.get(k, 0) / n * 100:.3f}%" '
        f'title="{k} {pct(cnt.get(k, 0) / n)}"></span>'
        for k, c in zip(keys, colors) if cnt.get(k, 0))
    return f'<div class="stack">{segs}</div>'


# ---------------------------------------------------------------- sections
cov = "".join(
    f'<tr><th>{NAME[c]}</th><td class="n">{D[c]["coverage"]["scored"]:,}</td>'
    f'<td class="n">{D[c]["coverage"]["possible"]:,}</td>'
    f'<td class="v no"><i style="--w:{D[c]["coverage"]["scored"] / D[c]["coverage"]["possible"] * 500:.1f}%"></i>'
    f'<b>{pct(D[c]["coverage"]["scored"] / D[c]["coverage"]["possible"])}</b></td></tr>'
    for c in CELLS)

q12 = ""
for c in CELLS:
    rows = "".join(
        bar_row(f"输入来自{side}", D[c]["q2"][side], KINDS + ["未评分"],
                D[c]["q2"][side]["n"])
        for side in ("自己", "同伴", "两者") if side in D[c]["q2"])
    rows += '<tr class="sep"><th colspan="7"></th></tr>'
    rows += "".join(
        bar_row(f"接收席位 {s}", v, KINDS + ["未评分"], v["n"])
        for s, v in sorted(D[c]["q1_seat"].items()))
    g = D[c]["q1_agreement"]
    q12 += (f'<h3>{NAME[c]}</h3><table class="grid">'
            f'<thead><tr><th></th>'
            + "".join(f"<th>{k}</th>" for k in KINDS + ["未评分"])
            + '<th class="n">曝光数</th></tr></thead><tbody>' + rows
            + '</tbody></table>'
            f'<p class="note">同一命题被两个以上接收者收到 {g["可比"]:,} 次留下了可评分关系，'
            f'其中 <b>{pct(g["接收者分歧"] / g["可比"])}</b> 的关系类型在接收者之间不一致。</p>')

q4 = "".join(
    f'<tr><th>{NAME[c]}</th><td>'
    + stack(D[c]["q4"], ["恢复到等价", "继续变弱", "反而更强", "转向别处", "端点未评分"],
            ["eq", "weak", "ref", "un", "no"])
    + "</td>" + "".join(
        f'<td class="n">{pct(D[c]["q4"].get(k, 0) / sum(D[c]["q4"].values()))}</td>'
        for k in ["恢复到等价", "继续变弱", "端点未评分"])
    + f'<td class="n">{sum(D[c]["q4"].values()):,}</td></tr>' for c in CELLS)

q5 = "".join(
    f'<tr><th>{NAME[c]}</th>' + "".join(
        f'<td class="v ref"><i style="--w:{D[c]["q5_rate"][r] * 400:.1f}%"></i>'
        f'<b>{pct(D[c]["q5_rate"][r])}</b></td>'
        f'<td class="v eq"><i style="--w:{D[c]["q5_joint"][r] * 400:.1f}%"></i>'
        f'<b>{pct(D[c]["q5_joint"][r])}</b></td>' for r in ("1", "2", "3"))
    + "</tr>" for c in CELLS)

K6 = ["等价重复", "已有内容的较弱表述", "细化候选", "无关于历史", "未匹配产出"]
q6 = "".join(
    f'<tr><th>{NAME[c]} · 第 {r} 轮</th><td>'
    + stack(D[c]["q6"][r], K6, ["eq", "weak", "ref", "un", "no"])
    + f'</td><td class="n">{sum(D[c]["q6"][r].values()):,}</td></tr>'
    for c in CELLS for r in ("2", "3"))

tests = "".join(
    f'<tr><th>{NAME[c]}</th><td class="n">{pct(T[f"redundant_refine|{c}"]["both"])}</td>'
    f'<td class="n">{pct(T[f"redundant_refine|{c}"]["peer"])}</td>'
    f'<td class="n hi">{T[f"redundant_refine|{c}"]["diff"] * 100:+.1f}pp</td>'
    f'<td class="n">p = {T[f"redundant_refine|{c}"]["p"]:.3f}</td></tr>'
    for c in CELLS if f"redundant_refine|{c}" in T)

asym = "".join(
    f'<tr><th>{NAME[c]} · {lab}</th>'
    f'<td class="n">{pct(T[f"asym|{c}|{lab}"]["self"])}</td>'
    f'<td class="n">{pct(T[f"asym|{c}|{lab}"]["peer"])}</td>'
    f'<td class="n">{T[f"asym|{c}|{lab}"]["diff"] * 100:+.1f}pp</td>'
    f'<td class="n{" hi" if T[f"asym|{c}|{lab}"]["p"] < .05 else ""}">'
    f'p = {T[f"asym|{c}|{lab}"]["p"]:.3f}</td></tr>'
    for c in CELLS for lab in ("细化", "弱化") if f"asym|{c}|{lab}" in T)

unsc = "".join(
    f'<tr><th>{"通用角色" if p == "generic" else "专家角色"}</th>'
    f'<td class="n">{pct(T[f"unscored|{p}"]["full"])}</td>'
    f'<td class="n">{pct(T[f"unscored|{p}"]["split"])}</td>'
    f'<td class="n hi">{T[f"unscored|{p}"]["diff"] * 100:+.1f}pp</td>'
    f'<td class="n hi">p &lt; 0.001</td></tr>' for p in ("generic", "specialist"))

HTML = f"""<title>改写方向的离线读数</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:#f7f7f5; --panel:#fffefc; --edge:#e0dfd9; --ink:#1b1f24;
  --dim:#6b7178; --faint:#9aa0a6;
  --eq:#1f7a6b; --weak:#c4703a; --ref:#3f5fa8; --un:#a8a6a0; --no:#cfcdc6;
  --hi:#1f7a6b;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme=light]) {{
    --ground:#15171a; --panel:#1c1f23; --edge:#2e3238; --ink:#e8e6e1;
    --dim:#9aa0a6; --faint:#6b7178;
    --eq:#4fbfa8; --weak:#e0955c; --ref:#7b96d8; --un:#6e6c67; --no:#3a3d42;
    --hi:#4fbfa8;
  }}
}}
:root[data-theme=dark] {{
  --ground:#15171a; --panel:#1c1f23; --edge:#2e3238; --ink:#e8e6e1;
  --dim:#9aa0a6; --faint:#6b7178;
  --eq:#4fbfa8; --weak:#e0955c; --ref:#7b96d8; --un:#6e6c67; --no:#3a3d42;
  --hi:#4fbfa8;
}}
body {{ background:var(--ground); color:var(--ink); margin:0;
  font:400 15px/1.65 "Noto Sans SC","PingFang SC",system-ui,sans-serif; }}
main {{ max-width:920px; margin:0 auto; padding:56px 24px 96px;
  display:flex; flex-direction:column; gap:44px; }}
h1 {{ font-size:30px; line-height:1.25; margin:0 0 6px; letter-spacing:-.01em;
  text-wrap:balance; }}
h2 {{ font-size:19px; margin:0 0 4px; letter-spacing:-.005em; }}
h3 {{ font-size:14px; margin:26px 0 8px; color:var(--dim); font-weight:500; }}
.eyebrow {{ font:500 11px/1 "IBM Plex Mono",monospace; letter-spacing:.14em;
  text-transform:uppercase; color:var(--faint); }}
.lede {{ color:var(--dim); font-size:15px; max-width:62ch; }}
section {{ display:flex; flex-direction:column; gap:10px; }}
.card {{ background:var(--panel); border:1px solid var(--edge); border-radius:6px;
  padding:20px 22px; overflow-x:auto; }}
.warn {{ border-left:3px solid var(--weak); }}
table {{ border-collapse:collapse; width:100%; font-size:13px;
  font-variant-numeric:tabular-nums; }}
th {{ text-align:left; font-weight:500; color:var(--dim); white-space:nowrap; }}
thead th {{ font:500 11px/1 "IBM Plex Mono",monospace; letter-spacing:.06em;
  color:var(--faint); padding-bottom:8px; border-bottom:1px solid var(--edge); }}
tbody th {{ padding:7px 16px 7px 0; color:var(--ink); }}
td {{ padding:5px 6px; }}
td.n {{ text-align:right; font-family:"IBM Plex Mono",monospace;
  color:var(--dim); white-space:nowrap; }}
td.n.hi {{ color:var(--hi); font-weight:500; }}
tr.sep th {{ padding:0; border-top:1px dashed var(--edge); height:10px; }}
td.v {{ position:relative; min-width:74px; text-align:right;
  font-family:"IBM Plex Mono",monospace; }}
td.v i {{ position:absolute; left:2px; top:4px; bottom:4px; width:var(--w);
  border-radius:2px; opacity:.24; background:currentColor; }}
td.v b {{ position:relative; font-weight:400; }}
td.v.eq {{ color:var(--eq); }} td.v.weak {{ color:var(--weak); }}
td.v.ref {{ color:var(--ref); }} td.v.un {{ color:var(--un); }}
td.v.no {{ color:var(--faint); }}
td.v.no i {{ background:repeating-linear-gradient(45deg,currentColor 0 2px,
  transparent 2px 5px); opacity:.5; }}
.stack {{ display:flex; height:17px; border-radius:3px; overflow:hidden;
  min-width:260px; background:var(--edge); }}
.stack span {{ width:var(--w); }}
.stack .eq {{ background:var(--eq); }} .stack .weak {{ background:var(--weak); }}
.stack .ref {{ background:var(--ref); }} .stack .un {{ background:var(--un); }}
.stack .no {{ background:repeating-linear-gradient(45deg,var(--no) 0 3px,
  transparent 3px 7px); }}
.key {{ display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--dim);
  font-family:"IBM Plex Mono",monospace; }}
.key i {{ display:inline-block; width:9px; height:9px; border-radius:2px;
  margin-right:5px; }}
.note {{ font-size:13px; color:var(--dim); margin:10px 0 0; max-width:66ch; }}
.note b {{ color:var(--ink); font-weight:500; }}
.big {{ font:500 34px/1 "IBM Plex Mono",monospace; color:var(--weak);
  letter-spacing:-.02em; }}
ul {{ margin:6px 0 0; padding-left:18px; color:var(--dim); font-size:14px;
  max-width:66ch; }}
li {{ margin:5px 0; }} li b {{ color:var(--ink); font-weight:500; }}
footer {{ font-size:12px; color:var(--faint); border-top:1px solid var(--edge);
  padding-top:16px; font-family:"IBM Plex Mono",monospace; }}
</style>
<main>
<header>
  <p class="eyebrow">IDRBench · 40 场 · deepseek-v4-pro · 3 轮</p>
  <h1>改写方向的离线读数</h1>
  <p class="lede">按 <code>2026-09-07-offline-factflow-analysis.md</code> 的第 1、2、4、5、6 节
  在缓存的 NLI 关系上计算。未评分的命题对一律记 unknown，方向不通过等价簇传递，
  同一命题对同一接收者只在首次可见的那一轮计一次曝光。</p>
</header>

<section>
  <p class="eyebrow">先看这个</p>
  <h2>可分析范围</h2>
  <div class="card warn">
    <p class="big">4.9%</p>
    <p class="note">blocker 只把约 5% 的命题对送进 NLI，其余 95% <b>从未被问过</b>。
    下面每个比率都只描述这 5% 之内可观察到的关系；把未评分当成"无关"会让稀疏条件
    看起来像是丢了内容。</p>
    <table class="grid" style="margin-top:16px">
      <thead><tr><th></th><th class="n">已评分对</th><th class="n">全部可能对</th>
      <th>占比</th></tr></thead><tbody>{cov}</tbody></table>
  </div>
</section>

<section>
  <p class="eyebrow">§1 · §2</p>
  <h2>同一个输入，谁在改写，改成什么方向</h2>
  <p class="lede">一次曝光可以同时命中多类关系（对某条输出等价、对另一条更弱），
  所以每行不求和到 100%。</p>
  <div class="card">
    <div class="key">
      <span><i style="background:var(--eq)"></i>等价</span>
      <span><i style="background:var(--weak)"></i>弱化</span>
      <span><i style="background:var(--ref)"></i>细化</span>
      <span><i style="background:var(--un)"></i>无关</span>
      <span><i style="background:var(--no)"></i>未评分</span>
    </div>
    {q12}
  </div>
</section>

<section>
  <p class="eyebrow">检验</p>
  <h2>三个结果</h2>
  <div class="card">
    <h3>一、自己说过又被同伴说过的命题，更常被细化</h3>
    <table><thead><tr><th></th><th class="n">两者都说过</th><th class="n">只来自同伴</th>
    <th class="n">差</th><th class="n">置换检验</th></tr></thead><tbody>{tests}</tbody></table>
    <p class="note">冗余不是浪费：<b>同一条命题从两个来源同时可见时，被细化的概率显著上升</b>，
    专家角色下的提升（+17.5pp）是通用角色的两倍多。分割来源的两个条件因为
    切断了这种重叠，"两者"的曝光只有 64 和 83 次，不足以进检验。</p>
    <p class="note">这不只是"两者更容易产出关系"。这类曝光留下任意关系的比例确实略高
    （全量通用 90.4%→97.5%，全量专家 88.8%→98.1%），但<b>在已经产生关系的曝光里，
    细化的份额仍然从 18.5% 升到 27.2%、从 18.9% 升到 34.1%</b>，
    细化对弱化的比值从 1.06 升到 1.33、从 1.11 升到 1.64。
    抬起来的不是产出量，是方向。</p>

    <h3>二、"对自己越说越具体、对同伴越转越宽泛"没有出现</h3>
    <table><thead><tr><th></th><th class="n">自己的历史</th><th class="n">同伴的历史</th>
    <th class="n">差</th><th class="n">置换检验</th></tr></thead><tbody>{asym}</tbody></table>
    <p class="note">四个条件里三个的细化与弱化都无差别。唯一显著的
    split-specialist 在<b>两个方向上同时</b>偏向自己的历史（细化 +4.5pp、弱化 +3.5pp），
    这是对自己内容的整体参与度更高，不是有向的改写偏差。§2 假设的不对称，在这批数据上不成立。</p>

    <h3>三、分割来源留下更多没有关系的产出</h3>
    <table><thead><tr><th></th><th class="n">全量来源</th><th class="n">分割来源</th>
    <th class="n">差</th><th class="n">置换检验</th></tr></thead><tbody>{unsc}</tbody></table>
    <p class="note">曝光之后本轮没有留下任何被判定关系的比例，从 9–11% 升到 21–25%。
    之前"分割降低每条事实的弱化率"的结论要接着这条读：<b>弱化变少，有一部分是因为
    命题之间根本没进入可比关系</b>，而不只是保真度更高。</p>
  </div>
</section>

<section>
  <p class="eyebrow">§4</p>
  <h2>中途变弱之后，端到端是什么关系</h2>
  <div class="card">
    <table><thead><tr><th></th><th>p → q → r 的端点关系</th>
    <th class="n">恢复等价</th><th class="n">继续变弱</th><th class="n">端点未评分</th>
    <th class="n">路径数</th></tr></thead><tbody>{q4}</tbody></table>
    <p class="note">两步都被判为弱化的路径里，端点关系能被读出来的只有三分之一。
    在这三分之一中，<b>恢复到等价几乎不发生（0–1%）</b>，继续变弱占 18–20%。
    可分析范围太窄，这里只报读数不下结论：单步弱化是否被后续抵消，需要更高的候选召回。</p>
  </div>
</section>

<section>
  <p class="eyebrow">§5</p>
  <h2>agent 之间的内容可替代性</h2>
  <div class="card">
    <table><thead><tr><th></th><th class="n" colspan="2">第 1 轮</th>
    <th class="n" colspan="2">第 2 轮</th><th class="n" colspan="2">第 3 轮</th></tr>
    <tr><th></th><th class="n">两两</th><th class="n">联合</th><th class="n">两两</th>
    <th class="n">联合</th><th class="n">两两</th><th class="n">联合</th></tr></thead>
    <tbody>{q5}</tbody></table>
    <p class="note">没有任何一场、任何一轮出现过一个 agent 的内容被另一个完全包含。
    其他所有 agent 合起来也只等价覆盖了某个 agent 输出的 <b>6–12%</b>。
    "大团队逐渐塌缩成一个主要表达者加若干重复者"在这批数据里没有发生。</p>
    <p class="note">第 1 轮分割来源 · 通用角色的 23.4% 是唯一的例外，而且值得单看：
    此时 <b>A 只读过 paper-b、B 只读过 paper-c、C 什么来源都没有</b>，三人没有任何
    共同文献，也还没看到彼此。持不同论文的 A 和 B 之间仍有 18.3% 和 24.8% 的
    有向等价重合。三个席位全部升高，不是没有来源的 C 单独拉起来的。
    换成专家角色，同一个 A↔B 掉到 8.2% 和 10.6%。<b>没有共同来源时，通用角色会用
    同一套泛泛的科研提案套话去填空白；角色分化把这层套话压掉了。</b></p>
  </div>
</section>

<section>
  <p class="eyebrow">§5 续</p>
  <h2>persona 只在一个地方起作用</h2>
  <div class="card">
    <table><thead><tr><th></th><th class="n">通用角色</th><th class="n">专家角色</th>
    <th class="n">差</th><th class="n">置换检验</th></tr></thead><tbody>
    <tr><th>全量来源 · 第 1 轮</th><td class="n">7.0%</td><td class="n">8.5%</td>
      <td class="n">+1.6pp</td><td class="n">p = 0.505</td></tr>
    <tr><th>全量来源 · 第 2 轮</th><td class="n">6.4%</td><td class="n">7.8%</td>
      <td class="n">+1.4pp</td><td class="n">p = 0.380</td></tr>
    <tr><th>全量来源 · 第 3 轮</th><td class="n">11.5%</td><td class="n">11.7%</td>
      <td class="n">+0.2pp</td><td class="n">p = 0.922</td></tr>
    <tr class="sep"><th colspan="5"></th></tr>
    <tr><th>分割来源 · 第 1 轮</th><td class="n">23.4%</td><td class="n">11.0%</td>
      <td class="n hi">−12.4pp</td><td class="n hi">p = 0.018</td></tr>
    <tr><th>分割来源 · 第 2 轮</th><td class="n">9.3%</td><td class="n">7.5%</td>
      <td class="n">−1.8pp</td><td class="n">p = 0.431</td></tr>
    <tr><th>分割来源 · 第 3 轮</th><td class="n">9.5%</td><td class="n">10.2%</td>
      <td class="n">+0.6pp</td><td class="n">p = 0.819</td></tr>
    </tbody></table>
    <p class="note">六个比较里只有一个显著：<b>分割来源的第 1 轮</b>。
    也就是说，角色分化能拉开内容差异的时刻，恰好是 agent 之间既没有共同来源、
    又还没有共同历史的那一刻。一轮交换之后效应就没了——共享历史盖过了角色设定。
    这是 persona × topology 的交互，不是 persona 的主效应。</p>
    <p class="note">轮次方向上，"第 2 轮降、第 3 轮升"只有全量通用一格通过检验
    （+5.1pp，p&lt;0.001），其余三格 p 在 0.14–0.93。每人每轮的命题数在 36–42 之间
    基本不变，所以那个升不是因为第 3 轮说得少。</p>
  </div>
</section>

<section>
  <p class="eyebrow">§6</p>
  <h2>继续生成买到了什么</h2>
  <div class="card">
    <table><thead><tr><th></th><th>相对于实际可见历史的分解</th>
    <th class="n">命题数</th></tr></thead><tbody>{q6}</tbody></table>
    <div class="key" style="margin-top:12px">
      <span><i style="background:var(--eq)"></i>等价重复</span>
      <span><i style="background:var(--weak)"></i>已有内容的较弱表述</span>
      <span><i style="background:var(--ref)"></i>细化候选</span>
      <span><i style="background:var(--un)"></i>无关于历史</span>
      <span><i style="background:var(--no)"></i>未匹配产出</span>
    </div>
    <p class="note">第 1 轮没有历史，按定义全部未匹配，已从表中略去。
    从第 2 轮到第 3 轮，<b>等价重复几乎翻倍</b>（全量通用 9.8%→18.9%，全量专家 12.3%→22.2%），
    与历史无关的产出同时从 38–44% 掉到 22–26%。轮次预算越往后，买到的越多是复述。
    这是离线的预算读数，不能据此断言提前停止不损任务质量。</p>
  </div>
</section>

<footer>
  由 experiments/analyze_offline_flow.py 与 experiments/test_offline_flow.py 生成。
  置换检验以「场」为独立单位（每条件 10 场），20000 次重排，双侧。
  没有干预，以上全部是表达变化与可见性相容性的描述，不是因果结论。
</footer>
</main>
"""

out = ROOT / "findings" / "2026-09-07-offline-flow.html"
out.write_text('<!doctype html><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width,initial-scale=1">'
               + HTML)
print(f"写入 {out}  ({out.stat().st_size / 1024:.0f} KB)")
