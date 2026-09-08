"""A horizontal funnel of the matching pipeline, drawn with the real corpus.

The report described the pipeline in four words per stage, which leaves the
one thing a reader needs to know invisible: how much is discarded where, and
by which of the two stages. Both the blocker and the NLI pass emit "unrelated",
and drawing them as a single sink is what makes them legible as two stages of
one classifier rather than a filter followed by a judge.

Objects (turns, mentions, facts) and pairs are different units, so they get
different lanes and are never put on one width scale. Band thickness inside
the pair lane is logarithmic -- linear thickness would make everything after
the blocker a hairline, since the first stage removes 96% of the mass.
"""

from __future__ import annotations

import math


def T(zh, en):
    """Both languages go into the page; CSS shows one. See mentor_report_i18n.js."""
    return f'<span class="zh">{zh}</span><span class="en">{en}</span>'

N = {"debates": 40, "turns": 354, "mentions": 14444, "split": 479,
     "facts": 11201, "possible": 2682292, "scored": 94843,
     "eq": 7566, "ab": 8273, "ba": 9773, "unrel_nli": 69231}
N["rejected"] = N["possible"] - N["scored"]
N["oneway"] = N["ab"] + N["ba"]
N["related"] = N["eq"] + N["oneway"]
N["unrel_all"] = N["rejected"] + N["unrel_nli"]


def thick(n: float) -> float:
    """Log thickness: the first stage removes 96% of the pairs, so a linear
    scale would leave every later band under one pixel."""
    lo, hi = 3.5, math.log10(N["possible"])
    return 7 + 47 * max(0.0, (math.log10(n) - lo)) / (hi - lo)


def fmt(n: int) -> str:
    return f"{n:,}"


def _ribbon(x1, t1, x2, t2, cy, cls=""):
    return (f'<path class="{cls}" d="M{x1},{cy-t1/2:.1f} L{x2},{cy-t2/2:.1f} '
            f'L{x2},{cy+t2/2:.1f} L{x1},{cy+t1/2:.1f} Z"/>')


def svg() -> str:
    CY = 168
    tp, ts, tr = thick(N["possible"]), thick(N["scored"]), thick(N["related"])
    teq, tow, tun = thick(N["eq"]), thick(N["oneway"]), thick(N["unrel_nli"])
    p = []
    a = p.append

    a('<svg viewBox="0 0 1120 430" role="img" '
      'aria-label="匹配管线的横向漏斗：268 万个可能的命题对，经 blocker 与 NLI 两级，'
      '剩下 2.6 万条有关系的边和 11201 个事实簇">')
    a('<defs><marker id="pl-a" viewBox="0 0 10 10" refX="9" refY="5" '
      'markerWidth="7" markerHeight="7" orient="auto">'
      '<path d="M0 0 L10 5 L0 10" fill="currentColor"/></marker></defs>')

    # ---- object lane -----------------------------------------------------
    a('<text class="pl-lane" x="20" y="26">对象</text>')
    for x, w, t1, t2 in ((92, 128, f'{N["debates"]} 场辩论', f'{N["turns"]} 个 turn'),
                         (300, 150, f'{fmt(N["mentions"])} mention',
                          f'其中 {N["split"]} 条来自拆分'),
                         (836, 150, f'{fmt(N["facts"])} fact',
                          'mention 合并 22.5%')):
        a(f'<rect class="pl-obj" x="{x}" y="12" width="{w}" height="46" rx="6"/>')
        a(f'<text class="pl-n" x="{x+w/2}" y="32" text-anchor="middle">{t1}</text>')
        a(f'<text class="pl-s" x="{x+w/2}" y="48" text-anchor="middle">{t2}</text>')
    a('<line x1="224" y1="35" x2="292" y2="35" stroke="currentColor" '
      'stroke-width="1.4" marker-end="url(#pl-a)"/>')
    a('<text class="pl-e" x="258" y="27" text-anchor="middle">抽取 + 原子化</text>')

    # ---- pair lane -------------------------------------------------------
    a('<text class="pl-lane" x="20" y="164">命题对</text>')
    a(f'<line x1="375" y1="62" x2="375" y2="{CY-tp/2-6:.0f}" stroke="currentColor" '
      'stroke-width="1.4" stroke-dasharray="3 3" marker-end="url(#pl-a)"/>')
    a('<text class="pl-e" x="384" y="105">两两配对</text>')

    a(f'<rect class="pl-band" x="92" y="{CY-tp/2:.1f}" width="188" '
      f'height="{tp:.1f}" rx="3"/>')
    a(f'<text class="pl-n" x="186" y="{CY+4}" text-anchor="middle">'
      f'{fmt(N["possible"])} 个可能对</text>')

    a(f'<rect class="pl-proc" x="300" y="{CY-38}" width="150" height="76" rx="6"/>')
    a(f'<text class="pl-t" x="375" y="{CY-16}" text-anchor="middle">第一级 · blocker</text>')
    a(f'<text class="pl-s" x="375" y="{CY+2}" text-anchor="middle">bge 余弦 ≥ 0.62</text>')
    a(f'<text class="pl-s" x="375" y="{CY+18}" text-anchor="middle">每 mention 取 top-12</text>')
    a(_ribbon(280, tp, 300, tp, CY, "pl-flow"))

    a(_ribbon(450, ts, 520, ts, CY, "pl-flow"))
    a(f'<rect class="pl-band pl-keep" x="520" y="{CY-ts/2:.1f}" width="150" '
      f'height="{ts:.1f}" rx="3"/>')
    a(f'<text class="pl-n" x="595" y="{CY+4}" text-anchor="middle">'
      f'{fmt(N["scored"])}</text>')
    a(f'<text class="pl-s" x="595" y="{CY+ts/2+15:.0f}" text-anchor="middle">'
      f'已评分对 · 占全部 {N["scored"]/N["possible"]:.1%}</text>')

    a(f'<rect class="pl-proc" x="700" y="{CY-38}" width="150" height="76" rx="6"/>')
    a(f'<text class="pl-t" x="775" y="{CY-16}" text-anchor="middle">第二级 · NLI</text>')
    a(f'<text class="pl-s" x="775" y="{CY+2}" text-anchor="middle">f(a,b) 与 f(b,a)</text>')
    a(f'<text class="pl-s" x="775" y="{CY+18}" text-anchor="middle">共享阈值 t = 5.28</text>')
    a(_ribbon(670, ts, 700, ts, CY, "pl-flow"))

    # ---- three outcomes --------------------------------------------------
    ys = [CY - 46, CY, CY + 52]
    outs = [("等价", N["eq"], teq, "pl-eq"),
            (T("单向蕴含 A⊨B / B⊨A", "one-way A⊨B / B⊨A"), N["oneway"], tow, "pl-ow"),
            ("无关", N["unrel_nli"], tun, "pl-un")]
    for (lab, n, t, cls), y in zip(outs, ys):
        a(f'<path class="pl-flow" d="M850,{CY:.0f} C880,{CY:.0f} 880,{y:.0f} '
          f'910,{y:.0f}" fill="none" stroke="currentColor" stroke-width="1.2" '
          'opacity=".35"/>')
        a(f'<rect class="pl-band {cls}" x="910" y="{y-t/2:.1f}" width="120" '
          f'height="{max(t,14):.1f}" rx="3"/>')
        a(f'<text class="pl-n" x="1040" y="{y+4:.0f}">{fmt(n)}</text>')
        a(f'<text class="pl-s" x="910" y="{y-max(t,14)/2-5:.0f}">{lab} · '
          f'{n/N["scored"]:.1%}</text>')

    # Route the clustering arrow above the outcome labels rather than through
    # them: a curve across this corner crossed both the 等价 label and its count.
    a(f'<path d="M1000,{ys[0]-max(teq,14)/2-18:.0f} L1000,40 L992,40" fill="none" '
      'stroke="currentColor" stroke-width="1.4" stroke-dasharray="4 3" '
      'marker-end="url(#pl-a)" opacity=".8"/>')
    a('<text class="pl-e" x="1010" y="44">只有等价边参与聚类</text>')

    # ---- the shared unrelated sink ---------------------------------------
    SY = 376
    a(f'<rect class="pl-sink" x="92" y="{SY-20}" width="938" height="42" rx="6"/>')
    a(f'<text class="pl-t" x="112" y="{SY-2}">这套仪器判为「无关」：{fmt(N["unrel_all"])}</text>')
    a(f'<text class="pl-s" x="112" y="{SY+15}">'
      f'占全部可能对 {N["unrel_all"]/N["possible"]:.2%}；两级共同给出，不是缺失数据</text>')
    a(f'<path d="M375,{CY+38} L375,{SY-22}" fill="none" stroke="currentColor" '
      'stroke-width="1.4" marker-end="url(#pl-a)" opacity=".75"/>')
    a(f'<text class="pl-e" x="384" y="{CY+72}">第一级否决 {fmt(N["rejected"])}</text>')
    a(f'<path d="M970,{ys[2]+14} L970,{SY-22}" fill="none" stroke="currentColor" '
      'stroke-width="1.4" marker-end="url(#pl-a)" opacity=".75"/>')
    a(f'<text class="pl-e" x="962" y="{SY-32}" text-anchor="end">'
      f'第二级判无关 {fmt(N["unrel_nli"])}</text>')
    a('</svg>')
    return "".join(p)


CSS = """<style>
.pl-obj{fill:none;stroke:currentColor;stroke-width:1.1;opacity:.55}
.pl-proc{fill:none;stroke:currentColor;stroke-width:1.4}
.pl-band{fill:currentColor;opacity:.16}
.pl-band.pl-keep{opacity:.3}
.pl-eq{fill:#1f7a6b;opacity:.75}
.pl-ow{fill:#3f5fa8;opacity:.7}
.pl-un{fill:currentColor;opacity:.22}
.pl-flow{fill:currentColor;opacity:.12}
.pl-sink{fill:none;stroke:currentColor;stroke-width:1.1;stroke-dasharray:5 4;opacity:.6}
.pl-n{font:500 13px "IBM Plex Mono",monospace;fill:currentColor}
.pl-t{font:500 12px "PingFang SC",system-ui,sans-serif;fill:currentColor}
.pl-s{font:400 11px "PingFang SC",system-ui,sans-serif;fill:currentColor;opacity:.68}
.pl-e{font:400 11px "PingFang SC",system-ui,sans-serif;fill:currentColor;opacity:.75}
.pl-lane{font:500 11px "IBM Plex Mono",monospace;fill:currentColor;opacity:.45;
 letter-spacing:.08em}
figure.pl{margin:0}
figure.pl figcaption{font-size:12.5px;color:var(--dim);margin-top:12px;max-width:80ch}
</style>"""


def build(tbl, card, note, section, pct):
    fig = (CSS + '<figure class="pl">' + svg()
           + '<figcaption>IDRBench 全部 40 场的实际流量。'
             '<b>对象</b>与<b>命题对</b>是两种单位，分两条泳道，不放在同一个宽度刻度上；'
             '命题对泳道的带宽按<b>对数</b>绘制——第一级就移走了 96.5% 的量，'
             '线性刻度会让后面每一段都细成一条线。'
             '两个「无关」出口汇入同一个方框，因为它们是同一个分类器的两级判决。'
             '</figcaption></figure>')
    return section(
        "pipeline-funnel", T("一条发言如何变成图上的一条边", "How one utterance becomes an edge"),
        fig
        + note("三个读法。<b>一、0.96%</b>：268 万个可能对里，最终只有 "
               f"{N['related']:,} 条带关系（等价 + 单向），"
               "所以「关系稀疏」是这个测量的常态而不是异常。"
               "<b>二、两级都在判无关</b>，第一级 258.7 万、第二级 6.9 万，"
               "合计 99.04%；把第一级的否决记成「未知」会把仪器自己的判决"
               "再当成一次不确定。"
               "<b>三、只有等价边参与聚类</b>——18,046 条单向边一条都不合并事实，"
               "它们只进方向分析，所以 fact 数不会因为单向关系变多而下降。"))
