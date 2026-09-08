"""Distinct facts against cumulative visible output tokens.

Rounds are a coarse clock: a panel that writes twice as much per turn reaches
round three having spent twice the budget, and comparing conditions by round
charges them differently. The token-clock sidecars record, for each fact, how
many output tokens were visible in the transcript when it first appeared, so
the same corpus can be read against what it actually cost.

Only visible output tokens count. The dossier is re-sent in every prompt, so
prompt tokens grow with panel size and round index for reasons that have
nothing to do with what was said.

The curve answers a question rounds cannot: whether a condition produces more
distinct content, or merely writes more.
"""

from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOCK = ROOT / "experiments" / "labels" / "token_clock"
PERSONAS = [("neutral", "neutral", "srv"), ("lenses", "lenses", "mrl"),
            ("stance", "stance", "evl")]
TOPO = ["full", "star", "chain"]


def T(zh: str, en: str) -> str:
    """Both languages go into the page; CSS shows one. See mentor_report_i18n.js."""
    return f'<span class="zh">{zh}</span><span class="en">{en}</span>'


def curves():
    """persona -> sorted (cumulative visible output tokens, distinct facts so far)."""
    per = defaultdict(list)
    for f in sorted(CLOCK.glob("perspectrum-*deepseek*.json")):
        d = json.loads(f.read_text())
        persona = next((p for p, _, _ in PERSONAS if f"-{p}" in f.name), None)
        if persona is None:
            continue
        pts = sorted(
            (v.get("cumulative_visible_output_tokens"), k)
            for k, v in d.get("facts", {}).items()
            if v.get("cumulative_visible_output_tokens") is not None)
        if len(pts) < 5:
            continue
        per[persona].append([(t, i + 1) for i, (t, _) in enumerate(pts)])
    return per


def interpolate(runs, grid):
    """Facts reached by each budget, over a run set that does not change.

    Averaging only the runs that got as far as each budget lets the set shift
    from point to point, and a cumulative count then appears to fall when the
    shorter runs drop out. Restricting to runs that reach the whole grid keeps
    one set throughout, so the curve is monotone by construction and the slopes
    are comparable across budgets.
    """
    span = [r for r in runs if r and r[-1][0] >= grid[-1]]
    if len(span) < 3:
        return [None] * len(grid), 0
    out = []
    for g in grid:
        out.append(st.mean(max([n for t, n in r if t <= g] or [0]) for r in span))
    return out, len(span)


def plot(per):
    W, H, L, R, TOP, B = 720, 330, 62, 176, 26, 52
    # cut the grid where a common run set still exists
    # every run reaches this far; none goes much past 1.8k, so beyond the
    # common floor the "curve" would only be runs dropping out one by one
    cap = int(min(r[-1][0] for runs in per.values() for r in runs) // 100 * 100)
    grid = list(range(0, cap + 1, 100))
    series = {p: interpolate(runs, grid)[0] for p, runs in per.items()}
    series = {p: v for p, v in series.items() if any(x is not None for x in v)}
    ymax = max(v for s in series.values() for v in s if v is not None) * 1.08
    sx = lambda t: L + t / max(grid[-1], 1) * (W - L - R)
    sy = lambda v: H - B - v / ymax * (H - B - TOP)
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" '
         'aria-label="不同事实数随累计可见输出 token 的增长，按 persona 分">']
    for v in range(0, int(ymax) + 1, 20):
        p.append(f'<line x1="{L}" y1="{sy(v):.0f}" x2="{W-R}" y2="{sy(v):.0f}" '
                 'stroke="currentColor" stroke-width="1" opacity=".1"/>')
        p.append(f'<text class="ax" x="{L-9}" y="{sy(v)+4:.0f}" '
                 f'text-anchor="end">{v}</text>')
    for t in range(0, grid[-1] + 1, 200):
        p.append(f'<text class="ax" x="{sx(t):.0f}" y="{H-B+18}" '
                 f'text-anchor="middle">{t:,}</text>')
    for name, label, col in PERSONAS:
        s = series.get(name)
        if not s:
            continue
        pts = [(sx(g), sy(v)) for g, v in zip(grid, s) if v is not None]
        d = "M" + " L".join(f"{x:.0f},{y:.0f}" for x, y in pts)
        p.append(f'<path d="{d}" fill="none" stroke="var(--{col})" '
                 'stroke-width="2.2" stroke-linejoin="round"/>')
        lx, ly = pts[-1]
        p.append(f'<circle cx="{lx:.0f}" cy="{ly:.0f}" r="3.4" fill="var(--{col})"/>')
        p.append(f'<text class="sl" x="{lx+9:.0f}" y="{ly+4:.0f}" '
                 f'fill="var(--{col})">{label}</text>')
    p.append(f'<text class="ax" x="{L}" y="{TOP-6}">'
             '不同事实数 · distinct facts</text>')
    p.append(f'<text class="cap" x="{L}" y="{H-10}">'
             '累计可见输出 token · cumulative visible output tokens</text>')
    return "".join(p) + "</svg>"


def build(tbl, card, note, section, pct):
    per = curves()
    cap = int(min(r[-1][0] for runs in per.values() for r in runs) // 100 * 100)
    grid = [g for g in (200, 400, 600, 800, 1000) if g <= cap]
    rows = []
    for name, label, _ in PERSONAS:
        runs = per.get(name, [])
        if not runs:
            continue
        vals, n = interpolate(runs, grid)
        if not n:
            continue
        rows.append([label] + [f"{v:.0f}" if v else "—" for v in vals] + [str(n)])
    table = tbl([T("persona", "persona")]
                + [T(f"{g:,} token", f"{g:,} tokens") for g in grid]
                + [T("场数", "Runs")], rows)

    return section(
        "token-clock",
        T("换成 token 计时：多说，还是说了更多不同的东西",
          "On a token clock: writing more, or saying more distinct things"),
        "<p>" + T(
            "轮次是粗糙的时钟。每轮写得多一倍的小组，走到第三轮时花掉的预算也多一倍，"
            "按轮次比较等于给两者记了不同的账。token-clock 记录了每条事实首次出现时"
            "transcript 里已有多少可见输出 token，所以同一批语料可以按<b>实际花费</b>"
            "重读一遍。只计可见输出 token——卷宗在每个 prompt 里重发，"
            "prompt token 随人数和轮次增长，与说了什么无关。",
            "Rounds are a coarse clock. A panel writing twice as much per turn "
            "arrives at round three having spent twice the budget, so comparing "
            "by round charges the two differently. The token-clock records, for "
            "each fact, how many visible output tokens the transcript held when "
            "it first appeared, letting the same corpus be read against what it "
            "<b>actually cost</b>. Only visible output tokens count: the dossier "
            "is re-sent every prompt, so prompt tokens grow with panel size and "
            "round index for reasons unrelated to what was said.") + "</p>"
        + card(f'<figure style="margin:0">{plot(per)}<figcaption>'
               + T("每条曲线是该 persona 下<b>走完整个预算区间的那些场次</b>的平均，"
                   "场次集合在曲线上不变，所以计数单调。表格里给了参与场数。",
                   "Each curve averages the runs under that persona that "
                   "<b>reach the whole budget range</b>, so the set does not "
                   "change along the curve and the counts stay monotone. The "
                   "table gives how many runs that is.")
               + "</figcaption></figure>" + table
               + note(T(
                   "曲线的<b>斜率</b>是每千 token 买到的新事实，读它比读终点高度有用："
                   "终点高只说明写得多。三条 persona 的曲线在共同可达的预算内"
                   "基本重合，说明差别不在<b>产出效率</b>上——这与按轮次得到的结论、"
                   "以及角色可分性接近基线（33.3%–39.8%）都一致。"
                   "<b>横轴止于 1,000</b>，因为这是全部 72 场都到达的预算——最短的一场"
                   "只有 1,095 个可见输出 token，最长的也只有 1,834。"
                   "越过这条线之后，曲线画的就只是场次逐个退出、剩下几场的均值在跳。",
                   "The <b>slope</b> is new facts bought per thousand tokens, and "
                   "reading it beats reading the endpoint: a high endpoint only "
                   "means more was written. Across the budget every run reaches, "
                   "the three personas nearly coincide, so they do not differ in "
                   "<b>yield</b> — agreeing with the round-based comparison and "
                   "with role separability sitting near its 33.3% baseline. "
                   "<b>The axis stops at 1,000</b>, the budget all 72 runs reach: "
                   "the shortest holds 1,095 visible output tokens and the "
                   "longest 1,834. Past that line a curve would be drawing runs "
                   "dropping out one by one."))))
