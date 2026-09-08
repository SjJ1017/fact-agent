"""Round-resolved tables for the mentor report.

The report's tables aggregate over rounds, so a reader can compare personas
within a round but cannot follow one setting across rounds -- which is where
most of the movement is.  `analysis.json` already carries the round on every
row (2,160 turns, 2,880 interaction rows), so this only re-aggregates.

Every mean is taken over cases, not over rows: a case contributes one value,
so a debate that produced more propositions cannot pull a condition on its
own.  With the table chips, selecting one condition leaves its three rounds
adjacent, which is the comparison that was missing.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def T(zh, en):
    """Both languages go into the page; CSS shows one. See mentor_report_i18n.js."""
    return f'<span class="zh">{zh}</span><span class="en">{en}</span>'

ROOT = Path(__file__).resolve().parents[1]


def _mean(rows, num, den=None):
    """Case-level mean of num/den, or of num when den is None.

    A share that the upstream analysis left as null means the denominator was
    zero for that row, not zero uptake, so those rows are skipped rather than
    counted as 0 -- averaging them in would drag every sparse cell down.
    """
    per = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        v = r.get(num)
        if v is None:
            continue
        d = r.get(den) if den else 1
        if den and (d is None or d == 0):
            continue
        per[r["case_id"]][0] += v
        per[r["case_id"]][1] += d
    vals = [a / b for a, b in per.values() if b]
    return st.mean(vals) if vals else None


def _fmt(v, nd):
    return "—" if v is None else f"{v:.{nd}f}"


def build(tbl, card, note, section, pct):
    d = json.loads((ROOT / "findings/data/idrbench-discipline/analysis.json").read_text())
    conds = sorted({r["condition"] for r in d["turns"]})

    # --- 1. what each turn is made of, by round
    rows = []
    for c in conds:
        for rnd in (1, 2, 3):
            sel = [r for r in d["turns"] if r["condition"] == c and r["round"] == rnd and r["scope"] == "output"]
            if not sel:
                continue
            rows.append([c, rnd, _fmt(_mean(sel, "n"), 1),
                         pct(_mean(sel, "B", "n")), pct(_mean(sel, "C", "n")),
                         pct(_mean(sel, "X", "n")), pct(_mean(sel, "G", "n")),
                         pct(_mean(sel, "proposal_share"))])
    t1 = tbl([T("条件", "Condition"), T("轮次", "Round"), T("命题数 / turn", "Propositions per turn"), T("B 侧主题", "B-side topic"), T("C 侧主题", "C-side topic"),
              T("两篇共有", "Shared by both"), T("通用", "Generic"), T("新提案占比", "New-proposal share")], rows)

    # --- 2. directed relations between agents, by the round the input came from
    rows = []
    for c in conds:
        for rnd in (1, 2):
            sel = [r for r in d["interaction"]
                   if r["condition"] == c and r["source_round"] == rnd
                   and r["sender"] != r["receiver"] and r["topic"] == "*"]
            if not sel:
                continue
            rows.append([c, f"r{rnd}→r{rnd + 1}",
                         pct(_mean(sel, "scored")),
                         pct(_mean(sel, "equivalent")),
                         pct(_mean(sel, "weaken")),
                         pct(_mean(sel, "strengthen"))])
    t2 = tbl([T("条件", "Condition"), T("跨轮", "Across rounds"), T("输出有候选评分", "Output had a scored candidate"), T("等价覆盖", "Equivalent cover"), T("弱化覆盖", "Weakened cover"), T("细化覆盖", "Sharpened cover")], rows)

    # --- 3. uptake of what was actually visible, by round
    rows = []
    for c in conds:
        for rnd in (2, 3):
            sel = [r for r in d["uptake"] if r["condition"] == c and r["round"] == rnd]
            if not sel:
                continue
            rows.append([c, rnd, _fmt(_mean(sel, "n"), 0),
                         pct(_mean(sel, "used_n", "n")),
                         pct(_mean(sel, "used_n", "scored_n"))])
    t3 = tbl([T("条件", "Condition"), T("轮次", "Round"), T("可见命题 / 场", "Visible propositions per run"), T("被后续表达采纳", "Taken up later"), T("仅在已评分中", "Among scored only")], rows)

    return section(
        "by-round", T("按轮次展开：同一设置跨轮怎么变", "By round: how one setting moves across rounds"),
        "<p>下面三张表把原本聚合掉的轮次维度放回行里。用完整条件下拉框选中一个"
        "设置，它的各轮就会连在一起；点列头排序，点第三次回到原始顺序。</p>"
        + card("<h3>每个 turn 的内容构成</h3>" + t1
               + note("均值以「场」为单位，先在场内合计再跨场平均。"
                      "新提案占比来自归属轴的 P 值，不代表这些提案成立。"))
        + card("<h3>跨轮的有向关系（只统计不同 agent 之间）</h3>" + t2
               + note("这些列的分母都是目标输出事实数；一条输出可被多个关系类型覆盖，不能相加。"
                      "无候选评分的输出仍在分母中，不能据此认定其确实未被使用。"))
        + card("<h3>可见内容被后续表达接住的比例</h3>" + t3
               + note("左列以全部可见命题为分母，右列只用已评分的那部分；"
                      "右列是条件于已有候选评分的比率；两者不是召回率的上下界。")))
