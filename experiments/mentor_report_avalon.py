"""Part III of the report: the adversarial paradigm, measured on Avalon.

Everything the Avalon corpus produced goes here in full rather than summarised,
because the point of the paradigm framing is that these results are not a
separate study -- they are what the same instrument reads when objectives stop
being shared.

Order follows what the measurement needs rather than what the game does. The
uptake digraph comes first because it is the only result that requires nothing
but the graph; the truth results come next because they need ground truth on
top of the graph; the profile results come last because they need both plus a
classifier.

One correction is stated rather than quietly dropped. An earlier version of the
truth-flow result reported sixteen false-to-true edges and no true-to-false, and
read that as the system repairing falsehoods. It does not survive a parser that
refuses conditionals and disjunctions: "If Player 3 is Good then Player 1 is
Evil" is not an assertion that Player 1 is Evil. Seventeen percent of the
matches were of that kind.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["Merlin", "Servant", "Evil"]
POS = {"Merlin": (150, 78), "Servant": (400, 78), "Evil": (275, 250)}
COL = {"Merlin": "mrl", "Servant": "srv", "Evil": "evl"}


def T(zh: str, en: str) -> str:
    return f'<span class="zh">{zh}</span><span class="en">{en}</span>'


def digraph(m: dict) -> str:
    lo = min(v["mean"] for v in m.values())
    hi = max(v["mean"] for v in m.values())
    w = lambda v: 1.2 + 7.0 * (v - lo) / (hi - lo)
    p = ['<svg viewBox="0 0 560 336" role="img" '
         'aria-label="Avalon 承接有向图，边宽为每千组合的承接条数"><defs>']
    for c in set(COL.values()):
        p.append(f'<marker id="av-{c}" viewBox="0 0 10 10" refX="9" refY="5" '
                 f'markerWidth="6" markerHeight="6" orient="auto">'
                 f'<path d="M0 0 L10 5 L0 10" fill="var(--{c})"/></marker>')
    p.append("</defs>")
    for src in ORDER:
        for rcv in ORDER:
            key = f"{src}->{rcv}"
            if key not in m:
                continue
            v = m[key]["mean"]
            x1, y1 = POS[src]
            x2, y2 = POS[rcv]
            c = COL[src]
            if src == rcv:
                p.append(f'<path d="M{x1-26},{y1+18} a26,22 0 1,0 52,0" fill="none" '
                         f'stroke="var(--{c})" stroke-width="{w(v):.1f}" opacity=".62" '
                         f'marker-end="url(#av-{c})"/>')
                p.append(f'<text class="ev" x="{x1}" y="{y1+62}" '
                         f'text-anchor="middle">{v:.1f}</text>')
                continue
            dx, dy = x2 - x1, y2 - y1
            L = (dx * dx + dy * dy) ** .5
            ox, oy = -dy / L * 15, dx / L * 15
            sx, sy = x1 + dx * .21 + ox, y1 + dy * .21 + oy
            ex, ey = x1 + dx * .79 + ox, y1 + dy * .79 + oy
            p.append(f'<line x1="{sx:.0f}" y1="{sy:.0f}" x2="{ex:.0f}" y2="{ey:.0f}" '
                     f'stroke="var(--{c})" stroke-width="{w(v):.1f}" opacity=".62" '
                     f'marker-end="url(#av-{c})"/>')
            p.append(f'<text class="ev" x="{(sx+ex)/2+ox*.6:.0f}" '
                     f'y="{(sy+ey)/2+oy*.6:.0f}" text-anchor="middle">{v:.1f}</text>')
    for r in ORDER:
        x, y = POS[r]
        p.append(f'<circle cx="{x}" cy="{y}" r="31" fill="var(--panel)" '
                 f'stroke="var(--{COL[r]})" stroke-width="2.4"/>')
        p.append(f'<text class="nd" x="{x}" y="{y+5}" text-anchor="middle">{r}</text>')
    p.append('<text class="cap" x="14" y="322">'
             '边宽 = 每千（先说 × 后说）组合的承接条数 · '
             'width = uptake per 1k speak-then-speak combinations</text></svg>')
    return "".join(p)


def build(tbl, card, note, section, pct):
    ab = json.loads((ROOT / "findings/data/avalon-absorption.json").read_text())

    corpus = tbl(
        [T("项", "Item"), T("值", "Value")],
        [[T("对局", "Games"), T("10 局五人局，deepseek-v4-pro，thinking 关闭",
                               "ten five-player games, deepseek-v4-pro, thinking off")],
         [T("角色", "Roles"), "Merlin · Servant ×2 · Minion · Assassin"],
         [T("命题", "Propositions"),
          T("4,961 条（公开 3,057 · 待揭示 1,646 · 秘密 258）",
            "4,961 (public 3,057 · private-until-reveal 1,646 · secret 258)")],
         [T("匹配", "Matching"),
          T("Qwen3-14B，阈值 5.28，top-k 20，65,262 条已评分关系",
            "Qwen3-14B, threshold 5.28, top-k 20, 65,262 scored relations")],
         [T("结局", "Outcomes"),
          T("邪恶方 8 胜；其中 7 局是好人过三关后刺客认出梅林",
            "Evil won eight; seven of those were Good completing three quests "
            "and the Assassin then finding Merlin")]])

    absorb_rows = tbl(
        [T("承接者 ← 发送者", "Taker ← speaker"), T("每千组合", "Per 1k"),
         T("95% CI", "95% CI"), "n", T("接收方是否知道发送方阵营", "Receiver knows alignment")],
        [[T(f"{r} ← {s}", f"{r} ← {s}"), f"{ab[f'{s}->{r}']['mean']:.1f}",
          f"[{ab[f'{s}->{r}']['lo']:.1f}, {ab[f'{s}->{r}']['hi']:.1f}]",
          str(ab[f"{s}->{r}"]["n"]),
          T("知道", "yes") if r in ("Merlin", "Evil") else T("不知道", "no")]
         for r, s in [("Merlin", "Servant"), ("Servant", "Merlin"),
                      ("Servant", "Servant"), ("Merlin", "Evil"),
                      ("Evil", "Servant"), ("Servant", "Evil"),
                      ("Evil", "Merlin"), ("Evil", "Evil")]
         if f"{s}->{r}" in ab])

    know = tbl([T("边的类型", "Edge type"), T("每千组合", "Per 1k"),
                T("95% CI", "95% CI"), "n", T("置换检验", "Permutation")],
               [[T("知晓边（接收方是梅林或坏人）",
                   "informed (receiver is Merlin or Evil)"), "4.6", "[4.0, 5.3]", "120",
                 "p = 0.067"],
                [T("无知边（接收方是侍从）", "blind (receiver is a Servant)"),
                 "5.5", "[4.9, 6.2]", "80", ""],
                [T("知晓边内：知道确切角色", "informed: exact role known"),
                 "4.9", "[3.7, 6.2]", "40", "p = 0.576"],
                [T("知晓边内：只知阵营", "informed: alignment only"),
                 "4.5", "[3.8, 5.2]", "80", ""]])

    truth = tbl([T("种子命题", "Seed proposition"), T("跨 agent 扇出", "Fan-out"),
                 T("扇出 ≥ 1 的比例", "Share reaching anyone"), "n",
                 T("置换检验", "Permutation")],
                [[T("真", "true"), "0.61", "40.6%", "69", "p = 0.010"],
                 [T("假", "false"), "0.06", "5.6%", "18", ""]])

    channel = tbl([T("真实身份", "True role"),
                   T("公开 ↔ 私有 等价率", "Public-to-private equivalence"),
                   T("场数", "Runs")],
                  [["Servant", "2.9%", "20"], ["Merlin", "1.8%", "10"],
                   ["Assassin", "1.8%", "10"], ["Minion", "0.8%", "9"]])

    ident = tbl([T("轮次", "Quest"), T("身份话题占比", "Share of talk"),
                 T("推测", "Speculation"), T("断言", "Assertion"),
                 T("信念", "Belief"), T("准确率", "Accuracy"),
                 T("接着更早的说", "Building on an earlier claim")],
                [["1", "0.3%", "0.0%", "100.0%", "0.0%", "100.0%", "0.0%"],
                 ["2", "3.9%", "15.8%", "73.7%", "5.3%", "78.9%", "0.0%"],
                 ["3", "5.6%", "17.4%", "43.5%", "36.2%", "73.9%", "20.3%"],
                 ["4", "4.8%", "16.7%", "42.9%", "40.5%", "81.0%", "54.8%"]])

    assassin = tbl([T("刺客理由回响的公开命题，按作者身份",
                      "Public propositions the Assassin's rationale echoes, by author"),
                    T("条数", "Count"), T("占比", "Share")],
                   [["Servant", "21", "51.2%"], ["Minion", "8", "19.5%"],
                    ["Merlin", "8", "19.5%"], ["Assassin", "4", "9.8%"],
                    [T("<b>随机基线（五人中梅林一人）</b>",
                       "<b>chance (one Merlin in five)</b>"), "—", "<b>20.0%</b>"],
                    [T("<b>按局平均的梅林占比</b>", "<b>Merlin's share, averaged by run</b>"),
                     "—", "<b>12.1%</b>"]])

    return section(
        "adversarial", T("adversarial：Avalon 的十局", "adversarial: ten games of Avalon"),
        "<p>" + T(
            "目标真实对立，一方赢另一方必输；身份私有且歪曲身份是合法招法；"
            "但信道<b>完全公开</b>——五个人听到的一模一样。所以这里测到的每一个"
            "承接差异都不是「能不能听到」，而是<b>选择接住什么</b>。",
            "Objectives are genuinely opposed, one side wins only if the other "
            "loses; identity is private and misrepresenting it is a legal move; "
            "but the channel is <b>completely public</b> — all five hear the same "
            "thing. Every difference in uptake measured here is therefore not "
            "access but <b>choice</b>.") + "</p>"
        + card("<h3>" + T("语料", "The corpus") + "</h3>" + corpus
               + note(T(
                   "抽取用 minimax-m2.5，与其余语料相同，所以事实数可跨数据集比较；"
                   "但匹配用了 top-k 20（其余为 12），<b>关系比例不可跨数据集横比</b>——"
                   "多放进来的候选本就相似度较低，会压低等价率。",
                   "Extraction uses minimax-m2.5 as everywhere else, so fact "
                   "counts are comparable across corpora; matching used top-k 20 "
                   "where the others used 12, so <b>relation shares are not</b>. "
                   "The extra candidates are lower-similarity by construction and "
                   "depress the equivalence rate.")))
        + card("<h3>" + T("承接有向图", "The uptake digraph") + "</h3>"
               + f'<figure style="margin:0">{digraph(ab)}<figcaption>'
               + T("箭头由发送者指向后来承接它的座位；自环是同类角色内部（梅林只有一人，"
                   "故无自环）。每条边是 10 局、20–40 个有序座位对的均值。",
                   "Arrows run from a speaker to the seat that later restates "
                   "them; loops are within-role (Merlin is alone, so has none). "
                   "Each edge averages 20 to 40 ordered seat pairs over ten games.")
               + "</figcaption></figure>" + absorb_rows
               + note(T(
                   "<b>梅林承接坏人 5.8，坏人承接梅林 3.3。</b>梅林知道那两个是谁，"
                   "要引导好人就必须回应他们；不知情的侍从承接坏人只有 4.3，"
                   "多出的三成是知识的痕迹。<b>只看图不看内容，统计谁在不成比例地"
                   "回应哪两个座位，就能反推梅林是谁。</b>"
                   "<b>坏人内部 3.1 是全图最低</b>——互知身份、目标一致，却最不互相承接，"
                   "因为公开承接同伴会留下可读的呼应关系。"
                   "组内 5.8 对跨组 4.5（p=0.041），但这个组内优势完全由好人贡献。",
                   "<b>Merlin takes up Evil at 5.8; Evil takes up Merlin at "
                   "3.3.</b> Merlin knows who they are and must answer them to "
                   "steer the table; an uninformed Servant takes up Evil at 4.3, "
                   "and the extra third is the trace of knowledge. <b>An observer "
                   "reading only the graph can find Merlin by asking which seat "
                   "answers those two disproportionately.</b> <b>Evil-to-Evil is "
                   "the lowest edge at 3.1</b>: they know each other and share a "
                   "goal, yet take each other up least, because visible mutual "
                   "uptake is itself readable. Within-team runs 5.8 against 4.5 "
                   "across (p=0.041), but that advantage is entirely the Good "
                   "side's.")))
        + card("<h3>" + T("知晓边与无知边", "Informed and blind edges") + "</h3>" + know
               + note(T(
                   "梅林和坏人都知道全场每个人的阵营（各自知道己方，补集即对方），"
                   "侍从什么都不知道。<b>知晓边反而更低</b>，而且知识的精细程度不重要"
                   "（知道确切角色 vs 只知阵营，p=0.576）。原因是两股相反的力抵消了："
                   "梅林的知识<b>推高</b>承接（5.8，要引导），坏人的知识<b>压低</b>承接"
                   "（3.3 和 3.1，要隐藏）。决定行为的不是知不知道，"
                   "而是知道之后的目标要求你靠近还是远离。",
                   "Merlin and Evil both know everyone's alignment — each knows "
                   "its own side, and the complement gives the other — while a "
                   "Servant knows nothing. <b>Informed edges are the lower "
                   "ones</b>, and how precise the knowledge is does not matter "
                   "(exact role against alignment only, p=0.576). Two opposite "
                   "forces cancel: Merlin's knowledge <b>raises</b> uptake (5.8, "
                   "to steer) and Evil's <b>lowers</b> it (3.3 and 3.1, to hide). "
                   "What governs behaviour is not knowing but whether the goal "
                   "then requires approach or avoidance.")))
        + card("<h3>" + T("真假命题的传播", "How true and false propositions travel")
               + "</h3>" + truth
               + note(T(
                   "种子是公开发言中真值可判的身份主张，真值取自 trace 的角色分配。"
                   "<b>18 条假命题里只有 1 条被别的 agent 接住过</b>，真命题是 40.6%。"
                   "跨轮边全部是真→真，共 42 条，<b>没有一条假→真，也没有一条真→假</b>。",
                   "Seeds are public alignment claims whose truth the trace's own "
                   "role assignment settles. <b>Of eighteen false propositions "
                   "exactly one is ever taken up by another agent</b>, against "
                   "40.6% of true ones. All 42 cross-round edges run true to "
                   "true: <b>not one false-to-true, and not one true-to-false</b>."))
               + "<p>" + T(
                   "<b>更正。</b>本节早先的版本报告过「假→真 16 条、真→假 0 条，"
                   "系统在纠错」。那不成立：那 16 条来自把条件句与析取句读成无条件断言"
                   "（<code>If Player 3 is Good then Player 1 is Evil</code>、"
                   "<code>At least one of Player 3 or Player 4 is Evil</code>），"
                   "宽松解析的误抓率为 17%。用只接受完整无条件单主语从句的解析器重算后，"
                   "纠错链消失。<b>正确的说法是：系统不纠正假信息，假信息只是传不出去。</b>",
                   "<b>Correction.</b> An earlier version of this section reported "
                   "sixteen false-to-true edges and no true-to-false, and read it "
                   "as the system repairing falsehoods. It does not hold. Those "
                   "edges came from conditionals and disjunctions read as "
                   "unconditional assertions — <code>If Player 3 is Good then "
                   "Player 1 is Evil</code>, <code>At least one of Player 3 or "
                   "Player 4 is Evil</code> — a 17% false-catch rate. With a "
                   "parser that accepts only whole, unconditional, single-subject "
                   "clauses the repair chain disappears. <b>The system does not "
                   "correct false information; false information simply fails to "
                   "travel.</b>") + "</p>")
        + card("<h3>" + T("说的和想的：跨频道比对",
                          "Said and thought: comparing the channels") + "</h3>"
               + channel
               + note(T(
                   "同一 agent 的公开命题与其私有理由之间被判为等价的比例。"
                   "方向符合预期——坏人最不一致，Minion 只有 0.8%——但"
                   "<b>邪恶方对侍从只有 −1.6pp、p=0.069，不显著</b>，10 局的功效不够。"
                   "这个指标的价值在于<b>完全不需要自评标注</b>：现有文献靠说话者自己打"
                   "欺骗标记加旁人打怀疑分，这里只需要 trace 同时保存公开发言和私有理由。",
                   "The share of an agent's public-to-private pairs judged "
                   "equivalent. The direction is as expected — Evil is least "
                   "consistent, the Minion at 0.8% — but <b>Evil against Servant "
                   "is only −1.6pp at p=0.069</b>, and ten games do not have the "
                   "power. What makes the measure worth having is that it "
                   "<b>needs no self-report</b>: the existing literature asks "
                   "speakers to flag their own deception and peers to rate "
                   "suspicion, where this needs only a trace that keeps both "
                   "channels.")))
        + card("<h3>" + T("身份话题随轮次的变化",
                          "How identity talk changes across rounds") + "</h3>" + ident
               + note(T(
                   "四条曲线里只有最后一条单调且幅度大：<b>接着更早的身份主张说的比例"
                   "从 0 升到 54.8%</b>。同时断言让位给信念（73.7% → 42.9%，"
                   "信念 5.3% → 40.5%），而<b>准确率纹丝不动</b>（78.9% → 81.0%）。"
                   "讨论在收敛，却没有在变对——而这只有把图（谁接着谁）、"
                   "情态（断言还是信念）、真值（对不对）三层叠起来才看得出来。",
                   "Only the last column moves monotonically and far: <b>the "
                   "share of identity claims that build on an earlier one goes "
                   "from zero to 54.8%</b>. Assertion gives way to belief over "
                   "the same stretch (73.7% to 42.9%, belief 5.3% to 40.5%), "
                   "while <b>accuracy does not move</b> (78.9% to 81.0%). The "
                   "discussion converges without getting closer to the truth — "
                   "visible only with the graph, the modality and the ground "
                   "truth layered together.")))
        + card("<h3>" + T("刺客在读什么", "What the Assassin was reading")
               + "</h3>" + assassin
               + note(T(
                   "把刺客在刺杀时写下的私有理由，与此前所有公开命题做匹配。"
                   "<b>梅林占比 12.1%，低于 20% 的随机基线——他没有说漏嘴。</b>"
                   "而两局失手恰是梅林被回响最多的两局（各 3 条），7 局命中里有 5 局"
                   "梅林贡献 0 条。被回响的命题中 <b>58.5% 是 record</b>——"
                   "投票记录和任务成败这类可核验的客观痕迹，不是任何人的主张。"
                   "所以泄露梅林的不是他的言辞，是他的<b>行为记录</b>。",
                   "The Assassin's private rationale at the kill, matched against "
                   "every earlier public proposition. <b>Merlin's share is 12.1%, "
                   "below the 20% chance baseline — he does not talk himself into "
                   "it.</b> The two misses are the two runs where Merlin is echoed "
                   "most (three each), and five of the seven hits echo him not at "
                   "all. <b>58.5% of what is echoed is record</b> — vote tallies "
                   "and quest outcomes, verifiable traces rather than anyone's "
                   "claim. What gives Merlin away is not his speech but his "
                   "<b>behavioural record</b>.")))
        + card("<h3>" + T("这一节能支持什么，不能支持什么",
                          "What this section supports, and what it does not")
               + "</h3>"
               + tbl([T("结论", "Claim"), T("证据强度", "Strength")],
                     [[T("全公开信道下存在选择性接收",
                         "selective uptake exists on a fully public channel"),
                       T("稳：组内 5.8 对跨组 4.5，p=0.041",
                         "solid: 5.8 within against 4.5 across, p=0.041")],
                      [T("坏人回避互相承接", "Evil avoid taking each other up"),
                       T("稳：3.1 为全图最低，对其余 p=0.011",
                         "solid: lowest edge at 3.1, p=0.011 against the rest")],
                      [T("梅林的承接模式泄露身份",
                         "Merlin's uptake pattern leaks his identity"),
                       T("中：5.8 对侍从 4.3，方向清楚但未做检验",
                         "moderate: 5.8 against a Servant's 4.3, clear in "
                         "direction but not tested")],
                      [T("假命题传不出去", "false propositions fail to travel"),
                       T("稳：扇出 0.06 对 0.61，p=0.010",
                         "solid: fan-out 0.06 against 0.61, p=0.010")],
                      [T("坏人说想更不一致", "Evil are less self-consistent"),
                       T("弱：p=0.069，10 局功效不足",
                         "weak: p=0.069, ten games lack the power")],
                      [T("系统会纠正假信息",
                         "the system corrects false information"),
                       T("<b>已推翻</b>：源于解析错误，见上",
                         "<b>refuted</b>: an artifact of parsing, see above")]])
               + note(T(
                   "所有比例的分母是<b>已评分对</b>，不是全部可能对。"
                   "承接量目前<b>未排除接收方此前已表达过的内容</b>，"
                   "那部分应算再现而非首次采纳；这会让所有承接率一致偏高，"
                   "条件之间的比较不受影响，绝对值受影响。",
                   "Every share is over <b>scored pairs</b>, not all possible "
                   "pairs. Uptake counts do <b>not yet exclude content the "
                   "receiver had already stated</b>, which is re-expression "
                   "rather than first adoption; this inflates every uptake rate "
                   "equally, leaving comparisons between conditions intact and "
                   "absolute values overstated."))))
