"""Formal definitions for every quantity the mentor report tabulates.

The report was readable but not self-contained: a column called "每千组合成边"
or "向外关联的目标事实" tells a reader neither what is counted nor what it is
divided by, and two measures that both look like "how much was picked up" can
have different denominators. Every row below gives the numerator, the
denominator, the unit that is averaged, and the script that computes it, so a
number in a table can be traced to an expression.

Where a measure comes from a script this module cannot restate exactly, the
definition points at the file and the predicate rather than paraphrasing it.
Guessing a definition would be worse than naming its source.
"""

from __future__ import annotations

PRIM = [
    ("mention", "一次发言里出现的一条原子命题，带出处 (agent, round, channel)。"
                "同一条内容在不同发言里出现算不同的 mention。", "—"),
    ("fact", "一组被判为两两等价的 mention 经并查集聚成的簇，取一个代表文本。"
             "「不同事实数」即簇数，不是 mention 数。", "match_traces.py"),
    ("已评分对 (scored pair)",
     "被 blocker（bge 余弦 ≥0.62，每个 mention 取 top-12）检索为候选、"
     "并真正送进 NLI 判过的 mention 对。IDRBench 上约 4.9% 的可能对被评分过，"
     "所有比率都只描述这一部分。<b>丢弃的部分不是无界的</b>：关系率随余弦单调下降"
     "（0.95+ 为 85.5%，0.85–0.90 为 36.1%，贴近截断线的 0.60–0.65 只有 5.7%），"
     "所以被丢弃的对判为有关系的概率 ≲6% 并继续下降。",
     "match_traces.py --match"),
    ("top-k 截断（已知缺口）",
     "58.9% 的 mention 用满了 top-12。它们保留的最弱候选余弦中位数是 0.692，"
     "所以多数截断发生在关系率约 8% 的低相似区，影响有限。"
     "但其中 206 个（2.4%）连最弱保留候选都 ≥0.80，而 0.80–0.85 这一档的关系率是 "
     "23.8%——<b>这批是可定位的真实漏检</b>，提高这些 mention 的 k 即可修复。",
     "match_traces.py --top-k"),
    ("f(a,b)",
     "判官对有序对给出的蕴含判定：margin = log P(YES) − log P(NO)，"
     "margin ≥ t 记为 a ⊨ b。t 是<b>单一共享阈值</b>（当前 5.28），"
     "两个方向用同一个 t。", "match_traces.py --calibrate"),
    ("四类关系",
     "等价 = f(a,b) ∧ f(b,a)；A⊨B = f(a,b) ∧ ¬f(b,a)；"
     "B⊨A = ¬f(a,b) ∧ f(b,a)；无关 = 两个方向都不成立。"
     "方向<b>不通过等价簇做传递闭包</b>：只读实际评过分的那一对。", "match_traces.py"),
    ("投递 / 可见",
     "delivery 里 peer_turns 是<b>本轮送达</b>的发言，"
     "visible_peer_turns / visible_self_turns 是<b>累积可见</b>的历史。"
     "Perspectrum 是 peer-only 记忆（不重看自己），IDRBench 是 cumulative。"
     "一条边成立要求源发言当时确实对接收方可见。", "run_debate.py"),
    ("弱化 / 细化（相对时间）",
     "对时间上在后的命题 q 与在先且可见的 p：q 弱化 = p ⊨ q 且 ¬(q ⊨ p)；"
     "q 细化 = q ⊨ p 且 ¬(p ⊨ q)。语义强弱与时间先后是两个维度，"
     "只有叠加可见性才叫「后述弱化」。", "analyze_offline_flow.py"),
]

MEAS = [
    ("不同事实数 / 场", "一场辩论里全部 agent 输出去重后的 fact 簇数",
     "—（计数）", "场（claim / case）", "analyze_perspectrum_topology.py"),
    ("每千组合成边",
     "分子：(p, q) 中被判为四类关系之一的对数，p 是实际投递给接收者的命题、"
     "q 是该接收者本轮输出的命题",
     "分母：同样范围内的全部 (p, q) 组合数 × 1/1000",
     "场", "analyze_perspectrum_topology.py"),
    ("等价 / 弱化 / 细化（格子表）",
     "分子：上述 (p, q) 中该类关系的对数",
     "分母：上述被判为四类之一的全部对数（即已评分且落在投递范围内的）",
     "场", "analyze_perspectrum_topology.py"),
    ("细化 − 弱化", "两个上述比例之差，正值表示后述整体更强", "—（差值）",
     "场", "analyze_perspectrum_topology.py"),
    ("同轮等价重合",
     "分子：某 agent 本轮输出里、存在等价对应物出现在<b>同轮其他 agent</b>输出中的事实数",
     "分母：该 agent 本轮输出的事实数", "先在 (agent, round) 上取比例再对场取均值",
     "analyze_perspectrum_topology.py"),
    ("向外关联的目标事实",
     "分子：从该 agent 早前发言出发、连到<b>其他</b> agent 第 r 轮输出的"
     "peer_screened 边所覆盖的<b>去重目标事实</b>数",
     "—（计数，非比例）", "(case, round∈{2,3}) 取均值",
     "analyze_network_balance.py / analyze_discipline_network.py"),
    ("向内关联的目标事实", "同上，方向相反：其他 agent 指向该 agent 的输出",
     "—（计数）", "同上", "analyze_network_balance.py"),
    ("向外 − 向内", "两者之差，正值表示该座位在这张跨轮关系图上是净供应端",
     "—（差值）", "同上", "analyze_network_balance.py"),
    ("专业化指数",
     "分子：本侧主题命题占比 − 对侧主题占比，按座位的名义归属取符号"
     "（A 属 paper-b 侧，B 属 paper-c 侧，C 综合席不计）",
     "分母：该 turn 有标签的命题数", "先在 turn 上算再对场取均值",
     "test_topic_specialisation.py"),
    ("新提案占比", "归属轴取值为 P 的命题占比；P 表示不归属于任一源论文的新提出内容",
     "分母：该范围内有标签的命题数", "场", "analyze_topic_flow.py"),
    ("被后续表达采纳",
     "分子：实际可见的命题中、后续输出里存在任一被判定关系与之对应的条数",
     "分母：全部实际可见的命题数（右列改用已评分的那部分作分母，"
     "两列之差即候选召回带来的不确定范围）", "场",
     "mentor_report_rounds.py"),
    ("置换检验",
     "以<b>场</b>为独立单位重排条件标签 20000 次，双侧，"
     "统计量是两组场均值之差。边数不作独立样本。",
     "—", "—", "test_offline_flow.py / analyze_perspectrum_topology.py"),
    ("事实 / 千 token", "该场输出的不同事实数 ÷ 输出 token 数 × 1000",
     "分母：该条件下的输出 token 总数", "场", "flow-profile.json · facts_per_k"),
    ("有同伴输入的后续 turn 比例",
     "分子：R2/R3 中实际收到过同伴发言的 turn 数", "分母：全部 R2/R3 turn 数",
     "场", "flow-profile.json · reception"),
    ("后续输出中 peer-adopted 份额",
     "分子：与可见同伴同簇、且此前未由自己说过的输出事实数",
     "分母：有同伴输入的后续输出事实数", "场",
     "flow-profile.json · adopted_share_recv（SAME/DIFF 管线）"),
    ("接触→R2 复述（uptake）",
     "分子：R1 被投递出去、且在 R2 被接收者复述的事实数",
     "分母：R1 被投递出去的事实数", "场",
     "rq-extensions.json · round2_transmission_funnel.uptake"),
    ("R2 复述→R3 保留",
     "分子：上一步复述过、且在 R3 仍出现的事实数", "分母：R2 复述的事实数",
     "场", "rq-extensions.json · post_adoption_retention"),
    ("接触→R3 保留（端到端）", "上面两步的乘积口径：R1 投递且 R3 仍在",
     "分母：R1 被投递出去的事实数", "场", "rq-extensions.json · end_to_end"),
    ("R1→R2 重合 Jaccard",
     "分子：两个 agent 该轮输出事实集合的交集大小",
     "分母：并集大小；对全部 agent 两两取均值", "场（此处 n=8，仅 1 个病人）",
     "2026-09-03-clinicalbench-pilot-metrics.json · roundN_agent_jaccard"),
    ("R1 事实 R2 保留",
     "分子：R1 出现且 R2 仍出现的事实数", "分母：R1 的事实数", "场",
     "clinicalbench-pilot-metrics.json · round1_retention"),
    ("源材料独占事实跨界复述 / 场",
     "分子：只出现在某一 agent 私有材料中的事实、被<b>没有该材料</b>的 agent 复述的条数",
     "—（计数）", "场", "clinicalbench-pilot-metrics.json · source_exclusive_uptake_facts"),
    ("无后续等价传播的事实比例",
     "分子：首次出现后、再没有任何其他 agent 给出等价表述的事实数",
     "分母：确实有机会被别人看到的事实数（排除末轮出生的截尾）", "条件（合并 10 场）",
     "idrbench-graph/summary.json · equivalence_dead_rate"),
    ("加入单向关系后仍无后续关联",
     "同上，但把单向蕴含也算作「有后续关联」", "同上", "条件",
     "idrbench-graph/summary.json · typed_dead_rate"),
    ("仅在有传播者中的不同接收者数",
     "在至少被一个人接住的事实上，平均有多少个<b>不同</b>接收者",
     "—（计数）", "条件", "idrbench-graph/summary.json · typed_spread_fanout"),
    ("全量已评分 mention 对",
     "40 场合并后各类关系的原始条数；<b>不是</b>全部可能对，只是被 blocker 检索到的那部分",
     "—（计数）", "全语料", "idrbench-entailment/summary.json · counts"),
    ("支持类占比：输入→输出",
     "分子：立场标注为 SUPPORT 的命题数，左侧在<b>可见输入</b>上算、右侧在<b>本轮输出</b>上算",
     "分母：对应范围内有立场标注的命题数", "turn（918 个有效 turn，非独立样本）",
     "context-output.json · by[persona].SUPPORT"),
    ("削弱类占比：输入→输出", "同上，立场标注为 UNDERMINE", "同上", "同上",
     "context-output.json · by[persona].UNDERMINE"),
    ("命题数 / turn", "该 (条件, 轮次) 下每个 turn 的原子命题数", "—（计数）",
     "先在场内合计再跨场平均", "mentor_report_rounds.py · analysis.json turns.n"),
    ("B 侧主题 / C 侧主题 / 两篇共有 / 通用",
     "主题轴取值 B / C / X / G 的命题占比。主题轴说命题<b>谈的是哪一侧的内容</b>，"
     "与归属轴（陈述某篇论文 vs 新提案）相互独立",
     "分母：该范围内有标签的命题数", "场",
     "label_fact_discipline.py · topic ∈ {B,C,X,G,U}"),
    ("peer-adopted 份额（旧表）",
     "与可见同伴同簇、且此前未由自己说过的输出事实占比。"
     "出自 SAME/DIFF 管线的等价簇，与上面四类关系不是同一种测量。",
     "分母：有同伴输入的后续输出", "场", "flow-profile.json（已折叠的旧表）"),
]


def build(tbl, card, note, section, pct):
    prim = tbl(["术语", "定义", "出处"], [[a, b, f"<code>{c}</code>"] for a, b, c in PRIM])
    meas = tbl(["量", "分子", "分母", "聚合单位", "计算脚本"],
               [[a, b, c, d, f"<code>{e}</code>"] for a, b, c, d, e in MEAS])
    return section(
        "definitions", "量的形式定义",
        "<p>本节把报告里出现的量写成可追溯的表达式。看表时如果不确定某一列的"
        "分母是什么，回到这里；<b>表头上带虚下划线和 ⓘ 的列名，悬停可看定义，按住 Alt 点击可跳到对应行</b>（普通点击是排序）。</p>"
        + card("<h3>基元</h3>" + prim
               + note("最容易误读的是「已评分对」：所有比率的全集都是它，"
                      "而不是全部可能的命题对。blocker 没检索到的对一律 unknown，"
                      "把它们当成「无关」会让稀疏条件看起来像丢了内容。"))
        + card("<h3>各量的分子与分母</h3>" + meas
               + note("不同量的分母不同，因此「关系密度」「重合率」「采纳率」"
                      "三类数字<b>不能互相比较</b>，只能在同一列内跨条件比较。"
                      "计算脚本一列给出唯一事实来源；表里的口径与脚本不一致时以脚本为准。")))
