"""Formal definitions for every quantity the mentor report tabulates.

The report was readable but not self-contained: a column called "每千组合成边"
or "向外关联的目标事实" tells a reader neither what is counted nor what it is
divided by, and three measures that all read as "how much was picked up" turned
out to have three different denominators. Every row below gives the numerator,
the denominator, the unit that is averaged, and the script or JSON field that
computes it, so a number in a table can be traced back to an expression.

Where a measure comes from a script this module cannot restate exactly, the row
names the field rather than paraphrasing it. A guessed definition would be
worse than a pointer.

Every string is carried in both languages. The reader is an English-speaking
supervisor, so the English is written as English rather than as a gloss on the
Chinese, and `L` switches between them.
"""

from __future__ import annotations


def T(zh: str, en: str) -> str:
    """Both languages go into the page; CSS shows one. See mentor_report_i18n.js."""
    return f'<span class="zh">{zh}</span><span class="en">{en}</span>'


# term, zh definition, en definition, source
PRIM = [
    ("mention",
     "一次发言里出现的一条原子命题，带出处 (agent, round, channel)。"
     "同一条内容在不同发言里出现算不同的 mention。",
     "One atomic proposition as it appears in one utterance, carrying its "
     "provenance (agent, round, channel). The same content said twice is two "
     "mentions.",
     "—"),
    ("fact",
     "一组被判为两两等价的 mention 经并查集聚成的簇，取一个代表文本。"
     "「不同事实数」即簇数。",
     "A cluster of mentions judged pairwise equivalent, union-found together "
     "and given one representative text. \"Distinct facts\" counts clusters.",
     "match_traces.py"),
    ("匹配管线的判决",
     "blocker 与 NLI 是<b>同一个分类器的两级</b>，不是两个独立步骤：第一级用 bge 余弦"
     "（≥0.62，每个 mention 取 top-12）廉价地否掉明显无关的对，第二级对剩下的做双向蕴含"
     "裁决。<b>被第一级否掉的对，这套仪器的判决是「无关」。</b>",
     "The blocker and the NLI pass are <b>two stages of one classifier</b>, not "
     "two independent steps: the first rejects the plainly unrelated cheaply "
     "using bge cosine (≥0.62, top-12 per mention), and the second adjudicates "
     "what survives with a bidirectional entailment call. <b>A pair the "
     "first stage rejects has been judged unrelated by the instrument.</b>",
     "match_traces.py --match"),
    ("已评分对 (scored pair)",
     "进入第二级、由 NLI 实际裁决过的 mention 对（IDRBench 上约占全部可能对的 4.9%）。"
     "报「等价占已评分对的百分之几」时全集是它；报「一条命题有没有被接住」时"
     "全集是全部对，第一级的否决计入无关。<b>两种分母各自成尺。</b>",
     "A mention pair that reached the second stage and was actually adjudicated "
     "by the NLI model — about 4.9% of all possible pairs on IDRBench. It is "
     "the universe when reporting \"equivalent as a share of scored pairs\"; "
     "the universe is <em>all</em> pairs, with first-stage rejections counted "
     "as unrelated, when asking whether a proposition was taken up at all. "
     "<b>Each is its own scale.</b>",
     "match_traces.py --match"),
    ("第一级的召回损失",
     "第一级会误否一部分真有关系的对，其规模可以从已存的 blocker 余弦读出。"
     "按已存的 blocker 余弦测算，关系率随余弦单调下降——0.95 以上 85.5%，"
     "0.85–0.90 为 36.1%，紧贴 0.62 截断线的 0.60–0.65 只有 5.7%——"
     "所以<b>被余弦否掉的对，判为有关系的概率 ≲6% 且继续下降</b>。",
     "The first stage wrongly rejects some genuinely related pairs, and the "
     "stored blocker cosines say how many. Relatedness falls monotonically — "
     "85.5% above 0.95, 36.1% in 0.85–0.90, 5.7% just above the 0.62 cut — so "
     "<b>a pair the cosine rejects would be judged related with probability "
     "under about 6%, and falling</b>.",
     "measured here from blocker_cosine"),
    ("top-k 截断（真实缺陷）",
     "名额用尽是预算耗尽，不是关于相关性的判断，所以这一类是伪影。"
     "58.9% 的 mention 用满 top-12；多数截断在余弦中位 0.692 处，落在关系率约 8% 的区间，"
     "影响有限。但其中 206 个（2.4%）连最弱保留候选都 ≥0.80，而该档关系率为 23.8%——"
     "这批是可定位的真实漏检，提高这些 mention 的 k 即可修复。",
     "Exhausting a budget is not a judgement about relatedness, so this "
     "class is an artifact. "
     "58.9% of mentions used the full top-12; most truncate at a median cosine "
     "of 0.692, inside the band where roughly 8% of pairs are related, so the "
     "loss is small. But 206 of them (2.4%) truncate with their weakest kept "
     "candidate at 0.80 or above, where a quarter of scored pairs turn out "
     "related. That is a locatable miss with a cheap fix: raise k for those "
     "mentions.",
     "match_traces.py --top-k"),
    ("f(a,b)",
     "判官对有序对给出的蕴含判定：margin = log P(YES) − log P(NO)，"
     "margin ≥ t 记为 a ⊨ b。t 是<b>单一共享阈值</b>（当前 5.28），"
     "两个方向用同一个 t。",
     "The judge's entailment call on an ordered pair: margin = log P(YES) − "
     "log P(NO), and a ⊨ b when margin ≥ t. There is <b>one shared "
     "threshold</b> t (currently 5.28), used for both directions.",
     "match_traces.py --calibrate"),
    ("四类关系",
     "等价 = f(a,b) ∧ f(b,a)；A⊨B = f(a,b) ∧ ¬f(b,a)；"
     "B⊨A = ¬f(a,b) ∧ f(b,a)；无关 = 两个方向都不成立。"
     "方向<b>不通过等价簇做传递闭包</b>：只读实际评过分的那一对。",
     "Equivalent = f(a,b) ∧ f(b,a); A⊨B = f(a,b) ∧ ¬f(b,a); B⊨A = ¬f(a,b) ∧ "
     "f(b,a); unrelated = neither direction holds. Direction is <b>never "
     "propagated through an equivalence cluster</b>: only the pair actually "
     "scored is read.",
     "match_traces.py"),
    ("投递 / 可见",
     "delivery 里 peer_turns 是<b>本轮送达</b>的发言，"
     "visible_peer_turns / visible_self_turns 是<b>累积可见</b>的历史。"
     "Perspectrum 是 peer-only 记忆（不重看自己），IDRBench 是 cumulative。"
     "一条边成立要求源发言当时确实对接收方可见。",
     "In the delivery record, peer_turns are the utterances <b>delivered this "
     "round</b>; visible_peer_turns and visible_self_turns are the "
     "<b>cumulative</b> history. Perspectrum runs peer-only memory (an agent "
     "does not re-read itself); IDRBench is cumulative. An edge requires that "
     "the source utterance was genuinely visible to the receiver at the time.",
     "run_debate.py"),
    ("弱化 / 细化（相对时间）",
     "对时间上在后的命题 q 与在先且可见的 p：q 弱化 = p ⊨ q 且 ¬(q ⊨ p)；"
     "q 细化 = q ⊨ p 且 ¬(p ⊨ q)。语义强弱与时间先后是两个维度，"
     "只有叠加可见性才叫「后述弱化」。",
     "For a later proposition q and an earlier, visible p: q is weakened when "
     "p ⊨ q and not q ⊨ p; q is sharpened when q ⊨ p and not p ⊨ q. Semantic "
     "strength and temporal order are two separate dimensions — only with "
     "visibility layered on top does this become \"weakened in the retelling\".",
     "analyze_offline_flow.py"),
]

# quantity, zh numerator, en numerator, zh denominator, en denominator, unit, script
MEAS = [
    ("不同事实数 / 场",
     "一场辩论里全部 agent 输出去重后的 fact 簇数",
     "Fact clusters in one run, deduplicated across all agents' output",
     "—（计数）", "— (a count)", "场 / run", "analyze_perspectrum_topology.py"),
    ("每千组合成边",
     "(p, q) 中被判为四类关系之一的对数，p 是实际投递给接收者的命题、"
     "q 是该接收者本轮输出的命题",
     "Pairs (p, q) assigned any of the four relations, where p was actually "
     "delivered to the receiver and q is that receiver's output this round",
     "同样范围内的全部 (p, q) 组合数 × 1/1000",
     "All (p, q) combinations in that same scope, per thousand",
     "场 / run", "analyze_perspectrum_topology.py"),
    ("等价 / 弱化 / 细化（格子表）",
     "上述 (p, q) 中该类关系的对数",
     "Pairs of that relation among the (p, q) above",
     "上述被判为四类之一的全部对数（即已评分且落在投递范围内的）",
     "All pairs above assigned one of the four relations — scored, and inside "
     "the delivery scope",
     "场 / run", "analyze_perspectrum_topology.py"),
    ("细化 − 弱化",
     "两个上述比例之差，正值表示后述整体更强",
     "The difference of those two shares; positive means later statements are "
     "stronger on balance",
     "—（差值）", "— (a difference)", "场 / run", "analyze_perspectrum_topology.py"),
    ("同轮等价重合",
     "某 agent 本轮输出里、存在等价对应物出现在<b>同轮其他 agent</b>输出中的事实数",
     "Facts in one agent's output this round that have an equivalent in "
     "<b>another agent's output the same round</b>",
     "该 agent 本轮输出的事实数",
     "That agent's facts this round",
     "先在 (agent, round) 上取比例再对场取均值 / share per (agent, round), then "
     "averaged over runs",
     "analyze_perspectrum_topology.py"),
    ("向外关联的目标事实",
     "从该 agent 早前发言出发、连到<b>其他</b> agent 第 r 轮输出的"
     "peer_screened 边所覆盖的<b>去重目标事实</b>数",
     "<b>Distinct target facts</b> covered by peer_screened edges running from "
     "this agent's earlier utterances into <b>another</b> agent's round-r output",
     "—（计数，非比例）", "— (a count, not a share)",
     "(case, round∈{2,3}) 取均值 / averaged over case and round",
     "analyze_network_balance.py"),
    ("向内关联的目标事实",
     "同上，方向相反：其他 agent 指向该 agent 的输出",
     "The same, reversed: other agents pointing into this agent's output",
     "—（计数）", "— (a count)", "同上 / as above", "analyze_network_balance.py"),
    ("向外 − 向内",
     "两者之差，正值表示该座位在这张跨轮关系图上是净供应端",
     "Their difference; positive means the seat is a net supplier on this "
     "cross-round relation graph",
     "—（差值）", "— (a difference)", "同上 / as above", "analyze_network_balance.py"),
    ("专业化指数",
     "本侧主题命题占比 − 对侧主题占比，按座位的名义归属取符号"
     "（A 属 paper-b 侧，B 属 paper-c 侧，C 综合席不计）",
     "Own-side topic share minus other-side share, signed by the seat's "
     "nominal allegiance (A to paper b, B to paper c; the synthesis seat C is "
     "excluded)",
     "该 turn 有标签的命题数", "Labelled propositions in that turn",
     "先在 turn 上算再对场取均值 / computed per turn, then averaged over runs",
     "test_topic_specialisation.py"),
    ("新提案占比",
     "归属轴取值为 P 的命题占比；P 表示不归属于任一源论文的新提出内容",
     "Share of propositions whose attribution axis reads P — newly proposed "
     "content belonging to neither source paper",
     "该范围内有标签的命题数", "Labelled propositions in that scope",
     "场 / run", "analyze_topic_flow.py"),
    ("被后续表达采纳",
     "实际可见的命题中、后续输出里存在任一被判定关系与之对应的条数",
     "Visible propositions for which some later output carries any adjudicated "
     "relation",
     "全部实际可见的命题数（右列改用已评分的那部分作分母，"
     "两列之差即候选召回带来的不确定范围）",
     "All propositions actually visible; the right-hand column uses only the "
     "scored subset instead, and the gap between the columns is the "
     "uncertainty candidate recall introduces",
     "场 / run", "mentor_report_rounds.py"),
    ("置换检验",
     "以<b>场</b>为独立单位重排条件标签 20000 次，双侧，"
     "统计量是两组场均值之差。边数不作独立样本。",
     "Condition labels are shuffled 20,000 times with the <b>run</b> as the "
     "independent unit, two-sided, the statistic being the difference of group "
     "means over runs. Edges are never treated as independent samples.",
     "—", "—", "—", "test_offline_flow.py / analyze_perspectrum_topology.py"),
    ("事实 / 千 token",
     "该场输出的不同事实数 ÷ 输出 token 数 × 1000",
     "Distinct facts in the run divided by output tokens, per thousand",
     "该条件下的输出 token 总数", "Output tokens in that condition",
     "场 / run", "flow-profile.json · facts_per_k"),
    ("有同伴输入的后续 turn 比例",
     "R2/R3 中实际收到过同伴发言的 turn 数",
     "R2 and R3 turns that actually received a peer utterance",
     "全部 R2/R3 turn 数", "All R2 and R3 turns",
     "场 / run", "flow-profile.json · reception"),
    ("后续输出中 peer-adopted 份额",
     "与可见同伴同簇、且此前未由自己说过的输出事实数",
     "Output facts clustered with a visible peer's and not previously said by "
     "the speaker",
     "有同伴输入的后续输出事实数",
     "Facts in later output that had peer input",
     "场 / run", "flow-profile.json · adopted_share_recv (SAME/DIFF pipeline)"),
    ("接触→R2 复述（uptake）",
     "R1 被投递出去、且在 R2 被接收者复述的事实数",
     "R1 facts that were delivered and then restated by a receiver at R2",
     "R1 被投递出去的事实数", "R1 facts that were delivered",
     "场 / run", "rq-extensions.json · round2_transmission_funnel.uptake"),
    ("R2 复述→R3 保留",
     "上一步复述过、且在 R3 仍出现的事实数",
     "Facts restated at R2 that are still present at R3",
     "R2 复述的事实数", "Facts restated at R2",
     "场 / run", "rq-extensions.json · post_adoption_retention"),
    ("接触→R3 保留（端到端）",
     "上面两步的乘积口径：R1 投递且 R3 仍在",
     "The two steps composed: delivered at R1 and still present at R3",
     "R1 被投递出去的事实数", "R1 facts that were delivered",
     "场 / run", "rq-extensions.json · end_to_end"),
    ("R1→R2 重合 Jaccard",
     "两个 agent 该轮输出事实集合的交集大小",
     "Size of the intersection of two agents' fact sets that round",
     "并集大小；对全部 agent 两两取均值",
     "Size of the union, averaged over all agent pairs",
     "场（此处 n=8，仅 1 个病人）/ run (n=8 here, one patient)",
     "2026-09-03-clinicalbench-pilot-metrics.json · roundN_agent_jaccard"),
    ("R1 事实 R2 保留",
     "R1 出现且 R2 仍出现的事实数",
     "Facts present at R1 and still present at R2",
     "R1 的事实数", "Facts at R1",
     "场 / run", "clinicalbench-pilot-metrics.json · round1_retention"),
    ("源材料独占事实跨界复述 / 场",
     "只出现在某一 agent 私有材料中的事实、被<b>没有该材料</b>的 agent 复述的条数",
     "Facts appearing only in one agent's private material, restated by an "
     "agent who <b>did not hold</b> that material",
     "—（计数）", "— (a count)", "场 / run",
     "clinicalbench-pilot-metrics.json · source_exclusive_uptake_facts"),
    ("无后续等价传播的事实比例",
     "首次出现后、再没有任何其他 agent 给出等价表述的事实数",
     "Facts that, after first appearing, are never given an equivalent by any "
     "other agent",
     "确实有机会被别人看到的事实数（排除末轮出生的截尾）",
     "Facts that genuinely had a chance to be seen, excluding those born in "
     "the final round",
     "条件（合并 10 场）/ condition, pooling ten runs",
     "idrbench-graph/summary.json · equivalence_dead_rate"),
    ("加入单向关系后仍无后续关联",
     "同上，但把单向蕴含也算作「有后续关联」",
     "The same, but one-way entailment now counts as a later link",
     "同上", "as above", "条件 / condition",
     "idrbench-graph/summary.json · typed_dead_rate"),
    ("仅在有传播者中的不同接收者数",
     "在至少被一个人接住的事实上，平均有多少个<b>不同</b>接收者",
     "Among facts picked up by at least one agent, how many <b>distinct</b> "
     "receivers on average",
     "—（计数）", "— (a count)", "条件 / condition",
     "idrbench-graph/summary.json · typed_spread_fanout"),
    ("全量已评分 mention 对",
     "40 场合并后各类关系的原始条数，范围是 NLI 裁决过的对。"
     "第一级否决的对不在此表，它们计入「无关」的另一种分母。",
     "Raw counts per relation over all forty runs pooled, scoped to the "
     "pairs the NLI adjudicated. First-stage rejections are not in this "
     "table; they belong to the other denominator, as unrelated.",
     "—（计数）", "— (a count)", "全语料 / whole corpus",
     "idrbench-entailment/summary.json · counts"),
    ("支持类占比：输入→输出",
     "立场标注为 SUPPORT 的命题数，左侧在<b>可见输入</b>上算、右侧在<b>本轮输出</b>上算",
     "Propositions labelled SUPPORT — computed over <b>visible input</b> on "
     "the left and over <b>this round's output</b> on the right",
     "对应范围内有立场标注的命题数",
     "Stance-labelled propositions in the corresponding scope",
     "turn（918 个有效 turn，非独立样本）/ turn (918 valid turns, not "
     "independent samples)",
     "context-output.json · by[persona].SUPPORT"),
    ("削弱类占比：输入→输出",
     "同上，立场标注为 UNDERMINE", "The same, for propositions labelled UNDERMINE",
     "同上", "as above", "同上 / as above",
     "context-output.json · by[persona].UNDERMINE"),
    ("命题数 / turn",
     "该 (条件, 轮次) 下每个 turn 的原子命题数",
     "Atomic propositions per turn in that condition and round",
     "—（计数）", "— (a count)",
     "先在场内合计再跨场平均 / summed within a run, then averaged across runs",
     "mentor_report_rounds.py · analysis.json turns.n"),
    ("B 侧主题 / C 侧主题 / 两篇共有 / 通用",
     "主题轴取值 B / C / X / G 的命题占比。主题轴说命题<b>谈的是哪一侧的内容</b>，"
     "与归属轴（陈述某篇论文 vs 新提案）相互独立",
     "Share of propositions whose topic axis reads B, C, X or G. The topic axis "
     "says <b>whose subject matter</b> a proposition is about, independently of "
     "the attribution axis, which says whether it states a paper or proposes "
     "something new",
     "该范围内有标签的命题数", "Labelled propositions in that scope",
     "场 / run", "label_fact_discipline.py · topic ∈ {B,C,X,G,U}"),
    ("peer-adopted 份额（旧表）",
     "与可见同伴同簇、且此前未由自己说过的输出事实占比。"
     "出自 SAME/DIFF 管线的等价簇，与上面四类关系不是同一种测量。",
     "Share of output facts clustered with a visible peer's and not previously "
     "said by the speaker. It comes from the SAME/DIFF pipeline's equivalence "
     "clusters and is not the same measurement as the four relations above.",
     "有同伴输入的后续输出", "Later output that had peer input",
     "场 / run", "flow-profile.json (superseded table)"),
]


def build(tbl, card, note, section, pct):
    prim = tbl([T("术语", "Term"), T("定义", "Definition"), T("出处", "Source")],
               [[a, T(zh, en), f"<code>{src}</code>"] for a, zh, en, src in PRIM])
    meas = tbl([T("量", "Quantity"), T("分子", "Numerator"),
                T("分母", "Denominator"), T("聚合单位", "Unit averaged"),
                T("计算脚本", "Script")],
               [[q, T(nz, ne), T(dz, de), u, f"<code>{s}</code>"]
                for q, nz, ne, dz, de, u, s in MEAS])
    return section(
        "definitions",
        T("量的形式定义", "Formal definitions of every quantity"),
        "<p>" + T(
            "本节把报告里出现的量写成可追溯的表达式。看表时如果不确定某一列的"
            "分母是什么，回到这里；<b>表头上带虚下划线和 ⓘ 的列名，悬停可看定义，"
            "按住 Alt 点击可跳到对应行</b>（普通点击是排序）。",
            "This section writes every quantity in the report as an expression "
            "you can trace. When a column's denominator is unclear, come back "
            "here: <b>a header carrying a dotted underline and an ⓘ shows its "
            "definition on hover, and alt-clicking jumps to the row</b> (a "
            "plain click sorts).") + "</p>"
        + card("<h3>" + T("基元", "Primitives") + "</h3>" + prim
               + note(T(
                   "分母有两种口径，相差一个数量级：「占已评分对」只算 NLI 裁决过的，"
                   "「占全部对」把第一级的否决也计入无关。top-k 截断掉的那部分"
                   "两种口径都不含——那是唯一真正没被判过的。",
                   "There are two denominators, an order of magnitude apart. "
                   "\"Of scored pairs\" counts only what the NLI adjudicated; "
                   "\"of all pairs\" adds the first stage's rejections as "
                   "unrelated. What the top-k cap truncated is in neither, and "
                   "is the only genuinely unadjudicated part.")))
        + card("<h3>" + T("各量的分子与分母",
                          "Numerator and denominator of each quantity")
               + "</h3>" + meas
               + note(T(
                   "每一列在同一列内跨条件比较。「关系密度」「重合率」「采纳率」"
                   "三类的分母不同，各自成尺。"
                   "计算脚本一列给出唯一事实来源；表里的口径与脚本不一致时以脚本为准。",
                   "Read each column down, across conditions. Relation density, "
                   "overlap rate and uptake rate carry different denominators "
                   "and are each their own scale. The "
                   "script column is the single source of truth; where this "
                   "table and the script disagree, the script wins."))))
