"""Three offline results from the co-constructive corpus that the report lacked.

They were computed for the offline-flow brief and published in a separate file,
which meant they did not exist: the main report is what gets read. They belong
in Part I because all three are about what a panel with a shared objective does
with content over three rounds -- whether a weakening is later undone, whether
any agent's contribution is contained in another's, and what the extra rounds
buy.

Each reads off findings/data/offline-flow.json, so the numbers move when the
analysis is re-run rather than being retyped here.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ["full-generic", "full-specialist", "split-generic", "split-specialist"]
NAME = {"full-generic": "全量来源 · 通用角色", "full-specialist": "全量来源 · 专家角色",
        "split-generic": "分割来源 · 通用角色", "split-specialist": "分割来源 · 专家角色"}
EN = {"full-generic": "full sources, generic role",
      "full-specialist": "full sources, specialist role",
      "split-generic": "split sources, generic role",
      "split-specialist": "split sources, specialist role"}


def T(zh: str, en: str) -> str:
    """Both languages go into the page; CSS shows one. See mentor_report_i18n.js."""
    return f'<span class="zh">{zh}</span><span class="en">{en}</span>'


def build(tbl, card, note, section, pct):
    D = json.loads((ROOT / "findings/data/offline-flow.json").read_text())

    K4 = ["恢复到等价", "继续变弱", "反而更强", "转向别处", "端点未评分"]
    E4 = ["back to equivalent", "weaker still", "stronger instead",
          "turned elsewhere", "endpoints unscored"]
    q4 = tbl([T("条件", "Condition")] + [T(z, e) for z, e in zip(K4, E4)]
             + [T("路径数", "Paths")],
             [[T(NAME[c], EN[c])]
              + [pct(D[c]["q4"].get(k, 0) / max(sum(D[c]["q4"].values()), 1))
                 for k in K4]
              + [f'{sum(D[c]["q4"].values()):,}'] for c in CELLS if D[c]["q4"]])

    q5 = tbl([T("条件", "Condition"), T("第 1 轮", "Round 1"),
              T("第 2 轮", "Round 2"), T("第 3 轮", "Round 3")],
             [[T(NAME[c], EN[c])] + [pct(D[c]["q5_joint"][r]) for r in ("1", "2", "3")]
              for c in CELLS if D[c].get("q5_joint")])

    K6 = ["等价重复", "已有内容的较弱表述", "细化候选", "无关于历史", "未匹配产出"]
    E6 = ["exact repeat", "weaker restatement", "sharpening", "unrelated to history",
          "unmatched"]
    q6 = tbl([T("条件 · 轮次", "Condition and round")]
             + [T(z, e) for z, e in zip(K6, E6)] + [T("命题数", "Propositions")],
             [[T(f"{NAME[c]} · 第 {r} 轮", f"{EN[c]}, round {r}")]
              + [pct(D[c]["q6"][r].get(k, 0) / max(sum(D[c]["q6"][r].values()), 1))
                 for k in K6]
              + [f'{sum(D[c]["q6"][r].values()):,}']
              for c in CELLS for r in ("2", "3") if D[c]["q6"].get(r)])

    return section(
        "offline-flow",
        T("三轮之间：弱化、可替代性与轮次预算",
          "Between the rounds: weakening, substitutability, and what rounds buy"),
        "<p>" + T(
            "同一批 40 场的三个离线结果，都只用已缓存的 NLI 关系，没有新的模型调用。"
            "问的是共享目标的小组在三轮里对内容做了什么。",
            "Three offline results on the same forty runs, computed from the "
            "cached NLI relations with no further model calls. All three ask what "
            "a panel with a shared objective does to content over three rounds.")
        + "</p>"
        + card("<h3>" + T("一次弱化之后，端到端是什么关系",
                          "After a weakening, what holds end to end")
               + "</h3>" + q4
               + note(T(
                   "路径是 p → q → r，两步都被判为弱化，再看首尾这一对的直接判定。"
                   "<b>端点被评分的只有约三分之一</b>，在这三分之一里恢复到等价占 0–1%，"
                   "继续变弱占 18–20%。可分析范围窄，这里只报读数：单步弱化是否被后续"
                   "抵消，需要更高的候选召回才能回答。",
                   "A path is p → q → r with both steps judged weakenings; the "
                   "table then reads the direct verdict on the first and last. "
                   "<b>Only about a third of endpoints were scored</b>, and within "
                   "that third a return to equivalence is 0–1% while a further "
                   "weakening is 18–20%. The analysable range is narrow, so these "
                   "are readings rather than a conclusion: whether a single "
                   "weakening gets undone needs better candidate recall to answer.")))
        + card("<h3>" + T("其他 agent 合起来覆盖了某人输出的多少",
                          "How much of one agent's output the others cover between them")
               + "</h3>" + q5
               + note(T(
                   "<b>没有任何一场、任何一轮出现过一个 agent 的内容被另一个完全包含。</b>"
                   "其他所有 agent 合起来也只等价覆盖某人输出的 6–12%。"
                   "「大团队逐渐塌缩成一个主要表达者加若干重复者」在这批数据里没有发生。"
                   "分割来源第 1 轮的 23.4% 是唯一的例外，那一格三人零共同文献却仍然重合——"
                   "重合的内容里陈述论文的占 0.0%，全是各自独立编出来的新提案。",
                   "<b>Not once, in any run or any round, is one agent's content "
                   "wholly contained in another's.</b> Every other agent combined "
                   "covers only 6–12% of a given agent's output. A large panel "
                   "collapsing into one speaker and several repeaters does not "
                   "happen here. Split sources at round one is the exception at "
                   "23.4%: three agents with no shared paper still overlap, and "
                   "0.0% of that overlap states either source paper — it is "
                   "independently invented convergence.")))
        + card("<h3>" + T("继续生成买到了什么",
                          "What the extra rounds buy") + "</h3>" + q6
               + note(T(
                   "每一条输出相对<b>实际可见的历史</b>分类。第 1 轮没有历史，按定义全部"
                   "未匹配，已从表中略去。从第 2 轮到第 3 轮，<b>等价重复接近翻倍</b>"
                   "（全量通用 9.8% → 18.9%，全量专家 12.3% → 22.2%），"
                   "与历史无关的产出同时从 38–44% 掉到 22–26%。"
                   "轮次越往后，买到的越多是复述。这是离线的预算读数，"
                   "提前停止是否损失任务质量要另测。",
                   "Every output proposition classified against the history that "
                   "was <b>actually visible</b> to its author. Round one has no "
                   "history and is unmatched by definition, so it is omitted. "
                   "Between rounds two and three <b>exact repetition nearly "
                   "doubles</b> — 9.8% to 18.9% under full sources with generic "
                   "roles, 12.3% to 22.2% with specialists — while output "
                   "unrelated to the history falls from 38–44% to 22–26%. Later "
                   "rounds buy more restatement. This is a budget reading; "
                   "whether stopping early costs task quality is a separate "
                   "measurement.")))
        + note(T(
            "三张表的分母都是<b>已评分对</b>。第一张的「端点未评分」一列把可分析范围"
            "明写出来，因为它决定了那一节能说多少。",
            "All three tables are over <b>scored pairs</b>. The first keeps an "
            "explicit column for unscored endpoints, because that column is what "
            "bounds how much the section can claim.")))
