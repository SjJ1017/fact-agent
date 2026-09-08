"""Perspectrum re-measured under one judge, for the mentor report.

The report's original perspectrum tables came from the SAME/DIFF pipeline, and
the star and chain runs were never matched at all -- so the topology rows and
the full rows were not the same measurement. All 108 deepseek traces now carry
four-way NLI relations from the Qwen3-14B judge at one shared threshold, which
is what this section reports.

The old tables are kept rather than replaced: they are the frozen equivalence
-cluster analysis several earlier conclusions rest on, and a reader comparing
the two should be able to see both. They are folded into a details block.
"""

from __future__ import annotations

import json
from pathlib import Path


def T(zh, en):
    """Both languages go into the page; CSS shows one. See mentor_report_i18n.js."""
    return f'<span class="zh">{zh}</span><span class="en">{en}</span>'

ROOT = Path(__file__).resolve().parents[1]
TOPO = ["full", "star", "chain"]
PERS = ["neutral", "lenses", "stance"]

TOPO_TESTS = [("每千组合成边", "star", "+8.3", "p = 0.277", False),
              ("每千组合成边", "chain", "+18.7", "p = 0.016", True),
              (T("同轮等价重合", "Same-round equivalent overlap"), "star", "−2.5pp", "p = 0.159", False),
              (T("同轮等价重合", "Same-round equivalent overlap"), "chain", "−4.0pp", "p = 0.047", True),
              (T("细化减弱化", "Sharpened minus weakened"), "star", "−0.2pp", "p = 0.851", False),
              (T("细化减弱化", "Sharpened minus weakened"), "chain", "−1.7pp", "p = 0.197", False)]
PERS_TESTS = [("每千组合成边", "lenses", "−18.6", "p = 0.019", True),
              ("每千组合成边", "stance", "−15.8", "p = 0.062", False),
              (T("同轮等价重合", "Same-round equivalent overlap"), "lenses", "+2.0pp", "p = 0.236", False),
              (T("同轮等价重合", "Same-round equivalent overlap"), "stance", "+2.3pp", "p = 0.227", False),
              (T("细化减弱化", "Sharpened minus weakened"), "lenses", "−0.6pp", "p = 0.678", False),
              (T("细化减弱化", "Sharpened minus weakened"), "stance", "−0.1pp", "p = 0.924", False)]


def build(tbl, card, note, section, pct):
    D = json.loads((ROOT / "findings/data/perspectrum-topology.json").read_text())
    cells = D["cells"]

    rows = []
    for t in TOPO:
        for p in PERS:
            v = cells.get(f"{t}/{p}")
            if not v:
                continue
            rows.append([f"{t}/{p}",
                         f"{v['不同事实数']['mean']:.1f}",
                         f"{v['每千组合成边']['mean']:.1f}",
                         pct(v["等价"]["mean"]), pct(v["弱化"]["mean"]),
                         pct(v["细化"]["mean"]), pct(v["细化减弱化"]["mean"]),
                         pct(v["同轮等价重合"]["mean"])])
    grid = tbl([T("拓扑 / persona", "Topology / persona"), T("不同事实数 / 场", "Distinct facts per run"), "每千组合成边",
                T("等价", "Equivalent"), T("弱化", "Weakened"), T("细化", "Sharpened"), T("细化 − 弱化", "Sharpened − weakened"), T("同轮等价重合", "Same-round equivalent overlap")], rows)

    mix = D["relation_mix"]
    mrows = []
    for t in TOPO:
        for p in PERS:
            m = mix.get(f"{t}/{p}")
            if not m:
                continue
            s = sum(m.values()) or 1
            mrows.append([f"{t}/{p}", f"{s:,}",
                          pct(m.get("EQUIVALENT", 0) / s),
                          pct(m.get("A_ENTAILS_B", 0) / s),
                          pct(m.get("B_ENTAILS_A", 0) / s),
                          pct(m.get("UNRELATED", 0) / s)])
    mixt = tbl([T("拓扑 / persona", "Topology / persona"), T("已评分对", "Scored pairs"), T("等价", "Equivalent"), "A⊨B", "B⊨A", T("无关", "Unrelated")], mrows)

    def testtbl(ts, base):
        return tbl([T("测量", "Measure"), T("对照", "Contrast"), T("差值", "Difference"), T("置换检验", "Permutation test")],
                   [[k, f"{w} vs {base}", d, p] for k, w, d, p, _hi in ts])

    return section(
        "perspectrum-nli", "I-bis / Perspectrum：三个拓扑第一次由同一个判官测量",
        "<p>本节是新增结果，不替换上面的表。原来的 perspectrum 数字出自 SAME/DIFF "
        "管线，而 star 和 chain 当时<b>根本没有匹配过</b>——拓扑行与 full 行不是同一种"
        "测量，管线差异恰好落在拓扑效应应该出现的位置。现在 108 条 deepseek trace 全部"
        "用 Qwen3-14B entail 判官、单一共享阈值重跑，每格 12 个 claim。</p>"
        + card("<h3>九个格子</h3>" + grid
               + note("「每千组合成边」的分母是「实际投递给某接收者的命题 × 该接收者"
                      "本轮输出的命题」的组合数，所以链式投递少并不会自动让它显得低。"
                      "所有均值以 claim 为单位，不是把边当独立样本。"))
        + card("<h3>拓扑效应：链式每次投递被更用力地处理，但更少趋同</h3>"
               + testtbl(TOPO_TESTS, "full")
               + note("链式相对全连接，每千组合多出 18.7 条关系（p=0.016），"
                      "同轮等价重合却低 4.0 个百分点（p=0.047）。"
                      "<b>投递少不等于处理得浅：每一条到达的内容反而被更密集地接住，"
                      "但整体趋同更弱。</b>星型在三项上都不显著。"))
        + card("<h3>persona 效应：角色分化降低对输入的关系密度</h3>"
               + testtbl(PERS_TESTS, "neutral")
               + note("lenses 比 neutral 每千组合少 18.6 条关系（p=0.019），"
                      "stance 少 15.8（p=0.062）。同时它们产出的不同事实更多"
                      "（full 条件下 100.1 与 97.7，neutral 只有 76.2）。"
                      "<b>角色分化让 agent 说更多新内容、更少复述收到的东西</b>，"
                      "而方向上没有变化——细化减弱化在所有九格里都在 ±2.5% 以内。"))
        + card("<h3>与 IDRBench 的对照</h3>"
               + "<p>IDRBench 那边 persona 提示对内容构成完全没有可测影响（三轮 p 分别"
                 "0.69、0.88、0.84），起作用的是信息分割且只维持一轮。"
                 "这里 persona 有效应，但落在<b>参与量</b>而不是<b>方向</b>上。"
                 "两处一致的是：persona 都没有改变改写的方向结构。</p>"
               + note("两个数据集的判官相同，但阈值 5.28 是在科学论文域的 377 对上"
                      "拟合的。三个拓扑共用同一阈值，所以拓扑之间的比较不受影响；"
                      "跨数据集比较绝对速率时要记住这一点。"
                      "关系类型的总体分布见下表。")
               + mixt))
