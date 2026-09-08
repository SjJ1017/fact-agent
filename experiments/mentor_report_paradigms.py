"""Three discussion paradigms, and what separates them.

The datasets in this project were collected one at a time and reported the same
way, which hid the fact that they are not three instances of one thing. Two
properties generate the difference: whether the agents' objectives are shared,
and whether an utterance can be assumed sincere.

  co-constructive   objectives shared, contributions complementary  (IDRBench,
                    ClinicalBench)
  agonistic         opposition assigned as method, objective still shared;
                    Mouffe's adversary rather than enemy               (Perspectrum)
  adversarial       objectives genuinely opposed                       (Avalon)

Concealment is deliberately not folded into the third name. It is a separate
dimension -- negotiation and competitive review are adversarial with open
identities -- and naming it into the paradigm would assert that real opposition
implies hiding, which is exactly what this corpus cannot test. Keeping it as an
axis leaves the empty cell visible.

The section closes on the one measurement that needs all three: whether a
seat's assigned role can be recovered from its input/output profile. Avalon is
the positive control, because there a role fixes the win condition and the
private information and cannot fail to take effect. The no-role conditions are
the negative control. What lies between is the project's actual question.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def T(zh: str, en: str) -> str:
    """Both languages go into the page; CSS shows one. See mentor_report_i18n.js."""
    return f'<span class="zh">{zh}</span><span class="en">{en}</span>'


# ---------------------------------------------------------------- the figure
ORDER = ["Merlin", "Servant", "Evil"]
POS = {"Merlin": (150, 78), "Servant": (400, 78), "Evil": (275, 250)}
COL = {"Merlin": "mrl", "Servant": "srv", "Evil": "evl"}


def absorption_graph(m: dict) -> str:
    """Directed graph: edge width is uptake per 1k combinations, thin means less."""
    lo = min(v["mean"] for v in m.values())
    hi = max(v["mean"] for v in m.values())
    w = lambda v: 1.2 + 7.0 * (v - lo) / (hi - lo)
    p = ['<svg viewBox="0 0 560 336" role="img" '
         'aria-label="Avalon 承接有向图：边宽表示每千组合的承接条数">']
    p.append('<defs>')
    for r, c in COL.items():
        p.append(f'<marker id="ah-{c}" viewBox="0 0 10 10" refX="9" refY="5" '
                 f'markerWidth="6" markerHeight="6" orient="auto">'
                 f'<path d="M0 0 L10 5 L0 10" fill="var(--{c})"/></marker>')
    p.append('</defs>')

    for src in ORDER:
        for rcv in ORDER:
            key = f"{src}->{rcv}"
            if key not in m:
                continue
            v = m[key]["mean"]
            x1, y1 = POS[src]
            x2, y2 = POS[rcv]
            c = COL[src]
            if src == rcv:                       # self loop, drawn as a small arc
                p.append(f'<path d="M{x1-26},{y1+16} a26,22 0 1,0 52,0" fill="none" '
                         f'stroke="var(--{c})" stroke-width="{w(v):.1f}" '
                         f'opacity=".62" marker-end="url(#ah-{c})"/>')
                p.append(f'<text class="ev" x="{x1}" y="{y1+58}" '
                         f'text-anchor="middle">{v:.1f}</text>')
                continue
            dx, dy = x2 - x1, y2 - y1
            L = (dx * dx + dy * dy) ** .5
            ox, oy = -dy / L * 15, dx / L * 15   # offset so the two directions split
            sx, sy = x1 + dx * .21 + ox, y1 + dy * .21 + oy
            ex, ey = x1 + dx * .79 + ox, y1 + dy * .79 + oy
            p.append(f'<line x1="{sx:.0f}" y1="{sy:.0f}" x2="{ex:.0f}" y2="{ey:.0f}" '
                     f'stroke="var(--{c})" stroke-width="{w(v):.1f}" opacity=".62" '
                     f'marker-end="url(#ah-{c})"/>')
            mx, my = (sx + ex) / 2 + ox * .5, (sy + ey) / 2 + oy * .5
            p.append(f'<text class="ev" x="{mx:.0f}" y="{my:.0f}" '
                     f'text-anchor="middle">{v:.1f}</text>')

    for r in ORDER:
        x, y = POS[r]
        p.append(f'<circle cx="{x}" cy="{y}" r="31" fill="var(--panel)" '
                 f'stroke="var(--{COL[r]})" stroke-width="2.4"/>')
        p.append(f'<text class="nd" x="{x}" y="{y+5}" text-anchor="middle">{r}</text>')
    p.append('<text class="cap" x="14" y="322">'
             '边宽 = 每千（先说 × 后说）组合的承接条数 · width = uptake per 1k'
             '</text>')
    return "".join(p) + "</svg>"


SEP = [("Avalon 好人 / 坏人", "Avalon Good vs Evil", 86.0, 60.0, 50, "pos"),
       ("Avalon 四类角色", "Avalon four roles", 54.0, 40.0, 50, "pos"),
       ("Perspectrum neutral（无角色）", "Perspectrum neutral (no role)",
        35.2, 33.3, 108, "neg"),
       ("Perspectrum lenses", "Perspectrum lenses", 33.3, 33.3, 108, ""),
       ("Perspectrum stance", "Perspectrum stance", 39.8, 33.3, 108, ""),
       ("IDRBench full-generic（无角色）", "IDRBench full-generic (no role)",
        26.7, 33.3, 30, "neg"),
       ("IDRBench full-specialist", "IDRBench full-specialist", 30.0, 33.3, 30, ""),
       ("IDRBench split-generic（无角色）", "IDRBench split-generic (no role)",
        43.3, 33.3, 30, ""),
       ("IDRBench split-specialist", "IDRBench split-specialist",
        76.7, 33.3, 30, "hit")]


def build(tbl, card, note, section, pct):
    absorb = json.loads((ROOT / "findings/data/avalon-absorption.json").read_text())

    grid = tbl(
        ["", T("目标共享", "Shared objective"),
         T("对立是分派的", "Opposition assigned"),
         T("对立是实质的", "Opposition real"),
         T("发言可假定诚实", "Sincerity assumable")],
        [[T("<b>co-constructive</b> 协作共建", "<b>co-constructive</b>"),
          "✓", "—", "—", "✓"],
         [T("<b>agonistic</b> 竞胜型", "<b>agonistic</b>"), "✓", "✓", "—", "✓"],
         [T("<b>adversarial</b> 敌对型", "<b>adversarial</b>"), "—", "—", "✓", "—"]])

    cells = tbl([T("范式", "Paradigm"), T("身份公开", "Identities open"),
                 T("身份隐蔽", "Identities hidden")],
                [["co-constructive", "IDRBench · ClinicalBench", "—"],
                 ["agonistic", "Perspectrum",
                  T("结构性空缺：分派立场必须公开",
                    "structurally empty: an assigned stance must be declared")],
                 ["adversarial",
                  T("<b>空</b>——谈判、竞标、对抗性评审",
                    "<b>empty</b> — negotiation, bidding, adversarial review"),
                  "Avalon"]])

    sep = tbl([T("语料 · 条件", "Corpus and condition"),
               T("留一局准确率", "Leave-one-run-out accuracy"),
               T("多数类基线", "Majority baseline"), "n", T("角色", "Role")],
              [[T(zh, en), f"{acc:.1f}%", f"{ch:.1f}%", str(n),
                {"pos": T("阳性对照", "positive control"),
                 "neg": T("阴性对照", "negative control"),
                 "hit": T("超基线 43 点", "43 points over baseline"),
                 "": ""}[tag]]
               for zh, en, acc, ch, n, tag in SEP])

    return section(
        "paradigms",
        T("三种讨论范式", "Three discussion paradigms"),
        "<p>" + T(
            "这些数据集不是同一件事的三个实例。区分它们的是两条性质："
            "<b>目标是否共享</b>，以及<b>发言能否假定诚实</b>。",
            "These corpora are not three instances of one thing. Two properties "
            "separate them: <b>whether objectives are shared</b>, and "
            "<b>whether an utterance can be assumed sincere</b>.") + "</p>"
        + card(grid + note(T(
            "命名沿用 Mouffe 对 agonism 与 antagonism 的区分：agonistic 的对手是"
            "<b>adversary</b>——立场对立但共享规则与更高目标；adversarial 的对手是"
            "<b>enemy</b>，目标互斥。",
            "The names follow Mouffe's distinction between agonism and "
            "antagonism: an agonistic opponent is an <b>adversary</b>, opposed "
            "in position but sharing the rules and the higher objective; an "
            "adversarial one is an <b>enemy</b>, with incompatible goals.")))
        + card("<h3>" + T("隐蔽是另一条轴，不写进名字",
                          "Concealment is a separate axis, kept out of the name")
               + "</h3>" + cells
               + note(T(
                   "把 concealed 写进第三个名字，等于断言真实对立必然伴随隐蔽——"
                   "而这正是本语料<b>无法检验</b>的。Avalon 里坏人内部承接率最低（3.1），"
                   "是因为要赢还是因为要藏，现在分不开；那个空格填上才分得开。",
                   "Folding concealment into the third name would assert that "
                   "real opposition implies hiding, which is precisely what "
                   "this corpus <b>cannot test</b>. Evil's mutual uptake is the "
                   "lowest in Avalon at 3.1, and whether that is because they "
                   "must win or because they must hide is not separable until "
                   "that empty cell is filled.")))
        + card("<h3>" + T("adversarial 的承接图：全公开信道下的选择性接收",
                          "The adversarial uptake graph: selective uptake on an "
                          "open channel")
               + "</h3>"
               + f'<figure style="margin:0">{absorption_graph(absorb)}'
               + "<figcaption>" + T(
                   "五个人听到的完全一样，投递图是完全图，所以任何边宽差异都不是"
                   "「能不能听到」，而是<b>选择接住什么</b>。箭头由发送者指向承接者，"
                   "自环是同队内部。每条边是 10 局、20–40 个有序座位对的均值。",
                   "All five players hear everything, so the delivery graph is "
                   "complete and any difference in width is not access but "
                   "<b>selective uptake</b>. Arrows run from speaker to the seat "
                   "that later restates them; the loop is within-team. Each edge "
                   "averages 20 to 40 ordered seat pairs over ten games.")
               + "</figcaption></figure>"
               + note(T(
                   "<b>梅林承接坏人 5.8，坏人承接梅林 3.3。</b>梅林知道那两个是谁，"
                   "要引导好人就得回应他们；侍从不知道，承接坏人只有 4.3。"
                   "多出来的三成是知识的痕迹——只看图不看内容，统计谁在不成比例地"
                   "回应哪两个座位，就能反推梅林。"
                   "<b>坏人内部 3.1 是全图最低</b>：互知身份、目标一致，却最不互相承接，"
                   "因为公开承接同伴会留下可读的呼应关系。",
                   "<b>Merlin takes up Evil at 5.8; Evil takes up Merlin at "
                   "3.3.</b> Merlin knows who they are and must answer them to "
                   "steer the table; a Servant, who does not know, takes up Evil "
                   "at 4.3. The extra third is the trace of knowledge — an "
                   "observer reading only the graph can find Merlin by asking "
                   "which seat answers those two disproportionately. "
                   "<b>Evil-to-Evil is the lowest edge at 3.1</b>: they know "
                   "each other and share a goal, yet take each other up least, "
                   "because visible mutual uptake is itself readable.")))
        + card("<h3>" + T("跨范式：角色能不能从输入输出画像里读出来",
                          "Across paradigms: is a role recoverable from its "
                          "input/output profile?")
               + "</h3>" + sep
               + note(T(
                   "特征只有图上的量——说了多少、被接住多少、承接多少、传到几个人、"
                   "自我复述多少——<b>没有一个依赖他人的真实身份</b>，否则分类器会"
                   "读到答案本身。以局为单位留一交叉验证。",
                   "The features are graph quantities only: how much a seat says, "
                   "how much is taken up from it, how much it takes up, how many "
                   "others it reaches, how much it repeats itself. <b>None reads "
                   "anyone's true role</b>, which would hand the classifier the "
                   "answer. Cross-validation leaves out a whole run at a time."))
               + "<p>" + T(
                   "Avalon 的角色不是提示词，它决定胜利条件和私有信息，<b>不可能不生效</b>，"
                   "所以 86% 验证的是仪器可用。无角色条件贴着基线（35.2%、26.7%），"
                   "说明特征没有偷偷编码座位位置。中间那些才是真正的检验。",
                   "A role in Avalon is not a prompt: it fixes the win condition "
                   "and the private information and <b>cannot fail to take "
                   "effect</b>, so 86% establishes that the instrument works. The "
                   "no-role conditions sit on the baseline at 35.2% and 26.7%, so "
                   "the features are not quietly encoding seat position. What lies "
                   "between is the test.") + "</p>"
               + card(tbl([T("IDRBench 条件", "IDRBench condition"),
                           T("可分性", "Separability"), T("相对基线", "Over baseline")],
                          [[T("无分割 + 无角色", "no split, no role"), "26.7%", "—"],
                           [T("无分割 + 有角色", "no split, role"), "30.0%", "+3.3"],
                           [T("有分割 + 无角色", "split, no role"), "43.3%", "+16.6"],
                           [T("<b>有分割 + 有角色</b>", "<b>split and role</b>"),
                            "<b>76.7%</b>", "<b>+50.0</b>"]])
                      + note(T(
                          "<b>角色单独 +3.3，分割单独 +16.6，两者合起来 +50.0。</b>"
                          "超可加。角色提示词自己几乎不改变行为，只有当 agent 手里"
                          "真的只有那一半材料时，「你是 paper-b 的方法专家」才变成"
                          "可执行的指令。这是第三种独立方法得到的同一结论——"
                          "另外两种是主题构成（三轮 p = 0.69 / 0.88 / 0.84）"
                          "与关系密度。",
                          "<b>The role alone adds 3.3, the split alone 16.6, and "
                          "together they add 50.0.</b> Super-additive. A role "
                          "prompt on its own barely changes behaviour; \"you are "
                          "the method specialist for paper B\" becomes an "
                          "executable instruction only once the agent actually "
                          "holds only that half of the material. This is a third "
                          "independent method reaching the same conclusion as the "
                          "topic composition (p = 0.69 / 0.88 / 0.84 across the "
                          "three rounds) and the relation density.")))))
