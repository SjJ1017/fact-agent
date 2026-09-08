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

Concealment is a separate axis. Negotiation and competitive review are
adversarial with open identities; Avalon occupies both cells at once, which is
why its lowest edge cannot be attributed to winning or to hiding alone.

The section closes on the one measurement that needs all three: whether a
seat's assigned role can be recovered from its input/output profile. Avalon is
the positive control, because there a role fixes the win condition and the
private information and cannot fail to take effect. The no-role conditions are
the negative control. What lies between is the project's actual question.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def T(zh: str, en: str) -> str:
    """Both languages go into the page; CSS shows one. See mentor_report_i18n.js."""
    return f'<span class="zh">{zh}</span><span class="en">{en}</span>'


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
                   "隐蔽是独立的一条轴：谈判和对抗性评审都是真实对立而身份公开。"
                   "Avalon 同时占了两格，所以坏人内部 3.1 的最低承接率里，"
                   "「要赢」和「要藏」的贡献分不开；<b>填上左下那格才能分开</b>。",
                   "Concealment is its own axis: negotiation and adversarial "
                   "review are genuinely opposed with open identities. Avalon "
                   "occupies both cells at once, so in its lowest edge — Evil "
                   "taking up Evil at 3.1 — winning and hiding cannot be told "
                   "apart. <b>Filling the lower-left cell separates them.</b>"))))


def build_cross(tbl, card, note, section, pct):
    """Part IV: the same profile classifier turned on all three paradigms.

    It sits in its own part because it is not a property of any one corpus.
    Avalon supplies the positive control and the no-role conditions the
    negative one, so the null results in between mean something.
    """
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
        "cross-paradigm",
        T("角色能不能从输入输出画像里读出来",
          "Is a role recoverable from its input/output profile?"),
        card("<h3>" + T("跨范式：角色能不能从输入输出画像里读出来",
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
