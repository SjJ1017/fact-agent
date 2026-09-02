# 重构 pipeline 前必读：已确认的缺陷、证据、修法

> 这份文件记录**读原文查出来的**抽取 / 原子化 / 匹配缺陷，每一条都带可复现的证据和
> 对现有数字的影响估计。重构 pipeline 之前必须逐条看过。
> 最后更新 2026-09-02。发现方式：`findings/trace-viewer.html` 逐 turn 读原文。

---

## 缺陷一：省略指代没有补全（最严重）

**现象。** 同一场辩论（`perspectrum-171-deepseek-v4-flash-full-lenses`）：

```
A|2 原文  “E2 concerns language and multiculturalism, not symbols.”
          → 原子  E2 does not concern symbols            → f_ca4fdf8d3dd6（独占一簇）

B|3 原文  “E2 concerns language and multiculturalism, not religious symbols”
          → 原子  E2 does not concern religious symbols  → f_ad229dfb0d4a（5 处）
```

**这是同一个命题。** 整场辩论的命题就是 "Religious symbols in public buildings must be
banned"，A 说的 "symbols" 只能是 "religious symbols"。但两个原子被分进不同簇，于是
A|2 那次被记成「新事实」而不是「采纳」。

**责任在抽取，不在匹配。** 我一开始判成"agent 自己措辞不同，原子是忠实的"，这是错的。
去语境化的定义就是把原子写成脱离上下文也成立的形式——省略的名词短语和代词是同一类
问题，本来就该补全。匹配器拿到两个范围不同的原子，它没有上下文，只能判"不同"。

**证据（`/tmp/diag.py` 可复现）**

```
「E2 does not concern symbols」vs「E2 does not concern religious symbols.」
   余弦 0.824  包含度 0.800  互为第 2/5 近邻
   → blocking（阈值 0.70, top_k 12）确实提出了这一对，判决器判了 DIFFERENT
```

blocking 没问题，判决器在它能看到的信息下也没问题。**修抽取。**

**修法。** `EXTRACTION_SYSTEM` 里"解析每个代词"这条规则要扩展到**省略的名词短语**，
并且按 `AGENTS.md` 的教训（"没有 worked example 的规则会被忽略"）必须配 3–5 个例子
和反例：

```
命题: Religious symbols in public buildings must be banned
原文: E2 concerns language and multiculturalism, not symbols.
✓    E2 does not concern religious symbols.
✗    E2 does not concern symbols.            ← "symbols" 在此处是省略，须按命题补全

原文: The dossier does not show it reduces harm.
✓    The dossier does not show that banning religious symbols reduces harm.
✗    The dossier does not show it reduces harm.

反例（不要过度补全）:
原文: E3 argues the ban is unworkable.
✓    E3 argues the ban is unworkable.        ← "the ban" 已经足够具体，不要改写成别的
```

---

## 缺陷二：连词没拆干净，而且拆得不一致

**现象。** 同一场、同一个句式，一个 turn 拆了，另一个没拆：

```
B|3  “E2 concerns language and multiculturalism, not religious symbols”
     → E2 concerns multiculturalism. / E2 concerns language. / E2 does not concern religious symbols.   （三条）

A|2  “E2 concerns language and multiculturalism, not symbols”
     → E2 concerns language and multiculturalism / E2 does not concern symbols            （两条，连词没拆）
```

两边的 `provenance.extra` 都是 `{}`，说明**这两种结果都是抽取器直接产出的**，
不是 atomize 拆的。抽取器在同一个句式上不确定。

`E2 concerns language and multiculturalism`（4 处）与它自己的两个部分
`E2 concerns multiculturalism`（1 处）、`E2 concerns language`（2 处）**互不匹配**，
于是同一个 agent 前后说同一件事被记成两个不同的事实。

**规模（`/tmp/split.py`、`/tmp/conj.py` 可复现）**

```
output mention 总数                17990
  被 atomize 拆过的                 5077  = 28.2%
  没被拆、文本里还有 and/or           986  =  5.5%
含 and/or 的规范事实（agent 说的）   1311/13314 = 9.8%
```

样例都是明确该拆的：

```
· E1 and E2 contain claims that are not quantified against costs.       → 应为 2 条
· The dossier does not address tuition, debt, field of study, or individual goals.  → 应为 4 条
· Confounders including recession, pricing, and consumption shifts are unexamined in E1 and E2.
```

**修法。**

1. atomize 的正则预筛现在漏掉 5.5%。它只在"可疑"时才调模型，这个预筛太窄。
   连词拆分是纯句法工作，应该**无条件对每条 mention 跑一次**，或者把预筛放宽到
   任何含 ` and ` / ` or ` / `, ` 的 mention。这一步很便宜（本地正则 + 短提示）。
2. 抽取提示里要有 `E1 and E2 都 X` → 两条的 worked example，包括
   **不该拆的反例**（`black and white film`、`Bosnia and Herzegovina`、
   `terms and conditions`）——`AGENTS.md` 里记着只给正例会导致过度拆分。

---

## 缺陷三：粒度兄弟污染了「新事实」计数

**现象。** 同一场里，一条事实是另一条的严格子集：

```
· The verdict is UNCERTAIN.              /  The verdict on college value is UNCERTAIN
· Correlation is not causation.          /  Correlation in E1 is not proof of causation.
· Prohibition does not reduce harm.      /  Prohibition does not reduce net harm.
· The dossier does not address individual goals.  /  ... individual goals related to college
```

**规模（`/tmp/impact.py` 可复现）**

```
output 事实 9348，其中与同场另一条构成严格 token 子集关系的 892 = 9.5%
被记为「新事实」的共 10353，其中 994 条是粒度兄弟 = 9.6%
```

**影响是单向的**：虚高 novel、压低 held 和 adopted。报告里"人格越强新事实占比越高"
这条主结论正好建立在 novel 份额上，所以这个偏差落在最要紧的地方。方向已知，
大小约 10%，但**不能简单减掉**——有些子集对是真的不同（`about electoral mechanics`
vs `about voter attitudes`）。

**修法。** 缺陷一和二修好之后这一项应自然下降。修完必须重测这个 9.6%，
它是判断"重抽是否值得"的验收指标。

---

## 缺陷四：判决器在粒度不一致时无法自救

不是它的错，但要写下来：判决是**二值 SAME/DIFF**，且原子已脱离上下文。
当上游给出 `X and Y` 与 `X`、或 `symbols` 与 `religious symbols` 时，
判"不同"是它唯一合理的选择。

**结论：不要通过放宽判决阈值来修上面三条。** 放宽会把
`reduce deaths from overdoses` 和 `reduce deaths from impure drugs`（余弦 0.946）
错并成一条。24 场抽样里余弦 ≥0.85 却未合并的事实对有 827 个，
手工看过样例后**大多数是该分开的**——所以那个数字不能当作欠合并的证据。

---

## 缺陷五：所有轮次都带着完整卷宗，"传输"不可识别

**现象。** `delivery` 字段里每个 slot 都是

```
source_ids = ['claim','E1','E2','E3','E4']    ← 9 个 turn 全部如此
peer_turns = [...]                            ← 只有第 2、3 轮有
```

也就是说 **agent 在任何一轮都能重述 E1–E4，不需要任何人传给它**。

**后果。** 关于证据文档的事实（"E2 讲的是多元文化"）在两个 turn 同时出现，
既可能是传输，也可能是两人各自读了同一份卷宗。现有的"实中继"口径
（要求投递方明确先说过）只是下界，无法排除独立重推。

**内容构成（`/tmp/anchor.py` 可复现，deepseek 108 场）**

```
              点名 E1-E4   泛指材料/发言人   关于世界     合计
新事实          25.2%        28.6%         46.2%    10353
自持            15.9%        16.6%         67.5%     1365
采纳            23.2%        25.9%         50.8%     1119
```

**约一半的流动内容是在谈材料本身，不是在谈世界。**而 agent 自己反复重提的
（自持）最偏向世界层（67.5%）。换句话说：**agent 把关于证据的评论传来传去，
把关于世界的主张留给自己。**

**修法（实验设计层，不是代码层）。** 要让传输可识别，必须打破共享卷宗：

1. **私有证据消融**：给每个 agent 一份只有它能看到的证据。此时同一事实出现在
   两个 agent 处只可能来自传输。这是目前唯一能把传输和重推分开的设计。
2. **卷宗只在第 1 轮给**：第 2、3 轮的 prompt 去掉证据全文。这样第 3 轮还出现的
   证据类事实必然来自记忆或传输，而不是重读。
3. 在此之前，所有传输类结论都必须标注为"与观察相容，不排除独立重推"。

---

## 缺陷六：一个片段上的多个事实曾被丢弃（viewer，已修）

`build_trace_view.py` 原来在重叠 span 里只保留最长的一个，**全语料丢掉 3627 个事实**。
`E2 does not discuss religious symbols` 就是这么消失的，导致人工核对时看不到它。
已改为一个 span 携带全部原子（`aa293fd`）。**教训：任何可视化的去重都可能删掉数据，
要报数不要静默丢。**

---

## 修的顺序

1. **抽取提示**：省略指代补全 + 连词拆分，都要带 worked example 和反例（缺陷一、二）
2. **atomize 预筛放宽**到所有含连词的 mention（缺陷二）
3. 重跑**整个语料**（不是局部——见 `AGENTS.md` 的「别给部分数据打补丁」）
4. 重测缺陷三的 9.6%，作为验收指标
5. 实验设计：私有证据或卷宗只给第 1 轮（缺陷五）

## 复现脚本

```
experiments/diagnostics/diag.py        缺陷一：blocking 提出了这一对，判决判了不同
experiments/diagnostics/split.py       缺陷二：atomize 拆了多少、漏了多少
experiments/diagnostics/conj.py        缺陷二：含 and/or 的规范事实比例
experiments/diagnostics/impact.py      缺陷三：粒度兄弟占新事实的比例
experiments/diagnostics/anchor.py      缺陷五：流动内容里 E1-E4 / 泛指 / 世界的构成
experiments/diagnostics/undermerge.py  反例：为什么不能用余弦阈值判欠合并
```

全部离线、不调模型（除 undermerge.py 要加载本地 bge），从仓库根目录运行。
