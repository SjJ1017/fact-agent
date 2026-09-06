你接手 factflow 项目，做 IDRBench 上的事实流分析。之前没人跟你交接过，下面是全部必要背景。

## 先读

按顺序，不要跳：

1. `AGENTS.md` — 操作层的硬教训（成本陷阱、负面结果、别重复踩的坑）
2. `fix.md` — **重构 pipeline 前必读**。已确认的抽取/原子化/匹配缺陷，每条带可复现证据
3. `experiments/matcher_eval/README.md` — 匹配判决器的选型过程和当前配置
4. `findings/2026-09-01-current-data-rq-hypotheses.md` — 研究问题和假设的现状

## 项目在做什么

用**原子事实作为坐标系**追踪信息在多智能体辩论中的流动。抽取每一轮每个 agent 输出里的原子命题，跨 turn 匹配，得到一张图：谁说了什么、什么被传递、什么被改写、什么消失了。

**自变量是 persona 和 topology**，不是事实存活率。存活率是测量通道，不是研究问题。这一点历史上漂过好几次，注意别再漂。

## 当前的关键 findings

**匹配判决器已换成本地模型。** Qwen3-14B 用作 NLI 分类器，在 422 对（辩论+临床）和 377 对（论文域）两个人工标注 oracle 上都超过原来的托管 minimax：f 本身准确 0.951–0.966，推出的等价 F1 0.889–0.904、精确 0.911–0.971。而且快 5 倍以上。

**判决形式是「一个 f，用两次」**：`f(x,y) → 蕴含与否`，对每一对问 `f(a,b)` 和 `f(b,a)`，四格由两个结果推出：

```
双向成立 → EQUIVALENT      只有 a→b → A_ENTAILS_B
只有 b→a → B_ENTAILS_A     都不成立 → UNRELATED
```

阈值**两个方向共用一个**，在方向标注上拟合，不是在等价标注上。曾经用过两个独立阈值，那会退化（合取下把一个方向推到最低点等于关掉它），已修。

**单向蕴含边占比很高，这是最重要的新发现。** IDRBench 单个文件里：EQUIVALENT 334 / A_ENTAILS_B 318 / B_ENTAILS_A 372 / UNRELATED 2194。**非无关边里三分之二是单向的。** 人工抽查确认它们是真实的信息降级，不是噪声：

```
强: 5,000 randomized 2D CompuCell3D configurations are generated
弱: Training covers randomized 2D cell configurations.          （丢了数量）

强: Paper-c's conditional GAN generates steady-state transport fields.
弱: The cGAN generates steady-state transport fields.            （丢了归属）

强: The built-in PDE solver runs as fallback when the cGAN is not used.
弱: The cGAN replaces the classical PDE solver.                  （丢了回退机制，语义被弱化）
```

抽样误判率约 17%，和 f 的精确率吻合。

## 和 DelibTrace 的区分（核心任务之一）

DelibTrace（arXiv 2606.03032，*The Deliberative Illusion*）和本项目高度相似：原子事实、跨轮存活、full/tree/chain 三拓扑、三轮、fSystem/fAgent 两个 Jaccard 指标。**不要重复它的问题设定**（factual attrition + stance homogenization）。

已识别的三个区分点：

1. **考虑新生成的事实**，不只追踪一开始分发的种子事实。DelibTrace 只测种子事实的存活；这里 60–70% 的事实是 agent 自己造的，它的设计看不到
2. **agent 信息不对称**（dealing 层，`src/factflow/tasks.py`）
3. **自变量是 persona × topology**；DelibTrace 的 persona 只在附录 D.9 做了一次干预

**第四个、也是最强的一个，需要你来做实**：有向蕴含边。Jaccard 是集合成员关系，一条事实要么在要么不在，**没有「还在但被削弱了」这个状态**。而上面那 690 条单向边正是这个状态。DelibTrace 把它们要么算作同一条（信息损失不可见），要么算作两条不同（看起来像丢了又新造），两种记法都描述不了发生了什么。

你的任务是把这个从「有道理」变成「有数据支撑」。

## 数据在哪

`experiments/idrbench_generation_10x5_r3/` — 40 个 trace，每个有：

- `*.debate.json` — 原始辩论（transcript、prompts、delivery、roles、disclosure）
- `*.atomized.json` — 抽取+原子化后的 mentions（facts 为空）
- `*.store.json` 或 `*.nli.store.json` — 匹配结果

store 的结构（和历史上的 `.v2.json` 完全兼容，只多两块）：

```
facts            fact_id -> {canonical_text, mention_ids, polarity, properties}
mentions         mention_id -> {text, polarity, quote, qualifiers, provenance}
mention_to_fact  mention_id -> fact_id
relations        [{a, b, relation, confidence, rationale, properties}]
matching         模型、阈值、blocker 设置、四格计数   ← 新增
```

`relations[].properties` 里有 `margin_ab` / `margin_ba` 全精度分数和当时的阈值，**换阈值重新推导标签不需要再跑模型**。

`delivery` 在 debate 文件里，每个 `agent|round` 记录：`source_ids`（该 agent 拿到的材料，按 agent 分，不是整份卷宗）、`peer_turns`（本轮投递）、`visible_peer_turns` / `visible_self_turns`（上下文里实际可见的）。**区分「投递」和「可见」很重要**，cumulative memory 下两者差很多。

## 要做的分析

**先做清点。** 全部 40 个 trace 上四格的分布是多少？单向边占非无关边的比例？各 trace 之间波动多大？（前面那个 67% 来自单个文件，不能外推。）

**然后接上图结构。** 方向本身不等于时间顺序：`a⊨b` 只说 b 更弱，不说 b 是从 a 来的。要断言「降级」必须叠上 `delivery`——b 出现在后一轮，且说 b 的 agent 收到过 a。把这个连起来，得到有向的降级边。

**然后挂到自变量上。** persona 和 topology 怎么影响降级？具体问题：某些 persona 是不是更容易泛化别人的事实？星型的中心节点是不是降级的瓶颈？降级是不是随轮次累积？

**注意 IDRBench 的条件维度**在 `experiments/datasets/idrbench_*.yaml` 和文件名里（`full-generic` / `full-specialist` / `split-generic` / `split-specialist`），是信息划分 × 角色分化的 2×2。

## 图结构这条线值得往深里做

这是本项目相对 DelibTrace 最结构性的优势：**原子匹配做完之后才有图，DelibTrace 只能做线性的存活率变化。** 下面几个方向都还没人做过，选你觉得最扎实的推进，不必全做。

**节点不止 agent。** 除了每个 agent 和每份源材料，每个 agent 还应该配一个虚拟出点，表示「他新造的事实」——那部分占 60–70%，是图里的主体，不是补充。把它显式建出来之后，「产出」和「转述」才是两类不同的边。

**边是有类型的。** 目前至少四类：origin（材料→首次陈述）、transmission（agent→agent）、persistence（同一 agent 跨轮）、degradation（有向蕴含 + 投递证据）。DelibTrace 的 Jaccard 把这四类压成一个标量。

**可算而且没算过的量：**

- **死端率**。说完就没人再提的事实占比。已知的一个反直觉观察：分信息条件下 80–88% 的事实死在原地，全信息下只有 55–65%——**分信息提高产量、降低扩散**，而线性存活率会把这读成「保持率更低」，方向都错了
- **扇出**。传开的事实平均到达几个同伴。三人局上限是 2，实测均值只有 1.3–1.8，说明多数「传播」是单跳单向而不是扩散
- **路径长度**。一条事实最远走几跳？chain 拓扑下两跳需要两轮，三轮只够传一次（见 `fix.md` 缺陷八）
- **降级链**。同一条事实沿路径连续被弱化的链有多长？这是纯粹的新东西，Jaccard 里不存在
- **源头信用**。一条最终被多人接受的事实，最初是谁说的？现在的实现用字母序 tie-break，那是个已知 artifact（`analyze_effective_structure.py` 里的 `obs_credit` vs `exp_credit`）

**把 agent 当算子看。** 每个 (agent, round) 是一个作用在事实集上的算子：输入是「自己上一轮输出 ∪ 可见同伴输出 ∪ 自己拿到的材料」，输出是这一轮说的。分解输出的来源构成（转述源材料 / 转述同伴 / 自持 / 新造）就得到这个算子的画像。已有的一个观察：**full-specialist 条件下 A 和 B 的新造率跨轮几乎不变（22%→21%、34%→35%），而 generic 三个 agent 起点相近然后一起塌**——角色差异表现为算子画像的离散度，而且能跨轮存活。样本极小（一个病人、两个任务），只能当假设。

**别忘了对照 MMLU-Pro。** 那是对称信息的基线：没有私有信息，所有 agent 看到同样的东西。任何在不对称数据集上观察到的结构效应，都要能说明它在对称基线上不出现。

## 必须遵守

**别给部分数据打补丁。** 要么全跑要么不跑。局部重跑会引入看不见的偏差——除非你能证明重跑一部分和重跑全部结果完全一致。

**别把已有产物当真理。** `fix.md` 列的缺陷是真的，抽取和匹配都会错。任何结论出来之后，抽样读原文核对再报。

**报数时说清楚阈值是怎么来的。** 现在所有的准确率都是在同一批 oracle 上拟合又评估的，是乐观上界。要写进论文需要独立的调阈子集，那个还没做。

**这是共享仓库，有别的 agent 在同一个工作区写代码。** 提交前先 `git status` 看清楚哪些改动不是你的，只 `git add` 自己的文件，永远不要 `git add -A`，永远不要 `git stash -u`。
