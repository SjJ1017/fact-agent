# Agent 重合与内容交互：从个人轨迹到关系矩阵

2026-09-07。一次纯离线统计，复用全部 40 个 IDRBench NLI store 与对应 debate。无生成、抽取或模型评分调用。

## 研究落点

核心自变量仍是 persona 与 topology。问题是它们如何改变 agent 之间的关系：内容重合、跨轮可见复用、互惠、输入/输出不对称，以及关系随轮次的变化。

DelibTrace §3.4 的 fAgent 比较个体当前保留集合与自己的初始分配，fSystem 比较团队初始/当前的集合；跨 agent 的种子事实重合原则上可从其集合推算。因此，不写“它数学上不能计算交互”。更准确的区分是：追踪全部产出（含新生成命题）的跨 agent 直接语义关系，叠上真实可见历史，形成动态、有类型的关系矩阵。[原文](https://arxiv.org/html/2606.03032v1)

## 三种矩阵

1. O_t：同轮输出重合。保存同簇 Jaccard 与直接 EQ 的双侧平均覆盖率。二者不是同一个指标；不要用不同分母的数值直接相减。
2. I_t：相邻轮次 i→j。前一轮 i 的具体 mention 必须实际可见于 j，且与 j 下一轮 mention 存在直接已评分关系。按目标 turn 的 fact ID 去重，除以目标输出事实数。每一方向独立，目标可同时关联多个发送者，行列不必和为 1。
3. I_t 的 EQ/weakening/strengthening 分量。一个目标可能有不同类型的输入边，类型比例不保证互斥。

另保存 screened I：若源端或目标端与接收者实际可见的自身历史同簇或存在直接 EQ，就剔除该候选。这个筛选减少已检测到的自持/既有重合，但没有去除独立重推、源材料共同支持、未抽出或漏匹配的自持。不能称为纯净的新知识采纳或因果交互。

时间尺度固定为 r→r+1，避免 R3 多一轮 cumulative memory 直接扩大来源窗口。历史筛选使用完整 visible_self_turns。源端输出为空时没有边；目标为空时比率为缺失。全部 40 场保留，空提取共 6 turn，分布在 split 条件。

## 初步描述

每场先平均 agent 对，再平均该条件 10 个任务。第一列是 R3 同轮直接 EQ 双侧覆盖，第二列是 R2→R3 screened 目标覆盖。两列分别衡量同时相似与时序关联，分母不同；不是要比较它们的绝对大小。

| 信息/角色条件（全部 full 通信） | 同轮等价覆盖 | 筛选后的跨轮候选关联 |
|---|---:|---:|
| full-generic | 18.8% | 23.2% |
| full-specialist | 15.1% | 23.1% |
| split-generic | 14.7% | 24.8% |
| split-specialist | 13.6% | 19.7% |

这些均为探索性候选图统计，尚未证明条件差异稳定。值得追的是两类指标不必同步变化，例如 full 的 specialist 同轮重合较低，候选跨轮关联量与 generic 接近。不能据十场均值直接宣称专家提高了协作。

## 原文初读后的限制

审阅集中在自动保存的样例，非随机误差率评估：

- task12/full-generic，A2→C3，“cluster shape over time”→“cluster shape”：C1 原文已提过 cluster size/shape。screened 筛选没有识别这处原文已有内容；抽取/匹配漏检会虚高“此前未表达”的候选。
- task12/full-specialist，C1→A2，“Boundary masks are saved”→“Boundary masks are saved from simulations”：源原文已有模拟上下文，因此新增“from simulations”的方向差异至少部分由抽取上下文丢失造成，不能解释成真实细化。
- task12/split-specialist，C2→A3，训练中变化 domain shapes 的 EQ 样例在原文成立。两轮还可见 every 50 Monte Carlo steps 等具体方案对应，而 A2 原文是 at each tissue update。它支持存在具体内容复用候选，仍不单凭相似确定来源因果性。

因此，目前可以报告量的定义与候选分布，不能把 screened I 直接写成真实交互率。这些后验核对没有阻断全语料计算，也没有重跑局部数据。

## 通用的后续问题

- 人格差异能否降低重复，同时保持跨 agent 的内容交换？
- 初始重合 O_t 与下一轮 I_t 的关系：重合低是否真的意味着更有可交换内容，还是关系会减弱？不预设单调或倒 U。
- 相同总关联量的团队，是双向往来，还是少数 agent 持续向其他人输出？
- 配置是 full，实际 I_t 是否长期集中在某个 agent 对或中心？三人图先看加权矩阵与集中度，不强做社区发现。
- 关系是否随轮次从单向转为双向，还是始终由同一节点占据主要输入/输出位置？
- 等价互动很少但单向关联多的 agent 对，是否被二值等价指标错误地描述为没有交互？

“重合×交互”四象限用于提出假设：高重合/低新增关联是冗余候选；低重合/低关联是并行产出候选；低重合/高关联是互补交换候选；高重合/高关联是共同维护候选。它们不是自动质量标签。

## 产物与复现

运行 `python3 experiments/analyze_idrbench_interaction.py`。输出 `findings/data/idrbench-interaction/`：

- overlap.csv：360 个同轮 agent 对。
- interaction.csv：480 个相邻轮次有向 agent 对。
- per-trace.csv：40 场汇总。
- matrices.csv：各条件两次轮间的 3×3 候选关系矩阵。
- summary.json：均值、矩阵、输入路径与 matching 配置。
- examples.json：120 个自动截取的候选，附源/目标原文及目标自身可见历史；不是人工 gold。

直接 NLI 共用阈值 5.28，block threshold 0.62/top-k 12，原簇 union-min 0.85。历史 oracle 准确率来自同集拟合评估，是乐观上界。缺失候选不能当作经判定的无关，筛选后的“未匹配自身”不能当作经证明的首次表达。
