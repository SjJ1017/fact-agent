# 事实人口学：生成–淘汰轨迹区分了准确率相同的配置

*2026-08-28 · MMLU-Pro 难题子集 · deepseek-v4-flash via OpenCode Go*

网页版图表：https://claude.ai/code/artifact/ae8c4cea-e482-40c7-a445-e3be2ece8f67

---

## 1. 结论

两条，第二条更重要。

**(a) 分歧 persona 生得多、也杀得多。** 三个不同专业角色的 panel，第一轮产出的不同
事实是三个同质通用 agent 的 **2.2 倍**（23.6 vs 10.6），第二轮淘汰掉的是 **3.1 倍**
（15.5 vs 5.0）。开局就分歧，不是慢慢积累出来的。

**(b) 准确率相同的两个配置，事实轨迹完全不同。** `chain-generalist` 和
`full-generalist` 的准确率**一模一样**（都是 53.2%），任何 endpoint 指标都会判它们等价。
但第 1 轮事实的存活率一个是 **0.95**、一个是 **0.46**。链式配置几乎不淘汰任何事实——
下游 agent 只看到上游一个人的输出，没有第二个来源可以反驳，只能承接。
**它不是在辩论，是在传声。**

(b) 是这次唯一一个 accuracy 给不出、而且不能从拓扑定义直接推出来的东西。
"链式传播慢"是可预测的；"链式三轮约等于一轮，因为没有淘汰机制"是实测结果。

---

## 2. 数据在哪

```
/Users/shenjiajun/Desktop/EPFL/AXA/project/factflow/experiments/mmlu-pro_out/
```

| 文件 | 数量 | 内容 |
|---|---|---|
| `mmlu-<qid>-<config>.debate.json` | 156 | 原始 transcript、roles、final、`correct` 标签 |
| `mmlu-<qid>-<config>.store.json` | 24 | 抽取 + 匹配好的事实（`FactStore`） |
| `summary.json` | 1 | 47 题全集的准确率 |

**这个目录被 `.gitignore` 忽略，只在本机。** 见 [findings/README.md](README.md) 最后一节。

- **模型**：`deepseek-v4-flash`（OpenCode Go），抽取和匹配用同一个
- **题目**：MMLU-Pro，经 `run_qa.py --screen` 筛出的难题子集，47 题；
  其中 8 题（`mmlu-3526` … `mmlu-3533`）跑了事实抽取，所以人口学的表是 **n=8**
- **配置**：3 种 × 3 轮 × 3 个 agent

| config | 拓扑 | persona |
|---|---|---|
| `chain-generalist` | 链式 A→B→C | 三个相同的通用 agent |
| `full-generalist` | 全连接 | 三个相同的通用 agent |
| `full-specialists` | 全连接 | internist / pharmacologist / pathophysiologist |

准确率（47 题全集，`summary.json`）：单 agent 42.6% ｜ chain-generalist 53.2% ｜
full-generalist 53.2% ｜ full-specialists 66.0%

---

## 3. 怎么重跑

算完的数字已经存在 [`findings/data/mmlu-pro-demography.json`](data/mmlu-pro-demography.json)（已提交）。
重新生成：

```bash
cd /Users/shenjiajun/Desktop/EPFL/AXA/project/factflow
.venv/bin/python experiments/analyze_demography.py experiments/mmlu-pro_out \
    --json findings/data/mmlu-pro-demography.json
```

脚本 [`experiments/analyze_demography.py`](../experiments/analyze_demography.py) 会打印本文所有的表。
它依赖的两个模块：

| 模块 | 提供 |
|---|---|
| [`src/factflow/aggregate.py`](../src/factflow/aggregate.py) | `trajectory` 每轮 born/carried/died ｜ `survival_curve` 队列存活 ｜ `mean_trajectory` 按轮次对齐平均 ｜ `signature` 定长 run 向量 |
| [`src/factflow/graph.py`](../src/factflow/graph.py) | 事实蕴含图 / 时空图 / role 图的中心性 ｜ `load_bearing` ｜ `fragility` |

测试：`.venv/bin/pytest tests/test_aggregate_graph.py`（19 个，全离线）

要重新产生原始 trace（会花额度）：

```bash
# --screen N: 先用单 agent 跑 N 题, 只保留答错的 (加等量答对的作对照), 得到难题子集
# --trace:    跑完 debate 后再抽取 + 匹配事实, 产出 .store.json (慢, 花额度)
.venv/bin/python experiments/run_qa.py --dataset mmlu-pro \
    --provider opencode --model deepseek-v4-flash \
    --screen 200 -n 47 --rounds 3 --trace \
    --configs full/generalist,chain/generalist,full/specialists
```

没有 `--trace` 只会产出 `.debate.json`（trace + 对错），不会有事实。
本文的 24 个 `.store.json` 就是这么来的——47 题的 debate 全跑了，
只有 8 题带 `--trace` 抽了事实。

---

## 4. 结果

### 4.1 每轮事实人口（8 题平均）

`alive = born + carried`。`died` = 上一轮说过、本轮无人再提。

| config | r | alive | born | carried | died | cumulative | r1 矛盾数 |
|---|---|---|---|---|---|---|---|
| chain-generalist | 1 | 9.4 | 9.4 | 0.0 | 0.0 | 9.4 | 2.0 |
| | 2 | 24.2 | 15.5 | 8.8 | 0.6 | 24.9 | |
| | 3 | 24.4 | 8.9 | 15.1 | 9.1 | 33.8 | |
| full-generalist | 1 | 10.6 | 10.6 | 0.0 | 0.0 | 10.6 | 2.8 |
| | 2 | 26.2 | 20.6 | 5.6 | 5.0 | 31.2 | |
| | 3 | 32.6 | 20.9 | 11.1 | 15.1 | 52.1 | |
| full-specialists | 1 | **23.6** | **23.6** | 0.0 | 0.0 | 23.6 | **5.9** |
| | 2 | 25.6 | 17.5 | 8.1 | **15.5** | 41.1 | |
| | 3 | 26.8 | 13.4 | 11.9 | 13.8 | 54.5 | |

三个配置的**事实总产量接近**（33.8 / 52.1 / 54.5），但路径相反：
分歧 persona 是「开局爆发 + 大规模剪枝」，链式是「缓慢生成 + 几乎不剪」。

### 4.2 队列生存曲线

第 b 轮出生的事实，到第 r 轮还有多大比例在场：

| config | born r1 → r2 → r3 | born r2 → r3 |
|---|---|---|
| chain-generalist | 1.00 → 0.93 → **0.95** | 1.00 → 0.43 |
| full-generalist | 1.00 → 0.57 → 0.46 | 1.00 → 0.37 |
| full-specialists | 1.00 → 0.41 → 0.44 | 1.00 → **0.23** |

**这张表是结论 (b) 的全部证据。** chain 和 full-generalist 准确率相同，存活率 0.95 vs 0.46。

### 4.3 Run signature（定长向量，可跨 run 平均）

| config | n_facts | injected | multi 支持 | 最终存活 | contested | redundancy | **echo_rate** |
|---|---|---|---|---|---|---|---|
| chain-generalist | 33.8 | 0.963 | 0.321 | 0.729 | 0.153 | 0.627 | **0.094** |
| full-generalist | 52.1 | 0.979 | 0.298 | 0.649 | 0.154 | 0.524 | **0.206** |
| full-specialists | 54.5 | 0.971 | 0.213 | 0.529 | 0.169 | 0.507 | **0.457** |

`echo_rate` = 被多个 agent 说过的事实里，只有**一个独立来源**的比例；
也就是"看起来是共识、其实是回声"的那部分。链式最低（0.09）是因为它根本没有
"多人同时说同一件事"的场面。

具体例子（`mmlu-3526-full-specialists`，答对）——两个事实**最后都是三人一致**，
多数投票看到的都是"三票赞成"，但：

```
ECHO   C1 A2 C2 A3 B3            "A covalent network solid has a high melting point."
INDEP  A1 B1 C1 A2 B2 C2 B3 C3   "Poor electrical conductivity rules out metallic bonding."
```

第一条是 C 一人在 r1 提出、其余两人读到后跟随的**回声链**，独立来源只有 1 个，
C 错则全盘错；第二条是三人在 r1 各自独立说出的，是真正的三重支持。
这个区分只有 fact flow 能给。

### 4.4 成本

| config | est. tokens / 新事实 | est. tokens / 存活事实 | 准确率 |
|---|---|---|---|
| chain-generalist | 28 | 39 | 53.2% |
| full-generalist | 20 | 32 | 53.2% |
| full-specialists | 19 | 39 | 66.0% |

分歧 persona 每个新事实最便宜，但每个*存活*事实和链式一样贵——因为它把大半扔了。
而准确率高 13 个点。差别不在存活事实的**数量**，在于**经过淘汰筛选**这件事本身。

---

## 5. 站不住的地方

**n = 8。** 每个配置只错 1 题，而且三个配置错的是同一题（`mmlu-3531`）。
配置层面的差异干净，但"用轨迹预测单题对错"在这个样本量下**完全没有证据**——
`analyze_demography.py` 会打印 churn 的对错分组，那几个数不要引用。

**成本轴是估算的。** `run_qa.py` 没有记录真实 token，脚本 fallback 到
`输出字符数 ÷ 4`。相对比较可信，绝对值不可信。
**待办：给 `run_qa.py` 加 usage 记录**，脚本已经准备好读 `debate.json` 里的 `usage` 字段。

**specialists 的角色是错配的。** internist / pharmacologist / pathophysiologist
在讨论共价网络晶体——这是 MedQA 的角色被复用到 MMLU-Pro 时没有替换。
它仍然是准确率最高的配置。两种解释：
① role prompt 的**内容**基本是装饰，起作用的是"persona 之间不一样"；
② 领域相关的角色会更好，这里只是碰巧。
**这两种解释指向完全不同的结论，目前无法区分。**

**97% 的事实是 `injected`。** MMLU-Pro 的"文档"就是题干，没有源材料可传输，
所以每个 run 只有 0.8–1.2 个 gold 事实，`fragility` / `load_bearing` 是在
1 个事实上算的，**是噪声，不要引用**（脚本里那一节标了 meaningless）。
生成–淘汰这条线不依赖 gold，不受影响。

**这里没有 transport。** 早先设定的"知识 QA 已饱和、数学和代码没有信息传输"
的预警依然成立。这个发现走的是另一条路：不看事实从哪传到哪，只看
**各类事实随轮次/成本的生灭**。这条路不需要 asymmetric 的源材料，所以在
inference 型任务上也成立。

---

## 6. 下一步

**优先：区分"分歧本身"和"分歧内容"。** 同一批题跑两组分歧 persona，
一组用领域相关角色（化学家/材料学家/物理学家），一组用随机无关角色
（厨师/会计/园丁）。如果后者的 r1 born 和 churn 与前者相同，
就说明起作用的是**分歧本身**，role prompt 的内容是装饰。
这比换拓扑更能定位机制，47 题 × 2 配置在 OpenCode Go 上成本可控。

其次：
- 给 `run_qa.py` 加真实 token 记录，把成本轴换成真值
- 扩大 n（现在只有 8 题抽了事实，47 题的 trace 都在，缺的只是抽取额度）
- `broker` 中心性目前是平的（A=0.048 B=0.052 C=0.045），全连接 + 对称角色下必然如此。
  要让中心性说话需要非对称拓扑和稀疏连接
- 换到 deliberative-collaboration benchmark（`../dca`），那里 `O_i` 是 ground truth，
  gold 类不再是噪声，`fragility` / `load_bearing` 才有意义
