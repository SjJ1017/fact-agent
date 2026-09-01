# 主假设：多立场只提高探索，整合能力决定它是否转化为质量

*2026-09-01 · 待办清单，非结论*

## 假设

多立场 persona 本身只提高事实**探索**。只有当通信结构具备**整合能力**——能汇聚、
对照、限定并保留这些事实——它才转化为更高质量的辩论。否则多样性退化为
"垃圾多样性"：高产生率、高淘汰率、低吸收率。

## 已有证据支持前半句

全部在 all-to-all 拓扑、12 道 Perspectrum claim、deepseek-v4-flash 上。

| 指标 | lenses vs neutral | stance vs neutral |
|---|---|---|
| 每千 token 事实数 | +20.5 | — |
| **观察到的 uptake** | **−6.1 pp [−11.5, −0.2]** | **−6.9 pp [−13.0, −1.4]** |
| 最终正反平衡 B± | −0.011 [−0.174, 0.170] | **+0.176 [0.049, 0.320]** |
| 元陈述占比 | 38.1% | 30.9%（neutral 41.1%） |

两个 persona 条件的 uptake 都显著下降且 CI 不含零。**高产出、低吸收**在数据里已经存在。

见 [token lifecycle](2026-08-31-deepseek-token-lifecycle.md) 与
[stance balance](2026-08-31-stance-balance.md)。

## 但框架有一处必须改：自变量不是拓扑

**all-to-all 是连通性最高的拓扑，而它的 uptake 恰恰是低的。**

如果整合能力来自图的密度，all-to-all 应当整合最强。它不是。所以变量是
**有没有一个角色/步骤负责整合**，不是图的形状。

原框架把 centralized 与 two-party+synthesis 归为"强整合"、chain 归为"弱整合"，
暗含 整合能力 ≈ 连通性。改为：

```
自变量        integration mechanism
取值          none / synthesizer / adversarial + synthesis
拓扑          它的一种实现方式, 不是它本身
```

这样 **all-to-all 成为"高连通、零整合"的对照组**——正是把"信息可达"与
"信息被整合"分开所需要的那一格。

## 三个卡点，按优先级

### 1. Q 不存在（关键路径）

整个 `ΔQ/ΔH` 建立在质量测量上。`add_gold.py` 现在报 `reached 0/41`——
Perspectrum 的人工标注论点一条都没匹配进 store，gold 覆盖率那条线是坏的。

**没有 Q，垃圾多样性与有效多样性在数据上无法区分**，假设不可证伪。
这比增加拓扑条件重要得多。

### 2. shared dossier 下的 uptake 不是 transport

三个 agent 看同一份四条证据。"B 说了 A 说过的话"完全可能是各自从同一条证据
独立推出。设计文档已标注这一点。

**integration yield 要有意义，证据必须在 agent 之间分开**，否则测的是巧合率。
候选做法：每 agent 私有证据子集，或稀疏可见性。

### 3. "垃圾多样性"的定义与其自身注解冲突

原式 `early diversity × (1 − uptake) × death rate` 把 death 放在分子，
等于假定死亡是坏的；但注解正确地指出"被反驳后带限定重生"是 productive pruning。

三项相乘还让解释变难：三项都中等的 cell 与"一项极端两项接近零"的 cell 得分相同，
含义完全不同。

**建议改为两项**：`early diversity` 高 × `integration yield` 低。
死亡率通过 integration yield 的分母间接进入，且自动排除"被转化"的那部分。

```
integration yield = 最终保留或被实质性转化的早期 facts / 早期产生的 unique facts
实质性转化 = 被重申 / 被另一方吸收 / 被明确反驳后形成带限定的新事实
```

## 样本量

3 × 4 = 12 个 cell。今天在 12 claim × 3 persona 上，**十二次比较只有一个 CI 不含零**。
而**交互效应比主效应更难检出**，需要的正是交互。

12 道题铺到 12 个 cell、每 cell 一个 run，检不出任何东西。两条路：

- 大幅增加 claim 数（Perspectrum 有数百个）
- 先做 **2×2**（stance/neutral × 有综合者/无），测出效应量再扩

## 建议顺序

```
1. 修 add_gold 的 reached 0/41              → Q 可测
2. 证据在 agent 间分开                       → uptake 变成真的 transport
3. 2×2 确认交互效应存在且量级够大             → 再决定要不要 3×4
```

前两件都是当前流水线的缺陷。做完之后即使假设不成立，也得到了一个能测
"多样性是否被转化"的工具——那本身就是这套方法要证明的价值。

## 待核对

- **Beyond Frameworks** 被引用为 centralization 有效的证据。
  引用前需回去核对：它测的是什么任务、centralization 的定义是否与此处一致。
  不要凭记忆断言。
- **two-party 拆成两条件**（有/无 final synthesis）是好设计，
  它把"对抗产生的多样性"与"是否被整合节点转化"分开。保留。
