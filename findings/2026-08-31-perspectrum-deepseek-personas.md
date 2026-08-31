# Persona 改变事实人口学，不自动增加信息流

*2026-08-31 · Perspectrum · DeepSeek V4 Flash · Claude 重匹配的 v2 FactStore*

## 1. 结论

在共享同一份证据 dossier、full-context 的三人辩论中，persona 多样性确实改变了
fact lifecycle，但**没有增加可观测的跨 agent adoption**。

- `lenses`（因果 / 权衡 / 不确定性）让总事实量从 100.4 提升到 **118.6**，同时第二轮
  birth 增加 8.3、第三轮 death 增加 8.9。这是「更多探索 + 更多剪枝」。
- `stance`（支持者 / 反驳者 / 裁决者）降低了事实共享、独立多源支持和第二轮 cohort 的
  最终存活。它更像是**碎片化**，而不是把不同立场整合成更多共同知识。
- direct adoption（前一轮由 peer 说出、当前轮由一个此前未说过它的 agent 首次表达）在三种
  条件间没有可靠差异：neutral 14.3、lenses 13.3、stance 12.5 次/run。

因此当前证据支持一个更窄、但有用的命题：**persona 是 fact demography 的干预，不是
information transport 的干预。** 要研究知识在 agent 间流动，必须改变可见性拓扑或信息
分布；只在 full-context 下换 persona 不够。

## 2. 数据

原始数据和 rematch 后 store 都在本机，未作为本 finding 的 git 依赖：

```
experiments/perspectrum_pilot_full/
  perspectrum-*-deepseek-v4-flash-full-*.debate.json  # 36 个原始三轮辩论
  perspectrum-*-deepseek-v4-flash-full-*.v2.json      # 36 个 Claude 重匹配 FactStore
```

实际文件前缀为 `perspectrum-`。每个条件覆盖 Perspectrum test split 的同 12 个 claim。
每个 claim 给所有 agent 完全相同的 2 条 SUPPORT 和 2 条 UNDERMINE 证据；没有人为给
agent 分配私有 atomic facts。

| panel | A | B | C |
|---|---|---|---|
| neutral | 中立证据分析 | 中立证据分析 | 中立证据分析 |
| lenses | 因果证据 | 实施/权衡 | 范围/不确定性 |
| stance | 支持 claim | 反驳 claim | 裁决 |

三人均为 `deepseek-v4-flash`，3 轮，full topology。agent 的自然语言输出没有 citation
格式、fact 标记或台账要求。抽取、原子化和匹配在辩论结束后离线完成。

## 3. 怎么重跑

```bash
cd /Users/shenjiajun/Desktop/EPFL/AXA/project/factflow
PYTHONPATH=src .venv/bin/python experiments/analyze_perspectrum_personas.py \
  experiments/perspectrum_pilot_full \
  --out-json findings/data/perspectrum-deepseek-personas.json
```

脚本按 claim 配对 `lenses - neutral` 与 `stance - neutral`，报告 20,000 次 paired bootstrap
区间。`adoption` 直接按 agent id 算，而不使用 role label：neutral 的三个角色文本相同，
按 role 聚合会错误地把 peer edge 算成 self edge。

## 4. 结果

### 4.1 事实人口（12 claim 平均）

| panel | output facts | r1 alive | r2 born | r3 died | r2 -> r3 存活 | shared | multi-source | redundancy | adoption |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neutral | 100.4 | 36.3 | 35.2 | 31.2 | 20.4% | 15.4% | 4.9% | 21.0% | 14.3 |
| lenses | **118.6** | 39.7 | **43.5** | **40.1** | 14.8% | 13.6% | 4.4% | 18.4% | 13.3 |
| stance | 113.6 | **42.0** | 38.5 | 36.2 | **12.7%** | **10.7%** | **2.7%** | **16.5%** | 12.5 |

`shared` 是至少两个 agent 曾表达的事实比例；`multi-source` 要求这些 agent 在任何 peer
说出之前就独立表达过。`adoption` 只记可由上一轮 peer 输出解释的新表达，不能被当前 agent
自己前一轮已说过的事实冒充。

### 4.2 配对比较（相对于 neutral）

| 比较 | 指标 | 差值 | paired bootstrap 95% |
|---|---|---:|---:|
| lenses | 总事实 | +18.2 | [+1.2, +35.3] |
| lenses | r2 born | +8.3 | [+0.5, +17.1] |
| lenses | r3 died | +8.9 | [+0.8, +17.2] |
| lenses | adoption | -0.9 | [-6.6, +3.9] |
| stance | 总事实 | +13.2 | [+0.9, +27.7] |
| stance | shared | -4.7 pp | [-7.5, -1.7] |
| stance | multi-source | -2.2 pp | [-4.0, -0.5] |
| stance | r2 -> r3 存活 | -7.7 pp | [-15.1, -1.2] |
| stance | adoption | -1.8 | [-6.8, +3.5] |

`lenses` 的确定信号是 birth/death 同时上升，而不是更高的跨 agent 接受率。它把 discussion
空间打开，然后在下一轮淘汰更多。`stance` 的确定信号是 shared 与 multi-source 下降：它让
三个 agent 更少停在同一组事实上，且较少出现真正独立的重合支持。

这也给了对「分歧比专业更重要吗」更准确的当前回答：**两类 persona 差异都改变事实人口，
但方式不同。** 非对抗的 lenses 是扩张-剪枝；显式对抗的 stance 是低共享-低存活。目前不能
把这一结果解释成“分歧必然胜过专业性”，因为没有正交地控制领域专长和立场分歧。

### 4.3 槽位完整性的敏感性检查

25/36 runs 的 9 个 agent-turn 均有至少一个被抽取到的 fact；10 个有 8/9，1 个有 7/9。
只有 5 个 claim 在三种 persona 下都完整。这个小样本中：

- stance 的 shared 仍低 4.0 pp（5 题中 4 题下降）；r2 -> r3 存活低 10.3 pp（4 题下降）。
- stance 的总事实量差异变成 -1.6，方向不稳。

所以**不要引用“stance 产生更多事实”**作为稳健结论；它受 atomic extraction coverage
影响。共享和存活下降的方向较一致，但仍需要更完整的重抽取才能称为强证据。

## 5. 站不住的地方

1. **没有 source attribution 结论。** v2 source-to-output 匹配率只有约 0.7%–1.2%，
   这说明 source grounding 还没有经过可靠审计，而不说明 99% 的 agent 内容是 hallucination。
2. **没有 conflict-resolution 结论。** 36 个 store 的 `CONTRADICTS` relation 为 0；当前
   matcher 把支持和反驳更多编码为不同命题，而非显式逻辑矛盾。不能用这批数据说 persona
   解决了或加剧了矛盾。
3. **不是私有知识传输测试。** 所有人一开始就看到同一 dossier；这里的 adoption 是公共讨论中
   一个已表达 fact 被另一个 agent 采用，不是“弱模型从强模型获得原本看不到的世界知识”。
4. **n=12，且一个 extractor/匹配器。** paired bootstrap 只表达这个固定 claim 样本的稳定性，
   不是跨数据集泛化保证。

## 6. 下一步

最直接的拓扑实验不应再问“persona 有没有用”，而应检验机制：固定 `neutral` 和 `stance`
两种 panel，在同一 Perspectrum claim 上比较 `full / ring / star`。预测是：若 persona 主要
作用于表达选择，full 下已经看到 churn 差异但 adoption 不变；稀疏 topology 才会让 adoption、
cohort survival 与 broker dependence 拉开。这样可以把「事实人口学」和「信息运输」分成两个
独立、可检验的轴。
