# RQ review 与新增分析：接触、选择门与功能性整合

> 本文只记录相对 `effective-structure.html` 和 `2026-09-01-flow-profile.html`
> 新增的分析与推论，不复述两页已有结果。全部计算为事后离线分析，fact extraction、
> matching 和 stance labels 从未进入 debate prompt。

## 一句话结论

现有数据支持把研究主线从“persona 是否增加事实多样性”升级为：
**persona 先改变发现什么，网络决定能否接触，接收者再决定是否采纳，而真正的整合还要看采纳后是否保留并影响结果。**
chain 证明 persona 的事实增益依赖接触；两阶段漏斗证明低 uptake 不自动等于低整合；
star 则说明结构中心不自动成为功能性整合者。

## 数据与重算

- DeepSeek-V4-Flash：12 claims × 3 personas × 3 topologies = 108 场。
- full stores：`experiments/perspectrum_pilot_full/*-full-*.v2.json`，36 场。
- star/chain stores：`experiments/perspectrum_pilot_star_chain/*.v2.json`，72 场。
- 新增结果：`findings/data/rq-extensions.json`。
- effective-structure 的 chain 扩展：`findings/data/effective-structure.json`。

```bash
python experiments/analyze_effective_structure.py
python experiments/report/build_effective_structure.py
python experiments/analyze_rq_extensions.py
```

三个命令均离线、不联网、不调用模型。

## RQ 应该怎样改写

原先三个问题仍然成立，但第三问必须拆开：

1. **Discovery / composition：persona 让团队发现了什么不同的信息？**
   主要指标是 token-normalized fact richness、agent/source diversity、stance entropy/balance。
2. **Exposure / uptake：信息是否真的沿允许的边到达并被复述？**
   必须分别报告接触机会、条件采纳率、传输深度，不能用一个 aggregate uptake 混在一起。
3. **Integration：被采纳的信息是否留在接收者后续推理和最终组合里？**
   用 adoption→retention 与最终覆盖衡量，不能把即时复述直接称为整合。
4. **Utility / grounding：留下的信息是否有证据支持，并改善了结果？**
   这需要外部 outcome 或可靠 source entailment。当前 VERDICT 是 agent 自报立场，不是质量标签；
   当前 3–7% grounding match 又只是严重 under-match 的下界，因此这批数据不能回答第 4 问。

这四层分别是“看见更多”“传过去”“留得住”“有用”。将它们压成一个 scoreboard 总分会丢掉机制。

## 新结果一：chain 显著削弱 stance 的事实增益

做 claim-paired difference-in-differences：

`[(stance-neutral)_chain - (stance-neutral)_full]`

| 指标 | 交互差 | 95% bootstrap CI | 同方向 claims |
|---|---:|---:|---:|
| 事实数 | **−14.17** | **[−28.92, −4.50]** | 12/12 为负 |
| facts / kToken | **−8.59** | **[−18.80, −0.60]** | 10/12 为负 |
| 条件采纳率 | +3.57pp | [−2.73, +10.66] | 5/12 为正 |
| 嵌套度 | +0.058 | [−0.004, +0.123] | 8/12 为正 |
| 正反平衡 | −0.140 | [−0.375, +0.037] | 4/12 为正 |

full 中 stance−neutral 增加 21.0 个事实，chain 只增加 6.8 个；按 token 归一化后，
full 仍为 +5.90 facts/kToken，chain 变成 −2.68。star 与 full 的 stance 事实增益几乎相同，
对应交互只有 −0.92 个事实、−0.49 facts/kToken，区间都宽跨零。

**解释：**persona 的“丰富化”不是独立于交互结构的固定属性。full/star 保持每个后续 turn
都有 peer input，stance 会明显改变事实预算；chain 中 A 永远收不到、全局只有 4 次投递，
persona 对事实数量的增益显著衰减。这支持“persona 效应通过接触表达”，但尚不能区分
是上下文量、来源数还是角色位置造成的。

## 新结果二：采纳后的端到端留存只有 2%–6%

只看第 2 轮，定义一个严格的三段漏斗：

- exposed：peer 的第 1 轮事实已经投递给该 agent，且 agent 自己此前没说过；
- adopted：agent 在第 2 轮复述该事实；
- retained：同一个 agent 第 3 轮仍说该事实。

下表是每场比例的均值：

| topology/persona | exposed→adopted | adopted→retained | exposed→retained |
|---|---:|---:|---:|
| full / neutral | 15.0% | 28.7% | 4.3% |
| full / lenses | 10.4% | 27.9% | 3.4% |
| full / stance | 10.4% | 39.0% | 4.6% |
| star / neutral | 19.3% | 25.7% | 4.4% |
| star / lenses | 11.9% | 15.5% | 2.1% |
| star / stance | 10.8% | 19.8% | 2.8% |
| chain / neutral | 19.0% | 20.8% | 5.9% |
| chain / lenses | 15.7% | 27.2% | 4.2% |
| chain / stance | 14.2% | 35.0% | 4.2% |

stance−neutral 的即时 uptake 在 full 为 −4.62pp [−9.52, −0.14]，star 为
−8.48pp [−14.65, −1.83]；chain 为 −4.79pp，但区间 [−15.72, +6.98] 很宽。
然而，三个拓扑的 **post-adoption retention 差和 end-to-end 差全部跨零**。

这修正了“stance uptake 低，所以整合更差”的过强说法：stance 主要改变入口选择，
不一定让已经接住的事实更快死亡。full/stance 甚至以更高的后续留存抵消了较低 uptake，
使端到端率与 neutral 基本相同。因而“垃圾多样性”至少应定义成
**新增但未被采纳/保留、且不改善结果的事实**，不能只用低 uptake 或高死亡率单独判定。

## 新结果三：chain 的主要瓶颈在第一跳，不是逐跳均匀衰减

用 neutral 条件避免角色差异。取 A 在第 1 轮独有、B/C 均未说过的 102 个事实：

- 10/102（9.8%）在 B 的第 2 轮出现；
- 其中 7 个又在 C 的第 3 轮首次出现；
- 所以可观察的 A→B→C 两跳到达率为 7/102（6.9%），但在已经通过第一跳的事实中为 7/10。
- 作为相邻边对照，B 的 102 个第 1 轮独有事实有 14 个在 C 第 2 轮出现（13.7%）。

这不是“每多一跳固定衰减一次”。更像一个强选择门：绝大多数事实第一跳就消失，
少数被 B 选中的事实很容易继续级联。一个合理的新对象是 **cascade fitness**：
哪些事实属性能预测它通过第一跳，并在通过后继续传播。

但 7/10 样本很小，而且三人共享 dossier；严格时间顺序只能说明传输路径与观察相容，
不能排除 C 在第 3 轮独立重推。要识别因果传输，需要 private-information 或遮蔽消融。

## 新结果四：大量 late facts 更像事后理由，不像改判证据

108 场中有 105 场在第 1、3 轮都存在明确多数；其中 89 场多数 verdict 不变。
在这 89 场里，81 场（91%）同时满足“最终事实中至少一半到第 3 轮才首次出现”。
这个组合在九个条件格中都出现，不由某一个 topology/persona 单独驱动。

按 topology×persona 条件格去中心化后，stable 与 changed verdict 在以下指标上的差值
全部跨零：late-final share −0.045 [−0.105, +0.022]、r1 survival +0.022
[−0.023, +0.065]、adopted share +0.014 [−0.007, +0.036]、端到端传输
−0.010 [−0.040, +0.016]。终局是否 unanimous 也没有稳定的 cell-adjusted flow 差异。

因此当前最稳妥的读法是：**事实组合持续更新，但这种更新没有可测地连接到 verdict 更新。**
这可能是 post-hoc rationalization，也可能只是 verdict 三分类太粗；没有外部质量分数前不能二选一。

## 对“中心化能吸收多立场”假设的修订

当前 star 不能被当作“有整合者”的实验。它只有结构中心 A，没有让 A 执行综合、资格化、
证据审计或最终裁决；在 stance 条件里 A 还固定是支持方。旧页已经显示真实跨枝干中继每场
不到一条，新分析又显示 star 与 full 的 persona 事实增益几乎没有交互。

因此应区分：

- **structural centralization**：谁连接得多；
- **functional integration**：谁被要求比较冲突、保留关键限定、产出共享状态；
- **integration capacity**：该角色能处理多少异质输入而不丢失。

现有结果反驳的是“中心节点自然会整合”，没有反驳“被赋予整合功能的中心能减少垃圾多样性”。
这反而给出一个更干净的下一实验：固定 star 图，只操纵 hub 是否有 integrator prompt，
并轮换 stance/lens 与 A/B/C 的绑定。主要结果应是 retained balanced coverage 和外部 judge quality，
而不是总事实数。

## 最小下一步

1. 在现有 12 claims 上先做 `star × {ordinary hub, integrator hub} × role rotation`，不造新数据集。
2. 预注册四层指标，不合成总分：discovery、conditional uptake、retention、external utility。
3. 为 Perspectrum 增加独立 outcome：盲评最终回答的证据覆盖、冲突处理和 source support。
4. 加一小组 private-fact masking，用来校准“同事实晚出现”中有多少是真传输、多少是独立重推。
5. 扩大 claims 后再检验 cascade fitness；当前二跳只有 7 个事实，不适合做特征模型。

## 边界

- 只有 DeepSeek 有三拓扑；GLM 不能用于 topology interaction 复现。
- 每格 n=12，差分中的差分是探索性分析，未做多重比较校正。
- stance/lenses 的角色与 A/B/C 位置绑定，只有 neutral 的纯位置比较干净。
- 空抽取 turn：full 3/324、star 14/324、chain 17/324；小的 topology 差可能受此影响。
- exact canonical-fact uptake 会漏掉 qualified paraphrase；这里测到的是保守的显式复用。
