# 事实流剖面：三拓扑 × 三档人格

2026-09-01。报告页：https://claude.ai/code/artifact/756f477f-f148-409e-9434-301220ae3f56

12 个 claim × 3 档人格 × 3 种拓扑 = 108 场，全部 deepseek-v4-flash。
GLM 那批只有全连接、没有 star/chain 对照，**不进入本分析**。
**chain 的立场标注还在跑**，元陈述占比和正反平衡两行暂缺，其余指标完整。

## 结论

1. **拓扑只买到"听到的机会"。** full→chain 接收率 −24.1pp（区间几乎没宽度），
   条件采纳率只动 −5.2pp。star→full 接收率 **0.0pp**——star 少的是来源多样性，
   不是有没有来源，这解释了 star 和 full 在几乎所有指标上测不出差别。
   真正起作用的线路变量是**接收率**，不是边数。
2. **"人格越强产出的事实越多"是长度效应。** stance 比 neutral 多写 15% 的字
   （full 1348→1569 token）。原始事实数 +21.0 [+11.3,+30.3]；
   按 token 归一化后 +5.90 [−0.74,+11.94]，p=0.96——**仍然偏正，但效应被砍掉大半**。
3. **信息密度是倒 U，在 lenses 见顶。** lenses−neutral 事实密度 +15.97 [+6.95,+23.95]
   p=0.999（11/12 为正）；stance−lenses −10.06 [−16.62,−2.45] p=0.006。
   full 走势 53.7 → 69.7 → 59.6。给视角让 agent 看到不同的东西；
   再给立场只是把同样的东西说得更用力。
4. **采纳一路向下，归一化后依然成立。** 采纳密度 stance−neutral：
   full −2.65 p=0.017，star −3.04 p=0.008。条件采纳率：full −4.7pp p=0.008。
5. **人格效应要靠接触才表达。** 同一个 stance−neutral，chain 上处处更靠近零：
   事实数 +21.0 / +20.1 / **+6.8**；自持占比 −4.5 / −4.5 / **−0.7**；
   条件采纳率 −4.7 / −4.6 / **−1.2**。只有嵌套度和事实覆盖在 chain 上还留着，
   约为 full 的三分之一到一半。

## 统计口径（重要）

配对 bootstrap，报**点估计 + 95% 区间 + p**（bootstrap 均值落在零以上的比例）。

**读 p，不要读"区间是否跨零"。** n=12 时两者能差很远：+5.90 [−0.74,+11.94]
区间跨零，但 96% 的 bootstrap 质量在零以上、8/12 claim 为正。
把它叫"没有效应"等于扔掉绝大部分信息。p=0.96 偏正的程度和 p=0.04 偏负的程度一样。
`excludes_zero` 字段保留，但只是"95% 区间的字面意思"，不是判据。

没有做多重比较校正。可信的是**同时**在多个拓扑、多个相关指标上一致出现的模式。

## 数据在哪

| 东西 | 位置 | 在 git |
|---|---|---|
| full store | `experiments/perspectrum_pilot_full/*deepseek-v4-flash-full-*.v2.json` (36) | ✅ |
| star / chain store | `experiments/perspectrum_pilot_star_chain/*.v2.json` (72) | ✅ |
| 投递结构 | 同目录同名 `.debate.json` 的 `delivery` 字段 | ✅ |
| 立场标注 | `experiments/labels/stance/*.json` | ✅ full+star，chain 跑完补 |
| token clock | `experiments/labels/token_clock/*.json` | ✅ |
| 结果 | `findings/data/flow-profile.json`、`flow-profile-per-debate.csv` | ✅ |

## 怎么复现

```bash
python experiments/analyze_flow_profile.py --topologies full star chain
python experiments/report/build_flow_profile.py
```

不联网、不花钱。token 由 bge-base-en-v1.5 重新分词 transcript 得到——
**不要用 token_clock 层的 `turn_output_tokens` 当分母**，那层按"事实首次出现的 turn"
记数，抽不出事实的 turn 就不计（star 14 个，chain 更多），分母正好偏在最不该偏的地方。
重新分词在该层覆盖到的每个 turn 上和它逐个数字一致。

## 三种拓扑

```
full   12 次投递  A↔B↔C 全连接        接收率 66.7% (6/9 turn 有输入)
star    8 次投递  B、C 只经中心 A      接收率 66.7%
chain   4 次投递  A→B→C，无回边       接收率 42.6% (A 什么都收不到)
```

## 局限

- chain 立场标注未完成；GLM 无 star/chain 对照
- 抽取会漏 turn：star 14 个（4.3%，full 0.9%），chain 更多。
  同时压低接收率和事实数——star 的 lenses 格接收率 63.0% 而非 66.7% 就是这个
- 立场标注未经人工验证，且约一半有立场的事实是证据层而非世界层
- 事实匹配人工核对准确率 80%；条件间的**差**比每格的**绝对值**可信
- "采纳"只判事实同一性，不判态度
- **接收率只有 66.7 / 66.7 / 42.6 三个取值**，来自三种固定接线，不是连续变量。
  "接收率是起作用的变量"还需要一个能连续调接收率的设计来检验

## 待办

- chain 立场标注跑完 → `export_labels.py` → 重跑分析 → 重发报告
- VERDICT 在 324/324 个 turn 里可解析，还没当作结果变量用过
