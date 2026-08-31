# 数量测不到的，立场组成测到了

*2026-08-31 · Perspectrum 12 claim × 3 persona · deepseek-v4-flash · all-to-all · 三轮*

## 结论

在配对的 12 道 claim 上，**stance-diverse persona 让最终事实组合的正反平衡度比 neutral
高 +0.176，95% bootstrap 区间 [0.049, 0.320]，不含零。** 这是这批数据上第一个区间不含零的
persona 效应。

同一批 run 上，所有数量类指标——第一轮新生事实数、队列存活率、被多 agent 说过的比例——
的配对差方向都随题目摇摆，区间全部含零。**不是 persona 没有作用，是数量指标看不到它的作用。**

`epistemic lenses` 在所有指标上都与 neutral 不可区分。

## 数据与产物

```
输入   experiments/perspectrum_pilot_full/*deepseek*.v2.json     36 个匹配好的 store
标注   /tmp/factflow-stance-labeled-deepseek-judge/*.stance.json  36 个（本机，非 git 产物）
时钟   /tmp/factflow-token-lifecycle-deepseek.json                token 位置
结果   /tmp/factflow-stance-diversity.json
图     findings/data/factflow-stance-diversity.png
```

复现：

```bash
PYTHONPATH=src .venv/bin/python experiments/analyze_stance_diversity.py \
    /tmp/factflow-stance-labeled-deepseek-judge \
    --token-analysis /tmp/factflow-token-lifecycle-deepseek.json \
    --out-json /tmp/factflow-stance-diversity.json
.venv/bin/python experiments/plot_stance_diversity.py /tmp/factflow-stance-diversity.json
```

标注的口径写在 `label_fact_stance.py` 里，值得复述一遍，因为它决定了这个指标的含义：
**标的是「这条命题如果为真，会往哪个方向影响论点」，不是这条命题是否为真。**
标注在辩论结束之后进行，事实 id 和抽取结果从不回流给 agent。

## 结果

### 正反平衡度随 token 预算的变化

`B± = 1 − |n₊ − n₋| / (n₊ + n₋)`，只算有立场的事实，1 表示正反各半。

| 累计输出 token | 100 | 300 | 600 | 1000 | 最终 |
|---|---|---|---|---|---|
| neutral | 0.549 | 0.634 | 0.624 | 0.544 | **0.397** |
| lenses | 0.567 | 0.613 | 0.550 | 0.519 | **0.386** |
| stance | 0.471 | 0.652 | 0.636 | 0.643 | **0.573** |

三条曲线的形状不同，而不只是高低不同：

- **neutral 和 lenses 在 300–400 token 处达到峰值，然后单调下滑。** 辩论越往后
  越偏向一侧——是收敛，也可以说是同质化。
- **stance 起点最低**（0.471，比另外两个低约 0.09），**但在 300 token 之后就不再下滑**，
  一直保持在 0.62–0.65。

起点低是合理的：支持者和反驳者各自先说自己那一侧，第一轮结束前的事实池天然偏斜。
它的价值出现在之后——**分工的角色不让辩论收敛到单侧**。

### 配对差（12 题，vs neutral）

| 指标 | lenses | stance |
|---|---|---|
| 最终 B± | −0.011 [−0.174, 0.170] | **+0.176 [0.049, 0.320]** |
| 最终 H | −0.016 [−0.094, 0.071] | +0.062 [−0.024, 0.148] |
| B± @1000 tok | −0.025 [−0.166, 0.131] | +0.099 [−0.044, 0.269] |
| 双侧覆盖 | −0.083 [−0.333, 0.167] | 0.000 [0.000, 0.000] |

只有一个区间不含零。其余的最好当作方向提示，不要当作效应。

### 熵与双侧覆盖

`H` 最终 0.846 / 0.785 / 0.769（stance / neutral / lenses），有效立场数 2.57 / 2.42 / 2.36。
方向与 B± 一致但区间含零。

双侧覆盖（正反两侧都出现过）在 300–500 token 后三个条件都到 100%，
**这个指标在这个数据集上饱和了，区分不出东西**。它只在极低预算下有信息量。

## 为什么数量测不到而组成测得到

同一批 36 个 run，早先的配对检验：

| 指标 | stance − neutral | 12 题中为正 |
|---|---|---|
| 第一轮新生事实数 | +4.9 | 8/12 |
| 队列存活率 | −0.002 | 6/12 |
| 被多 agent 说过 | −0.044 | 3/12 |
| **最终 B±** | **+0.176** | **CI 不含零** |

三个条件产出的事实**数量**、**淘汰速度**、**复述率**都差不多。差别在于**这些事实分别站在哪一边**。
一个 panel 可以产出同样多的事实、以同样的速度淘汰它们，而最终留下的是清一色的一侧。

这也是这套方法相对 endpoint 指标的意义所在：准确率、事实总数都给不出这个区分。

## 站不住的地方

**只有一个区间不含零，而我做了十二次比较。** 没有做多重比较校正。
[0.049, 0.320] 的下界离零很近，n=12。这应当作为一个**待复制的方向**，不是已确立的效应。

**立场标注没有人工核对。** 标注由 `deepseek-v4-flash` 做，没有人类基准，
也没有第二个模型的一致性检验。整个结论建立在这个标注可信之上。
按今天在匹配阈值上的教训——**未经核对的参照不能当基准**——这是下一步必须补的。

**匹配质量的上界还没量清。** 人工标注的 59 对上，匹配准确率 80%；
后来改了归属剥离规则重跑，但用旧样本验证是被构造偏掉的（59 对里只有 23 对在新 store
里还存在，且恰好是不受该改动影响的那些）。**需要从新 store 重新抽样重标。**

**这只是 all-to-all 一种拓扑。** 设计文档里的 star / ring 还没跑，
所以现在说不了「拓扑如何影响立场平衡」。

## 下一步

按设计文档的三个问题，现在的位置是：

- **Explore**（persona 是否改变早期事实组合与正反覆盖）——**有初步证据，是 B± 这一条。**
- **Integrate**（事实是否跨 agent 被采纳）——已测，见
  [token lifecycle](2026-08-31-deepseek-token-lifecycle.md)，persona 改变 churn 多于 uptake。
- **Select**（早期丰富或平衡的事实组合是否预测更好的最终 grounded coverage）——**还没做**，
  因为 gold 覆盖率那条线还卡在 `add_gold` 报 `reached 0/41` 的 bug 上。

Select 是三个里唯一能回答「这些差异有没有用」的，也是最该优先修的。
在它跑通之前，B± 的差异只能说明 **persona 改变了事实组合**，
不能说明**改得更好**。
