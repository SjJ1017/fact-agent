# 事实流剖面：full × star × 三档人格

2026-09-01。报告页：https://claude.ai/code/artifact/756f477f-f148-409e-9434-301220ae3f56

12 个 claim × 3 档人格（neutral / lenses / stance）× 2 种拓扑 = 72 场，
全部 deepseek-v4-flash。GLM 那批只有全连接、没有星型对照，**不进入本分析**。
chain 还在抽取（见 `AGENTS.md` 的"当前在跑"）。

## 数据在哪

| 东西 | 位置 | 在 git 里 |
|---|---|---|
| 全连接 store | `experiments/perspectrum_pilot_full/*deepseek-v4-flash-full-*.v2.json` (36) | ✅ |
| 星型 store | `experiments/perspectrum_pilot_star_chain/*deepseek-v4-flash-star-*.v2.json` (36) | ✅ |
| 投递结构 | 同目录同名 `.debate.json` 的 `delivery` 字段 | ✅ |
| 立场标注 | `experiments/labels/stance/*.json` (105) | ✅ |
| token clock | `experiments/labels/token_clock/*.json` (105) | ✅ |
| 结果表 | `findings/data/flow-profile.json`、`flow-profile-per-debate.csv` | ✅ |

## 怎么复现

```bash
python experiments/analyze_flow_profile.py        # 不联网、不花钱，几秒钟
python experiments/report/build_flow_profile.py   # 出报告 HTML
```

标注层是从 enriched store 里导出来的（`experiments/export_labels.py`），
原来的 `/tmp/factflow-stance-all/` 和 `/tmp/factflow-token-clock-all/` 是 75 MB
逐字节副本，现在压到 8.7 MB 并进了 git。要挂回 store 用 `attach_labels()`。

## 指标

三族，全部只从 `(agents, rounds, delivery, facts)` 定义，
不依赖任何拓扑或 persona 特有的东西——这是能跨条件比较的前提。

1. **token 预算去向**（互斥穷尽）：`新事实`（自己没说过、也不在收到的投递里）、
   `自持`（自己上一轮说过）、`采纳`（首次出现在自己嘴里但已在收到的投递里）。
2. **流动图形状**：`嵌套度`（agent 事实集合两两平均包含率）、
   `事实覆盖`（每个事实被几个 agent 说过，3 个 agent 上限 3.0）、
   `流量基尼`（按 (speaker, listener) 聚合的搬运量的基尼系数）、
   `投递利用率`（有多大比例的投递事件真搬动了 ≥1 个事实）。
3. **事实群体**：`元陈述占比`（NEUTRAL 比例）、`正反平衡 = 1 − |n₊−n₋|/(n₊+n₋)`，
   都只算至少有一个 OUTPUT mention 的事实。

统计：按 claim 配对（同 claim 同拓扑），12 个差值做 20000 次 bootstrap，报 95% 百分位 CI。
**没做多重比较校正。**

## 结果

```
条件           事实数  新事实  采纳占比 自持占比 嵌套度 事实覆盖 流量基尼 投递利用率 元陈述 正反平衡
full/neutral    72.4   81.1   12.9%   15.5%  0.330   1.33    0.348    64%    33%   0.528
full/lenses     97.7  106.7   10.8%    9.2%  0.250   1.25    0.308    71%    39%   0.506
full/stance     93.4  101.3    9.7%   10.9%  0.231   1.23    0.400    60%    28%   0.657
star/neutral    79.2   89.5   10.2%   12.7%  0.292   1.30    0.296    69%    46%   0.573
star/lenses     86.8   94.8   10.1%    8.9%  0.243   1.24    0.328    57%    33%   0.390
star/stance     99.3  106.8    7.1%    8.2%  0.181   1.17    0.413    57%    30%   0.610
```

配对 stance − neutral（✓ = CI 不含零）：

```
        事实数              自持占比            嵌套度              事实覆盖            采纳占比
full  +21.0 [+11.3,+30.3]✓  -.045 [-.080,-.008]✓ -.098 [-.159,-.039]✓ -.104 [-.171,-.038]✓ -.032 [-.064,-.003]✓
star  +20.1 [ +2.0,+35.2]✓  -.045 [-.072,-.021]✓ -.111 [-.176,-.032]✓ -.124 [-.193,-.042]✓ -.030 [-.059,+.001]
```

拓扑对比（star − full，同 claim 同 persona，8 指标 × 3 persona = 24 个）：
**23 个跨零。** 唯一例外是 stance 下的自持占比 −0.027 [−0.053, −0.002]——
24 次未校正比较里的一个区间，不该单独相信。

## 结论

1. **人格梯度效应在两个拓扑上复现。** 人格越强，产出的事实越多，
   但进入共享池的越少。事实覆盖从 1.33 降到 1.17——最强人格下一个事实
   平均只被 1.2 个 agent 说过。嵌套度在 full 下 12 个 claim 只有 2 个上升，star 下只有 1 个。
2. **采纳和自持一起塌。** 这两个是产出里仅有的两种"回头"。都下降意味着
   stance 条件下的输出几乎全是新生成——辩论在横向铺开，没有可供收敛的公共基底。
3. **瓶颈在整合，不在生成。** 系统能产出的差异超过它能吸收的差异。
   这是 `2026-09-01-integration-capacity-hypothesis.md` 的直接证据，且是跨拓扑的。
4. **拓扑（3 agent，12→8 次投递）测不出效应。** 两种读法分不开：
   要么这个规模的拓扑真不重要，要么效应被人格效应的方差淹没
   （人格效应量 0.5–1.0σ，拓扑若只有 0.2σ，12 个 claim 检不出）。
   **chain 是最直接的检验。**

## 局限

- 只有 deepseek-v4-flash，模型间是否复现未知
- star 有 14 个空 turn（4.3% vs full 0.9%），会同时压低 star 的事实数和采纳数，
  是拓扑阴性结果的待排除解释，见 `AGENTS.md` 的"已知缺陷"
- 立场标注未经人工验证，且约一半有立场的事实是证据层（"E3 支持禁令"）而非世界层，
  正反平衡是混合口径，只能横向比条件
- 事实匹配人工核对准确率 80%（`2026-08-29-fixing-the-matcher.md`）。
  没有理由认为错误率随条件变化，所以条件间的**差**比每格的**绝对值**可信
- "采纳"只判事实同一性，不判态度——复述来反驳和复述来同意计为同一件事

## 待办

- chain 跑完后：立场标注 → `export_labels.py` → `analyze_flow_profile.py
  --topologies full star chain` → `build_flow_profile.py` → 重发同一个 artifact URL
- VERDICT 在 324/324 个 turn 里可解析，还没当作结果变量用过
