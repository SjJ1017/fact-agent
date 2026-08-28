# 匹配是瓶颈：Perspectrum 试点暴露的 infra 问题

*2026-08-28 · 状态：诊断完成，方案未实施。明天从这里继续。*

**明天的题目：拿 Perspectrum 这批数据作为 example，设计出既快又准的 fact 匹配。**

---

## 1. 一句话

抽取是 O(n)、LLM 有完整上下文；匹配是 O(n²)、LLM 在脱离上下文的情况下判断孤立句子。
**我们把最贵、最不可靠的判断放在了会平方增长的位置上。** 这是设计问题，不是调参问题。

---

## 2. 数据在哪（都在本机，`experiments/*_out/` 被 gitignore）

```
experiments/perspectrum_pilot_full/
  *.debate.json               69 个  = deepseek 36 + glm 33（GLM 有 1 claim 三条件被内容过滤）
  *.store.json                39 个  = 用 light_match 匹配的，合并率 5.1%，不可用于结论
  *.extraction_failures.json  每 run 的抽取失败清单
  manifest.json
findings/data/perspectrum-lightmatch.json    ← 用坏 store 算出的人口学数字（存档用，别引用）
```

实验设计（别人做的，设计本身是好的）：Perspectrum test split 12 个真实争议 claim，
每题 4 条证据（2 support / 2 undermine），三个 agent、3 轮、full topology。
persona 三条件：`neutral`（三个相同中立分析员）/ `lenses`（因果、实施权衡、范围不确定性
三种审视角度，不指定立场）/ `stance`（支持者、反驳者、裁决者）。

`debate.json` 里有 `delivery` 字段，逐 turn 记录了 `source_ids` 和 `peer_turns`——
**每个 agent 每轮实际看到什么是 ground truth，不需要推断。** 这个字段还没被用上。

---

## 3. 三个仪器问题

### 3.1 匹配（致命）

`run_perspectrum.light_match` 用 stemmed Jaccard ≥ 0.80、`top_k=1`、**完全不调 LLM**。

```
合并率 5.1%   (1716 mentions -> 1628 facts)
对照 mmlu-pro  51%   (129 -> 63)
```

同一个命题被判成三个不同 fact 的实例（claim 221 / neutral）：

```
"E4 claims violent criminals will still obtain guns."
"E4 claims violent criminals will obtain guns regardless of a ban."
"E4 claims criminals will still obtain guns despite a ban."
```

候选对数量对比（同一 store，88 mentions）：`.80/1` → 14 对；`.50/12` → 166 对。差 12 倍。

**这个错误有方向，不是保守的。** `light_match` 的 docstring 说"false merge 会伪造一个
transmission，false split 只是少数一个"——对数传输是对的，对人口学是错的：
A 在 r1 说 X，B 在 r2 换个说法说 X，没合并 → X 记为**死亡**，B 的说法记为**新生**。
漏合并同时抬高 birth 和 death，把 survival 压向 0。

### 3.2 抽取有 8 个事实的硬上限（更根本）

`run_perspectrum.py:76` 自带的 prompt：`"Extract at most 8 short, self-contained,
verifiable facts"`，绕过了调过的 `src/factflow/extract.py:EXTRACTION_SYSTEM`。
**87% 的 agent-turn 正好返回 8 个。**

它不只是压缩信号——一段话自然分解可能是 5 个也可能是 13 个，被逼到 8 个就得改变切分
粒度。于是同一命题在不同 turn 被切成不同颗粒度的句子，**它们本来就不是同一个原子事实**。
匹配难做有一部分是因为输入不该匹配得上。修它的收益是双份的：mention 数下降 →
候选对平方下降，同时对齐质量上升。

### 3.3 抽取失败 15–22%

每 run 9 个 output turn，失败 lenses 2.0 / neutral 1.4 / stance 1.3 次。
每次失败 = 那个 agent-turn 贡献 0 个 fact = 看起来像整轮全部放弃。
claim 233 已被正确剔除，散在其他 claim 的单点失败还在分母里。

---

## 4. 性能实测

| 测试 | glm-5.3-flash | deepseek-v4-flash |
|---|---|---|
| 小 chat（未缓存） | 3.4s | — |
| flat schema parse | 9.2–10.4s | 1.5–5.6s |
| nested `list[8]` schema（合成数据） | 24.5s | **4.0s** |
| **真实裁决批次（8 对）** | **44.5s** | **240s 超时，返回 0 个关系** |

240.4s ≈ 3 × 80s，对应 `backends.generate` 的 `max_repairs + 1 = 3` 次静默重试。

**未解之谜（明天第一件事）**：deepseek 在合成的 nested `list[8]` 上只要 4.0s，
却在真实裁决 batch 上三次重试全败。所以不是 schema 形状的问题，是
`ADJUDICATION_SYSTEM` 的长度或内容触发了什么。**没有定位到根因就换模型是盲目的。**

按 glm 的 44.5s/批算：140 对 ÷ 8 = 18 批，串行 13 分钟，concurrency 6 约 2–3 分钟/store，
36 个 store 约 **1.5 小时**。

### 我今天误判的地方（避免明天重蹈）

- 前后三次说 rematch"卡死"。第一次是真的（漏了 `with_options(timeout=..., max_retries=1)`，
  client 默认 `max_retries=2`，等于 3×timeout）。**第二、三次不是卡死，是慢 + 无进度输出**：
  `rematch.py` 要整个 store 的 18 个批次全跑完才打印一行。
- 我曾说"98 秒没触发 60 秒超时，所以卡点不在请求上"——**这个推理是错的**。超时是每个请求
  60 秒，6 个线程各自在跑，98 秒时它们正常工作，只是还没走完全部批次。
- 教训：**任何长任务必须有 per-batch 进度输出**，否则无法区分慢和死。

---

## 5. 今天唯一可信的结果：绕开匹配的分析

`experiments/analyze_perspectrum.py`（已提交）。agent 用字面 token `E1`–`E4` 引用证据，
是精确字符串，**不经抽取、不经匹配、不受 8 上限影响**。

```
=== LEAN: +1 只引用支持方证据, -1 只引用反对方 ===
stance  A(advocate)  r1=-0.03  r2=-0.05  r3=-0.04
        B(critic)    r1=-0.08  r2=+0.03  r3=+0.03
        C(judge)     r1=+0.03  r2=-0.04  r3=+0.03
neutral A/B/C 全在 ±0.03 内，spread=0.02（lenses spread=0.15，比 stance 的 0.12 还大）

=== DIVERGENCE: 三个 agent 引用分布的平均两两 L1 ===
          r1     r2     r3    r3-r1
neutral  0.143  0.165  0.102  -0.041
lenses   0.196  0.222  0.135  -0.061
stance   0.229  0.260  0.176  -0.053

=== 覆盖率: 每轮引用了 4 条证据中的几条 ===
三条件三轮全部 = 4.00 / 4
```

**(a) stance 的 role prompt 没有产生立场分化。** 提示词明写 advocate / critic，
但 advocate 的 lean 是 −0.03、critic 是 −0.08，差距完全在噪声内。
这回答了 [fact-demography](2026-08-28-fact-demography.md) 留下的问题：
**role prompt 的内容至少在"用哪些证据"上是装饰性的。**
*限制*：`lean` 数的是引用不是认同，advocate 引用反方证据可能是为了反驳。
只能说"注意力没分化"，不能说"立场没分化"。

**(b) divergence 排序对**：stance > lenses > neutral，三者都在 r3 收敛。
persona 改变了证据**权重分配**，没改变方向。

**(c) 覆盖率恒为 4.00/4** —— dossier 只有 4 条，agent 一轮就穷举完，天花板效应。
**要观察选择性注意，证据集必须大到 agent 不得不取舍**（Perspectrum 每个 claim
本来就有更多 perspective 可用）。

### 内部一致性检验（这是"坏 store 是仪器问题"的铁证）

同一批 run 里：引用覆盖率 **4.00/4**（每个 agent 每轮都碰全部四条证据，内容高度延续）
vs 事实生存率 **0.04–0.06**（r1 说的事实 96% 每轮死光）。
**两者不可能同时成立**，而引用那个数没有判断环节。

---

## 6. 明天的方案（未实施）

按杠杆排序。核心思路：**把 LLM 成本从匹配阶段搬到抽取阶段。**

1. **规范化前移（O(n²) → O(n)，最大收益）**
   让抽取器直接输出受控的谓词-论元结构，实体名取自源文档，而不是自由句。
   ```
   现在  "E4 claims violent criminals will still obtain guns."
         "E4 claims criminals will still obtain guns despite a ban."
   改后  {ref:E4, pred:asserts, arg:"criminals obtain guns despite ban"}   ← 字符串相等, 免费合并
   ```
   抽取那次调用本来就在付。

2. **只裁决不确定的带宽** — 高相似度自动合并、低相似度自动拒绝，LLM 只处理中间。
   标准实体消解做法。**带宽必须实测标定，不能拍**（今天标定脚本超时了，没跑出来）。
   若 60–70% 的对落在带外，白赚 3 倍。

3. **增量匹配（O(n²) → O(n·k)）** — 按时间顺序，每个新 turn 只跟**已有规范事实**比，
   不跟所有历史 mention 比。事实大量重复，k 增长远慢于 n。
   `match()` 已接受 `store` 参数，这个能力建了一半。也是唯一能在线跑的版本。

4. **砍掉 `rationale`** — `PairJudgement.rationale` 每对生成一句，下游没有任何代码读它。
   批量模式关掉，留 flag 给审计。

5. **先修 8 上限再谈 rematch。** 颗粒度错位的话，放松阈值只是让 matcher 在错位的句子间
   做勉强判断——**花 1.5 小时买一个仍然不可信的结果。**

建议顺序：定位 deepseek 裁决失败根因 → 修 8 上限重新抽取 → 标定带宽 + 增量匹配 → 再跑匹配。

---

## 7. 代码状态

| 文件 | 状态 |
|---|---|
| `experiments/analyze_perspectrum.py` | 可用，绕开匹配的分析 |
| `experiments/analyze_demography.py` | 可用，但喂坏 store 会得到坏数字 |
| `experiments/rematch.py` | 能跑但慢且无进度输出；**加 per-batch 进度再用** |
| `src/factflow/aggregate.py` / `graph.py` | 未改动，53 个测试全过 |

重跑今天的可信分析：
```bash
.venv/bin/python experiments/analyze_perspectrum.py experiments/perspectrum_pilot_full \
    --model deepseek-v4-flash
```
