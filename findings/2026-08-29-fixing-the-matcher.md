# 修好匹配：五个真因，和四次我自己推翻的解释

*2026-08-29 · 续 [2026-08-28-matching-is-the-bottleneck](2026-08-28-matching-is-the-bottleneck.md)*

那份记录停在"诊断完成，方案未实施"。这份记录实施的结果，以及一件比结果更值钱的东西：
**哪些判断方式今天反复出错。**

---

## 1. 五个真因

按发现顺序，每一个都推翻了我当时正在做的事。

### 1.1 抽取根本没要求原子性

`run_perspectrum.py` 用自带的 `LIGHT_EXTRACTION_SYSTEM`，全文四行：

```
Extract at most 8 short, self-contained, verifiable facts from the supplied text.
Preserve names, numbers, and conditions. Do not infer, repeat a fact, or emit
claims about the conversation. Use "negate" only for an explicit denial.
```

**它从头到尾没有要求原子。** 而同一个仓库里 `src/factflow/extract.py` 的 `EXTRACTION_SYSTEM`
规则 1 正是为此写的，连"E3 and E4"这种集合量词都点名要求逐成员拆开。

后果：2914 个 fact 里 **33% 含并列结构**，10% 是主语并列。同一段文本，
`LIGHT` 给 8 条（被 `at most 8` 截断），`EXTRACTION_SYSTEM` 给 **23 条**且完全原子。

**这是全部问题的根。** 非原子事实之间没有精确重复，于是匹配被迫做蕴含判断，
而蕴含判断昂贵、模棱两可，还会把"E3 说 X"和"E4 说 X"这种本该是集合运算的东西
变成需要裁决的难题。我为此建的标定带宽、bisect、lineage，全是在给 matcher 加能力
去处理**本不该送到它面前的输入**。

### 1.2 缓存把失败变成永久

`extract_facts` 每次返回 0 条。它没坏——早先某次调用因为推理吃光预算返回空，
`llm.parse` 把空结果缓存了，此后每次都在重放那次失败。静默、永久。
和 DDI 那次空 transcript 是同一类问题。

修：`parse` 加 `cache_if` 谓词，`extract_facts` 传 `lambda r: bool(r.facts)`。

### 1.3 推理预算是"模型 × 任务"的属性

空响应的直接原因：`completion_tokens=4000, reasoning_tokens=4000, content=0, finish=length`。
**全部预算进了推理，一个字都没剩给内容。**

但关键在于它随任务变化：

| 模型 | 任务 | reasoning | content |
|---|---|---|---|
| glm-5.3-flash | 五分类 @4000 | 4000 | **0** |
| glm-5.3-flash | 五分类 @16000 | 15999 | **0**（给多少烧多少） |
| glm-5.3-flash | 二值 @16000 | 3656 | 596 ✓ |
| minimax-m2.5 | 二值 @8000 | **694** | 612 ✓ |

**五分类会把任何预算烧光；二值把推理钉在一个上界。** 两件事都要做：二值提供上界，
`max_tokens ≥ 8000` 提供余量。

网关不支持关闭推理——`reasoning_effort`、`thinking:{type:disabled}`、
`enable_thinking:false` 全部 400。33 个模型全是前沿推理模型，没有传统非推理模型。

### 1.4 守卫把自己的占位符当成了确定性

`_guard` 逐个成员对代表复查，传 `1.0` 作为相似度**占位符**（它在提问，不在打分）。
我给它绑的 judge 带着 `auto_accept_above=0.95` → 1.0 ≥ 0.95 → **不问就放行**。
守卫要盘问的每一对都被自动通过，union-find 的传递闭包失去制约，
154 个 mention 塌成一个 **85 成员**的簇。

### 1.5 `auto_accept_above` 本身就建立在错误标定上

`candidate_pairs` 返回 `max(cosine, containment)`，而 **containment = 1.0 意味着
"B 的词全在 A 里"——正是"丢掉限定范围"那种子集关系**。所以
`"UBI reduced poverty by 12.3% in Kenya"` 和 `"UBI reduced poverty"` 会被自动合并。

而按二值规则，改变覆盖范围的是 DIFFERENT。我那张"上端 100% SAME"的标定表
**是把 ENTAILS 映射成 SAME 算的**——正是二值化后不再采用的映射。
标定口径和最终规则不一致，我没察觉。已默认关闭。

---

## 2. 现在的流水线

一条路径，每个环节都有实测依据。

```
extract_facts    EXTRACTION_SYSTEM + minimax-m2.5      23 条 vs LIGHT 的 8 条
atomize          正则预筛 -> 拆并列（安全网）           只 33% 需要调用
candidate_pairs  bge-base-en-v1.5, 阈值 0.70, top_k 12
identify         二值 SAME/DIFF, diff 字段前置          推理 3600 -> 694
cluster          union 只在 sim >= 0.90
```

`match.py` 从 789 行减到 319 行。删掉：五分类裁决器及两套 schema、传递性守卫及
为它穿进 `cluster` 的 judge 回调、为"内容过滤"假设写的 bisect、从没跑过的
`incremental_match`、`aggregate.lineages`。

### 每个参数的依据

| 参数 | 值 | 依据 | 花费 |
|---|---|---|---|
| blocking 阈值 | 0.70 | 128 个已裁决 pair，bge-base 零漏失省 94.9% | 0 次调用 |
| union 阈值 | **0.90** | 6 个 store，最大簇 111 → 8 的断崖 | 0 次调用 |
| top_k | 12（**未收紧**） | top_k=6 会丢 24% 真实合并对 | 0 次调用 |
| max_tokens | ≥ 8000 | 实测 reasoning 占 694–4000 | 4 次调用 |
| auto_accept | **关闭** | containment=1.0 是子集不是同一 | 0 次调用 |

**三次离线标定零调用，其中一次（top_k）推翻了我的直觉。**

---

## 3. 我今天错了几次，以及共同的毛病

### 3.1 四次机制解释，前三次被自己的数据推翻

| 我的解释 | 依据 | 推翻它的证据 |
|---|---|---|
| prompt 太长 | 一次长 prompt 失败 | 长 prompt + 合成数据 3.4s 正常 |
| 配额/限流 | 真实失败、合成成功 | 合成在失败前后都成功（7.4 / 9.2 / 13.2s） |
| 争议内容被过滤 | "毒对"批次失败 | 毒对**单独**跑成功（28.3s） |
| 退化生成 | 近似重复的长句 | 同上 |
| **推理吃光预算** | `reasoning_tokens=4000, content=0` | 成立 |

共同毛病：**拿到一个数据点就构造机制，而不是先问"什么样的观察能同时解释全部现象"。**
所有失败都恰好等于我设的超时值——这个特征一直摆在那里，我却拿它支持了三个不同的假设。

### 3.2 成本估算高了 3.5 倍

我用 **glm 跑五分类**测到的推理量（3600），套到 **minimax 跑二值**上（实测 694）。
**推理预算是"模型 × 任务"这一对的属性**，两个变量同时换掉还沿用旧数。

`backends.Usage` 一直在累计 token，从没被打印过。现已加上 `cost()` / `report()`，
模型不在价目表里报 "cost unknown" 而非当成免费。

### 3.3 三次"慢"被误判成"卡死"

只有第一次是真的（漏了 `with_options(timeout=..., max_retries=1)`，
client 默认 `max_retries=2` 等于 3×timeout）。后两次是慢 + **无进度输出**：
代码要整个 store 跑完才打印一行。我还曾据此推断"98 秒没触发 60 秒超时说明卡点不在请求上"——
这个推理本身是错的，超时是每请求的，14 个线程各自在跑。

**任何长任务必须有 per-batch 进度输出**，否则无法区分慢和死。

### 3.4 一个假通过的测试

守卫的测试断言"请求了哪个 schema"而不是"拿回了什么判决"。`FakeLLM` 没有
`IdentityResult` 的 handler，调用其实抛异常、被 bisect 吞掉，断言照样成立。

**测试要断言结果，不要断言过程。**

---

## 4. 可复用的判据

写在动手之前，不要用结果去凑解释。

1. **能离线标定的，不要用 LLM 调用去试。** 今天三次离线标定零成本，
   而所有靠调用去试的改动（守卫、bisect、lineage、auto_accept）最后全部返工或删除。
2. **单点测量不能跨条件外推。** 换了模型或换了任务，就得重测。
3. **改动前先写下判据。** 例如 union 阈值："若 6 个 store 都在 0.85–0.90 出现最大簇断崖
   就采用 0.90；若拐点漂移超过一档则单一阈值不成立，改用簇内平均相似度。"
4. **幅度反常就是信号。** 合并率 19% → 62% 不可能是纯改进，一查就是 85 成员的 blob。
5. **不要为绕开坏输入而给下游加能力。** lineage、bisect、守卫都是这么来的，
   修好抽取之后全部作废。

---

## 5. 数据与复现

```
experiments/perspectrum_pilot_full/*.debate.json   69 个原始 trace（gitignored，本机）
experiments/perspectrum_pilot_full/*.v2.json       重建的 store
experiments/retrace.py          重新抽取 + 原子化 + 匹配
experiments/add_gold.py         把 Perspectrum 标注论点作为 gold 匹配进 store
experiments/analyze_coverage.py 覆盖率 / 平衡性 / 何时出现 / 是否存活
experiments/analyze_demography.py 每轮人口 + 生存曲线
```

```bash
.venv/bin/python experiments/retrace.py experiments/perspectrum_pilot_full \
    --glob "*deepseek*.debate.json" --model minimax-m2.5 --concurrency 14
```

约 80 秒/store，36 个约 50 分钟。成本待实测填入（此前的估算不可信，见 3.2）。

**三条件的结果尚未产出**，等 36 个 store 跑完 + `add_gold` 之后单独记录。
