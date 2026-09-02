# AGENTS.md — working notes for factflow

Durable, hard-won context that isn't obvious from the code. Committed lessons live in
the README; this is the operational layer. **In git since 2026-09-01** — it was
untracked before, which meant a new agent could read the whole repository and still
miss every negative result and cost trap below. It holds no secrets, only key names.

---

## 接手这个项目 — 从这里开始（2026-09-01）

**这个项目由多个 agent 同时操作。已经存在的产物不要凭记忆重做——先 `ls` 目录确认。**

阅读顺序，读完这四样就够接管：

1. `README.md` — 库本身：数据模型、pipeline 的每一步、已定稿的结论
2. 本文件 — 操作层：账单规则、环境、反复重学的教训，以及
   **`## RETRACTED: every "generalist" number so far is contaminated`** 那一节——
   别再引用那批被缓存 bug 污染的数字
3. `findings/README.md` — 实验索引，每个发现一个 md，每个 md 自带"数据在哪 / 怎么复现"
4. `findings/data/` — 已经算好的结果表，不用重跑就能读

跑任何 pipeline 之前，问自己三件事：产物是不是已经存在？参数跟已有产物对不对得上？
这一步是不是在花钱？

---

## Billing — standing rule

**OpenCode Go is a flat-rate subscription. OpenAI and DeepSeek are metered.**
Every experiment default points at `--provider opencode --model glm-5.3-flash`. Metered
providers are opt-in and are for **benchmark comparison only**, never for a sweep, a trace
pass, or anything that runs per-question.

This was learned the expensive way: the OpenAI key was given for a one-off model comparison
and then became the default for `run_hotpot.py`, a 4-panel × 55-question MMLU sweep, and
several full trace passes. That drained the credits. A trace pass alone is ~20 extraction
calls plus ~150 adjudication calls *per question*.

Before running anything wider than a single question, check which provider the default
resolves to.

## Environment

`.env` (gitignored) holds three keys:

| var | what it reaches |
|---|---|
| `OPENAI_API_KEY` | OpenAI direct |
| `OPENCODE_API_KEY` | OpenCode gateway — **two tiers, different catalogues** |
| `DEEPSEEK_API_KEY` | DeepSeek direct — **balance exhausted** |
| — | OpenAI — **credits exhausted 2026-08-27**; reachable via OpenCode Go instead |

OpenCode endpoints: `https://opencode.ai/zen/go/v1` (Go subscription, 31 models) and
`https://opencode.ai/zen/v1` (pay-as-you-go — **also out of credits**). Use the Go tier.
DeepSeek v4 on Go needed a one-time region opt-in in the workspace; that is done.

```bash
set -a && . ./.env && set +a          # load keys
.venv/bin/python -m pytest tests/ -q  # 24 tests, all offline (FakeLLM)
```

## The rule that keeps being relearned

**A prompt rule with no worked example is ignored.** This has now happened four times:

1. "split conjunctions into separate facts" → `"an American director, screenwriter and
   producer"` came back as one fact.
2. "resolve every pronoun" → `"Both were American"` came back unresolved.
3. `PRECISION IS NOT EQUIVALENCE` → the model applied it to aspect and tense, splitting
   one fact into four clusters.
4. Rule 7 "skip meta-discourse" → `"The documents state that…"` came back as a fact.

Each fix was the same shape: add 3–5 worked examples **and the counter-cases that must
not be transformed** (`"black and white film"`, `"Bosnia and Herzegovina"`, `"Rex is a
poodle"` → `"Rex is a dog"` must stay entailment). Adding only positive examples causes
over-application, which is how #3 happened.

## Defects found by reading output, never by a test

Every single one. The tests came after.

- **Quantifiers are an atomicity defect, not a reference one.** `"Both were American"`
  is not under-specified — it asserts one thing per member and those facts are already
  extracted. Resolving it keeps a duplicate; distributing it removes one.
- **Aspect is not information.** `"was named ambassador"` / `"held the position of
  ambassador"` / `"served as ambassador"` are one fact. The adjudicator called them
  entailment, split one fact into four clusters, and hotpot-1 reported **0% gold
  retention** as a result. Operational test now in the prompt: *same row in a
  who-did-what database → EQUIVALENT.*
- **Blocking recall is not uniform across relation types.** Contradictions survive any
  threshold (one token apart); entailment collapses (97%→64% at .65) because a weakened
  restatement shares few tokens with its source. Never raise the threshold above `.50`
  if degradation analysis matters.
- **`DiskCache` raced.** One `.tmp` per key, concurrent writers, `FileNotFoundError`
  killed a whole question mid-run. Temp names are per-writer now.
- **Silent-failure hazard is real.** `llm.map(tolerate_failures=True)` dropped a debate
  turn; the next round indexed into the hole and raised `KeyError` several questions
  later, far from the cause. Debate turns now use `tolerate_failures=False`.

## Benchmark gotchas

- **Separate quality from reliability.** Three models scored 54–73% where *every*
  failure was a timeout. Score only over calls that returned; report reliability apart.
  minimax-m3 is 100/100 on the 30% of calls it answers.
- **Timeouts are load-bearing.** At 30s several Go models look broken; at 60–90s they
  are perfect. Always state the timeout with a reliability number.
- **Never type prices.** Hand-entered rates were 2–4× low across the whole gpt-5 family,
  flattering exactly the poor-value models. Fetch from `models.dev/api.json` (needs a
  `User-Agent` header or it 403s).
- **Re-run every model when a prompt changes.** Comparing a sweep from before a prompt
  edit against one from after is invalid; this was nearly published once.

## Dataset status — the important negative results

| dataset | single-agent | multi-agent | verdict |
|---|---|---|---|
| HotpotQA distractor | — | 100% (3/3, unanimous) | retrieval, not collaboration |
| MedQA-USMLE 4-opt | **92%** (37/40) | **92%** full, 95% chain, 92% star, 92% specialists | **saturated** |
| MMLU-Pro (phys/chem/eng) | 87% (26/30) | 90%, 29/30 unanimous | saturated |
| **MMLU-Pro hard subset** (27 fails + 20 controls) | **43%** | 53% full · 53% chain · **66% specialists** | **discriminates** |

MedQA at n=10 showed 80%→90% and looked like collaboration helping. At n=40 it was
92%→92%. **Ten questions is not a sample**; the whole apparent effect was noise.
Unanimity 37–39/40 means the agents are not really deliberating either way.

GPQA is gated on HF and needs a token to use.

**Screening is what made the experiment work.** `--screen N` runs single-agent first, keeps
the failures, adds matched controls. On the hard subset specialist roles strictly dominate
(+6 questions over generalists, -0), which was invisible at the 92% ceiling.

**Flow separates what accuracy cannot** — the first real payoff for the content-centric
view. `full` and `chain` both score 53%; chain transmits 4 facts/run and persists 36
(agents repeat themselves), full transmits 10 and persists 25 (agents exchange).
Specialists transmit 16 and keep 13% of source facts, and score 66%. n=8 per config.

**Output dirs collide.** Two sweeps writing to `experiments/<dataset>_out/` mix runs, and
`full-generalist` ended up with 53 files across two different question sets. Filter by the
qids of a config that only ran on the subset before comparing.

## Cost, measured

Adjudication is **92% of tokens** (178 calls vs 19 on one HotpotQA question). Choosing a
cheaper extraction model saves under 8%. The lever is pair count → blocking threshold.

Good picks: `glm-5.3-flash` ($0.0042/bench, 100% relations, fast), `gpt-5.4-mini`
($0.0277, 100/100, 9s — fastest), `deepseek-v4-flash` ($0.0174, 95.2%, slow but cheap).
Never `gpt-4.1-nano` for adjudication — the only model emitting false `EQUIVALENT`, the
one error union-find amplifies.

## RETRACTED: every "generalist" number so far is contaminated

The chat cache keyed on `(model, system, user, temperature)` with **no agent identity**.
Undifferentiated agents send byte-identical prompts, so agents B and C received A's reply
verbatim. Measured: round-1 output was byte-identical in **55/55** `full-generalist` runs,
**53/53** in the earlier set, and **47/47** in `chain-generalist` — at temperature 0.7.
`full-specialists` was 0/47 because role prompts differ, so its keys differ.

A three-agent generalist panel was therefore one agent counted three times. That
invalidates, for every generalist/chain condition:

- round-1 pairwise fact overlap (the 0.827 and the 1.000 are the cache, not the model)
- unanimity (guaranteed by construction, so 51/55 and 44/47 mean nothing)
- majority voting, hence the accuracy itself — probably depressed, since there was no
  diversity to aggregate

So "roles beat generalist debate by +13-18" is confounded: part of that gap is the control
condition being broken. The specialist and functional-role numbers are unaffected on their
own, but every comparison *against* a generalist baseline has to be re-run.

Fixed by threading `sample_id` through `LLM.chat`. Caching is right for analysis
(deterministic work on fixed text) and wrong for generation (independent draws).

## Role effects — measured before the cache bug was found

Topology held fixed (`full` both sides), so any difference is content, not structure.

**Round 1, before any agent has read any other:**

| | pairwise fact overlap | distinct facts on table | accuracy |
|---|---|---|---|
| full/generalist | 0.827 | 10.6 | 53% |
| full/specialists | 0.277 | 23.6 | 66% |

Generalist debate is three near-copies of one agent. Roles cut overlap to 0.28 and double the
evidence pool **before any communication** — a sampling effect, not a deliberation effect.

**Trajectories run opposite:** generalists start identical and diverge (0.83→0.23, union 10.6→32.6);
specialists start diverse and stay flat (0.28→0.30, union 23.6→26.8). Generalists end with MORE
facts and score 13 points lower — because the answer is already fixed: **45/53 generalist runs have
their final majority at round 1**. Late diversity is post-hoc justification.

**Unanimity is worthless without independence.** On 5 of the 6 questions specialists uniquely
solved, the generalist panel was unanimously *wrong*. At 0.83 overlap agreement carries no evidence.

**Operator profiles do not differ by role.** From hand-extracted slices: analyzer derive
50% / relay 32%, decomposer 52% / 32%, summarizer 50% / 33% — and domain roles were
similarly flat (derive 59-60%). Roles change *which* facts an agent produces, not *what
operation* it performs. The one real difference is `inherit`: 1.3-1.7 facts per agent-round
under pipeline against 0.2 under generalist — but that gap is exactly what the cache bug
would manufacture, since identical agents have nothing to inherit. Re-measure.

## Relation-typed metrics measure revision, not error

Using the four relation types instead of binary transmission/persistence:

| | degradation pairs/run | self-contradictions/run | accuracy |
|---|---|---|---|
| chain/generalist | 15.4 | 1.5 | 53% |
| full/generalist | 25.8 | 2.4 | 53% |
| full/specialists | **29.8** | **3.4** | **66%** |

The winning config degrades most and contradicts itself most. Treating those as quality signals
ranks the systems exactly backwards — they track how much the agents revise.

**Two traps hit here:**
- Counting raw entailment *edges* inflated degradation ~20× (one run: 132 edges, 6 distinct fact
  pairs). Facts recur across nine agent-rounds; always dedupe to fact pairs.
- On MMLU-Pro most "degradation" is paraphrase compression (`"Magnesium metal is oxidized"` →
  `"Magnesium is oxidized"`), not information loss. Degradation needs content carrying conditions,
  scopes and magnitudes — clinical, policy, drug interaction. Wrong dataset for it.
- Option-level selectivity was too sparse to read (2–4 matched facts/round across 8 runs): agents
  write `"Option E is 3Mg + 2Cr³⁺ → …"` rather than restating option text.

**The chain result is a tautology.** Chain's head agent has no predecessor, so of course it repeats
(23% exact-repeat rate, 9/16 at position A). Derivable from the adjacency matrix — it validates the
instrument, it is not a finding. Don't lead with it.

## The summarizer role makes filtering WORSE

Answering "why doesn't the summarizer drop facts". On MMLU-Pro its profile was flat
because there was nothing to drop (three clues, all relevant) and because `FINAL ANSWER`
forced it to re-derive rather than consolidate — reading its turns showed it restating the
whole argument every round in fewer words. Paraphrase compression, not fact selection.

The synthetic DDI dossiers remove both objections: four genuinely irrelevant facts per
drug, and the consolidating role does not have to produce a verdict. Result, split
condition, 16 cases per arm:

| config | agent A | agent B |
|---|---|---|
| no roles | 24% | 37% |
| A=analyzer, **B=summarizer** | 30% | **53%** |
| **A=summarizer**, B=analyzer | **42%** | 39% |

*(% of that agent's own irrelevant dossier facts carried forward)*

Whichever position holds the summarizer label retains MORE. Length-normalised (turn
lengths are 730-831 chars everywhere, so this is not verbosity): irrelevant facts per 1000
chars is **1.29 and 1.03 at summarizer positions** against 0.65 / 0.76 / 0.89 / 0.94
elsewhere. Consistent across both orderings, so it is the role and not the position.

Likely mechanism: "consolidate what the panel established" is read as *coverage*. A
summary should be complete, and that prior beats the explicit instruction to leave out what
does not bear on the question.

Accuracy was **100% in every arm**, so none of this is visible from a scoreboard.

**Role ORDER made no difference** to accuracy (100% both ways) and only swapped which
position showed the effect. Order is not the lever; the role label is.

## DDI becomes a real benchmark at 7 decoys — and then the trace pays off

At one dossier per agent the task is trivial (one candidate fact per side) and everything
scores 100%, which is the same saturation problem as HotpotQA/MedQA/MMLU-Pro. Burying the
real dossier among decoy drugs fixes it:

| decoys/agent | solo-both | split |
|---|---|---|
| 0 | 100% | 100% |
| 3 | 100% | 100% |
| **7** | **100%** | **75%** (positives 4/8) |

`solo-both` stays at 100% with all 16 dossiers in one context, so the information is
sufficient and the model can use it. Distributing it costs half the positives. That is the
HiddenBench gap with a mechanistically explicit cause, and it finally gives variance to
explain.

**And the decomposition is the payoff.** In all 4 failures the partner's enzyme reached
both agents. Transport succeeded; the join failed. Accuracy says "75%" and cannot tell you
whether to fix the topology or the reasoning. The trace says: do not touch the topology,
the information already arrives.

Decoys deliberately include lures sharing the *partner's* enzyme, so an agent that answers
by scanning for any enzyme match joins the wrong pair.

## Open

- Probe set has saturated (11/17 perfect on extraction, 8/17 on relations). Needs harder
  cases or it ranks nothing.
- Adjudicator accuracy on *real* data still unmeasured — the probes are hand-written from
  known failures, which is a floor not a rate.
- Under-merge persists: audit still flags `possible-missed-merge` pairs that are real.

---

## 当前实验状态 — Perspectrum 人格 × 拓扑（2026-09-01）

12 个 claim × 3 档人格（neutral / lenses / stance）× 3 种拓扑，全部 deepseek-v4-flash。

| 拓扑 | store | 状态 |
|---|---|---|
| full（全连接，每场 12 次投递） | `experiments/perspectrum_pilot_full/*-full-*.v2.json` | 36/36 ✅ |
| star（星型，8 次投递） | `experiments/perspectrum_pilot_star_chain/*-star-*.v2.json` | 36/36 ✅ |
| chain（链，最少连线） | `experiments/perspectrum_pilot_star_chain/*-chain-*.v2.json` | **抽取中**，见下 |

另有 GLM-5.3-flash 的 33 个全连接 store（缺 claim 233），**没有 star/chain 对照**，
所以不进入跨拓扑分析。

### 标注层怎么共享

`label_fact_stance.py` 和 `token_clock.py` 都是"复制整个 store + 加一个字段"，
两遍标注产生了 75 MB、99% 是仓库里已有 store 的逐字节副本，而且落在 `/tmp`，
重启就没了（重做要 ~$12 和几小时）。

现在标注单独存，keyed by fact，附带 canonical_text 以便重新聚类后还能挂回去：

```
experiments/labels/stance/<execution_id>.json        105 个，1.3 MB
experiments/labels/token_clock/<execution_id>.json   105 个，7.4 MB
```

导出与挂回都在 `experiments/export_labels.py`（`attach_labels()` 可直接 import）。
**新跑出来的标注请用它导出并提交，不要把 enriched store 塞进 git。**

### 统一的 pipeline 配置（`retrace.py` 默认值，跨条件比较必须一致）

```
抽取      EXTRACTION_SYSTEM + minimax-m2.5, max_tokens 8000
原子化    atomize（剥离归属外壳 + turn 内去重）, 正则预筛
blocking  bge-base-en-v1.5 @ 0.70, top_k 12
判决      二值 SAME/DIFF, batch 16, auto_accept 关闭
合并      union >= 0.85（人工读边标定, findings/2026-08-29-fixing-the-matcher.md）
立场      label_fact_stance.py --stance-mode truth（默认）, deepseek-v4-flash,
          batch 24, max_tokens 16000
```

**参数一致性警告。** `batch-size` 会改变判决：同一批事实放在 12 条 vs 24 条的上下文里，
模型看到的对照不同，已观察到文本未变而判决改变。`max_tokens` 同理——deepseek-v4-flash
推理约占 4000 token，batch 24 需要 >= 8000。**跑通不等于可比。**

### 怎么重算 / 重出报告

```bash
set -a && . ./.env && set +a                        # 只有 retrace / 标注需要 key
python experiments/analyze_flow_profile.py          # 读 store + labels，全离线
python experiments/report/build_flow_profile.py     # 填模板出 HTML
```

`analyze_flow_profile.py` 不联网、不花钱，几秒钟跑完，写出
`findings/data/flow-profile.json` 和 `flow-profile-per-debate.csv`。
报告页发布在 https://claude.ai/code/artifact/756f477f-f148-409e-9434-301220ae3f56
（同一个 URL 原地更新）。结论与局限见 `findings/2026-09-01-flow-profile.md`。

### 过时 / 不要用

- `*.store.json` — 原始 light_match 产物，合并率 5.1%，**不可用于任何结论**
- `*.v3.json` — 中间态，已删
- `manifest.json` 里的 `trace_model: glm-5.3-flash` — **过时**，实际抽取模型是 minimax-m2.5
- `/tmp/factflow-stance-all/`、`/tmp/factflow-token-clock-all/` — 已导出成
  `experiments/labels/`，`/tmp` 里那两份随时会没
- `/tmp/factflow-stance-labeled-*` 的几个旧目录 — 参数与现行不一致，两套不可混用

### 想看原文就用轨迹查看器（2026-09-02）

`findings/trace-viewer.html`（4.1 MB，双击即开，**全部 141 场**：deepseek 108 + glm 33）。
顶上有模型筛选。
按 claim/拓扑/人格选一场，逐 turn 显示**全部原文**，第 1 轮三人并排。
抽取到事实的片段高亮，hover 出规范文本、立场、全部出现位置，点按钮可跳转。

**它最有用的地方是判断"这是传输还是巧合"**：三人读同一份材料，
两个 turn 出现同一个事实完全可能是各自推出来的。查看器按投递记录把每一处标成
`并行`（第 1 轮互不可见）/ `收到过`（可能是传输）/ `未收到`（独立重推）/ `自持`，
连线也按这个用实线虚线区分。实际点几个就会看到大多数重合是巧合。

span 定位命中 94.9%（13537/14259；deepseek 95.8%、glm 92.7%）；
GLM 的 VERDICT 只有 79.5% 可解析（deepseek 972/972 全中），所以 GLM 的场次会有 turn 没有结论色块——
这是那批 run 本身的性质，不是渲染 bug，页面上按场标了出来。`quote` 对不上原文的会计入"未定位"并在页面上报出来，
不是静默丢弃。重建：`python experiments/build_trace_view.py` 然后
`python experiments/report/build_trace_viewer.py`。

### 报告页有两份产物，别搞混（2026-09-02）

`experiments/report/build_*.py` 每次写两个文件：

| 文件 | 用途 | 在 git |
|---|---|---|
| `findings/<name>.html` | **人打开的**，完整 HTML 文档 | ✅ |
| `experiments/report/<name>.fragment.html` | **发 Artifact 的**，无 doctype/html/body | ❌ 已 ignore |

原因：模板是给 Artifact 主机写的，主机发布时会自己包一层
`<!doctype>…<head>…</head><body>`。所以直接存盘的文件没有这半边，
**本地双击会掉进 quirks 模式、排版全乱**——之前三个页面都是这个状态。
反过来，把完整文档交给主机会嵌套两层（实测能渲染，但靠浏览器丢弃游离标签）。
两份分开就都不用赌。包装逻辑在 `experiments/report/wrap.py`。

### ⚠️ 代码与数据已经不同步（2026-09-02）

`extract_facts` 现在有 `attempts=3`（空结果重试），**但仓库里 108 个 store 是用
`attempts=1` 跑出来的**。任何新的 retrace 都会得到一个和现有语料不可比的语料——
不是因为参数不同，而是因为**抽取本身不确定**：重跑一场会把每个 turn 都重抽，
重新原子化、重新聚类，处处都变。

**所以：不要为了"修几个空 turn"去重跑局部。** 要么整个 108 场一起重跑成一个新语料，
要么保持现状。混合抽样比不修更糟，而且偏差和拓扑共变（受影响的 27 场是
12 star + 12 chain + 3 full）。

### 已知缺陷

- **star 有 14 个空 turn**（4.3%，full 只有 0.9%）。turn 有实质内容（93–128 词）
  但抽取返回空。persona 分布均匀（lenses 5 / neutral 5 / stance 4），
  位置 A(中心) 4 : B/C(外围) 10，大致按人数比。不系统偏向某条件，
  但缺失率高 5 倍是拓扑阴性结果的待排除解释。猜测：B/C 只看到 A 一份输入，
  输出更可能是低信息量的附和——**未验证**。
- `add_gold` 报 `reached 0/41`，gold coverage 那一行是坏的，没修。
- VERDICT 在 324/324 个 turn 里可解析，**还没当过结果变量用**——现成的 outcome measure。

### 耗时构成（实测，2026-09-01）

抽取+原子化 96s，identify 37s —— **抽取占 72%**。调用次数上 identify 是大头
（60 vs 17），但它是批量并发的。**要提速该动抽取，不是 identify。**
NLI 预筛省的是 identify 调用，方向是错的。

### 别的 agent 未提交的工作（2026-09-01 时点）

工作区里有另一个 agent 的两个未跟踪脚本，我**没有**提交，因为那是他们正在写的东西：

- `experiments/plot_factflow_dashboard.py`
- `experiments/plot_uptake_trajectory.py`

他们对 `experiments/run_perspectrum.py` 的改动（从 `.env` 自动读 `OPENCODE_API_KEY`）
我提交了——那是个完整的小改动，而且它是生成全部辩论的脚本，
不提交的话仓库里的版本就跑不出现有数据。

`experiments/perspectrum_pilot_star_chain/*chain*.v2.json` 只提交了跑完的那几个，
retrace 还在跑；跑完 36 个再一起提交。

### 当前在跑

- **chain retrace**（2026-09-01 16:15 起，约 2.5h，~$4）：

  ```bash
  set -a && . ./.env && set +a
  .venv/bin/python experiments/retrace.py experiments/perspectrum_pilot_star_chain \
      --glob '*chain*.debate.json' --model minimax-m2.5 --suffix .v2.json \
      --record-token-clock
  ```

  日志 `/tmp/chain_retrace.log`，进度看
  `ls experiments/perspectrum_pilot_star_chain/*chain*.v2.json | wc -l`（目标 36）。
  跑完还要：立场标注 → `export_labels.py` 导出 → `analyze_flow_profile.py
  --topologies full star chain` → 重出报告。

  ⚠️ 之前挂过一个"等 star 跑完自动启动 chain"的 monitor，它只是在等文件出现，
  **并没有真的启动 retrace**，白等了几小时。挂 monitor 时确认它真的会执行命令。
