# findings/

一个发现一个文件，命名 `YYYY-MM-DD-<slug>.md`。
带图表的发现可以是 `.html`（由 `experiments/report/` 下的模板 + `findings/data/` 的
JSON 生成，**不要手改**，改模板重跑生成器）。

每个文件必须做到：**只看这一个文件，就能找到数据在哪、结果在哪、怎么重跑。**
Perspectrum 那条线的三份分析已经合并进
[2026-09-02-fact-flow-report.html](2026-09-02-fact-flow-report.html)，
那份是自明的（不需要任何项目前置知识），其余三份保留作为过程记录。
不要求读者先读代码或翻聊天记录。写的时候按这个顺序：

1. **结论** — 一句话，可证伪的那种
2. **数据** — 精确路径、多少条、哪个模型跑的、是否在 git 里
3. **怎么重跑** — 能直接粘贴执行的命令
4. **结果** — 表和数字，以及它们来自哪个 JSON
5. **站不住的地方** — 样本量、混淆变量、估算代替真值的地方
6. **下一步** — 这个发现指向的下一个实验

## 索引

| 日期 | 发现 | 状态 |
|---|---|---|
| 2026-09-02 | [每个 agent 都在丢弃支持性证据](2026-09-02-support-attrition.md) | **卷宗平衡，输出偏斜；neutral 面板 SUP −14.2pp；只有支持方能持平** |
| 2026-09-02 | **[辩论轨迹查看器](trace-viewer.html)** 🔍 | **全部 141 场逐 turn 原文（含 GLM 33 场）；片段高亮、可点、可跳转；连线区分传输与巧合。做 sanity check 从这里开始** |
| 2026-09-02 | **[事实在多智能体辩论里怎么流动](2026-09-02-fact-flow-report.html)** 📊 | **统一报告，自明；合并下面三份，新读者从这里开始** |
| 2026-09-02 | [RQ review 与新增分析：接触、选择门与功能性整合](2026-09-02-rq-review-and-new-analyses.md) | **108 场；chain 交互、采纳留存漏斗、两跳级联、verdict 脱钩** |
| 2026-09-01 | [事实流剖面：三拓扑 × 三档人格](2026-09-01-flow-profile.html) 📊 | **108 场；拓扑买接收率，人格买条件采纳率；流动解释不了判决** |
| 2026-09-01 | [现有数据支持的 RQ 结论与新假设](2026-09-01-current-data-rq-hypotheses.md) | DeepSeek n=12、GLM n=11；探索性 |
| 2026-09-01 | [共享实验状态与运行前检查](2026-09-01-shared-experiment-status.md) | 操作前必读；记录 canonical artifact |
| 2026-08-31 | [Token 对齐的 lifecycle：persona 更改 churn，多于 uptake](2026-08-31-deepseek-token-lifecycle.md) | DeepSeek、full topology，n=12 |
| 2026-08-31 | [Persona、topology 与 fact flow 的实验设计](2026-08-31-persona-topology-fact-flow-design.md) | 设计备忘录，待预注册与验证 |
| 2026-08-31 | [Persona 改变事实人口学，不自动增加信息流](2026-08-31-perspectrum-deepseek-personas.md) | DeepSeek 初步分析，n=12 |
| 2026-09-01 | [主假设：整合能力决定多样性是否转化为质量](2026-09-01-integration-capacity-hypothesis.md) | 待办清单，非结论 |
| 2026-08-31 | [数量测不到的，立场组成测到了](2026-08-31-stance-balance.md) | 初步，唯一 CI 不含零的 persona 效应 |
| 2026-08-29 | [修好匹配：五个真因，和四次我自己推翻的解释](2026-08-29-fixing-the-matcher.md) | 流水线已重建，三条件结果待出 |
| 2026-08-28 | [匹配是瓶颈：Perspectrum 试点暴露的 infra 问题](2026-08-28-matching-is-the-bottleneck.md) | 诊断完成，方案待实施 |
| 2026-08-28 | [事实人口学：生成–淘汰轨迹区分了准确率相同的配置](2026-08-28-fact-demography.md) | 初步，n=8 |

## 关于原始数据

三层，越往下越小、越可靠：

| 层 | 位置 | 在 git 里 | 说明 |
|---|---|---|---|
| 原始 trace / sweep 产物 | `experiments/*_out/` | ❌ | 被 `.gitignore` 忽略，只在本机 |
| fact store | `experiments/perspectrum_pilot_*/*.v2.json` | ✅ | Perspectrum 的 105 场，36 MB |
| 标注层 | `experiments/labels/{stance,token_clock}/` | ✅ | 只存标注本身，8.7 MB |
| 算完的结果 | `findings/data/*.json`、`*.csv` | ✅ | 几十 KB，即使 store 丢了也对得上 |

`.html` 的发现由生成器产出，源模板在 `experiments/report/`，数据在 `findings/data/`。
重跑 `python experiments/report/build_flow_profile.py` 即可更新，别手改生成物。

**标注不要以 enriched store 的形式提交。** `label_fact_stance.py` 和 `token_clock.py`
都是"复制整个 store + 加一个字段"，两遍标注就是 75 MB 的逐字节副本。
用 `experiments/export_labels.py` 导出成 sidecar，用它的 `attach_labels()` 挂回去。

要让某个发现完全可复现，就得把它依赖的 `.store.json` 显式加进 git（`git add -f`）。
目前还没有这么做过。
