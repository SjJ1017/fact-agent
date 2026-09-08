# 这批语料作废，别用来做拓扑对比

生成时 loader 漏掉了 `Case.public`，提示词里少了这一行：

> IDRBench positive-pair idea-generation task. The pair has been selected as a
> valid integration candidate, but no target paper or reference explanation is
> available to the panel.

参考语料 `idrbench_generation_10x5_r3`（full 拓扑）**有**这一行，这批**没有**。
所以两批之间的差异混杂了拓扑和提示词框架，不可比。

证据：split 条件第 1 轮（此时尚无任何通信，拓扑在设计上不起作用）
同轮等价重合率，有框架行 17.2%、无框架行 2.3%，p<0.001；
而同一批内 star 与 chain 相差 0.4pp、p=0.80，说明测量本身稳定。

修复见 commit 560a040，回归测试 `tests/test_idrbench_prompt_parity.py`。

**保留的原因**：它和重新生成的 v2 构成一个"同任务、只差一行框架"的对照，
可以用来实测提示词框架的敏感度。钱已经花了，别浪费这个意外的对照组。
