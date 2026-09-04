# 原子事实匹配:本地模型评测集

判决任务和生产管线一致:给两条原子事实,判 `SAME`(断言同一命题)还是
`DIFFERENT`(实体、数量、条件之一不同,或一条只是从另一条推出)。

## 集合构成:152 对

| 来源 | 数量 | 标注者 |
|---|---:|---|
| `bench.RELATION_PROBES` | 21 | 仓库已有,手写并带理由 |
| `clinicalbench_calibration` | 30 | 仓库已有,含 bge 余弦和 minimax 判决 |
| 近邻难例 `near-*` | 61 | **Claude 读原文标注,需复核** |
| 管线合并对 `merge-*` | 40 | **Claude 读原文标注,需复核** |

gold 分布 SAME 46 / DIFFERENT 106。领域:临床 74、perspectrum 34、
多数据集 23、通用 21。

**101 对是我标的,不是人工金标准。** 先抽查再拿它下结论,尤其是
`contested` 那 26 对。

## contested 标记

标 `contested: true` 的是认真读也可能判反的:一边给了具体数值另一边只说
"升高"、一边带限定另一边脱落。用 `--contested drop` 得到保守估计,
`--contested only` 单看这批的分歧程度。

## 两类难例的来源不同

`near-*` 从 blocker 的 0.62–0.99 相似度带里分层抽样,是**判决器真正要做决定**
的区域,天然偏 DIFFERENT。

`merge-*` 是管线已经合并进同一簇的两条 mention 文本,本意是补 SAME,但读下来
其中 24 对是**错误合并**——例如"黄疸阳性"并进了"胆红素 41.5"、"头晕"并进了
"心悸"、"CT/MRI 显示食管静脉曲张"并进了"EGD 确认食管静脉曲张破裂"。这些是
集合里最有价值的负例,因为现有管线在它们上面是错的。

## 基线

```
bge-base-en-v1.5 (biencoder, 最佳阈值 0.813)
  准确率 0.605  SAME 精确率 0.424  召回 0.848  F1 0.565
```

单靠双编码器不够:它把"药物降低 65 岁以上患者死亡率"判成等同于"药物降低
死亡率"。这条基线的意义是给候选模型一个必须显著超过的下限。

## 跑法

```
python run_local_matcher.py --kind biencoder --model BAAI/bge-base-en-v1.5
python run_local_matcher.py --kind reranker  --model BAAI/bge-reranker-v2-m3
python run_local_matcher.py --kind llm --model Qwen/Qwen3-14B-Instruct --dtype bfloat16
```

`biencoder` 和 `reranker` 输出分数,脚本扫 41 个阈值取最佳 F1,并把整条曲线写进
`--out`;阈值是拟合出来的,所以那个 F1 是乐观上界,真要用得留独立的调阈子集。
`llm` 走约束输出,只取首词,没有阈值可调。
