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

要超过的是**生产判决器在同一个集合上的成绩**,不是标定集里那个 0.967
(那是 30 对同域临床配对,比这个集合容易得多):

```
minimax-m2.5  (hosted, 生产协议, 152 对)
  准确率 0.849  SAME 精确率 0.702  召回 0.870  F1 0.777   211s
  错 23 条，其中 12 条是 contested
  错判方向: DIFFERENT→SAME 17 条，SAME→DIFFERENT 6 条
```

它**偏向合并**,这解释了语料里那 24 组错误合并的来源。

下限:

```
bge-base-en-v1.5 (biencoder, 最佳阈值 0.813)
  准确率 0.605  SAME 精确率 0.424  召回 0.848  F1 0.565
```

自己复现 hosted 基线:

```
python baseline_hosted.py --model minimax-m2.5 --protocol production
python baseline_hosted.py --model minimax-m2.5 --protocol oneword
```

两种协议不同:生产是批量 JSON、先写一句差别再判;oneword 是本地脚本用的
单词输出。跨协议比 F1 会混进协议差异。

## 跳过已跑过的

默认跳过。判据是结果文件是否存在(同 模型/模式/contested/难度):

```
./run.sh                          # 已完成的直接跳过，只补没跑的
REDO=1 ./run.sh                   # 强制全部重跑
./run.sh --from Qwen/Qwen3-8B     # 从清单里这个开始，前面的不碰
```

换 `MODE` 或 `--difficulty` 会写不同文件名,所以不会被误判成"已跑过"。

## 结果聚合到一个文件

每次跑完除了写自己那份带预测的结果,还会 upsert 一行进
`results/summary.json`,主键是(模型, 模式, contested, 难度)。重跑同一格只会
覆盖那一行,不会留下两条。

`run.sh` 最后的表就是读这个文件,所以中途 Ctrl-C 再续跑也能看到完整的表:

```
模型                              类型/模式          集合   难度      F1    精确    召回    准确  ms/对
BAAI/bge-base-en-v1.5             biencoder/logit  drop   all    0.840  0.878  0.806  0.924    25
BAAI/bge-base-en-v1.5             biencoder/logit  keep   all    0.790  0.818  0.764  0.898    25
```

单独看表:

```
python -c "import json;[print(r['model'], r['mode'], r['f1']) for r in json.load(open('results/summary.json'))]"
```

## 怎么开 cot

```
MODE=cot ./run.sh                      # 整组都用 cot
MODE=cot ./run.sh Qwen/Qwen3-14B       # 只跑一个
```

单独跑:

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
  $VENV/bin/python run_local_matcher.py --model Qwen/Qwen3-14B --mode cot --batch 4
```

`MODE` 只作用于 LLM,reranker / biencoder 没有这个维度,会被忽略。

**结果存在不同文件里**,不会覆盖。命名是
`<模型>.<mode>.<contested>.<难度>.json`:

```
Qwen__Qwen3-14B.logit.keep.all.json
Qwen__Qwen3-14B.cot.keep.all.json
Qwen__Qwen3-14B.cot.keep.hard.json
```

`run.sh` 最后的汇总表有"类型/模式"列,同一模型的两种模式会并排列出,直接比。

cot 的结果文件里每条还多存一个 `note` 字段,是模型自己写的那句"差别在哪",
错判时也会打出来 —— 用来看它是判断错了还是理解错了。

## 三种推理模式

| mode | 做法 | 速度 | 何时用 |
|---|---|---|---|
| `logit` | 读首个回答 token 上 SAME/DIFFERENT 的概率差 | 最快 | 扫模型 |
| `oneword` | 解码一个词 | 中 | 看模型自由行为 |
| `cot` | 先写一句"差别在哪"再判,固定格式 | 最慢 | 定稿选型 |

`cot` 的 prompt 里写死了几条规则,其中数值容差是按实际需求定的:

- `12.3%` 和 `roughly 12%` 判 SAME;`41.5` 和 `elevated` 描述同一测量时判 SAME
- 名称变体、缩写、增删的模糊限定都算 SAME
- 改变**覆盖哪些情况**的限定词算 DIFFERENT
- 输出固定为 `DIFF: <≤8 词>` 换行 `ANSWER: SAME|DIFFERENT`

模型写的那句差别会存进结果文件,错判时一并打出来。

**注意 gold 和这条规则有冲突。** `probe-precision-loss-compatible`
(`12.3%` vs `roughly 12%`)在仓库原有的探针里标的是 DIFFERENT,而 cot 的
prompt 要求判 SAME。这类配对需要先定清楚要哪种语义,再改 gold 或改 prompt,
不能两边各行其是。

## 跑法

```
python run_local_matcher.py --kind biencoder --model BAAI/bge-base-en-v1.5
python run_local_matcher.py --kind reranker  --model BAAI/bge-reranker-v2-m3
python run_local_matcher.py --kind llm --model Qwen/Qwen3-14B-Instruct --dtype bfloat16
```

`biencoder` 和 `reranker` 输出分数,脚本扫 41 个阈值取最佳 F1,并把整条曲线写进
`--out`;阈值是拟合出来的,所以那个 F1 是乐观上界,真要用得留独立的调阈子集。
`llm` 走约束输出,只取首词,没有阈值可调。

## 在服务器上跑

```
./setup_env.sh                       # 建 venv，跑一次
./run.sh --check                     # 查 GPU 和 wheel 是否匹配
./run.sh                             # 跑推荐的一组
./run.sh BAAI/bge-reranker-v2-m3     # 只跑指定的（可给多个）
./run.sh --list                      # 只看清单
```

**都不用 source 任何东西。** `run.sh` 自己解析 `setup_env.sh` 建的解释器
(`$SCRATCH_ROOT/venv-matcher/bin/python`),找不到就报错并告诉你先跑
`setup_env.sh`。要用别的解释器就 `PYTHON=/path/to/python ./run.sh`。

`run.sh` 顶部有三行按机器改，其余全部由它们派生:

```
SCRATCH_ROOT=/scratch/users/jiajun    # 一切缓存的根
GPU_SMALL=3                          # biencoder / reranker 跑这块
GPU_LARGE=4                          # LLM 跑这块
```

`SCRATCH_ROOT` 不存在时脚本直接退出并说明，不会默默建一个错路径。

派生出来的环境变量:

| 变量 | 值 | 不设的后果 |
|---|---|---|
| `HF_HOME` | `$SCRATCH_ROOT/hf-datasets` | 权重进家目录，14B 一个就 30GB |
| `HF_HUB_CACHE` / `TRANSFORMERS_CACHE` | `$HF_HOME/hub` | 旧版库仍读这两个 |
| `HF_DATASETS_CACHE` | `$HF_HOME/datasets` | |
| `SENTENCE_TRANSFORMERS_HOME` | `$HF_HOME/sentence-transformers` | ST 自己另有一套缓存 |
| `TORCH_HOME` | `$SCRATCH_ROOT/torch` | torch.hub 权重 |
| `TRITON_CACHE_DIR` | `$SCRATCH_ROOT/triton` | 编译内核，会长到几个 GB |
| `XDG_CACHE_HOME` | `$SCRATCH_ROOT/xdg` | 兜底 |
| `TMPDIR` | `$SCRATCH_ROOT/tmp` | 分片下载的临时文件和权重同量级 |

## no kernel image is available for execution on the device

装好的 torch wheel 里**没有该卡架构的编译内核**。它在第一次 kernel launch 时才
报,那时 `torch.cuda.is_available()` 早已返回 True、权重也装完了,所以看起来像
运行时故障,其实是装环境时就注定的。

先查(不需要激活 venv,脚本自己找解释器):

```
./run.sh --check
```

它打出 wheel 编译支持的架构列表、每块卡的算力,并在每块卡上实际跑一次
kernel,最后给出建议的 `GPU_SMALL` / `GPU_LARGE`。`run.sh` 开跑前会自动做这件事。

这台机器上最可能出问题的是 **GPU 3(2080 Ti,sm_75,Turing)**:新版 torch 的
wheel 正在陆续去掉 Turing。两条路:

**一,不用那块卡。** 把小模型也放到 GPU 4——reranker 只要 2GB,和 LLM 分时跑
互不影响:

```
GPU_SMALL=4 ./run.sh
```

**二,换带 sm_75 的旧 wheel。** cu121 的 torch 2.4/2.5 仍带 Turing:

```
TORCH_INDEX=cu121 ./setup_env.sh
```

代价是那套 wheel 对 GPU 4(Ada,sm_89)也支持,但对 GPU 1/2 的 Blackwell
(sm_120)完全不支持。只跑 3 和 4 的话没问题。

推荐第一条:少一个变量,而且 reranker 在 46GB 卡上一样是秒级。

## triton / gcc 编译失败

```
subprocess.CalledProcessError: Command '['/usr/bin/gcc', '.../triton/backends/nvidia/driver.c', ...]'
returned non-zero exit status 1
```

Triton 首次使用时会用 gcc 现场编译一个 CUDA 驱动 shim。失败的常见原因:缺
CUDA 头文件、gcc 版本不合、或者 `TMPDIR` 指向的分区挂了 `noexec`(编出来的
`.so` 加载不了)。

这个评测不需要融合内核 —— 四百条短 prompt 各做一次前向,瓶颈不在这里。
`run.sh` 已经把这条路关掉:

```
TORCH_COMPILE_DISABLE=1  TORCHDYNAMO_DISABLE=1
TORCHINDUCTOR_DISABLE=1  DISABLE_KERNEL_MAPPING=1
FF_ATTN=eager                       # 注意力用 eager，不走 flash/triton
```

手动跑单个模型时要自己带上,至少 `FF_ATTN=eager`。想试别的注意力实现:
`FF_ATTN=sdpa`(纯 PyTorch,不用 triton)。

如果 scratch 确实是 noexec,把 `TMPDIR` 指回本地盘:

```
TMPDIR=/tmp ./run.sh
```

## CUDA 编号和 nvidia-smi 对不上

CUDA 运行时默认按 `FASTEST_FIRST` 枚举设备,所以 `CUDA_VISIBLE_DEVICES=4`
指的是**CUDA 自己排序里的第 4 块**,不是 nvidia-smi 显示的第 4 块。症状是
OOM 信息里报出的显存总量对不上你以为在用的卡:

```
GPU 0 has a total capacity of 10.57 GiB      # 这是 2080 Ti，不是 46GB 的 Ada
```

`run.sh` 和 `setup_env.sh` 都设了 `CUDA_DEVICE_ORDER=PCI_BUS_ID`,设了之后编号
与 nvidia-smi 一致。`./run.sh --check` 和开跑前的检查都会打出每个选定编号
**实际拿到的是哪块卡**,不用猜。

手动跑单个模型时记得自己带上:

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python run_local_matcher.py ...
```

## 两块卡怎么分

| GPU | 卡 | 状态 | 用途 |
|---|---|---|---|
| 0 | RTX 6000 Ada 46GB | 占用 20GB | 不用 |
| 1 | RTX PRO 6000 96GB | 满载 100% | 不用 |
| 2 | RTX PRO 6000 96GB | 显存空但利用率 100% | 不用（有东西在跑） |
| **3** | **2080 Ti 11GB** | **空闲** | **biencoder / reranker** |
| **4** | **RTX 6000 Ada 46GB** | **空闲** | **LLM** |

GPU 3 是 Turing 架构，**不支持 bf16**，所以只放 2GB 级的 reranker（走 fp32），
不要往上塞 LLM。GPU 4 的 46GB 放得下 bf16 的 14B（约 30GB），27B/32B 要
`--4bit`，脚本按模型名自动加。

## 看进度

每批打一行,带已用时间、预计剩余和速率,flush 过所以重定向到日志也能实时看:

```
>>> Qwen/Qwen3-14B  [llm]  152 对
  载入权重…（首次会先下载，几十 GB 时这一步最久）
  显存占用 29.4 GB
    打分 16/152     12s 已用     102s 剩余    1.3 对/秒
    打分 32/152     24s 已用      90s 剩余    1.3 对/秒
```

`run.sh` 另外打 `[第几个/共几个]`、每个模型的用时和总用时。要静默加 `--quiet`。

首次跑某个模型时,最久的一步是下载而不是推理。想先看下载进度可以单独拉:

```
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-14B')"
```

后台跑建议:

```
nohup ./run.sh > run.log 2>&1 &
tail -f run.log
```

## LLM 走的是 logit 不是生成

默认不解码,只做一次前向,比较首个回答 token 上 `SAME` 与 `DIFFERENT` 的
log 概率之和,取差值作为分数。三个好处:一次前向而不是解码循环,快得多;
模型不可能跑偏格式(今天 `deepseek-v4-pro` 就是在别的任务上返回空);
得到的是连续分数,所以和 reranker 一样能扫阈值、能看校准。

要看模型自由生成的行为就加 `--generate`,那条路径没有阈值。

## 阈值是拟合出来的

`biencoder` / `reranker` / logit 模式的 F1 都来自在本集合上扫 81 个阈值取最优,
**是乐观上界**。真要选型定阈值,得另留一个调阈子集,否则会高估。
`--generate` 模式没有这个问题。
