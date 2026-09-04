#!/usr/bin/env bash
# 原子事实匹配 —— 本地模型评测入口
#
#   ./run.sh                          跑推荐的一组候选
#   ./run.sh BAAI/bge-reranker-v2-m3  只跑指定模型（可给多个）
#   ./run.sh --list                   只列出推荐清单，不下载不运行
#   ./run.sh --check                  只查 GPU 和 wheel 是否匹配，不加载模型
#
# 都不需要先 source venv：脚本自己解析 setup_env.sh 建的解释器。
#
# 模型类型自动识别，小模型走 GPU_SMALL、LLM 走 GPU_LARGE。
set -euo pipefail

# ==== 服务器上只需要改这一段 =============================================
# 一切缓存的根。这里放的是你给的 scratch 目录。
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/users/jiajun}"

# 小模型（biencoder / reranker，2GB 级）跑哪块卡。
# GPU 3 是 2080 Ti，11GB，空闲；Turing 架构不支持 bf16，但这类模型走 fp32。
GPU_SMALL="${GPU_SMALL:-3}"

# LLM 跑哪块卡。GPU 4 是 RTX 6000 Ada，46GB 全空。
# GPU 0 只剩 25GB，GPU 1 满载，GPU 2 显存空但利用率 100%。
GPU_LARGE="${GPU_LARGE:-4}"
# =========================================================================

if [[ ! -d "$SCRATCH_ROOT" ]]; then
  echo "找不到 SCRATCH_ROOT=$SCRATCH_ROOT" >&2
  echo "改脚本顶部的 SCRATCH_ROOT，或 SCRATCH_ROOT=... ./run.sh" >&2
  exit 1
fi

# hf-datasets 是你已经建好的空目录。HF_HOME 之下会自动分出
# hub/（权重，14B bf16 约 30GB）、datasets/、modules/，所以只需要这一个变量。
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf-datasets}"

# 下面这些是旧版库仍在读的变量名，以及几个默认会写进家目录的缓存。
# 不设的话权重和临时文件会散落到 ~，几个模型就能把家目录分区塞满。
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME/sentence-transformers"
export TORCH_HOME="$SCRATCH_ROOT/torch"          # torch.hub
export XDG_CACHE_HOME="$SCRATCH_ROOT/xdg"        # 兜底
export TRITON_CACHE_DIR="$SCRATCH_ROOT/triton"   # torch.compile / triton 内核
export TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp}"     # 分片下载的临时文件也很大
export PIP_CACHE_DIR="$SCRATCH_ROOT/pip-cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1               # 装了 hf_transfer 才生效，没装无害
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$SENTENCE_TRANSFORMERS_HOME" \
         "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TMPDIR"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# setup_env.sh 建的 venv，找到就用。PYTHON=... 可以覆盖。
VENV_DIR="${VENV_DIR:-$SCRATCH_ROOT/venv-matcher}"
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$VENV_DIR/bin/python" ]]; then
  PY="$VENV_DIR/bin/python"
else
  echo "没找到 $VENV_DIR/bin/python" >&2
  echo "先跑 ./setup_env.sh，或 PYTHON=/path/to/python ./run.sh" >&2
  exit 1
fi

# 46GB 单卡放得下的候选，从便宜到贵。前三个几分钟出结果，
# 够好的话就不必上 LLM —— 判决器要跑几十万次，成本差两个数量级。
DEFAULT_MODELS=(
  "BAAI/bge-base-en-v1.5"                 # 基线：现有 blocker，F1 0.565
  "BAAI/bge-reranker-v2-m3"               # ~2.5GB
  "cross-encoder/nli-deberta-v3-large"    # ~1.8GB，NLI 训练过
  "Qwen/Qwen3-8B"                         # ~17GB
  "Qwen/Qwen3-14B"                        # ~30GB，46GB 卡的上限
  "microsoft/phi-4"                       # ~30GB
)

if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' "${DEFAULT_MODELS[@]}"
  echo
  echo "更大的会自动加 --4bit："
  echo "  ./run.sh google/gemma-3-27b-it"
  echo "  ./run.sh mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  exit 0
fi

# 只做 GPU 检查然后退出。不用先激活 venv —— 上面已经解析好解释器了。
if [[ "${1:-}" == "--check" ]]; then
  echo "python  $PY"
  echo "HF_HOME $HF_HOME"
  echo
  exec "$PY" "$HERE/check_gpu.py"
fi

MODELS=("$@")
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("${DEFAULT_MODELS[@]}")

echo "HF_HOME      = $HF_HOME"
echo "TMPDIR       = $TMPDIR"
echo "小模型 → GPU $GPU_SMALL   LLM → GPU $GPU_LARGE"
df -h "$SCRATCH_ROOT" 2>/dev/null | tail -1 | sed 's/^/  磁盘 /' || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader 2>/dev/null | sed 's/^/  /' || true
echo

# 装好的 wheel 未必带每块卡的内核。不先查的话，报错会在第一次 kernel
# launch 时才出现（no kernel image available），那时模型权重都已经装载完了。
echo "--- GPU 兼容性 ---"
"$PY" "$HERE/check_gpu.py" || exit 1
for g in "$GPU_SMALL" "$GPU_LARGE"; do
  if ! CUDA_VISIBLE_DEVICES="$g" "$PY" -c "
import torch,sys
try:
    torch.zeros(8, device='cuda').sum().item()
except Exception as e:
    sys.exit(str(e)[:120])
" 2>/dev/null; then
    echo "!! GPU $g 跑不了当前 torch。改 GPU_SMALL / GPU_LARGE，或换 torch wheel" >&2
    exit 1
  fi
done
echo

mkdir -p "$HERE/results"
T_ALL=$SECONDS
N=${#MODELS[@]}
i=0
for m in "${MODELS[@]}"; do
  i=$((i + 1))
  T_ONE=$SECONDS
  extra=()
  case "$m" in
    *27b*|*27B*|*32b*|*32B*|*24B*|*24b*|*70b*|*70B*) extra+=(--4bit) ;;
  esac
  kind="$("$PY" -c "import sys;sys.path.insert(0,'$HERE');\
from run_local_matcher import detect_kind;print(detect_kind('$m'))")"
  if [[ "$kind" == "llm" ]]; then
    dev="$GPU_LARGE"
  else
    dev="$GPU_SMALL"
  fi
  echo "======================================================================"
  echo "[$i/$N] $m   [$kind → GPU $dev] ${extra[*]:-}   $(date +%H:%M:%S)"
  # keep = 全部 152 对；drop = 去掉 26 对有争议的，保守估计
  for c in keep drop; do
    CUDA_VISIBLE_DEVICES="$dev" \
      "$PY" "$HERE/run_local_matcher.py" --model "$m" --contested "$c" "${extra[@]}" \
      || { echo "!!! $m ($c) 失败，跳过"; break; }
  done
  echo "[$i/$N] $m 用时 $((SECONDS - T_ONE))s"
done

echo
echo "全部完成，总用时 $((SECONDS - T_ALL))s"
echo "======================================================================"
"$PY" - "$HERE/results" <<'PYEOF'
import json, sys, glob
from pathlib import Path
rows = []
for f in sorted(glob.glob(str(Path(sys.argv[1]) / "*.json"))):
    d = json.load(open(f)); b = d["best"]
    rows.append((d["model"], d["kind"], d["contested"], b["f1"],
                 b["same_precision"], b["same_recall"], b["acc"],
                 d["seconds"] / max(1, b["n"]) * 1000))
if not rows:
    sys.exit("没有结果文件")
print(f'{"模型":<44}{"类型":<11}{"集合":<6}{"F1":>7}{"精确":>7}{"召回":>7}{"准确":>7}{"ms/对":>8}')
for r in sorted(rows, key=lambda x: -x[3]):
    print(f'{r[0][:43]:<44}{r[1]:<11}{r[2]:<6}{r[3]:>7.3f}{r[4]:>7.3f}'
          f'{r[5]:>7.3f}{r[6]:>7.3f}{r[7]:>8.0f}')
PYEOF
