#!/usr/bin/env bash
# 原子事实匹配 —— 本地模型评测入口
#
#   ./run.sh                          跑推荐的一组候选
#   ./run.sh BAAI/bge-reranker-v2-m3  只跑指定模型（可给多个）
#   ./run.sh --list                   只列出推荐清单，不下载不运行
#
# 模型类型自动识别；除下面这两行外不需要改任何东西。
set -euo pipefail

# ==== 服务器上只需要改这两行 ==============================================
# 模型权重下载到哪。14B 的 bf16 约 30 GB，27B 约 55 GB，别放家目录。
export HF_HOME="${HF_HOME:-/data/$USER/hf}"
# 用哪块卡。GPU 4 是 RTX 6000 Ada 46GB 且空闲；GPU 2 显存空但利用率 100%。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
# =========================================================================

# 这三个是旧版变量名，新版 transformers/sentence-transformers 读 HF_HOME，
# 但装了旧版的话它们仍然生效，一起设上免得权重散落到家目录。
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export SENTENCE_TRANSFORMERS_HOME="${HF_HOME}/sentence-transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TOKENIZERS_PARALLELISM=false

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

# 46GB 单卡放得下的候选，从便宜到贵。前两个几分钟出结果，
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
  echo "更大的要加 --4bit，例如："
  echo "  ./run.sh google/gemma-3-27b-it            # 脚本会自动补 --4bit"
  echo "  ./run.sh mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  exit 0
fi

MODELS=("$@")
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("${DEFAULT_MODELS[@]}")

echo "HF_HOME=$HF_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader 2>/dev/null | sed 's/^/  /' || true
echo

mkdir -p "$HERE/results"
for m in "${MODELS[@]}"; do
  extra=()
  # 24B 以上在 46GB 卡上必须量化
  case "$m" in
    *27b*|*27B*|*32b*|*32B*|*24B*|*24b*|*70b*|*70B*) extra+=(--4bit) ;;
  esac
  echo "======================================================================"
  echo ">>> $m ${extra[*]:-}"
  # keep = 全部 152 对；drop = 去掉 26 对有争议的，保守估计
  for c in keep drop; do
    "$PY" "$HERE/run_local_matcher.py" --model "$m" --contested "$c" "${extra[@]}" \
      || { echo "!!! $m ($c) 失败，跳过"; break; }
  done
done

echo
echo "======================================================================"
"$PY" - "$HERE/results" <<'PYEOF'
import json, sys, glob
from pathlib import Path
rows = []
for f in sorted(glob.glob(str(Path(sys.argv[1]) / "*.json"))):
    d = json.load(open(f))
    b = d["best"]
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
