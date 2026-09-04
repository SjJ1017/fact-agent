#!/usr/bin/env bash
# 建评测用的 venv。跑一次就行，之后 run.sh 会自动找到它。
#
#   ./setup_env.sh                     默认 cu126 的 torch
#   TORCH_INDEX=cu124 ./setup_env.sh   换 CUDA wheel
#   VENV_DIR=/somewhere ./setup_env.sh 换位置
set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/users/jiajun}"
if [[ ! -d "$SCRATCH_ROOT" ]]; then
  echo "找不到 SCRATCH_ROOT=$SCRATCH_ROOT" >&2
  echo "改本脚本顶部，或 SCRATCH_ROOT=... ./setup_env.sh" >&2
  exit 1
fi

# venv 建在 scratch 而不是家目录：带 CUDA 的 torch 解包后约 3-5GB。
VENV_DIR="${VENV_DIR:-$SCRATCH_ROOT/venv-matcher}"
# pip 的缓存同样能长到几个 GB。
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_ROOT/pip-cache}"
export TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# 驱动 580.95 / CUDA 13.0 向下兼容 cu12x 的 wheel。
# 评测只用 GPU 3 (Turing sm_75) 和 GPU 4 (Ada sm_89)，两者 cu12x 都支持。
# 要用 GPU 1/2 那两块 Blackwell (sm_120) 就得换更新的 wheel。
TORCH_INDEX="${TORCH_INDEX:-cu126}"
BASE_PY="${BASE_PY:-python3}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "venv        $VENV_DIR"
echo "pip cache   $PIP_CACHE_DIR"
echo "torch wheel $TORCH_INDEX"
df -h "$SCRATCH_ROOT" | tail -1 | sed 's/^/磁盘        /'
echo

"$BASE_PY" -m venv "$VENV_DIR"
PY="$VENV_DIR/bin/python"
"$PY" -m pip install -q --upgrade pip wheel

echo ">>> torch ($TORCH_INDEX)"
"$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"

echo ">>> 其余依赖"
"$PY" -m pip install -r "$HERE/requirements.txt"

echo
echo ">>> 自检"
"$PY" - <<'PYEOF'
import torch, transformers, sentence_transformers
print(f"  torch                 {torch.__version__}")
print(f"  transformers          {transformers.__version__}")
print(f"  sentence-transformers {sentence_transformers.__version__}")
print(f"  CUDA 可用             {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  编译时 CUDA           {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        bf16 = "bf16 ok" if p.major >= 8 else "无 bf16 (Turing)"
        print(f"    [{i}] {p.name}  {p.total_memory/2**30:.0f}GB  "
              f"sm_{p.major}{p.minor}  {bf16}")
else:
    print("  !! 没检测到 CUDA，检查 torch wheel 和驱动是否匹配")
try:
    import bitsandbytes
    print(f"  bitsandbytes          {bitsandbytes.__version__}  (--4bit 可用)")
except Exception as e:
    print(f"  bitsandbytes          不可用: {type(e).__name__}  (--4bit 会失败)")
PYEOF

echo
echo "装好了。直接跑："
echo "  ./run.sh"
echo "run.sh 会自动用 $VENV_DIR；也可以 PYTHON=$PY ./run.sh 显式指定。"
