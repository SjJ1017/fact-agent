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

# Triton compiles a CUDA shim against Python.h the first time a model touches
# a fused kernel.  A system python without its -dev package has no headers, and
# the failure surfaces much later as "gcc returned non-zero exit status 1" with
# no mention of a missing header.  Check now, while the fix is still cheap.
HDR="$("$BASE_PY" -c 'import sysconfig,os;print(os.path.join(sysconfig.get_paths()["include"],"Python.h"))')"
if [[ ! -f "$HDR" ]]; then
  echo "!! $BASE_PY 没有 Python.h（找的是 $HDR）" >&2
  echo "   triton 需要它编译 CUDA shim，缺了会在跑模型时才报 gcc 失败。" >&2
  echo "   两条路：" >&2
  echo "     sudo apt install python3-dev      # 需要 root" >&2
  echo "     BASE_PY=~/miniconda3/bin/python3 ./setup_env.sh   # conda 自带头文件" >&2
  echo "   下面会自动卸掉 triton 绕过它（这个评测不需要融合内核）。" >&2
fi
echo "Python.h    $HDR"

"$BASE_PY" -m venv "$VENV_DIR"
PY="$VENV_DIR/bin/python"
"$PY" -m pip install -q --upgrade pip wheel

echo ">>> torch ($TORCH_INDEX)"
"$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"

echo ">>> 其余依赖"
"$PY" -m pip install -r "$HERE/requirements.txt"

# Without Python.h, triton's shim build fails the first time a model reaches
# for a fused kernel.  Uninstalling it is the fix that needs no root: torch and
# transformers handle a genuinely absent triton, and nothing in this eval wants
# a fused kernel anyway.  (Blocking the import is NOT equivalent -- find_spec
# then raises instead of returning None, and has_triton() dies with it.)
if [[ ! -f "$HDR" || -n "${DROP_TRITON:-}" ]]; then
  echo ">>> 卸载 triton（缺 Python.h 或显式要求）"
  "$PY" -m pip uninstall -y triton 2>/dev/null || true
fi

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
