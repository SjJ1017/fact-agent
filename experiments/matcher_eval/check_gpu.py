"""Which of this box's GPUs the installed torch can actually run on.

`CUDA error: no kernel image is available for execution on the device` means
the wheel carries no compiled kernels for that card's compute capability.  It
surfaces at the first kernel launch, long after `torch.cuda.is_available()`
has already said True, so check here rather than mid-run.
"""
import sys

import torch

archs = torch.cuda.get_arch_list()
print(f"torch {torch.__version__}  (CUDA {torch.version.cuda})")
print(f"wheel 编译支持: {' '.join(archs)}")
print()

if not torch.cuda.is_available():
    sys.exit("CUDA 不可用")

usable = []
print(f"{'GPU':<4}{'名称':<34}{'显存':>8}{'算力':>8}{'bf16':>7}  可用")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    cap = f"sm_{p.major}{p.minor}"
    ok = any(a.startswith(f"sm_{p.major}{p.minor}") for a in archs)
    if not ok:  # a newer card can run on an older sm via PTX JIT
        ok = any(a.startswith("compute_") and
                 int(a.split("_")[1]) <= p.major * 10 + p.minor for a in archs)
    if ok:
        try:
            torch.zeros(8, device=f"cuda:{i}").sum().item()
        except Exception as exc:
            ok = False
            cap += f"  ({type(exc).__name__})"
    usable.append(ok)
    print(f"{i:<4}{p.name[:33]:<34}{p.total_memory/2**30:>7.0f}G{cap:>8}"
          f"{'有' if p.major >= 8 else '无':>7}  {'是' if ok else '否'}")

print()
good = [i for i, ok in enumerate(usable) if ok]
if not good:
    sys.exit("没有一块卡能用当前 torch，换 wheel（见 README）")
print(f"可用: {good}")
big = max(good, key=lambda i: torch.cuda.get_device_properties(i).total_memory)
p = torch.cuda.get_device_properties(big)
print(f"建议 GPU={big}  ({p.name}, {p.total_memory / 2**30:.0f}GB)")
