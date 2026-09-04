"""Why triton's gcc build fails, in the compiler's own words.

The traceback only says gcc returned 1.  This runs the same compile and shows
stderr, then checks the three things that usually cause it.
"""
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

print("=== Python 头文件 ===")
inc = sysconfig.get_paths()["include"]
h = Path(inc) / "Python.h"
print(f"  include 路径 {inc}")
print(f"  Python.h     {'在' if h.exists() else '缺失  <-- 装 python3-dev / python3.12-dev'}")

print("\n=== gcc ===")
try:
    v = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
    print("  " + v.stdout.splitlines()[0])
except FileNotFoundError:
    print("  没有 gcc")

print("\n=== libcuda.so.1 ===")
found = [p for d in ("/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu",
                     "/usr/lib64", "/usr/local/cuda/lib64")
         for p in Path(d).glob("libcuda.so*") if Path(d).is_dir()]
print("  " + ("\n  ".join(str(p) for p in found[:4]) if found
              else "找不到 libcuda.so.1  <-- 驱动库不在链接路径上"))

print("\n=== TMPDIR 能不能执行 ===")
tmp = Path(os.environ.get("TMPDIR", "/tmp"))
probe = tmp / "ff_exec_probe.sh"
try:
    probe.write_text("#!/bin/sh\nexit 0\n")
    probe.chmod(0o755)
    ok = subprocess.run([str(probe)], capture_output=True).returncode == 0
    print(f"  {tmp}  {'可执行' if ok else '不可执行 (noexec)  <-- 换 TMPDIR=/tmp'}")
except Exception as e:
    print(f"  {tmp}  测试失败: {type(e).__name__}: {e}")
finally:
    probe.unlink(missing_ok=True)

print("\n=== 复现 triton 的编译 ===")
try:
    import triton
except ImportError:
    sys.exit("  没装 triton，那报错来自别处")
drv = Path(triton.__file__).parent / "backends/nvidia/driver.c"
if not drv.exists():
    sys.exit(f"  找不到 {drv}")
out = tmp / "ff_triton_probe.so"
cmd = ["gcc", str(drv), "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", str(out),
       "-l:libcuda.so.1",
       f"-L{drv.parent}/lib", "-L/lib/x86_64-linux-gnu",
       f"-I{drv.parent}/include", f"-I{inc}"]
r = subprocess.run(cmd, capture_output=True, text=True)
print(f"  退出码 {r.returncode}")
if r.stderr.strip():
    print("  --- gcc stderr ---")
    for line in r.stderr.strip().splitlines()[:20]:
        print("  " + line)
out.unlink(missing_ok=True)
