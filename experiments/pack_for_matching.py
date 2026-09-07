#!/usr/bin/env python3
"""Bundle the atomized traces that still need matching into one tarball.

Matching runs on the GPU server, not here, so what has to travel is the
mention set and nothing else: the atomized files are a few hundred KB each
while the debates they came from are tens of MB.  Traces that already have a
`.store.json` beside them are skipped, so re-running this after a partial
match only ships what is left.

    python experiments/pack_for_matching.py experiments/perspectrum_pilot_star_chain
    # on the server, after untarring into the repo:
    ./experiments/matcher_eval/run.sh --match experiments/<dir>
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--suffix", default=".atomized.json")
    ap.add_argument("--store-suffix", default=".store.json")
    ap.add_argument("--all", action="store_true",
                    help="也打包已经有 store 的（默认只打没匹配过的）")
    a = ap.parse_args()
    a.dirs = [d if d.is_absolute() else (ROOT / d) for d in a.dirs]

    picked: list[Path] = []
    stats = Counter()
    for d in a.dirs:
        for p in sorted(d.glob(f"*{a.suffix}")):
            stem = p.name[: -len(a.suffix)]
            if not a.all and (p.parent / f"{stem}{a.store_suffix}").exists():
                stats["已匹配，跳过"] += 1
                continue
            picked.append(p)
            m = re.search(r"-(full|star|chain)-", stem)
            stats[m.group(1) if m else "?"] += 1

    if not picked:
        print("没有待匹配的 trace")
        return 1

    out = a.out or ROOT / f"{a.dirs[0].name}-atomized.tar.gz"
    mentions = 0
    manifest = []
    for p in picked:
        d = json.loads(p.read_text())
        n = len(d["mentions"])
        mentions += n
        manifest.append({"file": str(p.relative_to(ROOT)), "mentions": n,
                         "facts": len(d.get("facts", {}))})

    readme = f"""原子事实匹配包
=================
{len(picked)} 条 trace，{mentions:,} 条 mention。
拓扑分布：{', '.join(f'{k} {v}' for k, v in sorted(stats.items()))}

在服务器上：
  1. 解压到仓库根目录（会还原成 experiments/<dir>/*.atomized.json）
       tar xzf {out.name} -C /path/to/factflow
  2. 跑匹配（Qwen3-14B + entail，是之前定下来的判官）
       ./experiments/matcher_eval/run.sh --match experiments/{a.dirs[0].name}
  3. 产物是每条 trace 旁边的 .store.json，把它们拷回来即可。

注意 run.sh 顶部的 SCRATCH_ROOT 和 GPU 编号要跟当前机器对上。
"""
    manifest_path = ROOT / "pack-manifest.json"
    readme_path = ROOT / "README-matching.txt"
    manifest_path.write_text(json.dumps(
        {"n_traces": len(picked), "n_mentions": mentions,
         "topology": dict(stats), "files": manifest}, ensure_ascii=False, indent=1))
    readme_path.write_text(readme)

    with tarfile.open(out, "w:gz") as tf:
        for p in picked:
            tf.add(p, arcname=str(p.relative_to(ROOT)))
        tf.add(manifest_path, arcname="pack-manifest.json")
        tf.add(readme_path, arcname="README-matching.txt")
    manifest_path.unlink()
    readme_path.unlink()

    print(f"{len(picked)} 条 trace，{mentions:,} 条 mention")
    for k, v in sorted(stats.items()):
        print(f"  {k:<12}{v}")
    print(f"→ {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
