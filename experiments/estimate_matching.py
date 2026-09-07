#!/usr/bin/env python3
"""Run the blocker locally and estimate how long the GPU pass will take.

Blocking is cheap and CPU-only; the NLI pass is neither. Counting candidates
here means the GPU time is known before the job is queued, and a corpus that
would take a day instead of an hour can be resized rather than discovered
mid-run.

The rate comes from the observed perspectrum runs, where each entailment pass
scored about 15.5 pairs per second on the RTX 6000 Ada. Every pair costs two
passes -- f(a,b) and f(b,a) -- so the model is called twice per candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factflow.blocking import SbertBlocker, candidate_pairs  # noqa: E402
from factflow.types import FactMention  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--suffix", default=".atomized.json")
    ap.add_argument("--embed", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--threshold", type=float, default=0.62)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--rate", type=float, default=15.5,
                    help="每秒评分的对数（单向一遍），实测值")
    ap.add_argument("--partition", default="",
                    help="只在 provenance.extra 这个键相等的 mention 间配对")
    a = ap.parse_args()

    blk = SbertBlocker(a.embed)
    tot_m = tot_p = tot_e = 0
    rows = []
    for d in a.dirs:
        for f in sorted(d.glob(f"*{a.suffix}")):
            ms = [FactMention(**m) for m in json.loads(f.read_text())["mentions"].values()]
            t0 = time.time()
            if a.partition:
                groups: dict[str, list[FactMention]] = {}
                for m in ms:
                    groups.setdefault(
                        str((m.provenance.extra or {}).get(a.partition, "")), []).append(m)
                n = sum(len(candidate_pairs(g, blocker=blk, threshold=a.threshold,
                                            top_k=a.top_k)) for g in groups.values())
            else:
                n = len(candidate_pairs(ms, blocker=blk, threshold=a.threshold,
                                        top_k=a.top_k))
            # exact-match pre-pass: these cost no model call at all
            import re as _re, collections as _c
            b = _c.defaultdict(list)
            for i, m in enumerate(ms):
                b[(" ".join(_re.sub(r"[^a-z0-9 ]", " ", m.text.lower()).split()),
                   str(m.polarity))].append(i)
            ex = sum(len(v) * (len(v) - 1) // 2 for v in b.values() if len(v) > 1)
            rows.append((f.name, len(ms), n, ex, time.time() - t0))
            tot_m += len(ms)
            tot_p += n
            tot_e += ex

    print(f'{"trace":<40}{"mention":>9}{"候选对":>9}{"占全部对":>9}{"直接判等":>9}')
    for name, m, n, ex, dt in rows:
        poss = m * (m - 1) // 2
        print(f"{name[:40]:<40}{m:>9,}{n:>9,}{n/max(poss,1):>9.2%}{ex:>9,}")
    net = max(tot_p - tot_e, 0)
    secs = net * 2 / a.rate
    print(f"\n{len(rows)} 个 trace，{tot_m:,} 条 mention，{tot_p:,} 个候选对")
    print(f"其中归一化后完全相同、可直接判等: {tot_e:,} 对（省 {tot_e/max(tot_p,1):.1%}）")
    print(f"需要模型评分: {net:,} 对 × 2 遍 = {net*2:,} 次")
    print(f"按 {a.rate} 对/秒：单卡 {secs/60:.0f} 分钟（{secs/3600:.1f} 小时），"
          f"两卡并行 {secs/120:.0f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
