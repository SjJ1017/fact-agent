"""Turn matched stores back into matching inputs, keeping only the mentions.

Extraction and atomisation are expensive and already done; only the matching
step changes when the judge changes.  This drops facts, mention_to_fact and
relations so match_traces.py can redo that step alone, and records where the
input came from so a store is never mistaken for the original.

The output sits beside the source with a different suffix, so the existing
stores are untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--suffix", default=".v2.json", help="源文件后缀")
    ap.add_argument("--out-suffix", default=".atomized.json")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()

    n = skipped = 0
    total_mentions = 0
    for d in a.dirs:
        for f in sorted(d.glob(f"*{a.suffix}")):
            out = f.with_name(f.name.replace(a.suffix, a.out_suffix))
            if out.exists() and not a.redo:
                skipped += 1
                continue
            src = json.loads(f.read_text())
            ms = src["mentions"]
            if isinstance(ms, list):
                ms = {m["mention_id"]: m for m in ms}
            out.write_text(json.dumps(
                {"mentions": ms, "facts": {}, "mention_to_fact": {},
                 "relations": [],
                 "derived_from": {"file": f.name, "suffix": a.suffix,
                                  "note": "mentions kept, matching dropped"}},
                ensure_ascii=False, indent=1, sort_keys=True) + "\n")
            n += 1
            total_mentions += len(ms)
    print(f"写出 {n} 个（跳过 {skipped}），mention 共 {total_mentions}")
    if n:
        print(f"均 {total_mentions / n:.0f} mention/文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
