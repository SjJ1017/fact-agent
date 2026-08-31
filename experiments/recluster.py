"""Re-cluster existing stores at a different union threshold. No LLM calls.

Every SAME/DIFFERENT judgement is already in the store; only which of them are
allowed to chain changes. That makes the union threshold the one parameter that
can be revised for free after the fact - and worth revising, since the first
value was calibrated on the wrong thing (where clusters stopped looking like
blobs) on the wrong data (stores from before extraction produced atomic facts).

    python experiments/recluster.py experiments/perspectrum_pilot_full \
        --glob "*.v2.json" --threshold 0.85 --suffix .v3.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factflow.match import cluster  # noqa: E402
from factflow.types import FactStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--glob", default="*.v2.json")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--suffix", default=".v3.json")
    a = ap.parse_args()

    paths = [p for p in sorted(a.out_dir.glob(a.glob)) if not p.name.endswith(a.suffix)]
    if not paths:
        print(f"no {a.glob} under {a.out_dir}", file=sys.stderr)
        return 1

    for i, p in enumerate(paths, 1):
        old = FactStore.load(str(p))
        mentions = list(old.mentions.values())
        new = FactStore(mentions=old.mentions, relations=old.relations)
        for f in cluster(mentions, old.relations, union_min_similarity=a.threshold):
            new.assign(f)
        target = p.with_name(p.name.replace(a.glob.lstrip("*"), "") + a.suffix)
        new.save(str(target))
        print(f"[{i}/{len(paths)}] {p.name}  facts {len(old.facts)} -> {len(new.facts)}  "
              f"(merge {1 - len(new.facts)/len(mentions):.0%})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
