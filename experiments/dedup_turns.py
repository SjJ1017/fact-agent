"""Collapse exact repeats inside one turn. No LLM calls.

Stripping the attribution made a single sentence state the same thing twice:
"E3 and E4 argue against banning the veil" becomes two identical claims once
the sources are gone. Blocking never compares two mentions from the same slot -
extraction was assumed to de-duplicate within a turn - so nothing downstream
merges them, and every count is inflated. 3.4% of DeepSeek output mentions were
exact within-turn repeats.

Runs over finished stores because the judgements are already there: dropping a
duplicate mention only changes which mentions exist, and clustering is local.

    python experiments/dedup_turns.py experiments/perspectrum_pilot_full \
        --glob "*.v2.json" --suffix .v4.json
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factflow.match import cluster  # noqa: E402
from factflow.types import FactStore  # noqa: E402


def norm(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--glob", default="*.v2.json")
    ap.add_argument("--suffix", default=".v4.json")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the input instead of writing a new suffix")
    a = ap.parse_args()

    paths = [p for p in sorted(a.out_dir.glob(a.glob)) if not p.name.endswith(a.suffix)]
    if not paths:
        print(f"no {a.glob} under {a.out_dir}", file=sys.stderr)
        return 1

    dropped_total = kept_total = 0
    for i, p in enumerate(paths, 1):
        store = FactStore.load(str(p))
        seen: dict[tuple, set[str]] = defaultdict(set)
        keep: list = []
        dropped = 0
        # Slot order is the trace order, so the mention kept is the first one
        # said - which is the one whose token clock marks the birth.
        for m in sorted(store.mentions.values(),
                        key=lambda x: (x.provenance.round or 0,
                                       x.provenance.agent_id or "",
                                       x.mention_id)):
            pr = m.provenance
            slot = (pr.execution_id, pr.agent_id, pr.round, pr.channel)
            key = norm(m.text)
            if key in seen[slot]:
                dropped += 1
                continue
            seen[slot].add(key)
            keep.append(m)

        alive = {m.mention_id for m in keep}
        new = FactStore(
            mentions={m.mention_id: m for m in keep},
            relations=[r for r in store.relations if r.a in alive and r.b in alive],
        )
        for f in cluster(keep, new.relations):
            new.assign(f)
        target = p if a.in_place else p.with_name(p.name.split(".")[0] + a.suffix)
        new.save(str(target))
        dropped_total += dropped
        kept_total += len(keep)
        if dropped:
            print(f"[{i}/{len(paths)}] {p.name[:52]}  −{dropped} mentions  "
                  f"facts {len(store.facts)} -> {len(new.facts)}", flush=True)

    print(f"\ndropped {dropped_total} within-turn repeats "
          f"({dropped_total / max(dropped_total + kept_total, 1):.1%} of mentions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
