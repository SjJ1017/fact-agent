"""How much of the annotated argument did each persona condition reach?

Everything else measured here was measured against a definition we chose:
survival against a threshold, echo against a similarity band. Perspectrum ships
human-written perspectives per claim, labelled support or undermine, so this one
is measured against what people said the arguments were.

Four questions, in the order they matter:

    coverage   how many annotated perspectives did the debate reach at all
    balance    did it reach both sides, or only the one it started on
    when       round a perspective first appears - a panel that finds
               everything in round 1 is not deliberating, it is enumerating
    survival   did a perspective it reached still stand at the end, or did the
               debate raise it and let it go

Coverage is not accuracy: reaching a perspective is not agreeing with it, and
the annotation is a set of positions, not a set of truths. What it does support
is a comparison no endpoint metric can make - two conditions can land on the
same conclusion having explored three of the four arguments or one of them.

    python experiments/analyze_coverage.py experiments/perspectrum_pilot_full \
        --glob "*.v2.json"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factflow.types import Channel, FactStore  # noqa: E402

PERSONA = re.compile(r"(neutral|lenses|stance)")


def gold_facts(store: FactStore) -> dict[str, dict]:
    """-> fact_id: {stance, reached_by, first_round, survived}."""
    rounds = [r for r in store.rounds() if r and r > 0]
    last = max(rounds) if rounds else 0
    out: dict[str, dict] = {}
    for fid, fact in store.facts.items():
        gold_m = [store.mentions[m] for m in fact.mention_ids
                  if store.mentions[m].provenance.extra.get("gold")]
        if not gold_m:
            continue
        said = [store.mentions[m] for m in fact.mention_ids
                if store.mentions[m].provenance.channel == Channel.OUTPUT]
        out[fid] = {
            "stance": gold_m[0].provenance.extra.get("stance"),
            "text": fact.canonical_text,
            "reached_by": sorted({m.provenance.agent_id for m in said if m.provenance.agent_id}),
            "first_round": min((m.provenance.round for m in said), default=None),
            "survived": any(m.provenance.round == last for m in said),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--glob", default="*.v2.json")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    per: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(a.out_dir.glob(a.glob)):
        m = PERSONA.search(p.name)
        if not m:
            continue
        store = FactStore.load(str(p))
        g = gold_facts(store)
        if not g:
            continue
        claim = p.name.split("-")[1]
        per[m.group(1)].append({"claim": claim, "gold": g})

    if not per:
        print("no gold facts found - run experiments/add_gold.py first", file=sys.stderr)
        return 1

    print("=== COVERAGE: annotated perspectives the debate reached ===")
    print(f"  {'persona':<10}{'runs':>6}{'gold/run':>10}{'reached':>9}{'support':>9}"
          f"{'undermine':>11}{'both sides':>12}")
    result: dict = {}
    for name, runs in sorted(per.items()):
        tot = sum(len(r["gold"]) for r in runs)
        hit = [g for r in runs for g in r["gold"].values() if g["reached_by"]]
        sup = [g for g in hit if g["stance"] == "support"]
        und = [g for g in hit if g["stance"] in ("undermine", "refute")]
        both = sum(1 for r in runs
                   if {g["stance"] for g in r["gold"].values() if g["reached_by"]}
                   >= {"support", "undermine"})
        print(f"  {name:<10}{len(runs):>6}{tot/len(runs):>10.1f}{len(hit)/max(tot,1):>9.0%}"
              f"{len(sup):>9}{len(und):>11}{both}/{len(runs):>7}")
        result[name] = {"runs": len(runs), "gold_total": tot, "reached": len(hit),
                        "support": len(sup), "undermine": len(und), "both_sides": both}

    print("\n=== WHEN a perspective first appears, and whether it lasts ===")
    print(f"  {'persona':<10}{'r1':>6}{'r2':>6}{'r3':>6}{'survived':>10}")
    for name, runs in sorted(per.items()):
        hit = [g for r in runs for g in r["gold"].values() if g["reached_by"]]
        by = defaultdict(int)
        for g in hit:
            by[g["first_round"]] += 1
        surv = sum(1 for g in hit if g["survived"]) / max(len(hit), 1)
        print(f"  {name:<10}" + "".join(f"{by.get(r,0):>6}" for r in (1, 2, 3))
              + f"{surv:>10.0%}")
        result[name]["first_round"] = {str(r): by.get(r, 0) for r in (1, 2, 3)}
        result[name]["survived"] = surv

    print("\n=== HOW MANY AGENTS reached each perspective ===")
    print(f"  {'persona':<10}{'1 agent':>9}{'2 agents':>10}{'all 3':>8}")
    for name, runs in sorted(per.items()):
        hit = [g for r in runs for g in r["gold"].values() if g["reached_by"]]
        c = defaultdict(int)
        for g in hit:
            c[min(len(g["reached_by"]), 3)] += 1
        print(f"  {name:<10}" + "".join(f"{c.get(k,0):>9}" for k in (1, 2, 3)))
        result[name]["by_agent_count"] = {str(k): c.get(k, 0) for k in (1, 2, 3)}

    print("\n=== MISSED perspectives (examples) ===")
    for name, runs in sorted(per.items()):
        missed = [(r["claim"], g) for r in runs for g in r["gold"].values()
                  if not g["reached_by"]]
        print(f"  {name} - {len(missed)} missed")
        for claim, g in missed[:3]:
            print(f"    claim {claim} [{g['stance']}] {g['text'][:78]}")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
