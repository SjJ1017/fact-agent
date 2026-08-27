"""Gold vs distractor retention - the direction-carrying measure.

Raw source-fact retention is uninterpretable on HotpotQA: 8 of 10 paragraphs are
distractors, so most "lost" facts were correctly ignored.  Splitting retention by
whether the source paragraph is one HotpotQA labels as gold turns a
direction-ambiguous number into a directional one:

    selectivity = P(expressed | gold) - P(expressed | distractor)

1.0 is perfect discrimination, 0.0 is indifference to relevance.  It is
invariant to how much the system drops overall, which is what makes it usable
as a score where retention is not.
"""

from __future__ import annotations

import json
from pathlib import Path

from factflow import Channel, FactStore

OUT = Path(__file__).parent / "out"


def analyse(store: FactStore) -> dict:
    gold, dist = set(), set()
    for fid, fact in store.facts.items():
        for mid in fact.mention_ids:
            p = store.mentions[mid].provenance
            if p.channel != Channel.SOURCE:
                continue
            (gold if p.extra.get("gold") else dist).add(fid)
    dist -= gold  # a fact in both is credited as gold

    def expressed(pool, rounds=None):
        out = set()
        for fid in pool:
            for mid in store.facts[fid].mention_ids:
                p = store.mentions[mid].provenance
                if p.channel == Channel.OUTPUT and (rounds is None or p.round in rounds):
                    out.add(fid)
                    break
        return out

    rows = {}
    for label, rr in [("any", None), ("r1", {1}), ("r2", {2}), ("r3", {3})]:
        g = len(expressed(gold, rr)) / len(gold) if gold else 0.0
        d = len(expressed(dist, rr)) / len(dist) if dist else 0.0
        rows[label] = {"gold": g, "distractor": d, "selectivity": g - d}
    return {"n_gold": len(gold), "n_distractor": len(dist), "rates": rows}


results = {}
print("=" * 78)
print("GOLD vs DISTRACTOR SOURCE-FACT RETENTION")
print("=" * 78)
for p in sorted(OUT.glob("*.store.json")):
    name = p.stem.replace(".store", "")
    r = analyse(FactStore.load(str(p)))
    results[name] = r
    print(f"\n{name}  ({r['n_gold']} gold facts, {r['n_distractor']} distractor facts)")
    print(f"    {'':6} {'gold':>8} {'distr':>8} {'selectivity':>12}")
    for label, v in r["rates"].items():
        print(f"    {label:6} {v['gold']:>7.1%} {v['distractor']:>8.1%} {v['selectivity']:>+12.3f}")

agg = {}
for label in ("any", "r1", "r2", "r3"):
    g = sum(r["rates"][label]["gold"] * r["n_gold"] for r in results.values())
    gd = sum(r["n_gold"] for r in results.values())
    d = sum(r["rates"][label]["distractor"] * r["n_distractor"] for r in results.values())
    dd = sum(r["n_distractor"] for r in results.values())
    agg[label] = {"gold": g / gd, "distractor": d / dd, "selectivity": g / gd - d / dd}

print("\n" + "=" * 78)
print("POOLED")
print("=" * 78)
print(f"{'':6} {'gold':>8} {'distr':>8} {'selectivity':>12}")
for label, v in agg.items():
    print(f"{label:6} {v['gold']:>7.1%} {v['distractor']:>8.1%} {v['selectivity']:>+12.3f}")

(OUT / "selectivity.json").write_text(json.dumps({"per_run": results, "pooled": agg}, indent=2))
