"""What the fact trace shows, across the HotpotQA debate runs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from factflow import Channel, FactStore

OUT = Path(__file__).parent / "out"


def load_all():
    for p in sorted(OUT.glob("*.store.json")):
        yield p.stem.replace(".store", ""), FactStore.load(str(p))


def slot(store, mid):
    p = store.mentions[mid].provenance
    return ("SRC", 0) if p.channel == Channel.SOURCE else (p.agent_id, p.round)


print("=" * 84)
print("SOURCE-FACT SURVIVAL INTO AGENT OUTPUT, BY ROUND")
print("=" * 84)
print("A source fact 'survives' at round r if any agent expressed it at r.\n")
print(f"{'run':<10} {'src facts':>10} {'r1':>10} {'r2':>10} {'r3':>10}")
for name, store in load_all():
    src = {
        fid
        for fid, f in store.facts.items()
        if any(store.mentions[m].provenance.channel == Channel.SOURCE for m in f.mention_ids)
    }
    row = []
    for r in (1, 2, 3):
        alive = {
            fid
            for fid in src
            if any(store.mentions[m].provenance.round == r for m in store.facts[fid].mention_ids)
        }
        row.append(f"{len(alive)/len(src):.1%}" if src else "-")
    print(f"{name:<10} {len(src):>10} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

print()
print("=" * 84)
print("NEW FACTS INTRODUCED BY AGENTS (not in the source documents)")
print("=" * 84)
print("Facts an agent asserted that no source paragraph contains.\n")
for name, store in load_all():
    novel = [
        f
        for f in store.facts.values()
        if not any(store.mentions[m].provenance.channel == Channel.SOURCE for m in f.mention_ids)
    ]
    by_round = Counter(
        min(store.mentions[m].provenance.round or 9 for m in f.mention_ids) for f in novel
    )
    print(f"{name}: {len(novel)}/{len(store.facts)} facts are agent-introduced; first seen {dict(sorted(by_round.items()))}")

print()
print("=" * 84)
print("DEGRADATION: entailment edges where a specific fact was restated vaguely")
print("=" * 84)
shown = 0
for name, store in load_all():
    for r in store.relations:
        if r.relation not in ("A_ENTAILS_B", "B_ENTAILS_A"):
            continue
        spec_id, gen_id = (r.a, r.b) if r.relation == "A_ENTAILS_B" else (r.b, r.a)
        sa, sb = slot(store, spec_id), slot(store, gen_id)
        # Only later-round weakenings of earlier-round or source content.
        if sa[1] >= sb[1] or sb[1] == 0:
            continue
        print(f"\n[{name}] {sa[0]}r{sa[1]} -> {sb[0]}r{sb[1]}")
        print(f"   specific: {store.mentions[spec_id].text}")
        print(f"   weakened: {store.mentions[gen_id].text}")
        shown += 1
        if shown >= 6:
            break
    if shown >= 6:
        break

print()
print("=" * 84)
print("CLUSTER HEALTH")
print("=" * 84)
tot = Counter()
for name, store in load_all():
    sizes = [len(f.mention_ids) for f in store.facts.values()]
    rels = Counter(r.relation for r in store.relations)
    tot.update(rels)
    print(f"{name}: {len(store.mentions)} mentions -> {len(store.facts)} facts | "
          f"largest cluster {max(sizes)} | singletons {sizes.count(1)}")
print(f"\nrelations across all runs: {dict(tot)}")
eq = tot["EQUIVALENT"]
print(f"EQUIVALENT share: {eq}/{sum(tot.values())} = {eq/sum(tot.values()):.1%}")
