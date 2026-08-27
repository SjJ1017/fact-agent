"""Join detection, and the metric accuracy cannot produce.

A join has occurred when ONE agent, at ONE round, expresses both mechanism
facts:

    "fluconazole inhibits CYP2C9"  AND  "warfarin is metabolised by CYP2C9"

That is a binary, observable event with a timestamp. It licenses the split that
motivates this whole line of work:

    correct WITH join     the answer was derived by combining the evidence
    correct WITHOUT join  the answer was already known; the evidence was decoration
    wrong WITH join       the facts met and the inference still failed
    wrong WITHOUT join    the facts never met

Accuracy collapses all four into one number. The second cell is the one that
matters: a system scoring well there is not doing what its architecture claims,
and no scoreboard will ever say so.

The `solo-half` condition calibrates it. Whatever that condition scores is what
the model knew without being told, so a `split` accuracy at or below it means
the collaboration contributed nothing regardless of how the trace looks.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from factflow import Channel, FactStore

OUT = Path(__file__).parent / "out"


def find_join(store: FactStore, critical: tuple[str, str]) -> dict:
    """Locate the two critical facts and the first slot holding both."""
    ids = []
    for want in critical:
        norm = " ".join(want.lower().split())
        hit = None
        for fid, f in store.facts.items():
            for mid in f.mention_ids:
                m = store.mentions[mid]
                if m.provenance.channel == Channel.SOURCE and \
                        " ".join(m.text.lower().split()) == norm:
                    hit = fid
                    break
            if hit:
                break
        ids.append(hit)

    if not all(ids):
        return {"joined": False, "reason": "critical fact not found in store",
                "found": [bool(i) for i in ids]}

    fa, fb = ids
    said = defaultdict(set)
    for fid in (fa, fb):
        for mid in store.facts[fid].mention_ids:
            p = store.mentions[mid].provenance
            if p.channel == Channel.OUTPUT and p.agent_id:
                said[(p.agent_id, p.round)].add(fid)

    slots = sorted((r, a) for (a, r), fs in said.items() if {fa, fb} <= fs)
    reached = {fid: sorted({f"{a}r{r}" for (a, r), fs in said.items() if fid in fs})
               for fid in (fa, fb)}
    return {
        "joined": bool(slots),
        "join_round": slots[0][0] if slots else None,
        "join_agent": slots[0][1] if slots else None,
        "fact_a_expressed_at": reached[fa],
        "fact_b_expressed_at": reached[fb],
        "both_surfaced": bool(reached[fa]) and bool(reached[fb]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(OUT))
    ap.add_argument("--require-store", action="store_true",
                    help="fail loudly if a run has no matched store rather than skipping it")
    args = ap.parse_args()
    root = Path(args.dir)

    rows = []
    missing = 0
    for rp in sorted(root.glob("*.run.json")):
        run = json.loads(rp.read_text())
        sp = root / rp.name.replace(".run.json", ".store.json")
        if not sp.exists():
            missing += 1
            continue
        j = find_join(FactStore.load(str(sp)), tuple(run["critical_pair"]))
        rows.append({**{k: run[k] for k in
                        ("case_id", "condition", "gold", "correct", "verdict",
                         "named_enzyme", "drug_a", "drug_b")}, **j})

    if missing:
        msg = f"{missing} runs have no matched store (run trace_ddi.py first)"
        if args.require_store:
            raise SystemExit(f"ERROR: {msg}")
        print(f"NOTE: {msg}\n")
    if not rows:
        print("no traced runs to analyse")
        return 1

    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)

    print("=" * 88)
    print("JOIN RATE AND WHAT IT BUYS")
    print("=" * 88)
    print(f"{'condition':<12}{'n':>4}{'acc':>7}{'join rate':>11}{'both surfaced':>15}"
          f"{'join@r1':>9}{'join@r2':>9}")
    for cond, rs in sorted(by.items()):
        n = len(rs)
        joined = [r for r in rs if r["joined"]]
        print(f"{cond:<12}{n:>4}{sum(r['correct'] for r in rs)/n:>7.0%}"
              f"{len(joined)/n:>11.0%}{sum(r['both_surfaced'] for r in rs)/n:>15.0%}"
              f"{sum(1 for r in joined if r['join_round']==1):>9}"
              f"{sum(1 for r in joined if r['join_round']==2):>9}")

    print()
    print("=" * 88)
    print("THE SPLIT ACCURACY CANNOT MAKE")
    print("=" * 88)
    print(f"{'condition':<12}{'correct+join':>14}{'correct NO join':>17}"
          f"{'wrong+join':>12}{'wrong NO join':>15}")
    for cond, rs in sorted(by.items()):
        cj = sum(1 for r in rs if r["correct"] and r["joined"])
        cn = sum(1 for r in rs if r["correct"] and not r["joined"])
        wj = sum(1 for r in rs if not r["correct"] and r["joined"])
        wn = sum(1 for r in rs if not r["correct"] and not r["joined"])
        print(f"{cond:<12}{cj:>14}{cn:>17}{wj:>12}{wn:>15}")
    print()
    print("correct WITHOUT join = answered from prior knowledge; the evidence was decoration.")
    print("A system whose accuracy comes mostly from that column is not doing what its")
    print("architecture claims, and no scoreboard will say so.")

    solo_half = by.get("solo-half")
    split = by.get("split")
    if solo_half and split:
        floor = sum(r["correct"] for r in solo_half) / len(solo_half)
        got = sum(r["correct"] for r in split) / len(split)
        print()
        print("=" * 88)
        print(f"prior-knowledge floor (solo-half): {floor:.0%}")
        print(f"split condition:                   {got:.0%}")
        print(f"contributed by collaboration:      {got - floor:+.0%}")
        if got <= floor:
            print("  -> the split contributed nothing; a high join rate here would be")
            print("     the agents confirming what one of them already knew.")

    print()
    print("=" * 88)
    print("PER-CASE (split condition)")
    print("=" * 88)
    for r in sorted(by.get("split", []), key=lambda r: r["case_id"]):
        mark = "ok " if r["correct"] else "BAD"
        jj = f"join@{r['join_agent']}r{r['join_round']}" if r["joined"] else "NO JOIN"
        print(f"  {mark} {r['case_id']:<12} {r['drug_a'][:14]:<15}+ {r['drug_b'][:14]:<15}"
              f"gold={'Y' if r['gold'] else 'N'} said={r['verdict']:<3} {jj}")

    (root / "join_analysis.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {root / 'join_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
