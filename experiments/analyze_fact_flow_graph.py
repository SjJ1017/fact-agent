"""Each agent-turn as an operator on fact sets, and the weighted flow graph.

Two views of the same traces.

The operator view asks what one turn did: of the facts it stated, how many came
from source material it held, from a peer turn it could see, from its own
earlier turn, and how many appear nowhere before it.  That last class is the
bulk of the output and is invisible to any design that only tracks seeded
facts.

The graph view makes those four classes into weighted edges.  Nodes are the
source documents, the agents, and one virtual node per agent standing for what
it introduced; an edge's weight is the number of distinct facts that moved
along it.  Only then can the questions be asked that a survival rate cannot
express: how much of what is said reaches anyone else, how far it travels, and
who is holding the graph together.

Attribution rule, applied throughout: a fact is credited to the earliest
turn that stated it, and later turns are counted as receiving rather than
producing.  Ties within a round are left as ties instead of broken
alphabetically, since the alphabetical tie-break is a known artifact.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELL = re.compile(r"-(full|split)-(generic|specialist)-")
CELLS = ["full-generic", "full-specialist", "split-generic", "split-specialist"]


def cell_of(name: str) -> str:
    m = CELL.search(name)
    return f"{m.group(1)}-{m.group(2)}" if m else "?"


def one_debate(store: dict, debate: dict) -> dict:
    m2f = store["mention_to_fact"]
    mentions = store["mentions"]
    if isinstance(mentions, list):
        mentions = {m["mention_id"]: m for m in mentions}

    by_slot: dict[str, set[str]] = defaultdict(set)
    by_doc: dict[str, set[str]] = defaultdict(set)
    for mid, fid in m2f.items():
        p = (mentions.get(mid) or {}).get("provenance", {})
        if p.get("agent_id"):
            by_slot[f"{p['agent_id']}|{p['round']}"].add(fid)
        elif p.get("doc_id"):
            by_doc[p["doc_id"]].add(fid)

    delivery = debate.get("delivery", {})
    agents = sorted({s.split("|")[0] for s in by_slot})
    rounds = sorted({int(s.split("|")[1]) for s in by_slot})

    # --- operator profile -------------------------------------------------
    turns = []
    for ag in agents:
        for rnd in rounds:
            slot = f"{ag}|{rnd}"
            out = by_slot.get(slot, set())
            if not out:
                continue
            info = delivery.get(slot, {})
            held = set().union(*[by_doc.get(d, set())
                                 for d in info.get("source_ids", [])] or [set()])
            peers = set().union(*[by_slot.get(t, set())
                                  for t in info.get("visible_peer_turns",
                                                    info.get("peer_turns", []))]
                                or [set()])
            prior = set().union(*[by_slot.get(t, set())
                                  for t in info.get("visible_self_turns", [])]
                                or [set()])
            novel = out - held - peers - prior
            turns.append({
                "agent": ag, "round": rnd, "n": len(out),
                "from_source": len(out & held - prior - peers),
                "from_peer": len(out & peers - held - prior),
                "held_over": len(out & prior),
                "novel": len(novel),
            })

    # --- weighted graph ---------------------------------------------------
    # first appearance decides who produced a fact; later turns receive it
    first: dict[str, list[str]] = defaultdict(list)
    for rnd in rounds:
        for ag in agents:
            for f in by_slot.get(f"{ag}|{rnd}", set()):
                if f not in first or int(first[f][0].split("|")[1]) == rnd:
                    first.setdefault(f, []).append(f"{ag}|{rnd}")
    origin = {f: slots for f, slots in first.items()}

    edges: Counter = Counter()
    for ag in agents:
        for rnd in rounds:
            slot = f"{ag}|{rnd}"
            out = by_slot.get(slot, set())
            if not out:
                continue
            info = delivery.get(slot, {})
            held = set().union(*[by_doc.get(d, set())
                                 for d in info.get("source_ids", [])] or [set()])
            prior = set().union(*[by_slot.get(t, set())
                                  for t in info.get("visible_self_turns", [])]
                                or [set()])
            vis = info.get("visible_peer_turns", info.get("peer_turns", []))
            for f in out:
                if f in prior:
                    edges[(ag, ag, "persist")] += 1
                    continue
                srcs = [t for t in vis if f in by_slot.get(t, set())]
                if srcs:
                    for peer in {t.split("|")[0] for t in srcs}:
                        edges[(peer, ag, "transmit")] += 1
                elif f in held:
                    for d in info.get("source_ids", []):
                        if f in by_doc.get(d, set()):
                            edges[(d, ag, "read")] += 1
                else:
                    edges[(f"{ag}*", ag, "invent")] += 1

    # --- reach ------------------------------------------------------------
    said_by = defaultdict(set)
    for slot, fs in by_slot.items():
        for f in fs:
            said_by[f].add(slot.split("|")[0])
    spread = [f for f, ags in said_by.items() if len(ags) > 1]
    dead = [f for f, ags in said_by.items() if len(ags) == 1]
    fanout = [len(said_by[f]) - 1 for f in spread]

    return {"turns": turns, "edges": {f"{a}>{b}|{k}": v
                                      for (a, b, k), v in edges.items()},
            "n_facts": len(said_by), "dead_end": len(dead),
            "spread": len(spread),
            "fanout_mean": round(st.mean(fanout), 2) if fanout else 0.0,
            "agents": agents}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=ROOT / "experiments" / "idrbench_generation_10x5_r3")
    ap.add_argument("--suffix", default=".nli.store.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "findings" / "data" / "fact-flow-graph.json")
    a = ap.parse_args()

    per_cell = defaultdict(list)
    for p in sorted(a.dir.glob(f"*{a.suffix}")):
        deb_p = p.with_name(p.name.replace(a.suffix, ".debate.json"))
        if not deb_p.exists():
            continue
        r = one_debate(json.loads(p.read_text()), json.loads(deb_p.read_text()))
        r["id"] = p.name
        per_cell[cell_of(p.name)].append(r)

    # ---- operator table --------------------------------------------------
    print(f'{"条件":<18}{"轮":>3}{"输出":>7}{"源":>7}{"同伴":>7}{"自持":>7}'
          f'{"新造":>7}{"新造%":>8}')
    op = {}
    for c in CELLS:
        for rnd in (1, 2, 3):
            t = [x for r in per_cell[c] for x in r["turns"] if x["round"] == rnd]
            if not t:
                continue
            s = {k: sum(x[k] for x in t)
                 for k in ("n", "from_source", "from_peer", "held_over", "novel")}
            op[f"{c}|{rnd}"] = s
            print(f'{c:<18}{rnd:>3}{s["n"]:>7}{s["from_source"]:>7}'
                  f'{s["from_peer"]:>7}{s["held_over"]:>7}{s["novel"]:>7}'
                  f'{s["novel"]/max(1,s["n"]):>8.0%}')

    # ---- graph table -----------------------------------------------------
    print(f'\n{"条件":<18}{"事实":>7}{"死端%":>8}{"扇出":>7}'
          f'{"读源":>8}{"传递":>8}{"自持":>8}{"新造":>8}')
    graph = {}
    for c in CELLS:
        rows = per_cell[c]
        if not rows:
            continue
        ew = Counter()
        for r in rows:
            for k, v in r["edges"].items():
                ew[k.split("|")[1]] += v
        f = sum(r["n_facts"] for r in rows)
        dead = sum(r["dead_end"] for r in rows)
        fo = st.mean([r["fanout_mean"] for r in rows if r["fanout_mean"]])
        graph[c] = {"facts": f, "dead_end": dead, "dead_share": round(dead / f, 3),
                    "fanout": round(fo, 2), **{k: ew[k] for k in
                                               ("read", "transmit", "persist", "invent")}}
        g = graph[c]
        print(f'{c:<18}{f:>7}{g["dead_share"]:>8.0%}{g["fanout"]:>7.2f}'
              f'{g["read"]:>8}{g["transmit"]:>8}{g["persist"]:>8}{g["invent"]:>8}')

    # ---- per-agent, where the split makes seats differ -------------------
    print(f'\n{"条件":<18}{"agent":>6}{"输出":>7}{"源":>7}{"同伴":>7}{"新造":>7}{"新造%":>8}')
    seats = {}
    for c in CELLS:
        for ag in ("A", "B", "C"):
            t = [x for r in per_cell[c] for x in r["turns"] if x["agent"] == ag]
            if not t:
                continue
            s = {k: sum(x[k] for x in t)
                 for k in ("n", "from_source", "from_peer", "held_over", "novel")}
            seats[f"{c}|{ag}"] = s
            print(f'{c:<18}{ag:>6}{s["n"]:>7}{s["from_source"]:>7}'
                  f'{s["from_peer"]:>7}{s["novel"]:>7}{s["novel"]/max(1,s["n"]):>8.0%}')

    a.out.write_text(json.dumps(
        {"operator_by_round": op, "graph": graph, "seats": seats,
         "per_debate": {c: rows for c, rows in per_cell.items()}},
        ensure_ascii=False, indent=1))
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
