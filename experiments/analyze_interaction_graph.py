#!/usr/bin/env python3
"""The directed interaction graph between agent turns.

DelibTrace can only follow a seed fact down one agent's own timeline.  Once
propositions are matched across agents, the object of study becomes a graph:
who restates whom, who weakens whom, and who sharpens whom.  This builds that
graph at the (agent, round) level.

An edge p -> q exists when p and q were *directly* judged by the NLI pass, q
was said no earlier than p, and p was actually visible to q's author by then.
The type is read from q's side: equivalent, weaker (p entails q), or sharper
(q entails p).  X -> X across rounds is the agent holding its own content over;
that is kept as a self-edge because it is the baseline any cross-agent edge
should be compared against.

Counts are also reported per thousand judged pairs, because a condition that
simply produced more propositions would otherwise look more interactive.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_offline_flow import load, cell_of, CELLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TYPE = {"EQUIVALENT": "等价", "A_ENTAILS_B": "弱化", "B_ENTAILS_A": "细化"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path,
                    default=ROOT / "experiments" / "idrbench_generation_10x5_r3")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "findings" / "data" / "interaction-graph.json")
    a = ap.parse_args()

    edges = defaultdict(Counter)      # cell -> (src, dst, type) -> count
    node = defaultdict(Counter)       # cell -> slot -> facts
    seat = defaultdict(Counter)       # cell -> (X, Y, type) -> count
    judged = Counter()
    cases = Counter()

    for sp in sorted(a.dir.glob("*.nli.store.json")):
        dp = sp.with_name(sp.name.replace(".nli.store.json", ".debate.json"))
        if not dp.exists():
            continue
        deb, said, rel = load(sp, ".nli.store.json")
        c = cell_of(sp.name)
        cases[c] += 1
        judged[c] += len(rel) // 2
        agents = sorted(deb["roles"])
        d = deb["delivery"]
        rounds = sorted({int(s.split("|")[1]) for s in said})

        # earliest round at which each proposition was visible to each agent
        vis: dict[tuple[str, str], int] = {}
        for ag in agents:
            for rnd in rounds:
                s = d.get(f"{ag}|{rnd}", {})
                for t in (list(s.get("visible_self_turns", []))
                          + list(s.get("visible_peer_turns", []))):
                    for p in said.get(t, set()):
                        vis[(ag, p)] = min(vis.get((ag, p), 99), rnd)

        for ag in agents:
            for rnd in rounds:
                node[c][f"{ag}|{rnd}"] += len(said.get(f"{ag}|{rnd}", set()))

        for x in agents:
            for rx in rounds:
                for p in said.get(f"{x}|{rx}", set()):
                    for y in agents:
                        for ry in rounds:
                            if ry < rx or (ry == rx and y != x):
                                continue      # no same-round cross links
                            if ry == rx:
                                continue      # a turn against itself is not an edge
                            if vis.get((y, p), 99) > ry:
                                continue      # y could not have seen p yet
                            for q in said.get(f"{y}|{ry}", set()):
                                k = rel.get((p, q))
                                if k in TYPE:
                                    edges[c][(f"{x}|{rx}", f"{y}|{ry}", TYPE[k])] += 1
                                    seat[c][(x, y, TYPE[k])] += 1

    out = {}
    for c in CELLS:
        tot = sum(edges[c].values())
        print(f"\n{c}   {cases[c]} 场   {tot:,} 条边   "
              f"每千个已评分对 {tot / max(judged[c], 1) * 1000:.0f} 条")
        print(f'{"":<10}' + "".join(f"{t:>22}" for t in ("等价", "弱化", "细化")))
        for x in "ABC":
            for y in "ABC":
                v = [seat[c][(x, y, t)] for t in ("等价", "弱化", "细化")]
                if sum(v):
                    tag = f"{x}→{y}" + ("（自持）" if x == y else "")
                    print(f'{tag:<10}' + "".join(
                        f"{n:>10,}{n / tot:>10.1%}  " for n in v))
        out[c] = {
            "cases": cases[c], "judged_pairs": judged[c],
            "nodes": {k: v / cases[c] for k, v in node[c].items()},
            "edges": [{"src": s, "dst": d_, "type": t, "n": n, "per_case": n / cases[c]}
                      for (s, d_, t), n in sorted(edges[c].items())],
            "seat": {f"{x}->{y}|{t}": seat[c][(x, y, t)]
                     for x in "ABC" for y in "ABC" for t in ("等价", "弱化", "细化")
                     if seat[c][(x, y, t)]},
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
