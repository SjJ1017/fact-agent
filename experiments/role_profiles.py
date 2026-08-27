"""Per-role fact-operator profiles.

The question is not whether a role prompt changes accuracy, but whether it
changes what the agent DOES to facts. Four operators, defined against the
agent's own input rather than against the world:

    relay     a fact that was already in this agent's context and it says again
    derive    a fact new to this agent's context, first said by it
    drop      a fact that was in its context and it does not carry forward
    inherit   a fact it takes from a peer that it had not said before

A summarizer that only relays is the redundant node that topology-refinement
papers try to prune; an analyzer that never derives is not analysing. If role
prompts are decorative, the four profiles will be indistinguishable.

Context is computed, not observed: under full broadcast an agent's context at
round r is the source plus every peer message from rounds < r.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from factflow import Channel, FactStore


def profile(store: FactStore) -> dict[str, dict[str, float]]:
    agents = sorted({m.provenance.agent_id for m in store.mentions.values() if m.provenance.agent_id})
    rounds = sorted({m.provenance.round for m in store.mentions.values()
                     if m.provenance.round and m.provenance.channel == Channel.OUTPUT})
    said = {(a, r): set() for a in agents for r in rounds}
    source = set()
    roles = {}
    for fid, f in store.facts.items():
        for mid in f.mention_ids:
            p = store.mentions[mid].provenance
            if p.channel == Channel.SOURCE:
                source.add(fid)
            elif (p.agent_id, p.round) in said:
                said[(p.agent_id, p.round)].add(fid)
                roles.setdefault(p.agent_id, p.extra.get("role", "generalist"))

    out: dict[str, dict[str, float]] = {}
    for a in agents:
        acc = defaultdict(int)
        for r in rounds:
            # what this agent could see: source + everything anyone said earlier
            ctx = set(source)
            for b in agents:
                for rr in rounds:
                    if rr < r:
                        ctx |= said[(b, rr)]
            mine_before = set().union(*[said[(a, rr)] for rr in rounds if rr < r]) if r > rounds[0] else set()
            cur = said[(a, r)]
            acc["said"] += len(cur)
            acc["relay"] += len(cur & mine_before)
            acc["inherit"] += len((cur & ctx) - mine_before)
            acc["derive"] += len(cur - ctx)
            acc["drop"] += len(mine_before - cur)
            acc["rounds"] += 1
        n = max(acc["rounds"], 1)
        out[a] = {"role": roles.get(a, "generalist"),
                  "said": acc["said"] / n, "relay": acc["relay"] / n,
                  "inherit": acc["inherit"] / n, "derive": acc["derive"] / n,
                  "drop": acc["drop"] / n}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--configs", default="")
    args = ap.parse_args()
    root = Path(args.dir)

    byrole = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for p in sorted(root.glob("*.store.json")):
        cfg = "-".join(p.stem.replace(".store", "").split("-")[2:])
        if args.configs and cfg not in args.configs.split(","):
            continue
        for a, prof in profile(FactStore.load(str(p))).items():
            key = (cfg, prof["role"])
            counts[key] += 1
            for k in ("said", "relay", "inherit", "derive", "drop"):
                byrole[key][k] += prof[k]

    print(f"{'config':<20}{'role':<15}{'said':>7}{'derive':>8}{'relay':>7}{'inherit':>9}{'drop':>7}"
          f"{'derive%':>9}{'relay%':>8}")
    last = None
    for (cfg, role), acc in sorted(byrole.items()):
        n = counts[(cfg, role)]
        said = acc["said"] / n
        d, rl, inh, dr = (acc[k] / n for k in ("derive", "relay", "inherit", "drop"))
        if cfg != last:
            print("-" * 90)
            last = cfg
        print(f"{cfg:<20}{role:<15}{said:>7.1f}{d:>8.1f}{rl:>7.1f}{inh:>9.1f}{dr:>7.1f}"
              f"{d/max(said,1e-9):>9.0%}{rl/max(said,1e-9):>8.0%}")
    print("\nper agent-round averages. derive = facts new to this agent's whole context;")
    print("relay = its own earlier facts restated; inherit = taken from a peer; drop = its own facts abandoned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
