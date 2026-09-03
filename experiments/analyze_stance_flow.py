"""Weighted directed uptake graphs, one per condition, split by fact stance.

Every metric so far collapses the flow to a scalar. This keeps the shape: for
each (topology, persona) cell, sum over all 12 claims to get a weighted
directed graph on the agents, where the weight on speaker -> listener is the
number of facts the listener adopted that the speaker had delivered. Each cell
is then split three ways by the stance of the fact that moved — SUPPORT,
UNDERMINE, NEUTRAL — so the question is not only how much flows but what kind.

Credit for one adoption is split evenly among the peers who delivered that fact
into the turn. In `full` a fact can arrive from two peers at once and counting
it for both would inflate every total by up to 2x; in `chain` there is only
ever one source, so the two conventions agree there and would not elsewhere.
Splitting keeps the graph's total weight equal to the adopted-fact count, which
is what makes the three stance layers add up to the whole.

A raw layer total says little on its own: UNDERMINE dominates every flow, but
it also dominates the fact pool. Each cell therefore also carries the stance
mix of everything the agents said, and a lift — the flow's share of a stance
divided by that stance's share of the pool. Lift above 1 means facts of that
stance are taken up more often than their abundance would predict.

    python experiments/analyze_stance_flow.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from export_labels import attach_labels

ROOT = Path(__file__).resolve().parent
STORE_DIRS = [ROOT / "perspectrum_pilot_full", ROOT / "perspectrum_pilot_star_chain"]
OUT = ROOT.parent / "findings" / "data"
PANELS = ("neutral", "lenses", "stance")
STANCES = ("SUPPORT", "UNDERMINE", "NEUTRAL")

# Identical across all 108 debates; checked, not assumed. Under `stance` the
# node letters are not interchangeable — A argues for, B argues against — so a
# flow graph is unreadable without them.
ROLES = {
    "neutral": {"A": "中立分析者", "B": "中立分析者", "C": "中立分析者"},
    "lenses":  {"A": "因果证据", "B": "落地与权衡", "C": "适用范围与不确定性"},
    "stance":  {"A": "支持方", "B": "反对方", "C": "裁决者"},
}


def spoken_stance_mix(store: dict) -> dict[str, int]:
    """Stance counts over every fact any agent said, the pool uptake draws on."""
    output = {mid for mid, m in store["mentions"].items()
              if m["provenance"]["channel"] == "output"}
    counts: dict[str, int] = defaultdict(int)
    for fact in store["facts"].values():
        if set(fact["mention_ids"]) & output:
            counts[fact.get("properties", {}).get("stance") or "UNLABELLED"] += 1
    return dict(counts)


def uptake_edges(store: dict, debate: dict, execution_id: str):
    """Yield (speaker, listener, stance, credit, taken) for every fact offered.

    `taken` is True when the listener went on to say it. A fact the listener had
    already said before that round was never on offer, so it is skipped: the
    denominator has to be what could have been adopted, not what was sent.
    Credit is split the same way on both sides, which makes taken/offered a
    per-edge uptake rate rather than two differently-normalised counts.
    """
    attach_labels(store, execution_id, "stance")
    stance_of = {fid: f.get("properties", {}).get("stance")
                 for fid, f in store["facts"].items()}

    said: dict[tuple[str, int], set[str]] = defaultdict(set)
    for mid, mention in store["mentions"].items():
        prov = mention["provenance"]
        if prov["channel"] == "output" and prov.get("agent_id") and prov.get("round"):
            fid = store["mention_to_fact"].get(mid)
            if fid:
                said[(prov["agent_id"], prov["round"])].add(fid)
    rounds = sorted({r for _, r in said})
    if not rounds:
        return

    # A set, not a list. Credit is split evenly among the peers who could have
    # supplied the fact, so a peer that restated it in two visible rounds must
    # still count once — with a list, B saying it in B1 and B2 while C says it
    # once hands B two thirds of the credit and C one third, which is a
    # property of how often B repeated itself, not of who it came from.
    sources: dict[tuple[str, int], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set))
    for slot, info in debate.get("delivery", {}).items():
        listener, rnd = slot.split("|")
        # uptake 边问的是"听者本可以接走什么"，所以用窗口而不是投递事件。
        for peer in info.get("visible_peer_turns", info.get("peer_turns", [])):
            speaker, peer_round = peer.split("|")
            for fid in said[(speaker, int(peer_round))]:
                sources[(listener, int(rnd))][fid].add(speaker)

    for (listener, rnd), by_fact in sources.items():
        earlier = set().union(
            *[said[(listener, r)] for r in rounds if r < rnd]
        ) if rnd > rounds[0] else set()
        for fid, speakers in by_fact.items():
            if fid in earlier:
                continue
            credit = 1.0 / len(speakers)
            taken = fid in said[(listener, rnd)]
            stance = stance_of.get(fid) or "UNLABELLED"
            for speaker in sorted(speakers):
                yield speaker, listener, stance, credit, taken


def _summarise(values: list[float], draws: int = 20_000, seed: int = 0) -> dict:
    """Mean per-claim lift with a bootstrap interval and P(lift > 1)."""
    if len(values) < 3:
        return {"n": len(values)}
    rng = random.Random(seed)
    means = [st.mean(rng.choices(values, k=len(values))) for _ in range(draws)]
    above = sum(m > 1 for m in means) / draws
    means.sort()
    return {"mean": st.mean(values), "lo": means[int(0.025 * draws)],
            "hi": means[int(0.975 * draws)], "p_above_1": above,
            "positive": sum(1 for v in values if v > 1), "n": len(values)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--memory", default="peer-only",
                    choices=("peer-only", "self-last", "cumulative"))
    ap.add_argument("--topologies", nargs="+", default=["full", "star", "chain"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = OUT
    tag = "" if args.memory == "peer-only" else f"-{args.memory}"

    # cell -> stance -> "A>B" -> weight
    graphs: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float)))
    claims: dict[str, set[str]] = defaultdict(set)
    pool: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    offers: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float)))
    # per-claim lift, so the pooled number can be checked against its spread
    per_claim: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))

    for directory in STORE_DIRS:
        for path in sorted(directory.glob(f"*{args.model}*.v2.json")):
            name = path.name[: -len(".v2.json")]
            match = re.match(
                rf"perspectrum-(\d+)-{re.escape(args.model)}-(\w+?)-(neutral|lenses|stance)"
                rf"(?:-(cumulative|self-last))?$", name)
            if not match:
                continue
            claim, topology, panel, mem = match.groups()
            if (mem or "peer-only") != args.memory:
                continue
            if topology not in args.topologies:
                continue
            debate_path = path.with_name(f"{name}.debate.json")
            if not debate_path.exists():
                continue
            with path.open() as fh:
                store = json.load(fh)
            with debate_path.open() as fh:
                debate = json.load(fh)
            cell = f"{topology}/{panel}"
            claims[cell].add(claim)
            attach_labels(store, name, "stance")
            for stance, count in spoken_stance_mix(store).items():
                pool[cell][stance] += count
            here: dict[str, float] = defaultdict(float)
            for speaker, listener, stance, credit, taken in uptake_edges(
                    store, debate, name):
                offers[cell][stance][f"{speaker}>{listener}"] += credit
                if taken:
                    graphs[cell][stance][f"{speaker}>{listener}"] += credit
                    here[stance] += credit
            said_here = spoken_stance_mix(store)
            flow_here = sum(here.get(s, 0.0) for s in STANCES) or 0.0
            pool_here = sum(said_here.get(s, 0) for s in STANCES) or 0
            if flow_here and pool_here:
                for stance in STANCES:
                    if said_here.get(stance):
                        per_claim[cell][stance].append(
                            (here.get(stance, 0.0) / flow_here)
                            / (said_here[stance] / pool_here))

    result = {
        "roles": ROLES,
        "model": args.model,
        "memory": args.memory,
        "topologies": list(args.topologies),
        "stances": list(STANCES),
        "note": "edge weight = adopted facts, credit split evenly among the "
                "peers who delivered the fact into that turn; summed over claims",
        "cells": {},
    }
    for topology in args.topologies:
        for panel in PANELS:
            cell = f"{topology}/{panel}"
            if cell not in graphs:
                continue
            layers = {s: dict(graphs[cell].get(s, {})) for s in STANCES}
            offered = {s: dict(offers[cell].get(s, {})) for s in STANCES}
            totals = {s: round(sum(layers[s].values()), 2) for s in STANCES}
            flow_all = sum(totals.values()) or 1
            said = {s: pool[cell].get(s, 0) for s in STANCES}
            pool_all = sum(said.values()) or 1
            result["cells"][cell] = {
                "claims": len(claims[cell]),
                "layers": layers,
                "offered": offered,
                "offered_totals": {s: round(sum(offered[s].values()), 2)
                                   for s in STANCES},
                "totals": totals,
                "spoken": said,
                "flow_share": {s: totals[s] / flow_all for s in STANCES},
                "pool_share": {s: said[s] / pool_all for s in STANCES},
                "lift": {s: (totals[s] / flow_all) / (said[s] / pool_all)
                         if said[s] else None for s in STANCES},
                "unlabelled": round(
                    sum(graphs[cell].get("UNLABELLED", {}).values()), 2),
                "lift_by_claim": {
                    s: _summarise(per_claim[cell][s]) for s in STANCES
                },
            }

    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / f"stance-flow{tag}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n")

    edges = sorted({e for c in result["cells"].values()
                    for layer in c["layers"].values() for e in layer})
    head = f"{'condition':16s}" + "".join(f"{e:>7s}" for e in edges)
    head += f"{'|':>3s}" + "".join(f"{s[:3] + ' flow':>10s}" for s in STANCES)
    head += "".join(f"{s[:3] + ' lift':>10s}" for s in STANCES)
    print(head)
    print("-" * len(head))
    for cell, info in result["cells"].items():
        total = defaultdict(float)
        for layer in info["layers"].values():
            for e, w in layer.items():
                total[e] += w
        lift = lambda s: ("  —  " if info["lift"][s] is None
                          else f"{info['lift'][s]:.2f}")
        print(f"{cell:16s}"
              + "".join(f"{total.get(e, 0):7.1f}" for e in edges)
              + f"{'|':>3s}"
              + "".join(f"{info['totals'][s]:6.0f} {info['flow_share'][s]:3.0%}"
                        for s in STANCES)
              + "".join(f"{lift(s):>10s}" for s in STANCES))
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
