"""Flow profile: how atomic facts move through a debate, in metrics that survive
a change of topology or persona.

Every quantity here is defined from four things and nothing else — which agents
exist, which rounds exist, whose output was delivered to whom, and which atomic
facts each turn said. No metric may mention a hub, a lens, or a side of the
argument, because the point is to put full-connect against star against chain,
and neutral against lenses against stance, on one axis.

Three families:

  token budget   every fact a turn says is novel (nobody said it, the speaker
                 included), held (the speaker said it in an earlier round), or
                 adopted (first time from this speaker, but it arrived in a
                 delivery). Mutually exclusive and exhaustive, so `adopted` is
                 readable as "what share of this system's output is uptake".

                 Counts are also reported per thousand visible output tokens,
                 because a raw count cannot tell "said more distinct things"
                 from "wrote more words". Tokens come from re-tokenising the
                 transcript with bge-base-en-v1.5, which reproduces
                 `turn_output_tokens` in the token_clock layer exactly and,
                 unlike it, still counts a turn that yielded no facts.

                 `adopted_share` over all turns confounds two things that
                 topology moves in opposite directions: how often an agent takes
                 up what it hears, and how often it hears anything at all. Chain
                 is A->B->C with no back edges, so A never receives and only 4 of
                 9 turns have any input, against 6 of 9 for full and star. Its
                 raw adoption rate reads 5pp below full and is not an uptake
                 effect. `reception` reports the structural term on its own and
                 `adopted_share_recv` the behavioural one, restricted to turns
                 that actually received something. Compare those two, not the
                 raw share, whenever topology varies.

  graph shape    nestedness (mean pairwise containment of agents' fact sets),
                 reach (how many agents said a given fact), flow Gini (is the
                 carried volume spread over the wiring or piled on one edge),
                 delivery use (what fraction of delivery events moved anything).

  population     meta share (facts judged NEUTRAL — mostly verdicts about the
                 debate or about an evidence document rather than claims about
                 the world) and polarised balance 1 - |n+ - n-| / (n+ + n-).

Statistics are paired within claim: the same claim under two conditions, then a
percentile bootstrap over the 12 differences. Pairing matters because claims
differ far more from each other than conditions do. No multiple-comparison
correction is applied; read a single interval accordingly.

    python experiments/analyze_flow_profile.py
    python experiments/analyze_flow_profile.py --topologies full star chain
"""

from __future__ import annotations

import argparse
import csv
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
METRICS = ("n_facts", "facts_per_k", "novel", "novel_per_k", "adopted_per_k",
           "adopted_share", "adopted_share_recv", "reception",
           "held_share", "nestedness", "reach", "flow_gini", "delivery_use")

_TOKENIZER = None


def output_tokens(debate: dict) -> int:
    """Visible output tokens over the whole debate.

    Counted off the transcript rather than the token_clock layer: the layer
    keys its per-turn count by a fact's first mention, so a turn that produced
    no facts contributes nothing. Star has 14 such turns and chain more, which
    is exactly where an undercounted denominator would do the most damage.
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    return sum(len(_TOKENIZER.encode(text, add_special_tokens=False))
               for text in debate.get("transcript", {}).values())


def gini(values: list[float]) -> float:
    vs = sorted(v for v in values if v >= 0)
    total = sum(vs)
    if not vs or total == 0:
        return 0.0
    n = len(vs)
    return sum((2 * i - n + 1) * v for i, v in enumerate(vs)) / (n * total)


def spoken_by_turn(store: dict) -> dict[tuple[str, int], set[str]]:
    said: dict[tuple[str, int], set[str]] = defaultdict(set)
    for mid, mention in store["mentions"].items():
        prov = mention["provenance"]
        if prov["channel"] == "output" and prov.get("agent_id") and prov.get("round"):
            fid = store["mention_to_fact"].get(mid)
            if fid:
                said[(prov["agent_id"], prov["round"])].add(fid)
    return said


def profile(store: dict, debate: dict, execution_id: str) -> dict:
    said = spoken_by_turn(store)
    rounds = sorted({r for _, r in said})
    agents = sorted({a for a, _ in said})
    if not rounds or not agents:
        return {}

    received: dict[tuple[str, int], set[str]] = defaultdict(set)
    for slot, info in debate.get("delivery", {}).items():
        listener, rnd = slot.split("|")
        for peer in info.get("peer_turns", []):
            speaker, peer_round = peer.split("|")
            received[(listener, int(rnd))] |= said[(speaker, int(peer_round))]

    novel = held = adopted = 0
    recv_novel = recv_held = recv_adopted = 0
    receiving_turns = 0
    for agent in agents:
        for rnd in rounds:
            earlier = set().union(
                *[said[(agent, r)] for r in rounds if r < rnd]
            ) if rnd > rounds[0] else set()
            current = said[(agent, rnd)]
            fresh = current - earlier
            trio = (len(fresh - received[(agent, rnd)]),
                    len(current & earlier),
                    len(fresh & received[(agent, rnd)]))
            novel += trio[0]
            held += trio[1]
            adopted += trio[2]
            if received[(agent, rnd)]:
                receiving_turns += 1
                recv_novel += trio[0]
                recv_held += trio[1]
                recv_adopted += trio[2]
    total = novel + held + adopted
    recv_total = recv_novel + recv_held + recv_adopted

    sets = {a: set().union(*[said[(a, r)] for r in rounds]) for a in agents}
    spoken = set().union(*sets.values())
    nest = [len(sets[a] & sets[b]) / len(sets[b])
            for a in agents for b in agents if a != b and sets[b]]
    reach = [sum(1 for a in agents if fid in sets[a]) for fid in spoken]

    # Carried volume per delivery event. Keying the edge weight by (a, b) alone
    # collapses the round dimension while `permitted` keeps it; that mismatch
    # once inflated delivery_use from 64% to 86%.
    carried: list[int] = []
    edge: dict[tuple[str, str], int] = defaultdict(int)
    for slot, info in debate.get("delivery", {}).items():
        listener, rnd = slot.split("|")
        rnd = int(rnd)
        before = set().union(
            *[said[(listener, r)] for r in rounds if r < rnd]
        ) if rnd > rounds[0] else set()
        for peer in info.get("peer_turns", []):
            speaker, peer_round = peer.split("|")
            moved = len((said[(speaker, int(peer_round))] & said[(listener, rnd)]) - before)
            carried.append(moved)
            edge[(speaker, listener)] += moved

    tokens = output_tokens(debate)
    per_k = (lambda n: 1000 * n / tokens) if tokens else (lambda n: 0.0)

    attach_labels(store, execution_id, "stance")
    output_mentions = {mid for mid, m in store["mentions"].items()
                       if m["provenance"]["channel"] == "output"}
    stances = [f["properties"]["stance"]
               for f in store["facts"].values()
               if set(f["mention_ids"]) & output_mentions
               and f.get("properties", {}).get("stance")]
    plus = sum(1 for s in stances if s == "SUPPORT")
    minus = sum(1 for s in stances if s == "UNDERMINE")

    return {
        "n_facts": len(spoken),
        "facts_per_k": per_k(len(spoken)),
        "output_tokens": tokens,
        "novel": novel,
        "novel_per_k": per_k(novel),
        "adopted_per_k": per_k(adopted),
        "held": held,
        "adopted": adopted,
        "adopted_share": adopted / total if total else 0.0,
        "adopted_share_recv": recv_adopted / recv_total if recv_total else 0.0,
        "reception": receiving_turns / (len(agents) * len(rounds)),
        "held_share": held / total if total else 0.0,
        "nestedness": st.mean(nest) if nest else 0.0,
        "reach": st.mean(reach) if reach else 0.0,
        "flow_gini": gini(list(edge.values())),
        "delivery_use": sum(1 for c in carried if c > 0) / len(carried) if carried else 0.0,
        "deliveries": len(carried),
        "n_agents": len(agents),
        "meta_share": (sum(1 for s in stances if s == "NEUTRAL") / len(stances)
                       if stances else None),
        "balance": (1 - abs(plus - minus) / (plus + minus)) if plus + minus else None,
    }


def collect(model: str, topologies: tuple[str, ...]) -> dict[tuple[str, str, str], dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for directory in STORE_DIRS:
        for path in sorted(directory.glob(f"*{model}*.v2.json")):
            name = path.name[: -len(".v2.json")]
            match = re.match(rf"perspectrum-(\d+)-{re.escape(model)}-(\w+?)-(\w+)$", name)
            if not match:
                continue
            claim, topology, panel = match.groups()
            if topology not in topologies:
                continue
            debate_path = path.with_name(f"{name}.debate.json")
            if not debate_path.exists():
                continue
            with path.open() as fh:
                store = json.load(fh)
            with debate_path.open() as fh:
                debate = json.load(fh)
            row = profile(store, debate, name)
            if row:
                out[(topology, claim, panel)] = row
    return out


def bootstrap(values: list[float], draws: int = 20_000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def contrast(data, keys_a, keys_b, claims) -> dict:
    out = {}
    for metric in METRICS:
        diffs = [data[ka(c)][metric] - data[kb(c)][metric]
                 for c in claims
                 for ka, kb in [(keys_a, keys_b)]
                 if ka(c) in data and kb(c) in data]
        if len(diffs) < 3:
            continue
        lo, hi = bootstrap(diffs)
        out[metric] = {
            "delta": st.mean(diffs), "lo": lo, "hi": hi,
            "positive": sum(1 for d in diffs if d > 0), "n": len(diffs),
            "excludes_zero": lo > 0 or hi < 0,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--topologies", nargs="+", default=["full", "star"])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    topologies = tuple(args.topologies)

    data = collect(args.model, topologies)
    if not data:
        raise SystemExit("no debates matched; check --model and --topologies")
    claims = sorted({c for _, c, _ in data})

    cells = {}
    for topology in topologies:
        for panel in PANELS:
            rows = [data[(topology, c, panel)] for c in claims
                    if (topology, c, panel) in data]
            if not rows:
                continue
            cell = {"n": len(rows)}
            for key in rows[0]:
                vals = [r[key] for r in rows if r[key] is not None]
                cell[key] = st.mean(vals) if vals else None
            cells[f"{topology}/{panel}"] = cell

    paired = {}
    for topology in topologies:
        for a, b in (("lenses", "neutral"), ("stance", "neutral"), ("stance", "lenses")):
            for metric, stat in contrast(
                data, lambda c, t=topology, p=a: (t, c, p),
                lambda c, t=topology, p=b: (t, c, p), claims
            ).items():
                paired[f"{topology}|{a}-{b}|{metric}"] = stat
    for panel in PANELS:
        for i, base in enumerate(topologies):
            for other in topologies[i + 1:]:
                for metric, stat in contrast(
                    data, lambda c, t=other, p=panel: (t, c, p),
                    lambda c, t=base, p=panel: (t, c, p), claims
                ).items():
                    paired[f"topology|{panel}|{other}-{base}|{metric}"] = stat

    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model, "topologies": list(topologies), "claims": claims,
        "cells": cells, "paired": paired,
        "per_debate": {f"{t}|{c}|{p}": r for (t, c, p), r in data.items()},
    }
    (args.out / "flow-profile.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n")

    with (args.out / "flow-profile-per-debate.csv").open("w", newline="") as fh:
        fields = ["topology", "claim", "panel"] + list(next(iter(data.values())))
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for (topology, claim, panel), row in sorted(data.items()):
            writer.writerow({"topology": topology, "claim": claim,
                             "panel": panel, **row})

    head = f"{'condition':16s}{'n':>3s}{'facts':>7s}{'f/kTok':>8s}{'tokens':>8s}"
    head += f"{'adopt':>8s}{'a|recv':>8s}{'recv':>7s}{'held':>7s}"
    head += f"{'nest':>8s}{'reach':>7s}{'gini':>7s}{'d.use':>7s}{'meta':>7s}{'bal':>7s}"
    print(head)
    print("-" * len(head))
    for name, c in cells.items():
        pct = lambda v: "  —  " if v is None else f"{v:.1%}"
        num = lambda v: "  —  " if v is None else f"{v:.3f}"
        print(f"{name:16s}{c['n']:3d}{c['n_facts']:7.1f}{c['facts_per_k']:8.1f}"
              f"{c['output_tokens']:8.0f}{pct(c['adopted_share']):>8s}"
              f"{pct(c['adopted_share_recv']):>8s}{pct(c['reception']):>7s}"
              f"{pct(c['held_share']):>7s}{num(c['nestedness']):>8s}"
              f"{c['reach']:7.2f}{num(c['flow_gini']):>7s}"
              f"{pct(c['delivery_use']):>7s}{pct(c['meta_share']):>7s}"
              f"{num(c['balance']):>7s}")

    print("\npaired contrasts whose 95% bootstrap CI excludes zero:")
    for key, stat in paired.items():
        if stat["excludes_zero"]:
            print(f"  {key:44s} {stat['delta']:+9.3f} "
                  f"[{stat['lo']:+.3f}, {stat['hi']:+.3f}]  "
                  f"{stat['positive']}/{stat['n']} positive")
    print(f"\nwrote {args.out / 'flow-profile.json'} and flow-profile-per-debate.csv")


if __name__ == "__main__":
    main()
