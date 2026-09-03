"""Four more things the fact graph can carry, none of them needing a new label.

The uptake graph answers "what moves between agents". These use the same
machinery on data already in the repository to ask three other questions, and
one of them is the only place a debate's *outcome* enters the analysis at all.

  verdict    Every turn ends in a parseable VERDICT — 972 of 972, no misses.
             This reports the level per condition, and the round-to-round
             transitions as a flow on {SUPPORT, UNCERTAIN, UNDERMINE}, which is
             a graph of the same kind with a different set of nodes. It also
             tests the obvious hypothesis, that what an agent absorbed in a
             round predicts whether its verdict moved. It does not.

  survival   A fact first said in round r: is it ever said again? Splits the
             existing nestedness/reach numbers by when the fact appeared, so
             the question becomes *when* integration fails rather than whether.

  grounding  A fact whose cluster also contains a mention from an evidence
             document is traceable to the dossier. Only 12.7% of source facts
             merge with anything an agent said, so the rate here is a floor on
             grounding and mostly a statement about the matcher — but the
             *composition* is still readable, and it says the assigned stance
             does not change which documents an agent draws on.

  position   Where in its own output a turn puts an adopted fact, from the
             token clock that has been sitting unused. Front-loaded would mean
             engagement, back-loaded a closing nod. It is neither.

    python experiments/analyze_flow_extras.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from export_labels import attach_labels

ROOT = Path(__file__).resolve().parent
STORE_DIRS = [ROOT / "perspectrum_pilot_full", ROOT / "perspectrum_pilot_star_chain"]
OUT = ROOT.parent / "findings" / "data"
PANELS = ("neutral", "lenses", "stance")
STANCES = ("SUPPORT", "UNDERMINE", "NEUTRAL")
VERDICTS = ("SUPPORT", "UNCERTAIN", "UNDERMINE")
VERDICT_RE = re.compile(r"VERDICT\s*[:：]\s*([A-Za-z_ ]+)")


def bootstrap(values, draws=20_000, seed=0):
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    means = [st.mean(rng.choices(values, k=len(values))) for _ in range(draws)]
    above = (sum(m > 0 for m in means) + 0.5 * sum(m == 0 for m in means)) / draws
    means.sort()
    return {"delta": st.mean(values), "lo": means[int(0.025 * draws)],
            "hi": means[int(0.975 * draws)], "p_positive": above,
            "positive": sum(1 for v in values if v > 0), "n": len(values)}


def turn_facts(store: dict):
    said = defaultdict(set)
    for mid, mention in store["mentions"].items():
        prov = mention["provenance"]
        if prov["channel"] == "output" and prov.get("agent_id") and prov.get("round"):
            fid = store["mention_to_fact"].get(mid)
            if fid:
                said[(prov["agent_id"], prov["round"])].add(fid)
    return said


def received(said, debate):
    got = defaultdict(set)
    for slot, info in debate.get("delivery", {}).items():
        listener, rnd = slot.split("|")
        # 漏斗的 exposed 一段是"本可以被采纳"，那是窗口不是到达。
        for peer in info.get("visible_peer_turns", info.get("peer_turns", [])):
            speaker, peer_round = peer.split("|")
            got[(listener, int(rnd))] |= said[(speaker, int(peer_round))]
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--memory", default="peer-only",
                    choices=("peer-only", "self-last", "cumulative"),
                    help="只分析这一种记忆条件；混着平均没有意义")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = OUT
    tag = "" if args.memory == "peer-only" else f"-{args.memory}"

    verdict_level = defaultdict(Counter)
    verdict_moves = defaultdict(lambda: defaultdict(int))
    move_rows = []
    per_claim_unc = defaultdict(dict)
    survival = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    per_claim_surv = defaultdict(dict)
    ground = defaultdict(lambda: [0, 0])
    ex_flow = defaultdict(lambda: defaultdict(float))
    role_ground = defaultdict(lambda: [0, 0])
    position = defaultdict(lambda: defaultdict(list))

    for directory in STORE_DIRS:
        for path in sorted(directory.glob(f"*{args.model}*.v2.json")):
            name = path.name[: -len(".v2.json")]
            m = re.match(
                rf"perspectrum-(\d+)-{re.escape(args.model)}-(\w+?)-(neutral|lenses|stance)"
                rf"(?:-(cumulative|self-last))?$", name)
            if not m:
                continue
            claim, topology, panel, mem = m.groups()
            if (mem or "peer-only") != args.memory:
                continue
            debate_path = path.with_name(f"{name}.debate.json")
            if not debate_path.exists():
                continue
            with path.open() as fh:
                store = json.load(fh)
            with debate_path.open() as fh:
                debate = json.load(fh)
            attach_labels(store, name, "stance")
            attach_labels(store, name, "token_clock")
            cell = f"{topology}/{panel}"

            said = turn_facts(store)
            got = received(said, debate)
            rounds = sorted({r for _, r in said})
            agents = sorted({a for a, _ in said})
            stance = {fid: f.get("properties", {}).get("stance")
                      for fid, f in store["facts"].items()}

            # --- verdict ---
            verdict = {}
            for slot, text in debate["transcript"].items():
                hit = VERDICT_RE.search(text)
                if hit:
                    a, r = slot.split("|")
                    verdict[(a, int(r))] = hit.group(1).strip().upper()
            for value in verdict.values():
                verdict_level[cell][value] += 1
            seen = list(verdict.values())
            if seen:
                per_claim_unc[cell][claim] = sum(
                    1 for v in seen if v == "UNCERTAIN") / len(seen)
            for agent in agents:
                for i in range(1, len(rounds)):
                    r0, r1 = rounds[i - 1], rounds[i]
                    if (agent, r0) not in verdict or (agent, r1) not in verdict:
                        continue
                    a0, a1 = verdict[(agent, r0)], verdict[(agent, r1)]
                    verdict_moves[panel][f"{a0}>{a1}"] += 1
                    earlier = set().union(*[said[(agent, r)] for r in rounds if r < r1])
                    adopted = (said[(agent, r1)] - earlier) & got[(agent, r1)]
                    counts = Counter(stance.get(f) for f in adopted)
                    total = sum(counts[s] for s in STANCES)
                    move_rows.append({
                        "topology": topology, "panel": panel, "changed": a0 != a1,
                        "adopted": total,
                        "undermine_share": counts["UNDERMINE"] / total if total else None,
                    })

            # --- survival ---
            first = {}
            for (agent, rnd), facts in said.items():
                for fid in facts:
                    first[fid] = min(first.get(fid, 99), rnd)
            alive = defaultdict(lambda: [0, 0])
            for fid, r0 in first.items():
                later = any(fid in said[(a, r)] for a, r in said if r > r0)
                survival[panel][r0][1] += 1
                alive[r0][1] += 1
                if later:
                    survival[panel][r0][0] += 1
                    alive[r0][0] += 1
            if alive[rounds[0]][1]:
                per_claim_surv[cell][claim] = alive[rounds[0]][0] / alive[rounds[0]][1]

            # --- grounding ---
            source_doc = {mid: mv["provenance"].get("doc_id")
                          for mid, mv in store["mentions"].items()
                          if mv["provenance"]["channel"] == "source"}
            for fact in store["facts"].values():
                ids = set(fact["mention_ids"])
                outs = [store["mentions"][i] for i in ids
                        if store["mentions"][i]["provenance"]["channel"] == "output"]
                if not outs:
                    continue
                docs = {source_doc[i] for i in ids
                        if i in source_doc and source_doc[i] not in (None, "claim")}
                who = {o["provenance"]["agent_id"] for o in outs}
                ground[cell][1] += 1
                for agent in who:
                    role_ground[f"{panel}/{agent}"][1] += 1
                if docs:
                    ground[cell][0] += 1
                    for agent in who:
                        role_ground[f"{panel}/{agent}"][0] += 1
                        for doc in docs:
                            ex_flow[cell][f"{doc}>{agent}"] += 1 / len(docs)

            # --- position within the turn ---
            where = {}
            for mid, mention in store["mentions"].items():
                prov = mention["provenance"]
                clock = prov.get("extra", {}).get("token_clock") or {}
                if (prov["channel"] == "output" and clock.get("turn_output_tokens")):
                    fid = store["mention_to_fact"].get(mid)
                    if fid:
                        where.setdefault(
                            (fid, prov["agent_id"], prov["round"]),
                            clock["output_prefix_tokens"] / clock["turn_output_tokens"])
            for agent in agents:
                for rnd in rounds:
                    earlier = set().union(
                        *[said[(agent, r)] for r in rounds if r < rnd]
                    ) if rnd > rounds[0] else set()
                    for fid in said[(agent, rnd)]:
                        value = where.get((fid, agent, rnd))
                        if value is None:
                            continue
                        if fid in earlier:
                            kind = "held"
                        elif fid in got[(agent, rnd)]:
                            kind = "adopted"
                        else:
                            kind = "novel"
                        position[panel][kind].append(min(1.0, value))

    claims = sorted({c for cell in per_claim_unc.values() for c in cell})
    contrasts = {}
    for source, key in ((per_claim_unc, "uncertain_share"),
                        (per_claim_surv, "round1_survival")):
        for topology in ("full", "star", "chain"):
            for a, b in (("stance", "neutral"), ("lenses", "neutral")):
                ka, kb = f"{topology}/{a}", f"{topology}/{b}"
                diffs = [source[ka][c] - source[kb][c] for c in claims
                         if c in source.get(ka, {}) and c in source.get(kb, {})]
                stat = bootstrap(diffs)
                if stat:
                    contrasts[f"{topology}|{a}-{b}|{key}"] = stat

    if not move_rows:
        raise SystemExit(
            f"--memory {args.memory} 没有可分析的数据。可能是：\n"
            f"  · 该条件的 .v2.json 还没生成\n"
            f"  · 或者生成了但还没做立场标注"
            f"（experiments/labels/stance/ 里要有对应 execution_id 的文件）\n"
            f"检查：ls experiments/perspectrum_pilot_*/*{args.model}*"
            + ("" if args.memory == "peer-only" else f"-{args.memory}") + ".v2.json")
    changed = [r for r in move_rows if r["changed"]]
    same = [r for r in move_rows if not r["changed"]]
    picked = lambda rows, k: [r[k] for r in rows if r[k] is not None]

    result = {
        "memory": args.memory,
        "verdict": {
            "level": {cell: {v: verdict_level[cell][v] for v in VERDICTS}
                      for cell in verdict_level},
            "moves": {p: dict(verdict_moves[p]) for p in PANELS},
            "n_transitions": len(move_rows),
            "n_changed": len(changed),
            "undermine_share_when_changed": st.mean(picked(changed, "undermine_share")),
            "undermine_share_when_same": st.mean(picked(same, "undermine_share")),
            "adopted_when_changed": st.mean([r["adopted"] for r in changed]),
            "adopted_when_same": st.mean([r["adopted"] for r in same]),
        },
        "survival": {p: {str(r): survival[p][r] for r in sorted(survival[p])}
                     for p in PANELS},
        "grounding": {
            "rate": {cell: ground[cell][0] / ground[cell][1] for cell in ground},
            "n": {cell: ground[cell][1] for cell in ground},
            "by_role": {k: v[0] / v[1] for k, v in role_ground.items()},
            "ex_flow": {cell: dict(ex_flow[cell]) for cell in ex_flow},
        },
        "position": {p: {k: {"mean": st.mean(v), "n": len(v)}
                         for k, v in position[p].items()} for p in PANELS},
        "contrasts": contrasts,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / f"flow-extras{tag}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n")

    print("VERDICT level")
    for cell, counts in result["verdict"]["level"].items():
        n = sum(counts.values())
        print(f"  {cell:16s}" + "".join(f"{counts[v] / n:>11.1%}" for v in VERDICTS))
    v = result["verdict"]
    print(f"\ntransitions {v['n_changed']}/{v['n_transitions']} changed; "
          f"undermine share of that round's uptake "
          f"{v['undermine_share_when_changed']:.1%} when changed vs "
          f"{v['undermine_share_when_same']:.1%} when not")
    print("\nround-1 facts ever said again")
    for p in PANELS:
        k, n = result["survival"][p]["1"]
        print(f"  {p:9s} {k / n:6.1%}  ({n} facts)")
    print("\ngrounding rate by role")
    for k in sorted(result["grounding"]["by_role"]):
        print(f"  {k:16s} {result['grounding']['by_role'][k]:6.1%}")
    print("\ncontrasts whose interval excludes zero:")
    for key, stat in contrasts.items():
        if stat["lo"] > 0 or stat["hi"] < 0:
            print(f"  {key:38s} {stat['delta']:+7.3f} "
                  f"[{stat['lo']:+.3f}, {stat['hi']:+.3f}] p={stat['p_positive']:.3f} "
                  f"{stat['positive']}/{stat['n']}")
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
