"""What an agent has in front of it, against what it chooses to say.

Every metric so far counts output. But an agent's context is knowable too: the
dossier's own facts are in it on every turn, and from round 2 the facts its
peers said are in it as well. Treating the agent as a node with an input side
and an output side makes a different question askable — not how much moves, but
how the stance mix changes as it passes through.

    context(a, r) = facts extracted from the evidence documents
                  + facts said in the peer turns delivered into a|r
                  + facts the agent itself said last round, when the run was
                    generated with self-last memory (delivery.self_turn)
    output(a, r)  = facts said in a|r

Both sides carry the same SUPPORT / UNDERMINE / NEUTRAL labels, so the two mixes
are directly comparable, and the difference is a selection: of everything this
agent could have repeated, what did it choose. `tilt` is that difference in
percentage points, positive when the output leans more that way than the input.

The dossier is balanced by construction — two documents for the claim, two
against — so a tilt is not inherited from the materials.

    python experiments/analyze_context_output.py
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
OUT = ROOT.parent / "findings" / "data" / "context-output{memory}.json"
PANELS = ("neutral", "lenses", "stance")
STANCES = ("SUPPORT", "UNDERMINE", "NEUTRAL")
ROLE = {"neutral": {"A": "中立", "B": "中立", "C": "中立"},
        "lenses": {"A": "因果证据", "B": "落地与权衡", "C": "范围与不确定性"},
        "stance": {"A": "支持方", "B": "反对方", "C": "裁决者"}}


def bootstrap(values, draws=20_000, seed=0):
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    means = [st.mean(rng.choices(values, k=len(values))) for _ in range(draws)]
    above = (sum(m > 0 for m in means) + 0.5 * sum(m == 0 for m in means)) / draws
    means.sort()
    return {"delta": st.mean(values), "lo": means[int(0.025 * draws)],
            "hi": means[int(0.975 * draws)],
            "confidence": max(above, 1 - above),
            "positive": sum(1 for v in values if v > 0), "n": len(values)}


def mix(fids, stance_of):
    counts = {s: 0 for s in STANCES}
    for fid in fids:
        s = stance_of.get(fid)
        if s in counts:
            counts[s] += 1
    total = sum(counts.values())
    return ({s: counts[s] / total for s in STANCES} if total else None), total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--memory", default="peer-only",
                    choices=("peer-only", "self-last", "cumulative"))
    # Default output carries the condition. Three analyses writing one path
    # means the last one silently replaces the other two.
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        tag = "" if args.memory == "peer-only" else f"-{args.memory}"
        args.out = Path(str(OUT).format(memory=tag))

    rows = []
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
            stance_of = {fid: f.get("properties", {}).get("stance")
                         for fid, f in store["facts"].items()}

            said = defaultdict(set)
            dossier = set()
            for mid, mention in store["mentions"].items():
                prov = mention["provenance"]
                fid = store["mention_to_fact"].get(mid)
                if not fid:
                    continue
                if prov["channel"] == "output" and prov.get("agent_id") and prov.get("round"):
                    said[(prov["agent_id"], prov["round"])].add(fid)
                elif prov["channel"] == "source" and prov.get("doc_id") not in (None, "claim"):
                    dossier.add(fid)

            for slot, info in debate.get("delivery", {}).items():
                agent, rnd = slot.split("|")
                rnd = int(rnd)
                # visible_*, not peer_turns. A delivery event is what arrived
                # this round; the context window is what the model can still
                # read. Under cumulative those differ — full's A|3 is handed B2
                # and C2 but the older user turns still hold B1 and C1 — and
                # using the delivery would count facts the agent can see as
                # facts it cannot.
                peers = set()
                for peer in info.get("visible_peer_turns", info.get("peer_turns", [])):
                    a, r = peer.split("|")
                    peers |= said[(a, int(r))]
                # Under peer-only memory `self_turn` is absent and context is
                # the dossier plus peers. Under self-last it is present, the
                # agent's own previous answer is genuinely in front of it, and
                # leaving it out would count facts the agent could see as
                # facts it could not — inflating every tilt by whatever the
                # agent had already said.
                own = set()
                self_visible = info.get("visible_self_turns")
                if self_visible is None:
                    self_visible = [info["self_turn"]] if info.get("self_turn") else []
                for turn in self_visible:
                    a, r = turn.split("|")
                    own |= said[(a, int(r))]
                context = dossier | peers | own
                cmix, cn = mix(context, stance_of)
                omix, on = mix(said[(agent, rnd)], stance_of)
                if not cmix or not omix:
                    continue
                rows.append({
                    "claim": claim, "topology": topology, "panel": panel,
                    "agent": agent, "round": rnd, "role": ROLE[panel][agent],
                    "context": cmix, "output": omix,
                    "n_context": cn, "n_output": on,
                    "peers_only": mix(peers, stance_of)[0] if peers else None,
                    "own_prior": mix(own, stance_of)[0] if own else None,
                    # The run's condition, not this turn's. Round 1 has no
                    # prior turn under any setting, so inferring from self_turn
                    # would label every first round peer-only.
                    "memory": debate.get("memory",
                                         "self-last" if debate.get("self_history")
                                         else "peer-only"),
                })

    # The headline is every turn, because the question is what an agent does
    # with whatever is in front of it, and from round 2 most of that is what
    # peers said rather than the dossier. Rounds are also reported separately,
    # which answers a different question: whether the tilt is a one-off
    # reaction to the materials or something the agent does on every pass. Note
    # that the later-round baseline is peer output that already tilted, so a
    # smaller slope there is not evidence the drift is slowing.
    def summarise(subset, key):
        out = {}
        for s in STANCES:
            tilt = [r["output"][s] - r["context"][s] for r in subset]
            stat = bootstrap(tilt)
            out[s] = {"context": st.mean(r["context"][s] for r in subset),
                      "output": st.mean(r["output"][s] for r in subset),
                      "tilt": stat}
        out["n"] = len(subset)
        return out

    result = {"model": args.model, "n_turns": len(rows), "by": {}}
    first_round = min(r["round"] for r in rows)
    all_rounds = sorted({r["round"] for r in rows})
    scopes = [("", lambda r: True)]
    scopes += [(f"r{n}", (lambda n: lambda r: r["round"] == n)(n)) for n in all_rounds]
    for scope, keep in scopes:
        for panel in PANELS:
            sub = [r for r in rows if r["panel"] == panel and keep(r)]
            tag = f"{scope}|" if scope else ""
            if sub:
                result["by"][f"{tag}{panel}"] = summarise(sub, panel)
            for agent in "ABC":
                s2 = [r for r in sub if r["agent"] == agent]
                if s2:
                    result["by"][f"{tag}{panel}/{agent}"] = summarise(s2, agent)
    result["rows"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")

    print(f"{len(rows)} 个 turn\n")
    for scope, title in (
            ("", "全部 turn：上下文 = 卷宗 ∪ 同伴投递 ∪ 自己上一轮（若该条件有记忆）"),
            ("r1", "只看第 1 轮（此时上下文只有卷宗）"),
            ("r2", "只看第 2 轮"),
            ("r3", "只看第 3 轮")):
        print(f"■ {title}")
        hdr = f"{'条件':14s}{'角色':11s}" + "".join(f"{s[:3]+' 入→出':>17s}" for s in STANCES)
        print(hdr); print("-" * len(hdr))
        tag = f"{scope}|" if scope else ""
        for panel in PANELS:
            for key, label in ([(f"{tag}{panel}", "全部")] +
                               [(f"{tag}{panel}/{a}", ROLE[panel][a]) for a in "ABC"]):
                c = result["by"].get(key)
                if not c:
                    continue
                line = f"{panel if label=='全部' else '':14s}{label:11s}"
                for st_ in STANCES:
                    t = c[st_]["tilt"]
                    mark = "*" if t and t["confidence"] >= 0.975 else " "
                    line += (f"{c[st_]['context']:6.0%}→{c[st_]['output']:5.0%}"
                             f"{mark}{t['delta']*100:+5.1f}")
                print(line)
            print()
        print()

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
