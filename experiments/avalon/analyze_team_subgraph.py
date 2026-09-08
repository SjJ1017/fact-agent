#!/usr/bin/env python3
"""Does information flow denser inside a team than across, when all talk is public?

Every utterance in this game is heard by everyone, so the delivery graph is
complete and identical for all five players. Any difference in how much content
actually moves between two seats is therefore not a channel difference -- it is
selective uptake. If Evil restates Evil more than it restates Good, that is two
agents who know each other's alignment choosing to build on each other in the
open.

Group sizes differ (three Good, two Evil), so raw edge counts are dominated by
how many ordered pairs each block has. Everything here is edges per thousand
(delivered proposition x later output proposition) combinations, the same
normalisation the perspectrum and idrbench topology comparisons use, computed
inside each game and then averaged over games.

Merlin is separated from the Servants throughout: Merlin knows the Evil pair
and the Servants know nothing, so folding them together would average an
informed player in with uninformed ones.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIL = {"Minion", "Assassin"}


def team(role: str) -> str:
    return "Evil" if role in EVIL else ("Merlin" if role == "Merlin" else "Servant")


def boot(x, y, n=20000):
    if len(x) < 4 or len(y) < 4:
        return None, None
    obs = st.mean(x) - st.mean(y); pool = list(x) + list(y); k = 0
    for _ in range(n):
        random.shuffle(pool)
        if abs(st.mean(pool[:len(x)]) - st.mean(pool[len(x):])) >= abs(obs): k += 1
    return obs, (k + 1) / (n + 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=ROOT / "experiments" / "avalon_5p_deepseek_v4_flash")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "findings" / "data" / "avalon-team-subgraph.json")
    a = ap.parse_args()

    per_game = defaultdict(list)      # "Sender>Receiver" -> per-game rate
    kinds = defaultdict(Counter)

    for sp in sorted(a.dir.glob("*.nli.store.json")):
        d = json.loads(sp.read_text())
        tr = json.loads(sp.with_name(sp.name.replace(".nli.store.json",
                                                     ".trace.json")).read_text())
        roles = tr["roles"]
        ms = d["mentions"]
        ms = ms if isinstance(ms, dict) else {m["mention_id"]: m for m in ms}

        # public propositions only, indexed by agent and quest
        pub = defaultdict(list)
        for k, m in ms.items():
            e = m["provenance"]["extra"]
            ag = m["provenance"].get("agent_id")
            if ag and e["visibility"] == "public" and e.get("quest") is not None:
                pub[(ag, e["quest"])].append(k)

        agents = sorted({a_ for a_, _ in pub})
        quests = sorted({q for _, q in pub})
        idx = {k: (ag, q) for (ag, q), ks in pub.items() for k in ks}

        edges = Counter()
        opp = Counter()
        for x in agents:
            for y in agents:
                if x == y:
                    continue
                key = f"{team(roles[x])}>{team(roles[y])}"
                for qx in quests:
                    for qy in quests:
                        if qy <= qx:
                            continue
                        opp[key] += len(pub.get((x, qx), [])) * len(pub.get((y, qy), []))
        for r in d.get("relations", []):
            if r["relation"] == "UNRELATED":
                continue
            ia, ib = idx.get(r["a"]), idx.get(r["b"])
            if not ia or not ib or ia[0] == ib[0]:
                continue
            (x, qx), (y, qy) = (ia, ib) if ia[1] < ib[1] else (ib, ia)
            if qx == qy:
                continue
            key = f"{team(roles[x])}>{team(roles[y])}"
            edges[key] += 1
            kinds[key][r["relation"]] += 1
        for key, o in opp.items():
            if o >= 200:
                per_game[key].append(edges[key] / o * 1000)

    order = ["Evil>Evil", "Servant>Servant", "Servant>Evil", "Evil>Servant",
             "Merlin>Evil", "Evil>Merlin", "Merlin>Servant", "Servant>Merlin"]
    print("每千个（先说的公开命题 × 后说的公开命题）组合里，成边多少条")
    print(f'{"发送 → 接收":<22}{"成边率":>9}{"场数":>7}   关系构成")')
    for k in order:
        v = per_game.get(k)
        if not v:
            continue
        c = kinds[k]; n = sum(c.values()) or 1
        mix = f"等价 {c['EQUIVALENT']/n:.0%} 单向 {(n-c['EQUIVALENT'])/n:.0%}"
        print(f'{k:<22}{st.mean(v):>9.1f}{len(v):>7}   {mix}')

    random.seed(0)
    print("\n组内 vs 跨组（把 Merlin 与 Servant 都算好人）")
    within = [x for k, v in per_game.items() if k.split(">")[0] == k.split(">")[1]
              or set(k.split(">")) <= {"Merlin", "Servant"} for x in v]
    cross = [x for k, v in per_game.items() if "Evil" in k.split(">")
             and set(k.split(">")) != {"Evil"} for x in v]
    df, p = boot(within, cross)
    if df is not None:
        print(f"  组内 {st.mean(within):.1f}   跨组 {st.mean(cross):.1f}   "
              f"差 {df:+.1f}   p={p:.3f}")
    ee, other = per_game.get("Evil>Evil", []), [x for k, v in per_game.items()
                                                if k != "Evil>Evil" for x in v]
    df, p = boot(ee, other)
    if df is not None:
        print(f"  坏人→坏人 {st.mean(ee):.1f}   其余 {st.mean(other):.1f}   "
              f"差 {df:+.1f}   p={p:.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({k: {"mean": st.mean(v), "n": len(v)}
                                 for k, v in per_game.items()},
                                ensure_ascii=False, indent=1))
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
