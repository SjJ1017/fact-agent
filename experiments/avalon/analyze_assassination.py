#!/usr/bin/env python3
"""What the Assassin was reading when it named Merlin.

Good completed three quests in eight of ten games and still lost seven of them,
which is the shape AvalonBench reports from win rates alone. A win rate cannot
say why. A proposition-level record can: the Assassin's own reasoning at the
assassination is in the trace as private text, the public speech that preceded
it is there too, and the same NLI judge that matched everything else will say
which earlier public propositions its reasoning restates.

The question this exists to ask is whether Merlin talks himself into being
identified -- whether the public propositions the Assassin's rationale echoes
are disproportionately Merlin's own.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REL = {"EQUIVALENT": "等价", "A_ENTAILS_B": "单向", "B_ENTAILS_A": "单向"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=ROOT / "experiments" / "avalon_5p_deepseek_v4_flash")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "findings" / "data" / "avalon-assassination.json")
    a = ap.parse_args()

    games, echo_author, echo_mod, target_talk = [], Counter(), Counter(), []
    merlin_share = []

    for sp in sorted(a.dir.glob("*.nli.store.json")):
        d = json.loads(sp.read_text())
        tr = json.loads(sp.with_name(sp.name.replace(".nli.store.json",
                                                     ".trace.json")).read_text())
        roles = tr["roles"]
        merlin = next(k for k, v in roles.items() if v == "Merlin")
        assassin = next(k for k, v in roles.items() if v == "Assassin")
        ev = next((e for e in tr["events"] if e["phase"] == "assassination"), None)
        if ev is None:
            continue
        target = f"P{ev['action']}" if str(ev.get("action", "")).isdigit() else None
        hit = target == merlin

        ms = d["mentions"]
        ms = ms if isinstance(ms, dict) else {m["mention_id"]: m for m in ms}
        # the Assassin's reasoning written at the assassination itself
        rationale = {k for k, m in ms.items()
                     if m["provenance"].get("agent_id") == assassin
                     and m["provenance"]["extra"].get("phase") == "assassination"}
        public = {k: m for k, m in ms.items()
                  if m["provenance"]["extra"]["visibility"] == "public"
                  and m["provenance"].get("agent_id")}

        echoed = defaultdict(set)
        for r in d.get("relations", []):
            for x, y in ((r["a"], r["b"]), (r["b"], r["a"])):
                if x in rationale and y in public and r["relation"] in REL:
                    echoed[y].add(REL[r["relation"]])
        by_author = Counter(public[k]["provenance"]["agent_id"] for k in echoed)
        n_echo = sum(by_author.values())
        for k in echoed:
            echo_author[roles.get(public[k]["provenance"]["agent_id"], "?")] += 1
            echo_mod[public[k]["provenance"]["extra"]["modality"]] += 1
        if n_echo:
            merlin_share.append(by_author.get(merlin, 0) / n_echo)

        # how much public talk mentioned the target at all
        about_t = sum(1 for m in public.values()
                      if int(target[1:]) in (m["provenance"]["extra"].get("about") or []))
        target_talk.append((about_t, len(public)))
        games.append({"game": sp.name.split("-")[-1][:2], "merlin": merlin,
                      "assassin": assassin, "target": target, "hit": hit,
                      "rationale_props": len(rationale), "echoed_public": n_echo,
                      "merlin_authored": by_author.get(merlin, 0)})

    hits = sum(g["hit"] for g in games)
    print(f"{len(games)} 局，刺杀命中 {hits} 局（{hits/len(games):.0%}）\n")
    print(f'{"局":<4}{"梅林":>5}{"刺客":>6}{"目标":>6}{"命中":>6}'
          f'{"刺客理由命题":>13}{"回响到的公开命题":>17}{"其中梅林所说":>13}')
    for g in games:
        print(f'{g["game"]:<4}{g["merlin"]:>5}{g["assassin"]:>6}{str(g["target"]):>6}'
              f'{"是" if g["hit"] else "否":>6}{g["rationale_props"]:>13}'
              f'{g["echoed_public"]:>17}{g["merlin_authored"]:>13}')

    print("\n刺客理由所回响的公开命题，按其作者的真实身份:")
    n = sum(echo_author.values())
    for k, v in echo_author.most_common():
        print(f'  {k:<10}{v:>5}{v/n:>8.1%}')
    print(f"\n随机基线：五人中梅林一人，若与身份无关应为 20.0%")
    if merlin_share:
        print(f"实际梅林占比（按局平均）: {st.mean(merlin_share):.1%}")

    print("\n所回响命题的情态:")
    m = sum(echo_mod.values())
    for k, v in echo_mod.most_common():
        print(f'  {k:<12}{v:>5}{v/m:>8.1%}')

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"games": games,
                                 "echo_author": dict(echo_author),
                                 "echo_modality": dict(echo_mod)},
                                ensure_ascii=False, indent=1))
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
