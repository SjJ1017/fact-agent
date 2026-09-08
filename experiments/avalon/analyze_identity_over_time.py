#!/usr/bin/env python3
"""How talk about identity changes as the game goes on.

Four things move together and are worth separating, because a rising share of
identity talk could mean the panel is converging on an answer or merely that it
has run out of anything else to say.

1. What share of what is said is about someone's alignment at all.
2. Within that, the modality mix -- does hedged speculation give way to flat
   assertion as quests resolve and evidence accumulates?
3. Whether those claims get any more accurate, scored against the trace's roles.
4. The flow-native one: what share of identity claims are graph-connected to an
   earlier claim rather than raised fresh. A discussion that is genuinely
   accumulating should show later claims building on earlier ones; one that is
   merely repeating itself under pressure shows the opposite.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIL = {"Minion", "Assassin"}
ALIGN = re.compile(r"\bPlayer (\d)\b[^.]*?\bis\b\s*(not\s+)?"
                   r"(Evil|Good|Merlin|the Assassin|a Servant|a loyal Servant|"
                   r"the Minion)", re.I)


def truth_of(claim, subject, negated, roles):
    actual = roles.get(f"P{subject}")
    if actual is None:
        return None
    c = claim.lower()
    if "evil" in c:       holds = actual in EVIL
    elif "good" in c:     holds = actual not in EVIL
    elif "merlin" in c:   holds = actual == "Merlin"
    elif "assassin" in c: holds = actual == "Assassin"
    elif "minion" in c:   holds = actual == "Minion"
    elif "servant" in c:  holds = actual == "Servant"
    else: return None
    return (not holds) if negated else holds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=ROOT / "experiments" / "avalon_5p_deepseek_v4_flash")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "findings" / "data" / "avalon-identity-time.json")
    a = ap.parse_args()

    share = defaultdict(list)          # quest -> per-game share of identity talk
    mod = defaultdict(Counter)         # quest -> modality counts within identity
    acc = defaultdict(Counter)         # quest -> true/false
    linked = defaultdict(Counter)      # quest -> connected / fresh

    for sp in sorted(a.dir.glob("*.nli.store.json")):
        d = json.loads(sp.read_text())
        tr = json.loads(sp.with_name(sp.name.replace(".nli.store.json",
                                                     ".trace.json")).read_text())
        roles = tr["roles"]
        ms = d["mentions"]
        ms = ms if isinstance(ms, dict) else {m["mention_id"]: m for m in ms}

        ident, meta, total = {}, {}, Counter()
        for k, m in ms.items():
            e = m["provenance"]["extra"]
            if e["visibility"] != "public" or not m["provenance"].get("agent_id"):
                continue
            q = e.get("quest")
            if q is None:
                continue
            meta[k] = q
            total[q] += 1
            h = ALIGN.search(m["text"])
            if h:
                t = truth_of(h.group(3), h.group(1),
                             bool(h.group(2)) or m.get("polarity") == "negate", roles)
                ident[k] = (q, e["modality"], t)

        n_by_q = Counter(q for q, _, _ in ident.values())
        for q in sorted(total):
            if total[q] >= 20:
                share[q].append(n_by_q.get(q, 0) / total[q])
        for k, (q, md, t) in ident.items():
            mod[q][md] += 1
            if t is not None:
                acc[q]["true" if t else "false"] += 1

        # graph link: does this identity claim relate to an EARLIER one?
        back = defaultdict(bool)
        for r in d.get("relations", []):
            if r["relation"] == "UNRELATED":
                continue
            for x, y in ((r["a"], r["b"]), (r["b"], r["a"])):
                if x in ident and y in ident and ident[y][0] < ident[x][0]:
                    back[x] = True
        for k, (q, _, _) in ident.items():
            linked[q]["接着早前的说" if back[k] else "新起"] += 1

    qs = sorted(q for q in share if len(share[q]) >= 5)
    print("一、身份类命题占公开命题的比例")
    print(f'{"轮次":<8}{"占比":>9}{"场数":>7}')
    for q in qs:
        print(f'Quest {q:<3}{st.mean(share[q]):>9.1%}{len(share[q]):>7}')

    print("\n二、身份命题的情态构成（推测是否让位于断言）")
    ks = ["speculation", "assertion", "belief", "record", "directive"]
    print(f'{"轮次":<8}' + "".join(f"{k:>13}" for k in ks) + f'{"n":>7}')
    for q in qs:
        c = mod[q]; n = sum(c.values()) or 1
        print(f'Quest {q:<3}' + "".join(f"{c[k]/n:>13.1%}" for k in ks) + f"{n:>7}")

    print("\n三、身份主张的准确率（对照 trace 真值）")
    print(f'{"轮次":<8}{"真":>6}{"假":>6}{"准确率":>9}')
    for q in qs:
        c = acc[q]; n = c["true"] + c["false"]
        if n:
            print(f'Quest {q:<3}{c["true"]:>6}{c["false"]:>6}{c["true"]/n:>9.1%}')

    print("\n四、身份命题里，有多少是接着更早的身份命题说的（需要匹配才能判定）")
    print(f'{"轮次":<8}{"接着说":>9}{"新起":>8}{"接着说占比":>12}')
    for q in qs:
        c = linked[q]; n = sum(c.values()) or 1
        print(f'Quest {q:<3}{c["接着早前的说"]:>9}{c["新起"]:>8}'
              f'{c["接着早前的说"]/n:>12.1%}')

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"share": {q: st.mean(v) for q, v in share.items()},
         "modality": {q: dict(v) for q, v in mod.items()},
         "accuracy": {q: dict(v) for q, v in acc.items()},
         "linked": {q: dict(v) for q, v in linked.items()}},
        ensure_ascii=False, indent=1))
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
