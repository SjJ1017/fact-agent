#!/usr/bin/env python3
"""Do false propositions travel further than true ones?

This is the question the debate corpora cannot ask and Avalon can. There, a
weakening edge means a qualifier was lost; here a proposition about a player's
alignment has a truth value taken from the trace's own role assignment, so an
edge can be classified by what it does to truth as well as to strength.

Everything here is graph-shaped and needs the matching by construction:

*Fan-out.* A seed proposition is one an agent stated about someone's alignment.
Its fan-out is the number of OTHER agents who later stated something the judge
matched to it. Counting requires the matcher; no amount of string comparison
finds "Player 3 is Evil" restated as "Player 3 is not to be trusted on a quest".

*Survival.* How many rounds later the last matched restatement appears.

*Truth-flipping edges.* A pair where the earlier proposition is true and the
later one false, or the reverse. That edge type exists only where relations and
ground truth are both available, and it is the Avalon-specific addition to the
equivalent / weakened / sharpened taxonomy the other corpora use.
"""

from __future__ import annotations

import argparse
import json
import random
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
    if "evil" in c:      holds = actual in EVIL
    elif "good" in c:    holds = actual not in EVIL
    elif "merlin" in c:  holds = actual == "Merlin"
    elif "assassin" in c: holds = actual == "Assassin"
    elif "minion" in c:  holds = actual == "Minion"
    elif "servant" in c: holds = actual == "Servant"
    else: return None
    return (not holds) if negated else holds


def boot(x, y, n=20000):
    if len(x) < 5 or len(y) < 5:
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
                    default=ROOT / "findings" / "data" / "avalon-truth-flow.json")
    a = ap.parse_args()

    fan = {"true": [], "false": []}
    surv = {"true": [], "false": []}
    flip = Counter()
    edge_kind = defaultdict(Counter)
    seeds_n = Counter()

    for sp in sorted(a.dir.glob("*.nli.store.json")):
        d = json.loads(sp.read_text())
        tr = json.loads(sp.with_name(sp.name.replace(".nli.store.json",
                                                     ".trace.json")).read_text())
        roles = tr["roles"]
        ms = d["mentions"]
        ms = ms if isinstance(ms, dict) else {m["mention_id"]: m for m in ms}

        truth, meta = {}, {}
        for k, m in ms.items():
            e = m["provenance"]["extra"]
            ag = m["provenance"].get("agent_id")
            meta[k] = (ag, e.get("quest"), e["visibility"], e["modality"])
            if not ag or e["visibility"] != "public":
                continue
            h = ALIGN.search(m["text"])
            if not h:
                continue
            t = truth_of(h.group(3), h.group(1),
                         bool(h.group(2)) or m.get("polarity") == "negate", roles)
            if t is not None:
                truth[k] = t

        # graph: who later matched what
        nxt = defaultdict(list)
        for r in d.get("relations", []):
            if r["relation"] == "UNRELATED":
                continue
            for x, y in ((r["a"], r["b"]), (r["b"], r["a"])):
                if x in meta and y in meta:
                    nxt[x].append((y, r["relation"]))
            # truth-flipping edges
            ta, tb = truth.get(r["a"]), truth.get(r["b"])
            if ta is not None and tb is not None:
                qa, qb = meta[r["a"]][1], meta[r["b"]][1]
                if qa is None or qb is None or qa == qb:
                    continue
                early, late = ((r["a"], r["b"]) if qa < qb else (r["b"], r["a"]))
                e_t, l_t = truth[early], truth[late]
                flip[f"{'真' if e_t else '假'}→{'真' if l_t else '假'}"] += 1
                edge_kind[f"{'真' if e_t else '假'}→{'真' if l_t else '假'}"][
                    r["relation"]] += 1

        for k, t in truth.items():
            ag0, q0 = meta[k][0], meta[k][1]
            others, last = set(), q0
            for y, _rel in nxt.get(k, []):
                ag1, q1 = meta[y][0], meta[y][1]
                if ag1 and ag1 != ag0 and q1 is not None and q0 is not None and q1 >= q0:
                    others.add(ag1)
                    last = max(last, q1)
            key = "true" if t else "false"
            fan[key].append(len(others))
            surv[key].append((last - q0) if q0 is not None else 0)
            seeds_n[key] += 1

    random.seed(0)
    print(f"种子命题（公开的身份主张，真值可判）：真 {seeds_n['true']}，假 {seeds_n['false']}\n")
    print("一、跨 agent 扇出：后来有多少个「其他 agent」说了被判官匹配上的话")
    for k, lab in (("true", "真命题"), ("false", "假命题")):
        v = fan[k]
        print(f"  {lab}  均值 {st.mean(v):.2f}  中位 {st.median(v):.0f}  "
              f"扇出≥1 的比例 {sum(1 for x in v if x)/len(v):.1%}  n={len(v)}")
    df, p = boot(fan["false"], fan["true"])
    if df is not None:
        print(f"  假 − 真 = {df:+.2f}  p={p:.3f}")

    print("\n二、存活轮数：最后一次被匹配上的表述，距首次出现几轮")
    for k, lab in (("true", "真命题"), ("false", "假命题")):
        v = surv[k]
        print(f"  {lab}  均值 {st.mean(v):.2f} 轮  存活>0 的比例 "
              f"{sum(1 for x in v if x)/len(v):.1%}")
    df, p = boot(surv["false"], surv["true"])
    if df is not None:
        print(f"  假 − 真 = {df:+.2f} 轮  p={p:.3f}")

    print("\n三、跨轮边对真值做了什么（只有同时有关系和真值才定义得出来）")
    tot = sum(flip.values())
    for k in ("真→真", "真→假", "假→真", "假→假"):
        if flip[k]:
            kinds = "  ".join(f"{r} {n}" for r, n in edge_kind[k].most_common())
            print(f"  {k}  {flip[k]:>4} ({flip[k]/tot:>5.1%})   {kinds}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"fanout": fan, "survival": surv,
                                 "flip": dict(flip),
                                 "flip_by_relation": {k: dict(v) for k, v in edge_kind.items()}},
                                ensure_ascii=False, indent=1))
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
