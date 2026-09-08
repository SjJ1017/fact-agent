#!/usr/bin/env python3
"""Is a seat's role recognisable from its input/output profile alone?

If it is, then a profile built from labelled atomic facts works as an identity
signature, and the same construction can be turned on the other paradigms to
ask whether assigned roles stay distinguishable across rounds -- role diversity
measured as separability rather than as a stance label the config asserts.

Two rules keep this honest.

*No label leakage.* A feature may not depend on anyone else's true role. "How
much this seat takes up from Evil" would classify perfectly and prove nothing,
because it already contains the answer. Every feature here is computable by an
observer who sees only the public transcript and this seat's own private notes:
the modality and scope mix of what it says, how much of its own earlier content
it repeats, how much others take up from it and it from them, and whether its
public and private lines agree.

*The game is the unit.* Fifty observations, ten games, so classification is
leave-one-game-out: a seat is never predicted by a model that saw any seat from
the same game. Chance is reported alongside, and with n this small the interval
matters more than the point estimate.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
EVIL = {"Minion", "Assassin"}
MODS = ["record", "assertion", "belief", "speculation", "intention",
        "directive", "strategy"]
SCOPES = ["game", "quest", "state"]


def ci(v, n=5000):
    if len(v) < 3:
        m = st.mean(v) if v else 0.0
        return m, m
    rng = random.Random(0)
    b = sorted(st.mean([v[rng.randrange(len(v))] for _ in v]) for _ in range(n))
    return b[int(.025 * n)], b[int(.975 * n)]


def profiles(d: Path):
    rows = []
    for sp in sorted(d.glob("*.nli.store.json")):
        s = json.loads(sp.read_text())
        roles = json.loads(sp.with_name(sp.name.replace(".nli.store.json",
                                                        ".trace.json")).read_text())["roles"]
        ms = s["mentions"]
        ms = ms if isinstance(ms, dict) else {m["mention_id"]: m for m in ms}
        game = sp.name.split("-")[-1][:2]

        by_agent = defaultdict(list)
        for k, m in ms.items():
            ag = m["provenance"].get("agent_id")
            if ag:
                by_agent[ag].append((k, m))
        idx = {k: (m["provenance"]["agent_id"], m["provenance"]["extra"])
               for k, m in ms.items() if m["provenance"].get("agent_id")}

        # graph degrees, normalised by opportunity, and self-consistency
        pub = {k: (a, e["quest"]) for k, (a, e) in idx.items()
               if e["visibility"] == "public" and e.get("quest") is not None}
        byaq = defaultdict(list)
        for k, (a, q) in pub.items():
            byaq[(a, q)].append(k)
        ags = sorted({a for a, _ in pub.values()})
        qs = sorted({q for _, q in pub.values()})
        opp_out, opp_in = Counter(), Counter()
        for x in ags:
            for y in ags:
                if x == y:
                    continue
                for q1 in qs:
                    for q2 in qs:
                        if q2 > q1:
                            n = len(byaq[(x, q1)]) * len(byaq[(y, q2)])
                            opp_out[x] += n
                            opp_in[y] += n
        deg_out, deg_in, cross = Counter(), Counter(), Counter()
        for r in s.get("relations", []):
            if r["relation"] == "UNRELATED":
                continue
            A, B = pub.get(r["a"]), pub.get(r["b"])
            if A and B and A[0] != B[0] and A[1] != B[1]:
                (x, _), (y, _) = (A, B) if A[1] < B[1] else (B, A)
                deg_out[x] += 1
                deg_in[y] += 1
            # public vs own private agreement
            ia, ib = idx.get(r["a"]), idx.get(r["b"])
            if ia and ib and ia[0] == ib[0]:
                vs = {ia[1]["visibility"], ib[1]["visibility"]}
                if "public" in vs and vs != {"public"}:
                    cross[(ia[0], "n")] += 1
                    if r["relation"] == "EQUIVALENT":
                        cross[(ia[0], "eq")] += 1

        for ag, items in by_agent.items():
            n = len(items)
            mod = Counter(e["modality"] for _, m in items
                          for e in [m["provenance"]["extra"]])
            sc = Counter(e["scope"] for _, m in items
                         for e in [m["provenance"]["extra"]] if "scope" in e)
            vis = Counter(m["provenance"]["extra"]["visibility"] for _, m in items)
            about_self = sum(1 for _, m in items
                             if int(ag[1:]) in (m["provenance"]["extra"].get("about") or []))
            about_any = sum(1 for _, m in items
                            if m["provenance"]["extra"].get("about"))
            f = {f"mod_{k}": mod[k] / n for k in MODS}
            f.update({f"scope_{k}": sc[k] / max(sum(sc.values()), 1) for k in SCOPES})
            f["public_share"] = vis["public"] / n
            f["about_self"] = about_self / max(about_any, 1)
            f["out_deg"] = deg_out[ag] / max(opp_out[ag], 1) * 1000
            f["in_deg"] = deg_in[ag] / max(opp_in[ag], 1) * 1000
            f["say_think_eq"] = cross[(ag, "eq")] / max(cross[(ag, "n")], 1)
            f["props"] = n
            rows.append({"game": game, "agent": ag, "role": roles[ag],
                         "team": "Evil" if roles[ag] in EVIL else "Good", **f})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=ROOT / "experiments" / "avalon_5p_deepseek_v4_flash")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "findings" / "data" / "avalon-role-profile.json")
    a = ap.parse_args()

    rows = profiles(a.dir)
    feats = [k for k in rows[0] if k not in ("game", "agent", "role", "team")]
    print(f"{len(rows)} 个 (局, 座位) 观测，{len(feats)} 个特征，全部不依赖他人真实身份\n")

    print("按真实身份的特征均值（括号内 95% 自助 CI）")
    groups = ("Merlin", "Servant", "Minion", "Assassin")
    show = ["out_deg", "in_deg", "say_think_eq", "mod_belief", "mod_speculation",
            "mod_record", "scope_game", "about_self", "props"]
    print(f'{"特征":<18}' + "".join(f"{g:>20}" for g in groups))
    for f in show:
        line = f'{f:<18}'
        for g in groups:
            v = [r[f] for r in rows if r["role"] == g]
            lo, hi = ci(v)
            line += f"{st.mean(v):>9.2f} [{lo:.2f},{hi:.2f}]"[:20].rjust(20)
        print(line)

    # leave-one-game-out classification, three ways
    X = np.array([[r[f] for f in feats] for r in rows], dtype=float)
    games = np.array([r["game"] for r in rows])
    for label, y, chance in (("好人 / 坏人", np.array([r["team"] for r in rows]), None),
                             ("四类角色", np.array([r["role"] for r in rows]), None),
                             ("是不是梅林", np.array([r["role"] == "Merlin" for r in rows]), None)):
        base = Counter(y).most_common(1)[0][1] / len(y)
        ok = 0
        for g in sorted(set(games)):
            tr, te = games != g, games == g
            if len(set(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
            clf.fit(sc.transform(X[tr]), y[tr])
            ok += (clf.predict(sc.transform(X[te])) == y[te]).sum()
        print(f"\n{label}：留一局交叉验证准确率 {ok/len(rows):.1%}   "
              f"多数类基线 {base:.1%}   n={len(rows)}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
