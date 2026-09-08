#!/usr/bin/env python3
"""Can a seat's assigned role be recovered from its input/output profile?

The point is not the accuracy. It is that Avalon supplies a positive control
and the no-role conditions supply a negative one, so a null result elsewhere
becomes interpretable instead of ambiguous.

In Avalon a role is not a prompt: it fixes the win condition and the private
information, so it cannot fail to take effect. If the profile classifier
separates the teams there, the instrument works. In `neutral` and
`full-generic` no role is assigned at all, so a classifier that beats chance
there would mean the features are picking up seat position or some artifact,
and every other number would be suspect.

Between those two lie the conditions the project actually asks about --
`stance`, `lenses`, `specialist` -- where a role was written into the prompt
and may or may not have changed behaviour. A null there is a finding about the
prompt, not about the method.

Features are computable from mentions and relations alone, so the same profile
is built for every corpus: how much a seat says, how much of it others take up,
how much it takes up, how far its content spreads, and how much it repeats
itself. None of them reads any label.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def profile(sp: Path):
    """One feature row per seat in one run, plus the run id for grouping."""
    s = json.loads(sp.read_text())
    ms = s["mentions"]
    ms = ms if isinstance(ms, dict) else {m["mention_id"]: m for m in ms}
    slot = {}
    for k, m in ms.items():
        p = m["provenance"]
        ag, rd = p.get("agent_id"), p.get("round")
        if ag and rd is not None:
            slot[k] = (ag, int(rd))
    if not slot:
        return []
    byaq = defaultdict(list)
    for k, (ag, r) in slot.items():
        byaq[(ag, r)].append(k)
    ags = sorted({a for a, _ in slot.values()})
    rds = sorted({r for _, r in slot.values()})

    opp_out, opp_in = Counter(), Counter()
    for x in ags:
        for y in ags:
            if x == y:
                continue
            for r1 in rds:
                for r2 in rds:
                    if r2 > r1:
                        n = len(byaq[(x, r1)]) * len(byaq[(y, r2)])
                        opp_out[x] += n
                        opp_in[y] += n
    d_out, d_in, spread, selfrep = Counter(), Counter(), defaultdict(set), Counter()
    for r in s.get("relations", []):
        if r["relation"] == "UNRELATED":
            continue
        A, B = slot.get(r["a"]), slot.get(r["b"])
        if not A or not B or A[1] == B[1]:
            continue
        (x, _), (y, _) = (A, B) if A[1] < B[1] else (B, A)
        if x == y:
            selfrep[x] += 1
        else:
            d_out[x] += 1
            d_in[y] += 1
            spread[x].add(y)

    rows = []
    for ag in ags:
        mine = [k for k, (a, _) in slot.items() if a == ag]
        n = len(mine)
        lens = [len(ms[k]["text"].split()) for k in mine]
        r1 = sum(1 for k in mine if slot[k][1] == min(rds))
        rows.append({
            "seat": ag,
            "props": n,
            "props_r1_share": r1 / n,
            "mean_len": st.mean(lens) if lens else 0.0,
            "out_deg": d_out[ag] / max(opp_out[ag], 1) * 1000,
            "in_deg": d_in[ag] / max(opp_in[ag], 1) * 1000,
            "spread": len(spread[ag]),
            "self_repeat": selfrep[ag] / max(n, 1),
        })
    return rows


FEATS = ["props", "props_r1_share", "mean_len", "out_deg", "in_deg",
         "spread", "self_repeat"]


def evaluate(rows, label_of, name):
    """Leave-one-run-out logistic regression; the run is never split."""
    if len(rows) < 12:
        print(f"  {name:<34} 样本不足（{len(rows)}）")
        return None
    X = np.array([[r[f] for f in FEATS] for r in rows], float)
    y = np.array([label_of(r) for r in rows])
    g = np.array([r["run"] for r in rows])
    base = Counter(y).most_common(1)[0][1] / len(y)
    ok = 0
    for run in sorted(set(g)):
        tr, te = g != run, g == run
        if len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        ok += int((clf.predict(sc.transform(X[te])) == y[te]).sum())
    acc = ok / len(rows)
    flag = "  ←超基线" if acc > base + .08 else ""
    print(f"  {name:<34}{acc:>7.1%}   基线 {base:>5.1%}   n={len(rows):<4}{flag}")
    return {"acc": acc, "chance": base, "n": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "findings" / "data" / "role-separability.json")
    a = ap.parse_args()
    res = {}

    def gather(pattern, key_of, suffix=".nli.store.json"):
        buckets = defaultdict(list)
        for sp in sorted(ROOT.glob(pattern)):
            deb = sp.with_name(sp.name.replace(suffix, ".debate.json"))
            tra = sp.with_name(sp.name.replace(suffix, ".trace.json"))
            meta = json.loads((deb if deb.exists() else tra).read_text())
            for r in profile(sp):
                r["run"] = sp.name
                buckets[key_of(meta)].append({**r, "meta": meta})
        return buckets

    print("阳性对照 · Avalon（角色即规则，必定生效）")
    av = gather("experiments/avalon_5p_deepseek_v4_flash/*.nli.store.json",
                lambda m: "avalon")
    rows = av["avalon"]
    for r in rows:
        r["role"] = r["meta"]["roles"][r["seat"]]
    res["avalon_team"] = evaluate(
        rows, lambda r: "Evil" if r["role"] in ("Minion", "Assassin") else "Good",
        "好人 / 坏人")
    res["avalon_role"] = evaluate(rows, lambda r: r["role"], "四类角色")

    print("\n阴性对照与检验 · Perspectrum（座位 A/B/C）")
    ps = gather("experiments/perspectrum_pilot_*/*deepseek*.nli.store.json",
                lambda m: m.get("panel", "?"))
    for panel in ("neutral", "lenses", "stance"):
        if ps.get(panel):
            tag = "（阴性对照：无角色）" if panel == "neutral" else ""
            res[f"perspectrum_{panel}"] = evaluate(
                ps[panel], lambda r: r["seat"], f"{panel} 分座位{tag}")

    print("\n阴性对照与检验 · IDRBench（座位 A/B/C）")
    ib = gather("experiments/idrbench_generation_10x5_r3/*.nli.store.json",
                lambda m: m.get("condition", "?"))
    for cond in ("full-generic", "full-specialist", "split-generic",
                 "split-specialist-aligned"):
        if ib.get(cond):
            tag = "（阴性对照：无角色）" if cond.endswith("generic") else ""
            res[f"idrbench_{cond}"] = evaluate(
                ib[cond], lambda r: r["seat"], f"{cond} 分座位{tag}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
