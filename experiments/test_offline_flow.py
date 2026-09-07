"""Case-level permutation tests on the contrasts analyze_offline_flow.py finds.

The debate is the independent unit, not the edge -- 10 per cell.  Every rate
here is computed within a case first, then compared across cases, so a single
verbose debate cannot carry a contrast on its own.
"""
import json, random, statistics as st, sys
from collections import defaultdict
sys.path.insert(0, "experiments")
from analyze_offline_flow import load, cell_of, kinds_between, CELLS
from pathlib import Path

D = Path("experiments/idrbench_generation_10x5_r3")
per = defaultdict(lambda: defaultdict(list))
for sp in sorted(D.glob("*.nli.store.json")):
    if not sp.with_name(sp.name.replace(".nli.store.json", ".debate.json")).exists():
        continue
    deb, said, rel = load(sp, ".nli.store.json")
    c = cell_of(sp.name)
    agents = sorted(deb["roles"]); d = deb["delivery"]
    rounds = sorted({int(s.split("|")[1]) for s in said})
    exp = {}
    for ag in agents:
        for rnd in rounds:
            s = d.get(f"{ag}|{rnd}", {})
            for turns, side in ((set(s.get("visible_self_turns", [])), "自己"),
                                (set(s.get("visible_peer_turns", [])), "同伴")):
                for t in turns:
                    for p in said.get(t, set()):
                        if (ag, p) in exp:
                            r0, s0 = exp[(ag, p)]
                            if r0 == rnd and s0 != side: exp[(ag, p)] = (r0, "两者")
                        else: exp[(ag, p)] = (rnd, side)
    cnt = defaultdict(lambda: [0, 0])
    for (ag, p), (rnd, side) in exp.items():
        ks = kinds_between(rel, p, said.get(f"{ag}|{rnd}", set()))
        cnt[side][1] += 1
        if "细化" in ks: cnt[side][0] += 1
        cnt[side + "|弱"][1] += 1
        if "弱化" in ks: cnt[side + "|弱"][0] += 1
        cnt[side + "|未"][1] += 1
        if not ks: cnt[side + "|未"][0] += 1
    for k, (a, b) in cnt.items():
        if b >= 20: per[c][k].append(a / b)

def boot(x, y, n=20000):
    """One-sided permutation test on the difference of case means."""
    obs = st.mean(x) - st.mean(y); pool = x + y; k = 0
    for _ in range(n):
        random.shuffle(pool)
        if abs(st.mean(pool[:len(x)]) - st.mean(pool[len(x):])) >= abs(obs): k += 1
    return obs, (k + 1) / (n + 1)

random.seed(0)
OUT = {}
print("细化率：同一命题自己也说过且同伴也说过（两者） vs 只来自同伴")
for c in CELLS:
    a, b = per[c].get("两者", []), per[c].get("同伴", [])
    if len(a) >= 5:
        d, p = boot(list(a), list(b))
        print(f"  {c:<18} 两者 {st.mean(a):.1%} (n={len(a)}场)  同伴 {st.mean(b):.1%}"
              f"   差 {d:+.1%}  p={p:.3f}")
        OUT[f"redundant_refine|{c}"] = {"both": st.mean(a), "peer": st.mean(b),
                                        "diff": d, "p": p, "cases": len(a)}

print("\n改写方向不对称：自己的历史 vs 同伴的历史")
for c in CELLS:
    for lab, suf in (("细化", ""), ("弱化", "|弱")):
        a, b = per[c].get("自己" + suf, []), per[c].get("同伴" + suf, [])
        if len(a) >= 5:
            d, p = boot(list(a), list(b))
            print(f"  {c:<18}{lab}  自己 {st.mean(a):.1%}  同伴 {st.mean(b):.1%}"
                  f"   差 {d:+.1%}  p={p:.3f}")
            OUT[f"asym|{c}|{lab}"] = {"self": st.mean(a), "peer": st.mean(b),
                                      "diff": d, "p": p}

print("\n未评分（曝光后本轮没有任何被判定的关系）：full vs split")
for pers in ("generic", "specialist"):
    a = [v for c in CELLS if c.startswith("full") and c.endswith(pers)
         for v in per[c].get("同伴|未", [])]
    b = [v for c in CELLS if c.startswith("split") and c.endswith(pers)
         for v in per[c].get("同伴|未", [])]
    d, p = boot(a, b)
    print(f"  {pers:<12} full {st.mean(a):.1%}  split {st.mean(b):.1%}"
          f"   差 {d:+.1%}  p={p:.3f}")
    OUT[f"unscored|{pers}"] = {"full": st.mean(a), "split": st.mean(b),
                               "diff": d, "p": p}

out = Path("findings/data/offline-flow-tests.json")
out.write_text(json.dumps(OUT, ensure_ascii=False, indent=1))
print(f"\n写入 {out}")
