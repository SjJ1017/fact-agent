#!/usr/bin/env python3
"""Does the role prompt differentiate content, or does only the split do it?

Specialisation index for a turn: share of B-topic propositions minus share of
C-topic ones, signed by the seat's nominal side (A is the paper-B side, B the
paper-C side; seat C is the synthesis seat and is left out).  A seat that
really works its own side scores high; a seat indistinguishable from its
neighbour scores near zero.

Both `full-specialist` and `split-*` assign A to paper B and B to paper C, so
the index is comparable across them; `full-generic` has no assignment and its
seats are read under the same nominal mapping, which is exactly what makes it
the null.
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_fact_discipline as L  # noqa: E402
from analyze_offline_flow import load, cell_of, CELLS  # noqa: E402
from analyze_topic_flow import load_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "experiments" / "idrbench_generation_10x5_r3"


def boot(x, y, n=20000):
    obs = st.mean(x) - st.mean(y)
    pool = list(x) + list(y)
    k = 0
    for _ in range(n):
        random.shuffle(pool)
        if abs(st.mean(pool[:len(x)]) - st.mean(pool[len(x):])) >= abs(obs):
            k += 1
    return obs, (k + 1) / (n + 1)


def main() -> int:
    uid2lab = load_labels()
    man = json.loads((L.EVAL / "manifest.json").read_text())
    idx = defaultdict(list)

    for sp in sorted(D.glob("*.nli.store.json")):
        dp = sp.with_name(sp.name.replace(".nli.store.json", ".debate.json"))
        if not dp.exists():
            continue
        deb, said, _rel = load(sp, ".nli.store.json")
        c = cell_of(sp.name)
        papers = man["tasks"][deb["case_id"]]["papers"]
        store = json.loads(sp.read_text())
        lab = {}
        for fid, f in store["facts"].items():
            if f.get("canonical_text"):
                lab[fid] = uid2lab.get(
                    L.digest([deb["case_id"], papers, f["canonical_text"]])[:20])
        for rnd in (1, 2, 3):
            vals = []
            for seat, sign in (("A", +1), ("B", -1)):
                fs = [lab.get(f) for f in said.get(f"{seat}|{rnd}", set())]
                fs = [x for x in fs if x]
                if len(fs) < 10:
                    continue
                b = sum(x["topic"] == "B" for x in fs) / len(fs)
                cc = sum(x["topic"] == "C" for x in fs) / len(fs)
                vals.append(sign * (b - cc))
            if vals:
                idx[f"{c}|{rnd}"].append(st.mean(vals))

    random.seed(0)
    print("专业化指数：本侧主题占比 − 对侧主题占比（A、B 两席平均，一场一个观测）\n")
    print(f'{"":<20}{"第 1 轮":>10}{"第 2 轮":>10}{"第 3 轮":>10}')
    for c in CELLS:
        print(f'{c:<20}' + "".join(
            f'{st.mean(idx[f"{c}|{r}"]):>10.1%}' for r in (1, 2, 3)))

    print("\n对照检验")
    for r in (1, 2, 3):
        d, p = boot(idx[f"full-specialist|{r}"], idx[f"full-generic|{r}"])
        print(f'  r{r}  只加角色（full-specialist vs full-generic）   {d:>+7.1%}  p={p:.3f}')
    for r in (1, 2, 3):
        d, p = boot(idx[f"split-generic|{r}"], idx[f"full-generic|{r}"])
        print(f'  r{r}  只分信息（split-generic vs full-generic）     {d:>+7.1%}  p={p:.3f}')
    for r in (1, 2, 3):
        d, p = boot(idx[f"split-generic|{r}"], idx[f"split-specialist|{r}"])
        print(f'  r{r}  分信息下再加角色                              {d:>+7.1%}  p={p:.3f}')

    out = ROOT / "findings" / "data" / "topic-specialisation.json"
    out.write_text(json.dumps(
        {k: {"mean": round(st.mean(v), 4), "n": len(v)} for k, v in idx.items()},
        ensure_ascii=False, indent=1))
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
