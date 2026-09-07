#!/usr/bin/env python3
"""Perspectrum under one judge: three topologies x three personas.

The earlier perspectrum numbers came from the SAME/DIFF pipeline, so the full
condition and the star/chain conditions were never measured the same way --
the pipeline difference sat exactly where the topology effect was supposed to
be. All 108 deepseek traces have now been re-matched with the Qwen3-14B entail
judge at one shared threshold, which is what makes this grid comparable.

Perspectrum runs peer-only memory: an agent does not see its own earlier turn,
only what was delivered. `delivery` records `peer_turns` and no visible-history
keys, so an edge is admitted when the source turn was actually delivered to the
receiver -- which is what makes chain's first node correctly have no input.

Every rate is a case mean over the 12 claims, never a pooled count over edges.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_offline_flow import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOPO = ["full", "star", "chain"]
PERS = ["neutral", "lenses", "stance"]
AS_Q = {"EQUIVALENT": "等价", "A_ENTAILS_B": "弱化", "B_ENTAILS_A": "细化",
        "UNRELATED": "无关"}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "findings" / "data" / "perspectrum-topology.json")
    a = ap.parse_args()

    per = defaultdict(lambda: defaultdict(list))     # cell -> metric -> per-case
    rel_mix = defaultdict(Counter)

    for sp in sorted(ROOT.glob("experiments/perspectrum_pilot_*/*.nli.store.json")):
        if a.model not in sp.name:
            continue
        deb, said, rel = load(sp, ".nli.store.json")
        cell = f"{deb['topology']}/{deb['panel']}"
        agents = sorted(deb["roles"])
        rounds = sorted({int(s.split("|")[1]) for s in said})
        d = deb.get("delivery", {})

        rel_mix[cell].update(r["relation"] for r in json.loads(sp.read_text())
                             .get("relations", []))

        # delivered edges only: peer_turns is the record of what arrived
        deliv = {f"{ag}|{r}": set(d.get(f"{ag}|{r}", {}).get(
            "visible_peer_turns", d.get(f"{ag}|{r}", {}).get("peer_turns", [])))
            for ag in agents for r in rounds}

        cnt = Counter()
        opp = 0
        for ag in agents:
            for r in rounds:
                out = said.get(f"{ag}|{r}", set())
                for t in deliv[f"{ag}|{r}"]:
                    for p in said.get(t, set()):
                        opp += len(out)
                        for q in out:
                            k = rel.get((p, q))
                            if k:
                                cnt[AS_Q[k]] += 1
        scored = sum(cnt.values())
        if not scored:
            continue
        per[cell]["每千组合成边"].append(scored / max(opp, 1) * 1000)
        for k in ("等价", "弱化", "细化"):
            per[cell][k].append(cnt[k] / scored)
        per[cell]["细化减弱化"].append((cnt["细化"] - cnt["弱化"]) / scored)

        # how much of an agent's round output another agent also stated
        cov = []
        for r in rounds:
            for x in agents:
                fx = said.get(f"{x}|{r}", set())
                if not fx:
                    continue
                others = set().union(*[said.get(f"{y}|{r}", set())
                                       for y in agents if y != x] or [set()])
                cov.append(sum(1 for f in fx if any(
                    rel.get((f, g)) == "EQUIVALENT" for g in others)) / len(fx))
        if cov:
            per[cell]["同轮等价重合"].append(st.mean(cov))
        per[cell]["不同事实数"].append(len({f for s in said.values() for f in s}))

    keys = ["不同事实数", "每千组合成边", "等价", "弱化", "细化", "细化减弱化",
            "同轮等价重合"]
    print(f"{'条件':<16}" + "".join(f"{k:>13}" for k in keys) + f"{'n':>5}")
    for t in TOPO:
        for p in PERS:
            c = f"{t}/{p}"
            if not per[c]:
                continue
            row = []
            for k in keys:
                v = st.mean(per[c][k])
                row.append(f"{v:>13.1f}" if k in ("不同事实数", "每千组合成边")
                           else f"{v:>13.1%}")
            print(f"{c:<16}" + "".join(row) + f"{len(per[c]['等价']):>5}")

    random.seed(0)
    print("\n拓扑效应（同 persona 内配对，合并三个 persona）")
    for k in ("每千组合成边", "同轮等价重合", "细化减弱化"):
        for t in ("star", "chain"):
            x = [v for p in PERS for v in per[f"{t}/{p}"][k]]
            y = [v for p in PERS for v in per[f"full/{p}"][k]]
            df, pv = boot(x, y)
            unit = "" if k == "每千组合成边" else "%"
            fmt = (lambda z: f"{z:+.1f}") if k == "每千组合成边" else (lambda z: f"{z*100:+.1f}pp")
            print(f"  {k:<12}{t:<6} vs full  {fmt(df)}  p={pv:.3f}")

    print("\npersona 效应（同拓扑内配对，合并三个拓扑）")
    for k in ("每千组合成边", "同轮等价重合", "细化减弱化"):
        for p in ("lenses", "stance"):
            x = [v for t in TOPO for v in per[f"{t}/{p}"][k]]
            y = [v for t in TOPO for v in per[f"{t}/neutral"][k]]
            df, pv = boot(x, y)
            fmt = (lambda z: f"{z:+.1f}") if k == "每千组合成边" else (lambda z: f"{z*100:+.1f}pp")
            print(f"  {k:<12}{p:<6} vs neutral  {fmt(df)}  p={pv:.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"cells": {c: {k: {"mean": st.mean(v), "n": len(v)} for k, v in m.items()}
                   for c, m in per.items()},
         "relation_mix": {c: dict(v) for c, v in rel_mix.items()}},
        ensure_ascii=False, indent=1))
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
