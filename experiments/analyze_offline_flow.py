"""The offline analyses in findings/2026-09-07-offline-factflow-analysis.md.

Covers §1 (one input, several receivers), §2 (own vs peer rewriting), §4 (a
local weakening seen end to end), §5 (directed coverage between agents) and
§6 (what the generation budget buys).  All of it reads the cached NLI store;
nothing here calls a model.

Three rules from the brief govern every count.

*Unscored is not unrelated.*  The blocker proposes ~6% of the possible pairs
in a debate, so most were never asked about.  Those stay `unknown` and are
reported next to every rate -- folding them into UNRELATED would make sparse
conditions look like they had lost content.

*Direction does not pass through equivalence.*  Relations are read only
between the two propositions actually judged; no transitive closure, because
a chain of paraphrases carries directions neither endpoint was scored for.

*One exposure per proposition per receiver.*  History is cumulative, so a
proposition stays visible for every later round; it is counted at the round
it first became visible, not once per round thereafter.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELL = re.compile(r"-(full|split)-(generic|specialist)-")
CELLS = ["full-generic", "full-specialist", "split-generic", "split-specialist"]
# relation as seen from the later proposition q, given input p
AS_Q = {"EQUIVALENT": "等价", "A_ENTAILS_B": "弱化",
        "B_ENTAILS_A": "细化", "UNRELATED": "无关"}
KINDS = ["等价", "弱化", "细化", "无关"]


def cell_of(name: str) -> str:
    m = CELL.search(name)
    return f"{m.group(1)}-{m.group(2)}" if m else "?"


def load(store_path: Path, suffix: str):
    store = json.loads(store_path.read_text())
    deb = json.loads(store_path.with_name(
        store_path.name.replace(suffix, ".debate.json")).read_text())
    mentions = store["mentions"]
    if isinstance(mentions, list):
        mentions = {m["mention_id"]: m for m in mentions}
    m2f = store["mention_to_fact"]

    said: dict[str, set[str]] = defaultdict(set)
    for mid, fid in m2f.items():
        p = (mentions.get(mid) or {}).get("provenance", {})
        if p.get("agent_id"):
            said[f"{p['agent_id']}|{p['round']}"].add(fid)

    rel: dict[tuple[str, str], str] = {}
    for r in store.get("relations", []):
        fa, fb = m2f.get(r["a"]), m2f.get(r["b"])
        if not fa or not fb or fa == fb:
            continue
        k = r["relation"]
        rel[(fa, fb)] = k
        rel[(fb, fa)] = ("EQUIVALENT" if k == "EQUIVALENT" else
                         "B_ENTAILS_A" if k == "A_ENTAILS_B" else
                         "A_ENTAILS_B" if k == "B_ENTAILS_A" else "UNRELATED")
    return deb, said, rel


def kinds_between(rel, p: str, outs: set[str]) -> set[str]:
    """Multi-hot: every judged relation between input p and this turn's output."""
    out = {AS_Q[rel[(p, q)]] for q in outs if (p, q) in rel}
    return out - {"无关"} or ({"无关"} if any((p, q) in rel for q in outs) else set())


def analyse_case(deb, said, rel, agents, rounds, acc):
    d = deb.get("delivery", {})

    # first exposure of each proposition to each receiver, with its source side
    exposure: dict[tuple[str, str], tuple[int, str]] = {}
    for ag in agents:
        for rnd in rounds:
            slot = d.get(f"{ag}|{rnd}", {})
            peer = set(slot.get("visible_peer_turns", slot.get("peer_turns", [])))
            selfs = set(slot.get("visible_self_turns", []))
            for turns, side in ((selfs, "自己"), (peer, "同伴")):
                for t in turns:
                    for p in said.get(t, set()):
                        key = (ag, p)
                        if key in exposure:
                            r0, s0 = exposure[key]
                            if r0 == rnd and s0 != side:
                                exposure[key] = (r0, "两者")
                        else:
                            exposure[key] = (rnd, side)

    # ---- §2 own vs peer, and the receiver table §1 is built from
    for (ag, p), (rnd, side) in exposure.items():
        ks = kinds_between(rel, p, said.get(f"{ag}|{rnd}", set()))
        acc["q2"][side]["n"] += 1
        if not ks:
            acc["q2"][side]["未评分"] += 1
        for k in ks:
            acc["q2"][side][k] += 1

    # ---- §1 the same proposition, visible to two or more receivers
    by_prop: dict[tuple[str, int], list[str]] = defaultdict(list)
    for (ag, p), (rnd, _side) in exposure.items():
        by_prop[(p, rnd)].append(ag)
    for (p, rnd), recv in by_prop.items():
        if len(recv) < 2:
            continue
        acc["q1_props"] += 1
        seen: dict[str, set[str]] = {}
        for ag in recv:
            ks = kinds_between(rel, p, said.get(f"{ag}|{rnd}", set()))
            seen[ag] = ks
            acc["q1_seat"][ag]["n"] += 1
            if not ks:
                acc["q1_seat"][ag]["未评分"] += 1
            for k in ks:
                acc["q1_seat"][ag][k] += 1
        nonempty = [k for k in seen.values() if k]
        if len(nonempty) >= 2:
            acc["q1_agree"]["可比"] += 1
            acc["q1_agree"]["接收者一致" if len(
                {frozenset(k) for k in nonempty}) == 1 else "接收者分歧"] += 1

    # ---- §4 p -> q -> r, classified on the directly judged endpoints
    step: dict[str, set[str]] = defaultdict(set)
    for rnd in rounds[:-1]:
        cur = set().union(*[said.get(f"{a}|{rnd}", set()) for a in agents] or [set()])
        nxt = {a2: said.get(f"{a2}|{rnd + 1}", set()) for a2 in agents}
        for p in cur:
            for a2, qs in nxt.items():
                if exposure.get((a2, p), (99, ""))[0] > rnd + 1:
                    continue          # p was not yet visible to a2 that round
                for q in qs:
                    if rel.get((p, q)) == "A_ENTAILS_B":
                        step[p].add(q)
    for p, mids in step.items():
        for q in mids:
            for r_ in step.get(q, ()):
                k = rel.get((p, r_))
                acc["q4"]["端点未评分" if k is None else
                          {"EQUIVALENT": "恢复到等价", "A_ENTAILS_B": "继续变弱",
                           "B_ENTAILS_A": "反而更强", "UNRELATED": "转向别处"}[k]] += 1

    # ---- §5 directed coverage between agents, per round
    for rnd in rounds:
        for x, y in permutations(agents, 2):
            fx, fy = said.get(f"{x}|{rnd}", set()), said.get(f"{y}|{rnd}", set())
            if not fx or not fy:
                continue
            xy = sum(1 for f in fx if any(rel.get((f, g)) == "EQUIVALENT" for g in fy))
            yx = sum(1 for g in fy if any(rel.get((g, f)) == "EQUIVALENT" for f in fx))
            acc["q5_pair"][rnd][
                "互相覆盖" if xy == len(fx) and yx == len(fy) else
                "单向包含" if xy == len(fx) or yx == len(fy) else
                "部分重叠" if xy or yx else "内容互补"] += 1
            acc["q5_rate"][rnd].append(xy / len(fx))
        for x in agents:
            fx = said.get(f"{x}|{rnd}", set())
            others = set().union(*[said.get(f"{y}|{rnd}", set())
                                   for y in agents if y != x] or [set()])
            if fx:
                acc["q5_joint"][rnd].append(
                    sum(1 for f in fx
                        if any(rel.get((f, g)) == "EQUIVALENT" for g in others))
                    / len(fx))

    # ---- §6 each turn's output against everything visible to it
    for ag in agents:
        for rnd in rounds:
            hist = {p for (a2, p), (r0, _) in exposure.items()
                    if a2 == ag and r0 <= rnd}
            for q in said.get(f"{ag}|{rnd}", set()):
                ks = {AS_Q[rel[(p, q)]] for p in hist if (p, q) in rel}
                acc["q6"][rnd]["等价重复" if "等价" in ks else
                               "已有内容的较弱表述" if "弱化" in ks else
                               "细化候选" if "细化" in ks else
                               "无关于历史" if ks else "未匹配产出"] += 1
                acc["q6_n"][rnd] += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path,
                    default=ROOT / "experiments" / "idrbench_generation_10x5_r3")
    ap.add_argument("--suffix", default=".nli.store.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "findings" / "data" / "offline-flow.json")
    a = ap.parse_args()

    def blank():
        return {"q1_seat": defaultdict(Counter), "q1_props": 0,
                "q1_agree": Counter(), "q2": defaultdict(Counter),
                "q4": Counter(), "q5_pair": defaultdict(Counter),
                "q5_joint": defaultdict(list), "q5_rate": defaultdict(list), "q6": defaultdict(Counter),
                "q6_n": Counter(), "cov": [0, 0], "cases": 0}

    acc = {c: blank() for c in CELLS}
    for sp in sorted(a.dir.glob(f"*{a.suffix}")):
        if not sp.with_name(sp.name.replace(a.suffix, ".debate.json")).exists():
            continue
        deb, said, rel = load(sp, a.suffix)
        c = cell_of(sp.name)
        agents = sorted(deb.get("roles", {})) or sorted({s.split("|")[0] for s in said})
        rounds = sorted({int(s.split("|")[1]) for s in said})
        nf = len({f for s in said.values() for f in s})
        acc[c]["cov"][0] += len(rel) // 2
        acc[c]["cov"][1] += nf * (nf - 1) // 2
        acc[c]["cases"] += 1
        analyse_case(deb, said, rel, agents, rounds, acc[c])

    def table(title, rows, keys, note=""):
        print(f"\n{title}")
        if note:
            print(f"  {note}")
        print(f'{"":<24}' + "".join(f"{k:>10}" for k in keys) + f'{"n":>9}')
        for label, cnt in rows:
            n = cnt.get("n") or sum(cnt[k] for k in keys)
            if not n:
                continue
            print(f'{label:<24}' + "".join(f"{cnt[k] / n:>10.1%}" for k in keys)
                  + f'{n:>9,}')

    print("可分析范围（被 NLI 直接评分的命题对占全部可能对的比例）")
    for c in CELLS:
        s, t = acc[c]["cov"]
        print(f'  {c:<18}{s:>8,} / {t:>9,} = {s / t:>6.2%}   '
              f'{acc[c]["cases"]} 场')
    print("  其余一律记 unknown，不计入任何比率的分母之外的类别。")

    k2 = KINDS + ["未评分"]
    for c in CELLS:
        table(f"§1/§2  {c}",
              [(f"输入来自{side}", acc[c]["q2"][side])
               for side in ("自己", "同伴", "两者")] +
              [(f"接收席位 {s}", acc[c]["q1_seat"][s])
               for s in sorted(acc[c]["q1_seat"])],
              k2, note="多标签，一次曝光可同时命中多类，行和可超 100%")
        g = acc[c]["q1_agree"]
        if g["可比"]:
            print(f'  §1 同一命题被多个接收者收到 {acc[c]["q1_props"]:,} 次；'
                  f'其中 {g["可比"]:,} 次有两个以上接收者留下可评分的关系，'
                  f'{g["接收者分歧"] / g["可比"]:.1%} 关系类型不一致')

    print("\n§4 一次弱化之后，端到端是什么关系")
    print(f'{"":<24}' + "".join(f"{k:>10}" for k in
                               ["恢复到等价", "继续变弱", "反而更强", "转向别处", "端点未评分"])
          + f'{"n":>9}')
    for c in CELLS:
        q, n = acc[c]["q4"], sum(acc[c]["q4"].values())
        if n:
            print(f'{c:<24}' + "".join(
                f"{q[k] / n:>10.1%}" for k in
                ["恢复到等价", "继续变弱", "反而更强", "转向别处", "端点未评分"]) + f'{n:>9,}')

    print("\n§5 同轮内 agent 之间的有向覆盖")
    print(f'{"":<24}' + "".join(f"{k:>10}" for k in
                               ["互相覆盖", "单向包含", "部分重叠", "内容互补"])
          + f'{"两两覆盖率":>9}{"联合覆盖":>10}{"n":>7}')
    for c in CELLS:
        for rnd in sorted(acc[c]["q5_pair"]):
            p, n = acc[c]["q5_pair"][rnd], sum(acc[c]["q5_pair"][rnd].values())
            j = acc[c]["q5_joint"][rnd]
            print(f'{c + " r" + str(rnd):<24}' + "".join(
                f"{p[k] / n:>10.1%}" for k in
                ["互相覆盖", "单向包含", "部分重叠", "内容互补"])
                + f'{st.mean(acc[c]["q5_rate"][rnd]):>10.1%}'+ f'{st.mean(j):>10.1%}{n:>7}')

    print("\n§6 每轮输出相对于实际可见历史的分解")
    ks6 = ["等价重复", "已有内容的较弱表述", "细化候选", "无关于历史", "未匹配产出"]
    print(f'{"":<24}' + "".join(f"{k:>12}" for k in ks6) + f'{"n":>8}')
    for c in CELLS:
        for rnd in sorted(acc[c]["q6"]):
            q, n = acc[c]["q6"][rnd], acc[c]["q6_n"][rnd]
            print(f'{c + " r" + str(rnd):<24}' + "".join(
                f"{q[k] / n:>12.1%}" for k in ks6) + f'{n:>8,}')

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        c: {"coverage": {"scored": acc[c]["cov"][0], "possible": acc[c]["cov"][1]},
            "q1_seat": {s: dict(v) for s, v in acc[c]["q1_seat"].items()},
            "q1_agreement": dict(acc[c]["q1_agree"]),
            "q2": {s: dict(v) for s, v in acc[c]["q2"].items()},
            "q4": dict(acc[c]["q4"]),
            "q5_pair": {r: dict(v) for r, v in acc[c]["q5_pair"].items()},
            "q5_rate": {r: round(st.mean(v), 4)
                        for r, v in acc[c]["q5_rate"].items()},
            "q5_joint": {r: round(st.mean(v), 4)
                         for r, v in acc[c]["q5_joint"].items()},
            "q6": {r: dict(v) for r, v in acc[c]["q6"].items()}}
        for c in CELLS}, ensure_ascii=False, indent=1))
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
