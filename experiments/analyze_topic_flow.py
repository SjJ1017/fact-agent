#!/usr/bin/env python3
"""Topic-conditioned flow analysis over the 40 IDRBench generation debates.

Reads the discipline labels straight out of the cached response files rather
than the exported sidecars, because three of the ten cases still have batches
the model could not answer inside the declared enum.  Those facts are carried
as an explicit `未标注` class instead of dropping their debates: the gap is
1.9% of the corpus and is not distributed evenly, so it has to stay visible.

The question this exists to settle: the split/round-1 result said generic
agents overlap far more than specialist ones even though they hold different
papers, and the proposed reading was that generic roles fill the gap with the
same boilerplate.  With a `paper` axis that separates "states a source paper"
from "proposes something new", that reading is now falsifiable -- if it is
right, the overlapping propositions should be the P and G ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_fact_discipline as L  # noqa: E402
from analyze_offline_flow import load, cell_of, CELLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOPIC = {"B": "B 侧主题", "C": "C 侧主题", "X": "两篇共有", "G": "通用", "U": "无法判定"}
PAPER = {"B": "陈述 B", "C": "陈述 C", "X": "比较两篇", "P": "新提案", "U": "不确定"}


def load_labels() -> dict[str, dict]:
    """uid -> {topic, paper}, taken from every cached production response."""
    summary = json.loads((L.EVAL / "production-summary.json").read_text())
    out: dict[str, dict] = {}
    for p in (L.EVAL / "responses" / summary["scope"]).glob("*.json"):
        rec = json.loads(p.read_text())
        out.update(rec.get("labels", {}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path,
                    default=ROOT / "experiments" / "idrbench_generation_10x5_r3")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "findings" / "data" / "topic-flow.json")
    a = ap.parse_args()

    uid2lab = load_labels()
    man = json.loads((L.EVAL / "manifest.json").read_text())
    print(f"缓存里可用标签 {len(uid2lab):,} 条")

    cover = Counter()
    seat_topic = defaultdict(lambda: defaultdict(Counter))   # cell -> seat|rnd -> topic
    ovl = defaultdict(lambda: defaultdict(Counter))          # cell|rnd -> 覆盖/未覆盖 -> paper
    ovl_t = defaultdict(lambda: defaultdict(Counter))        # same, topic axis
    edge_topic = defaultdict(lambda: defaultdict(Counter))   # cell -> 自持/交互 -> topic of q

    for sp in sorted(a.dir.glob("*.nli.store.json")):
        dp = sp.with_name(sp.name.replace(".nli.store.json", ".debate.json"))
        if not dp.exists():
            continue
        deb, said, rel = load(sp, ".nli.store.json")
        c = cell_of(sp.name)
        case = deb["case_id"]
        papers = man["tasks"][case]["papers"]
        store = json.loads(sp.read_text())
        lab = {}
        for fid, f in store["facts"].items():
            txt = f.get("canonical_text")
            if not txt:
                continue
            u = L.digest([case, papers, txt])[:20]
            lab[fid] = uid2lab.get(u)
            cover[c if lab[fid] else c + "|缺"] += 1

        agents = sorted(deb["roles"])
        rounds = sorted({int(s.split("|")[1]) for s in said})
        for ag in agents:
            for rnd in rounds:
                for f in said.get(f"{ag}|{rnd}", set()):
                    t = (lab.get(f) or {}).get("topic", "未标注")
                    seat_topic[c][f"{ag}|{rnd}"][t] += 1

        # what is the content that another agent independently restated?
        for rnd in rounds:
            for x in agents:
                fx = said.get(f"{x}|{rnd}", set())
                others = set().union(*[said.get(f"{y}|{rnd}", set())
                                       for y in agents if y != x] or [set()])
                for f in fx:
                    l = lab.get(f)
                    if not l:
                        continue
                    hit = any(rel.get((f, g)) == "EQUIVALENT" for g in others)
                    k = "被他人等价覆盖" if hit else "无人覆盖"
                    ovl[f"{c}|{rnd}"][k][l["paper"]] += 1
                    ovl_t[f"{c}|{rnd}"][k][l["topic"]] += 1

        # topic of the later proposition on self vs cross edges
        d = deb["delivery"]
        vis = {}
        for ag in agents:
            for rnd in rounds:
                s = d.get(f"{ag}|{rnd}", {})
                for t in (list(s.get("visible_self_turns", []))
                          + list(s.get("visible_peer_turns", []))):
                    for p in said.get(t, set()):
                        vis[(ag, p)] = min(vis.get((ag, p), 99), rnd)
        for x in agents:
            for rx in rounds:
                for p in said.get(f"{x}|{rx}", set()):
                    for y in agents:
                        for ry in rounds:
                            if ry <= rx or vis.get((y, p), 99) > ry:
                                continue
                            for q in said.get(f"{y}|{ry}", set()):
                                if rel.get((p, q)) in ("EQUIVALENT", "A_ENTAILS_B",
                                                       "B_ENTAILS_A"):
                                    l = lab.get(q)
                                    if l:
                                        edge_topic[c]["自持" if x == y else "交互"][
                                            l["topic"]] += 1

    print("\n标签覆盖")
    for c in CELLS:
        n, miss = cover[c], cover[c + "|缺"]
        print(f'  {c:<18}{n:>6,} 条已标注，{miss:>4,} 条缺 '
              f'({miss / (n + miss):.1%})')

    print("\n各席位每轮的主题构成（B 侧 / C 侧 / 共有 / 通用 / 未定 / 未标注）")
    for c in CELLS:
        print(f" {c}")
        for slot in sorted(seat_topic[c]):
            v = seat_topic[c][slot]
            n = sum(v.values())
            print(f'   {slot}  ' + "  ".join(
                f'{k}{v[k] / n:>6.1%}' for k in ("B", "C", "X", "G", "U", "未标注")))

    print("\n被别的 agent 等价复述的内容，是在陈述论文还是在提新方案")
    print(f'{"":<24}{"":<16}' + "".join(f"{PAPER[k]:>10}" for k in "BCXPU") + f'{"n":>8}')
    for c in CELLS:
        for rnd in (1, 2, 3):
            for k in ("被他人等价覆盖", "无人覆盖"):
                v = ovl[f"{c}|{rnd}"][k]
                n = sum(v.values())
                if n:
                    print(f'{c + " r" + str(rnd):<24}{k:<16}'
                          + "".join(f"{v[x] / n:>10.1%}" for x in "BCXPU")
                          + f'{n:>8,}')

    print("\n自持边与交互边所携带内容的主题")
    print(f'{"":<24}{"":<8}' + "".join(f"{TOPIC[k]:>10}" for k in "BCXGU") + f'{"n":>9}')
    for c in CELLS:
        for k in ("自持", "交互"):
            v = edge_topic[c][k]
            n = sum(v.values())
            if n:
                print(f'{c:<24}{k:<8}' + "".join(f"{v[x] / n:>10.1%}" for x in "BCXGU")
                      + f'{n:>9,}')

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "coverage": {c: {"labelled": cover[c], "missing": cover[c + "|缺"]}
                     for c in CELLS},
        "seat_topic": {c: {s: dict(v) for s, v in seat_topic[c].items()} for c in CELLS},
        "overlap_paper": {k: {kk: dict(vv) for kk, vv in v.items()}
                          for k, v in ovl.items()},
        "overlap_topic": {k: {kk: dict(vv) for kk, vv in v.items()}
                          for k, v in ovl_t.items()},
        "edge_topic": {c: {k: dict(v) for k, v in edge_topic[c].items()} for c in CELLS},
    }, ensure_ascii=False, indent=1))
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
