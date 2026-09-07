"""Graph structure of the entailment edges, by condition and by role.

The four idrbench cells cross information split with role differentiation, so
every count here is reported per cell rather than pooled: the whole point is
whether the structure moves with the manipulation.

One caveat governs the reading of every number below.  Under cumulative
memory every agent sees every earlier turn, so "could the speaker have seen
it" excludes nothing -- 0 of 9,202 cross-round edges failed that test.  A
degraded edge therefore means the weaker statement came later and the speaker
*could* have seen the stronger one, not that it came from it.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELL = re.compile(r"-(full|split)-(generic|specialist)-")


def cell_of(name: str) -> str:
    m = CELL.search(name)
    return f"{m.group(1)}-{m.group(2)}" if m else "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=ROOT / "findings" / "data" / "entail-view.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "findings" / "data" / "entail-graph.json")
    a = ap.parse_args()

    view = json.loads(a.data.read_text())
    per_cell: dict[str, list[dict]] = defaultdict(list)

    for db in view["debates"]:
        cell = cell_of(db["id"])
        edges = db["entail"]
        kinds = Counter(e["kind"] for e in edges)
        deg = [e for e in edges if e["kind"] == "degraded"]
        ref = [e for e in edges if e["kind"] == "refined"]

        # who weakens whose statements, and who is weakened
        out_deg = Counter(e["strong_agent"] for e in deg)
        in_deg = Counter(e["weak_agent"] for e in deg)
        cross = sum(1 for e in deg if not e["same_agent"])

        # chains: follow degraded edges strong -> weak and take the longest run
        nxt: dict[str, set[str]] = defaultdict(set)
        for e in deg:
            nxt[e["strong_fact"]].add(e["weak_fact"])
        seen_depth: dict[str, int] = {}

        def depth(f: str, stack: frozenset = frozenset()) -> int:
            if f in seen_depth:
                return seen_depth[f]
            if f in stack:          # a cycle: stop rather than recurse
                return 0
            d = 0
            for g in nxt.get(f, ()):
                d = max(d, 1 + depth(g, stack | {f}))
            seen_depth[f] = d
            return d

        chain = max((depth(f) for f in nxt), default=0)

        # how far a degraded edge reaches in rounds
        span = [e["weak_round"] - e["strong_round"] for e in deg]

        per_cell[cell].append({
            "id": db["id"], "kinds": dict(kinds),
            "n_facts": len(db["facts"]), "n_deg": len(deg), "n_ref": len(ref),
            "deg_cross_agent": cross,
            "out_degree": dict(out_deg), "in_degree": dict(in_deg),
            "longest_chain": chain,
            "mean_round_span": round(st.mean(span), 2) if span else 0.0,
        })

    print(f'{"条件":<18}{"场":>4}{"事实":>7}{"降级":>7}{"细化":>7}{"同轮":>7}'
          f'{"降级/事实":>10}{"跨agent%":>9}{"最长链":>7}{"轮跨度":>7}')
    summary = {}
    for cell in sorted(per_cell):
        rows = per_cell[cell]
        f = sum(r["n_facts"] for r in rows)
        d = sum(r["n_deg"] for r in rows)
        rf = sum(r["n_ref"] for r in rows)
        cc = sum(r["kinds"].get("concurrent", 0) for r in rows)
        cross = sum(r["deg_cross_agent"] for r in rows)
        chains = [r["longest_chain"] for r in rows]
        spans = [r["mean_round_span"] for r in rows if r["mean_round_span"]]
        summary[cell] = {
            "debates": len(rows), "facts": f, "degraded": d, "refined": rf,
            "concurrent": cc, "deg_per_fact": round(d / max(1, f), 3),
            "cross_agent_share": round(cross / max(1, d), 3),
            "longest_chain_mean": round(st.mean(chains), 2),
            "round_span_mean": round(st.mean(spans), 2) if spans else 0.0,
        }
        s = summary[cell]
        print(f'{cell:<18}{len(rows):>4}{f:>7}{d:>7}{rf:>7}{cc:>7}'
              f'{s["deg_per_fact"]:>10.3f}{s["cross_agent_share"]:>9.1%}'
              f'{s["longest_chain_mean"]:>7.2f}{s["round_span_mean"]:>7.2f}')

    # agent-level: is one seat the source of weakening more than others?
    print(f'\n{"条件":<18}{"agent":>6}{"被弱化":>8}{"弱化他人":>9}{"净":>7}')
    agents = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for cell, rows in per_cell.items():
        for r in rows:
            for ag, n in r["out_degree"].items():
                agents[cell][ag][0] += n
            for ag, n in r["in_degree"].items():
                agents[cell][ag][1] += n
    for cell in sorted(agents):
        for ag in sorted(agents[cell]):
            out, inn = agents[cell][ag]
            print(f'{cell:<18}{ag:>6}{out:>8}{inn:>9}{out - inn:>+7}')

    a.out.write_text(json.dumps(
        {"summary": summary, "per_cell": {k: v for k, v in per_cell.items()},
         "agents": {c: {a_: list(v) for a_, v in d.items()}
                    for c, d in agents.items()}},
        ensure_ascii=False, indent=1))
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
