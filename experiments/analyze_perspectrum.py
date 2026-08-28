"""What each agent does with the evidence, measured without the fact matcher.

The traces this reads were matched by `run_perspectrum.light_match`, which
merges only on stemmed Jaccard >= .80 and never adjudicates. At a 5% merge rate
every metric that depends on two agents having stated "the same" fact -
survival, carry, echo, sharing - is measuring paraphrase rather than content,
so none of it is reported here.

What survives is the evidence citation itself. Agents refer to the dossier by
literal `E1`..`E4` tokens, which are exact strings in the raw transcript: no
extraction, no matching, no 8-fact cap. Each evidence item carries a stance
label from Perspectrum, so a turn's citations reduce to two numbers that are
directly interpretable:

    coverage   how much of the dossier the turn engages with at all
    lean       (support cites - undermine cites) / total, in [-1, +1]

and across the three agents of a run:

    divergence mean pairwise L1 distance between their citation distributions

Divergence over rounds is the question the persona conditions exist to answer.
If role prompts are decorative, `stance` agents cite the dossier the same way
`neutral` agents do and the three conditions collapse. If they bite, the
advocate and the critic should lean in opposite directions from round 1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

EVID = re.compile(r"\bE([1-9])\b")


def turn_citations(text: str) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for m in EVID.finditer(text or ""):
        out["E" + m.group(1)] += 1
    return dict(out)


def lean(cites: dict[str, int], stance: dict[str, str]) -> float | None:
    """+1 = cites only supporting evidence, -1 = only undermining. None if silent."""
    pos = sum(n for e, n in cites.items() if stance.get(e) == "support")
    neg = sum(n for e, n in cites.items() if stance.get(e) in ("undermine", "refute"))
    return (pos - neg) / (pos + neg) if pos + neg else None


def l1(a: dict[str, int], b: dict[str, int], ids: list[str]) -> float:
    """L1 between citation distributions; 0 = identical focus, 2 = disjoint."""
    sa, sb = sum(a.values()) or 1, sum(b.values()) or 1
    return sum(abs(a.get(e, 0) / sa - b.get(e, 0) / sb) for e in ids)


def load(out_dir: Path, model: str):
    runs = []
    for p in sorted(out_dir.glob(f"*{model}*.debate.json")):
        d = json.loads(p.read_text())
        stance = {e["id"]: e["stance"] for e in d["evidence"]}
        ids = sorted(stance)
        turns = {}
        for slot, text in d["transcript"].items():
            agent, rnd = slot.split("|")
            turns[(agent, int(rnd))] = turn_citations(text)
        runs.append({"claim": d["claim_id"], "panel": d["panel"], "roles": d["roles"],
                     "stance": stance, "ids": ids, "turns": turns,
                     "agents": sorted({a for a, _ in turns}),
                     "rounds": sorted({r for _, r in turns})})
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    runs = load(a.out_dir, a.model)
    if not runs:
        print(f"no *{a.model}*.debate.json under {a.out_dir}", file=sys.stderr)
        return 1
    panels = sorted({r["panel"] for r in runs})
    print(f"{len(runs)} runs, {len(set(r['claim'] for r in runs))} claims, panels={panels}\n")

    result: dict = {"model": a.model, "panels": {}}

    print("=== EVIDENCE COVERAGE: distinct dossier items cited per turn (of 4) ===")
    print(f"  {'panel':<10}{'r1':>7}{'r2':>7}{'r3':>7}")
    for panel in panels:
        rs = [r for r in runs if r["panel"] == panel]
        row = []
        for rnd in (1, 2, 3):
            vals = [len([e for e, n in r["turns"].get((ag, rnd), {}).items() if n])
                    for r in rs for ag in r["agents"]]
            row.append(sum(vals) / len(vals) if vals else 0.0)
        print(f"  {panel:<10}" + "".join(f"{x:>7.2f}" for x in row))
        result["panels"].setdefault(panel, {})["coverage"] = row

    print("\n=== LEAN per agent: +1 cites only support, -1 only undermine ===")
    for panel in panels:
        rs = [r for r in runs if r["panel"] == panel]
        agents = sorted({ag for r in rs for ag in r["agents"]})
        print(f"  {panel}")
        per = {}
        for ag in agents:
            row = []
            for rnd in (1, 2, 3):
                vals = [v for r in rs
                        if (v := lean(r["turns"].get((ag, rnd), {}), r["stance"])) is not None]
                row.append(sum(vals) / len(vals) if vals else 0.0)
            role = rs[0]["roles"].get(ag, "")[:46]
            print(f"    {ag}  r1={row[0]:+.2f}  r2={row[1]:+.2f}  r3={row[2]:+.2f}   {role}")
            per[ag] = row
        spread = [max(per[x][i] for x in agents) - min(per[x][i] for x in agents) for i in range(3)]
        print(f"    spread (max-min across agents): "
              f"r1={spread[0]:.2f}  r2={spread[1]:.2f}  r3={spread[2]:.2f}")
        result["panels"][panel]["lean"] = per
        result["panels"][panel]["lean_spread"] = spread

    print("\n=== DIVERGENCE: mean pairwise L1 between agents' citation focus (0=identical) ===")
    print(f"  {'panel':<10}{'r1':>7}{'r2':>7}{'r3':>7}{'r3-r1':>8}")
    for panel in panels:
        rs = [r for r in runs if r["panel"] == panel]
        row = []
        for rnd in (1, 2, 3):
            vals = []
            for r in rs:
                ags = r["agents"]
                for i, x in enumerate(ags):
                    for y in ags[i + 1:]:
                        vals.append(l1(r["turns"].get((x, rnd), {}),
                                       r["turns"].get((y, rnd), {}), r["ids"]))
            row.append(sum(vals) / len(vals) if vals else 0.0)
        print(f"  {panel:<10}" + "".join(f"{x:>7.3f}" for x in row) + f"{row[2]-row[0]:>+8.3f}")
        result["panels"][panel]["divergence"] = row

    print("\n=== CITATION VOLUME per turn ===")
    print(f"  {'panel':<10}{'r1':>7}{'r2':>7}{'r3':>7}")
    for panel in panels:
        rs = [r for r in runs if r["panel"] == panel]
        row = []
        for rnd in (1, 2, 3):
            vals = [sum(r["turns"].get((ag, rnd), {}).values()) for r in rs for ag in r["agents"]]
            row.append(sum(vals) / len(vals) if vals else 0.0)
        print(f"  {panel:<10}" + "".join(f"{x:>7.1f}" for x in row))
        result["panels"][panel]["volume"] = row

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
