"""Unambiguous transfer detection on synthetic negatives.

On a negative case the two dossiers name DIFFERENT invented enzymes, so an agent
naming the other drug's enzyme can only have received it from its peer. There is
no inference route and no prior knowledge route: the names do not exist outside
this fixture.

That closes the hole in every earlier flow measurement in this project.
Transmission was previously inferred from co-expression under full broadcast,
where B saying what A said may just be B reading the same document. Here it is
observed.

`solo-half` is the control. That agent never receives a peer message, so a
non-zero rate there would mean the enzyme names are guessable after all.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import generate

ENZYME = re.compile(r"HTX-\d[A-Z]")


def enzymes(text: str) -> set[str]:
    return set(ENZYME.findall((text or "").upper()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).parent / "out_synthetic"))
    ap.add_argument("--n-pos", type=int, default=12)
    ap.add_argument("--n-neg", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dir)
    byid = {c.case_id: c for c in generate(args.n_pos, args.n_neg, seed=args.seed)}

    print("TRANSFER, measured on negative cases (the two enzymes differ)\n")
    print(f"{'condition':<11}{'cases':>7}{'A got B enzyme':>17}{'B got A enzyme':>17}"
          f"{'first round':>13}{'accuracy':>10}")
    rows = []
    for cond in ("solo-both", "solo-half", "split", "broadcast"):
        runs = [json.loads(p.read_text()) for p in sorted(root.glob(f"*-{cond}.run.json"))]
        neg = [r for r in runs if not r["gold"]]
        if not neg:
            continue
        ab = ba = 0
        first = []
        for r in neg:
            c = byid[r["case_id"]]
            ea, eb = enzymes(c.drug_a.mechanism), enzymes(c.drug_b.mechanism)
            for agent, other in (("A", eb), ("B", ea)):
                turns = {int(k.split("|")[1]): v for k, v in r["transcript"].items()
                         if k.startswith(agent)}
                hits = [n for n in sorted(turns) if other <= enzymes(turns[n])]
                if hits:
                    first.append(hits[0])
                    if agent == "A":
                        ab += 1
                    else:
                        ba += 1
        acc = sum(x["correct"] for x in runs) / len(runs)
        n = len(neg)
        rng = f"r{min(first)}-r{max(first)}" if first else "-"
        print(f"{cond:<11}{n:>7}{ab:>10} ({ab/n:>3.0%}){ba:>10} ({ba/n:>3.0%}){rng:>13}{acc:>10.0%}")
        rows.append({"condition": cond, "n_negative": n, "a_got_b": ab, "b_got_a": ba,
                     "first_rounds": first, "accuracy": acc})

    (root / "transfer.json").write_text(json.dumps(rows, indent=2))
    print("\nAll four conditions can score the same and differ completely in whether")
    print("information moved and whether that movement was load-bearing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
