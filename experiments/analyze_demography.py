"""Fact demography across debate configurations: births, deaths, survival.

Reads the paired `<qid>-<config>.debate.json` / `.store.json` files an experiment
run leaves behind, and reports the per-round fact population plus the cohort
survival curves. Everything is averaged over questions, which works because
rounds align across runs even when the transcripts share no wording.

    python experiments/analyze_demography.py experiments/mmlu-pro_out
    python experiments/analyze_demography.py experiments/mmlu-pro_out --json findings/data/x.json

Cost is estimated from output characters unless the runner recorded real token
counts under a "usage" key; the header of the printed table says which was used,
because the absolute numbers are only meaningful in the recorded case.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factflow.aggregate import (  # noqa: E402
    build_view, mean_signature, mean_trajectory, signature, survival_curve, trajectory,
)
from factflow import graph as G  # noqa: E402
from factflow.types import FactStore  # noqa: E402

CHARS_PER_TOKEN = 4.0
GLOB = ["*.store.json"]  # crude, and flagged as such wherever it is used


def load(out_dir: Path):
    """-> {config: [(qid, view, correct, cost_map)]}, and whether cost is real."""
    runs = defaultdict(list)
    real_usage = True
    for store_p in sorted(out_dir.glob(GLOB[0])):
        stem = store_p.name[: -len(GLOB[0]) + 1] if GLOB[0].startswith("*") else store_p.stem
        stem = store_p.name.split(".")[0]
        m = re.match(r"(.+?-\d+)-(.+)", stem)
        if not m:
            continue
        qid, config = m.groups()
        debate_p = store_p.with_name(stem + ".debate.json")
        if not debate_p.exists():
            continue
        dj = json.loads(debate_p.read_text())
        store = FactStore.load(str(store_p))
        execs = store.executions()
        if not execs:
            continue
        view = build_view(store, execs[0], dj.get("roles"))

        usage = dj.get("usage") or {}
        cost = {}
        for slot, text in dj["transcript"].items():
            agent, rnd = slot.split("|")
            key = (agent, int(rnd))
            if slot in usage:
                cost[key] = float(usage[slot].get("output_tokens", 0))
            else:
                real_usage = False
                cost[key] = len(text) / CHARS_PER_TOKEN
        runs[config].append((qid, view, bool(dj.get("correct")), cost))
    return runs, real_usage


def report(runs, real_usage: bool) -> dict:
    out = {"cost_basis": "recorded output tokens" if real_usage
           else f"estimated: output characters / {CHARS_PER_TOKEN:g}",
           "configs": {}}

    print(f"cost basis: {out['cost_basis']}\n")
    print("=== PER-ROUND FACT POPULATION (mean over questions) ===")
    for config, rs in sorted(runs.items()):
        views = [v for _, v, _, _ in rs]
        costs = {v.execution_id: c for _, v, _, c in rs}
        traj = mean_trajectory(views, costs)
        print(f"\n{config}   n={len(rs)} questions")
        print(f"  {'r':>2}{'alive':>7}{'born':>6}{'died':>6}{'carry':>7}{'cum':>6}"
              f"{'said':>6}{'contested':>10}{'cost':>8}")
        for row in traj:
            print(f"  {row['round']:>2}{row['alive']:>7.1f}{row['born']:>6.1f}"
                  f"{row['died']:>6.1f}{row['carried']:>7.1f}{row['cumulative']:>6.1f}"
                  f"{row['said']:>6.1f}{row['born_contested']:>10.1f}{row['cost']:>8.0f}")
        out["configs"][config] = {"n_questions": len(rs), "trajectory": traj}

    print("\n=== COHORT SURVIVAL: facts born at round b, fraction alive at r ===")
    for config, rs in sorted(runs.items()):
        acc = defaultdict(lambda: defaultdict(list))
        for _, v, _, _ in rs:
            for b, curve in survival_curve(v).items():
                for i, frac in enumerate(curve):
                    acc[b][b + i].append(frac)
        curves = {b: {r: sum(x) / len(x) for r, x in sorted(v.items())}
                  for b, v in sorted(acc.items())}
        print(f"  {config}")
        for b, pts in curves.items():
            print("    born r%d: %s" % (b, "  ".join(f"r{r}:{x:.2f}" for r, x in pts.items())))
        out["configs"][config]["survival"] = {str(b): {str(r): x for r, x in p.items()}
                                              for b, p in curves.items()}

    print("\n=== RUN SIGNATURE (mean) ===")
    keys = ["n_facts", "grounding:gold", "grounding:injected", "support:multi",
            "spread:shared", "fate:survived", "contested", "redundancy", "echo_rate"]
    print(f"  {'config':<20}" + "".join(f"{k.split(':')[-1][:10]:>12}" for k in keys))
    for config, rs in sorted(runs.items()):
        m = mean_signature([v for _, v, _, _ in rs])
        print(f"  {config:<20}" + "".join(f"{m[k]:>12.3f}" for k in keys))
        out["configs"][config]["signature"] = m

    print("\n=== CORRECT vs WRONG (watch the n; this is usually too small) ===")
    for config, rs in sorted(runs.items()):
        def churn(v):
            t = trajectory(v)
            tot = sum(r["alive"] for r in t)
            return sum(r["died"] for r in t) / tot if tot else 0.0
        ok = [churn(v) for _, v, c, _ in rs if c]
        no = [churn(v) for _, v, c, _ in rs if not c]
        fmt = lambda a: f"{sum(a)/len(a):.3f} (n={len(a)})" if a else "n=0"
        print(f"  {config:<20} churn correct {fmt(ok):<16} wrong {fmt(no)}")
        out["configs"][config]["churn"] = {
            "correct": {"mean": sum(ok) / len(ok) if ok else None, "n": len(ok)},
            "wrong": {"mean": sum(no) / len(no) if no else None, "n": len(no)},
        }

    print("\n=== FRAGILITY / LOAD-BEARING  (meaningless where gold facts are ~0) ===")
    for config, rs in sorted(runs.items()):
        gold = [sum(1 for c in v.classes.values() if c.grounding == "gold") for _, v, _, _ in rs]
        frag = [G.fragility(v, "gold") for _, v, _, _ in rs]
        print(f"  {config:<20} gold facts/run {sum(gold)/len(gold):>5.1f}"
              f"   fragility {sum(frag)/len(frag):.3f}")
        out["configs"][config]["gold_facts_per_run"] = sum(gold) / len(gold)
        out["configs"][config]["fragility_gold"] = sum(frag) / len(frag)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="directory of *.debate.json / *.store.json")
    ap.add_argument("--json", type=Path, help="also write the numbers here")
    ap.add_argument("--glob", default="*.store.json")
    a = ap.parse_args()

    GLOB[0] = a.glob
    runs, real = load(a.out_dir)
    if not runs:
        print(f"no paired .store.json / .debate.json under {a.out_dir}", file=sys.stderr)
        return 1
    result = report(runs, real)
    result["source_dir"] = str(a.out_dir)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
