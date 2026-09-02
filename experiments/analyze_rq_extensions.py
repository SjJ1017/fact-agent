"""Additional RQ-oriented analyses for the Perspectrum fact-flow pilot.

This script is fully offline. It reads the existing matched stores and debate
delivery records; it does not call a model or modify any debate condition.

Outputs findings/data/rq-extensions.json.
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_effective_structure import (
    AGENTS,
    PERSONAS,
    TOPOLOGIES,
    compute,
    load_labels,
    load_runs,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FLOW = ROOT / "findings" / "data" / "flow-profile.json"
OUT = ROOT / "findings" / "data" / "rq-extensions.json"
SEED = 20260902


def bootstrap(values: list[float], n: int = 20000) -> dict:
    if not values:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(SEED)
    a = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    means = a[idx].mean(axis=1)
    lo, hi = np.percentile(means, (2.5, 97.5))
    return {
        "mean": round(float(a.mean()), 6),
        "lo": round(float(lo), 6),
        "hi": round(float(hi), 6),
        "n": len(values),
        "positive": int((a > 0).sum()),
    }


def bootstrap_group_difference(left: list[float], right: list[float], n: int = 20000) -> dict:
    if not left or not right:
        return {"difference": None, "lo": None, "hi": None}
    rng = np.random.default_rng(SEED)
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    ai = rng.integers(0, len(a), size=(n, len(a)))
    bi = rng.integers(0, len(b), size=(n, len(b)))
    diffs = a[ai].mean(axis=1) - b[bi].mean(axis=1)
    lo, hi = np.percentile(diffs, (2.5, 97.5))
    return {
        "difference": round(float(a.mean() - b.mean()), 6),
        "lo": round(float(lo), 6),
        "hi": round(float(hi), 6),
    }


def rate(num: int, den: int) -> float | None:
    return num / den if den else None


def majority(verdicts: dict[str, str], round_no: int) -> str | None:
    vals = [verdicts.get(f"{a}|{round_no}") for a in AGENTS]
    vals = [v for v in vals if v]
    if len(vals) < 2:
        return None
    value, count = Counter(vals).most_common(1)[0]
    return value if count >= 2 else None


def funnel(run: dict) -> dict:
    """Round-2 exposure -> immediate uptake -> same-agent round-3 retention."""
    exposed = adopted = retained = 0
    for agent in AGENTS:
        prior = run["out"].get((agent, 1), set())
        eligible = run["delivered"].get((agent, 2), set()) - prior
        used = eligible & run["out"].get((agent, 2), set())
        kept = used & run["out"].get((agent, 3), set())
        exposed += len(eligible)
        adopted += len(used)
        retained += len(kept)
    return {
        "exposed": exposed,
        "adopted": adopted,
        "retained": retained,
        "uptake": rate(adopted, exposed),
        "post_adoption_retention": rate(retained, adopted),
        "end_to_end": rate(retained, exposed),
    }


def chain_path(run: dict) -> dict:
    """Strict observable paths for round-1 facts unique to one chain position."""
    r1 = {a: run["out"].get((a, 1), set()) for a in AGENTS}
    src_a = r1["A"] - r1["B"] - r1["C"]
    src_b = r1["B"] - r1["A"] - r1["C"]
    ab = src_a & run["out"].get(("B", 2), set())
    # C must first state the fact at r3, after B's r2 turn was delivered.
    abc = ab & run["out"].get(("C", 3), set()) - run["out"].get(("C", 2), set())
    bc = src_b & run["out"].get(("C", 2), set())
    return {
        "a_unique_r1": len(src_a),
        "a_to_b": len(ab),
        "a_to_b_to_c": len(abc),
        "b_unique_r1": len(src_b),
        "b_to_c": len(bc),
        "a_to_b_rate": rate(len(ab), len(src_a)),
        "second_hop_given_first": rate(len(abc), len(ab)),
        "two_hop_rate": rate(len(abc), len(src_a)),
        "b_to_c_rate": rate(len(bc), len(src_b)),
    }


def main() -> int:
    raw = load_runs()
    if len(raw) != 108:
        raise SystemExit(f"expected 108 runs, found {len(raw)}")
    stance, tok = load_labels()
    runs = {ex: compute(ex, value, stance, tok) for ex, value in raw.items()}

    # 1) Difference-in-differences: does topology moderate a persona effect?
    flow = json.loads(FLOW.read_text())["per_debate"]
    flow_by = {}
    for key, row in flow.items():
        topology, claim, persona = key.split("|")
        flow_by[(topology, claim, persona)] = row
    did = {}
    did_metrics = (
        "n_facts", "facts_per_k", "adopted_share_recv", "adopted_per_k",
        "nestedness", "reach", "balance", "meta_share",
    )
    claims = sorted({key[1] for key in flow_by})
    for persona in ("lenses", "stance"):
        for topology in ("star", "chain"):
            for metric in did_metrics:
                values = []
                for claim in claims:
                    needed = [
                        (topology, claim, persona), (topology, claim, "neutral"),
                        ("full", claim, persona), ("full", claim, "neutral"),
                    ]
                    if not all(key in flow_by for key in needed):
                        continue
                    topology_effect = (flow_by[needed[0]][metric]
                                       - flow_by[needed[1]][metric])
                    full_effect = flow_by[needed[2]][metric] - flow_by[needed[3]][metric]
                    values.append(topology_effect - full_effect)
                did[f"{topology}-full|{persona}-neutral|{metric}"] = bootstrap(values)

    # 2) A transmission funnel that does not mix access with response behavior.
    run_funnels = {ex: funnel(run) for ex, run in runs.items()}
    funnel_cells = {}
    for topology in TOPOLOGIES:
        for persona in PERSONAS:
            ids = [ex for ex, run in runs.items()
                   if run["topo"] == topology and run["persona"] == persona]
            rows = [run_funnels[ex] for ex in ids]
            sums = {key: sum(row[key] for row in rows)
                    for key in ("exposed", "adopted", "retained")}
            funnel_cells[f"{topology}/{persona}"] = {
                "n": len(rows),
                **sums,
                "uptake": rate(sums["adopted"], sums["exposed"]),
                "post_adoption_retention": rate(sums["retained"], sums["adopted"]),
                "end_to_end": rate(sums["retained"], sums["exposed"]),
                "per_run": {
                    metric: bootstrap([row[metric] for row in rows if row[metric] is not None])
                    for metric in ("uptake", "post_adoption_retention", "end_to_end")
                },
            }

    funnel_contrasts = {}
    indexed = {(run["topo"], run["claim"], run["persona"]): ex
               for ex, run in runs.items()}
    for topology in TOPOLOGIES:
        for persona in ("lenses", "stance"):
            for metric in ("uptake", "post_adoption_retention", "end_to_end"):
                values = []
                for claim in claims:
                    left = run_funnels[indexed[(topology, claim, persona)]][metric]
                    right = run_funnels[indexed[(topology, claim, "neutral")]][metric]
                    if left is not None and right is not None:
                        values.append(left - right)
                funnel_contrasts[f"{topology}|{persona}-neutral|{metric}"] = bootstrap(values)

    # 3) Content that can be observed traversing one and two chain hops.
    chain = {}
    for persona in PERSONAS:
        rows = [chain_path(run) for run in runs.values()
                if run["topo"] == "chain" and run["persona"] == persona]
        sums = {key: sum(row[key] for row in rows) for key in (
            "a_unique_r1", "a_to_b", "a_to_b_to_c", "b_unique_r1", "b_to_c")}
        chain[persona] = {
            "n": len(rows),
            **sums,
            "a_to_b_rate": rate(sums["a_to_b"], sums["a_unique_r1"]),
            "second_hop_given_first": rate(sums["a_to_b_to_c"], sums["a_to_b"]),
            "two_hop_rate": rate(sums["a_to_b_to_c"], sums["a_unique_r1"]),
            "b_to_c_rate": rate(sums["b_to_c"], sums["b_unique_r1"]),
            "per_run": {
                metric: bootstrap([row[metric] for row in rows if row[metric] is not None])
                for metric in ("a_to_b_rate", "second_hop_given_first", "two_hop_rate", "b_to_c_rate")
            },
        }

    # 4) Descriptive coupling between information dynamics and verdict outcomes.
    outcome_rows = []
    for ex, run in runs.items():
        m1, m3 = majority(run["verdicts"], 1), majority(run["verdicts"], 3)
        if m1 is None or m3 is None:
            continue
        r3 = [run["verdicts"].get(f"{a}|3") for a in AGENTS]
        outcome_rows.append({
            "execution": ex,
            "topology": run["topo"],
            "persona": run["persona"],
            "stable": m1 == m3,
            "unanimous": len(set(r3)) == 1,
            "late_final_share": run["final_first_round"][3],
            "r1_survival": run["r1_surv"],
            "adopted_share": run["adopted_share"],
            "end_to_end": run_funnels[ex]["end_to_end"],
        })

    verdict_coupling = {"n": len(outcome_rows), "by_cell": {}}
    for topology in TOPOLOGIES:
        for persona in PERSONAS:
            cell = [row for row in outcome_rows
                    if row["topology"] == topology and row["persona"] == persona]
            stable = [row for row in cell if row["stable"]]
            verdict_coupling["by_cell"][f"{topology}/{persona}"] = {
                "eligible": len(cell),
                "stable": len(stable),
                "stable_and_late": sum(row["late_final_share"] >= 0.5 for row in stable),
            }

    metrics = ("late_final_share", "r1_survival", "adopted_share", "end_to_end")
    cell_means = {}
    for topology in TOPOLOGIES:
        for persona in PERSONAS:
            cell = [row for row in outcome_rows
                    if row["topology"] == topology and row["persona"] == persona]
            for metric in metrics:
                vals = [row[metric] for row in cell if row[metric] is not None]
                cell_means[(topology, persona, metric)] = st.mean(vals) if vals else 0.0
    for split in ("stable", "unanimous"):
        yes = [row for row in outcome_rows if row[split]]
        no = [row for row in outcome_rows if not row[split]]
        verdict_coupling[split] = {
            "yes_n": len(yes), "no_n": len(no),
            "locked_late_n": sum(row["late_final_share"] >= 0.5 for row in yes)
                            if split == "stable" else None,
        }
        for metric in metrics:
            y = [row[metric] for row in yes if row[metric] is not None]
            n = [row[metric] for row in no if row[metric] is not None]
            y_adjusted = [row[metric] - cell_means[(row["topology"], row["persona"], metric)]
                          for row in yes if row[metric] is not None]
            n_adjusted = [row[metric] - cell_means[(row["topology"], row["persona"], metric)]
                          for row in no if row[metric] is not None]
            verdict_coupling[split][metric] = {
                "yes_mean": round(st.mean(y), 6) if y else None,
                "no_mean": round(st.mean(n), 6) if n else None,
                **bootstrap_group_difference(y, n),
                "cell_adjusted": bootstrap_group_difference(y_adjusted, n_adjusted),
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": "2026-09-02",
        "model": "deepseek-v4-flash",
        "n_runs": len(runs),
        "difference_in_differences": did,
        "round2_transmission_funnel": {"cells": funnel_cells, "contrasts": funnel_contrasts},
        "chain_paths": chain,
        "verdict_coupling": verdict_coupling,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
