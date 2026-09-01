#!/usr/bin/env python3
"""Compute token-aligned stance diversity and polarized balance from labeled facts."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from factflow.types import Channel, FactStore


STANCES = ("SUPPORT", "UNDERMINE", "NEUTRAL")
BUDGETS = tuple(range(100, 1001, 100))


def parse_name(path: Path, model: str) -> tuple[str, str, str]:
    stem = path.name.removesuffix(".stance.json")
    body = stem.removeprefix("perspectrum-")
    claim_id, rest = body.split(f"-{model}-", 1)
    topology, panel = rest.split("-", 1)
    return claim_id, topology, panel


def profile(labels: list[str]) -> dict[str, float]:
    counts = Counter(labels)
    total = len(labels)
    proportions = {stance: counts[stance] / total if total else 0.0 for stance in STANCES}
    entropy = -sum(p * math.log(p) for p in proportions.values() if p)
    polarized = counts["SUPPORT"] + counts["UNDERMINE"]
    balance = (1 - abs(counts["SUPPORT"] - counts["UNDERMINE"]) / polarized) if polarized else None
    return {
        "n_facts": total,
        "support": counts["SUPPORT"],
        "undermine": counts["UNDERMINE"],
        "neutral": counts["NEUTRAL"],
        "entropy_normalized": entropy / math.log(len(STANCES)),
        "effective_stances": math.exp(entropy),
        "polarized_balance": balance,
        "dual_sided": float(counts["SUPPORT"] > 0 and counts["UNDERMINE"] > 0),
    }


def one_run(path: Path, model: str, output_tokens: float) -> dict[str, Any]:
    claim_id, topology, panel = parse_name(path, model)
    store = FactStore.load(str(path))
    first: dict[str, int] = {}
    by_round: dict[int, set[str]] = defaultdict(set)
    for fact_id, fact in store.facts.items():
        if fact.properties.get("stance") not in STANCES:
            continue
        clock = fact.properties.get("first_output_token_clock")
        if clock:
            first[fact_id] = clock["cumulative_visible_output_tokens"]
        for mention_id in fact.mention_ids:
            mention = store.mentions[mention_id]
            if mention.provenance.channel == Channel.OUTPUT:
                by_round[mention.provenance.round].add(fact_id)
    token_curve = {}
    for budget in BUDGETS:
        active = [fact_id for fact_id, time in first.items() if time <= budget]
        token_curve[str(budget)] = {"eligible": output_tokens >= budget,
                                    **profile([store.facts[fact_id].properties["stance"] for fact_id in active])}
    last_round = max(by_round) if by_round else 0
    final = profile([store.facts[fact_id].properties["stance"] for fact_id in by_round[last_round]])
    by_round_profile = {
        str(round_): profile([store.facts[fact_id].properties["stance"] for fact_id in fact_ids])
        for round_, fact_ids in sorted(by_round.items())
    }
    return {"claim_id": claim_id, "panel": panel, "topology": topology, "path": path.name,
            "token_curve": token_curve, "final": final, "rounds": by_round_profile}


def mean(values: list[float | None]) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def bootstrap(values: list[float], seed: int, samples: int) -> list[float]:
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return [draws[int(samples * .025)], draws[int(samples * .975) - 1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--token-analysis", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()

    token_data = json.loads(args.token_analysis.read_text())
    output_tokens = {run["path"].removesuffix(".tokens.json").removesuffix(".v2"): run["metrics"]["output_tokens"]
                     for run in token_data["runs"]}
    paths = sorted(args.input_dir.glob(f"*{args.model}*.stance.json"))
    runs = [one_run(path, args.model, output_tokens[path.name.removesuffix(".stance.json")]) for path in paths]
    if not runs:
        parser.error(f"no labeled {args.model} stores in {args.input_dir}")
    panels = sorted({run["panel"] for run in runs})
    by_panel = {panel: [run for run in runs if run["panel"] == panel] for panel in panels}
    summary: dict[str, Any] = {}
    for panel, panel_runs in by_panel.items():
        summary[panel] = {"token_curve": {}, "final": {}}
        for budget in BUDGETS:
            eligible = [run["token_curve"][str(budget)] for run in panel_runs
                        if run["token_curve"][str(budget)]["eligible"]]
            summary[panel]["token_curve"][str(budget)] = {
                "n_eligible": len(eligible),
                **{metric: mean([row[metric] for row in eligible])
                   for metric in ("entropy_normalized", "effective_stances", "polarized_balance", "dual_sided")},
            }
        summary[panel]["final"] = {
            metric: mean([run["final"][metric] for run in panel_runs])
            for metric in ("entropy_normalized", "effective_stances", "polarized_balance", "dual_sided")
        }

    by_claim = {(run["claim_id"], run["panel"]): run for run in runs}
    paired: dict[str, Any] = {}
    for target in (panel for panel in panels if panel != "neutral"):
        claims = sorted(claim for claim, panel in by_claim if panel == "neutral" and (claim, target) in by_claim)
        paired[target] = {"n_claims": len(claims), "at_1000": {}, "final": {}}
        for field, location in (("at_1000", "token_curve"), ("final", "final")):
            for index, metric in enumerate(("entropy_normalized", "polarized_balance", "dual_sided")):
                differences = [
                    by_claim[(claim, target)][location]["1000"][metric] - by_claim[(claim, "neutral")][location]["1000"][metric]
                    if location == "token_curve" else
                    by_claim[(claim, target)][location][metric] - by_claim[(claim, "neutral")][location][metric]
                    for claim in claims
                    if (location != "token_curve" or by_claim[(claim, target)][location]["1000"]["eligible"]
                        and by_claim[(claim, "neutral")][location]["1000"]["eligible"])
                ]
                paired[target][field][metric] = {"mean_difference": mean(differences),
                                                  "bootstrap_95": bootstrap(differences, 47 + index, args.bootstrap_samples)}

    output = {"n_runs": len(runs), "n_claims": len({run["claim_id"] for run in runs}),
              "label_definition": "argumentative direction if the canonical fact were true; not factuality",
              "summary": summary, "paired_vs_neutral": paired, "runs": runs}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2) + "\n")
    print(f"runs={len(runs)} claims={output['n_claims']} -> {args.out_json}")
    for panel in panels:
        curve = summary[panel]["token_curve"]["1000"]
        final = summary[panel]["final"]
        print(f"{panel:<8} H@1000={curve['entropy_normalized']:.3f} B@1000={curve['polarized_balance']:.3f} "
              f"H_final={final['entropy_normalized']:.3f} B_final={final['polarized_balance']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
