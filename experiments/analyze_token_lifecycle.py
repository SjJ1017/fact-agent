#!/usr/bin/env python3
"""Token-aligned fact lifecycle analysis for annotated Perspectrum FactStores.

Input stores must have been produced by ``experiments/token_clock.py``. The
clock is an estimate over visible text, not provider billing: hidden reasoning
and same-round wall-clock ordering are deliberately out of scope.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from factflow.aggregate import build_view
from factflow.types import Channel, FactStore
from token_clock import VisibleTokenCounter, turn_clocks


BUDGETS = (500, 1000, 1500)
METRICS = (
    "output_tokens",
    "visible_total_tokens",
    "output_facts",
    "facts_per_1k_output_tokens",
    "final_facts",
    "final_facts_per_1k_output_tokens",
    "r2_born",
    "r3_born",
    "r2_died",
    "r3_died",
    "r1_final_survival",
    "novel_output_facts",
    "observed_uptake",
    "observed_uptake_rate",
    "mean_fact_span_tokens",
)


def parse_name(path: Path, model: str) -> tuple[str, str, str]:
    stem = path.name.removesuffix(".tokens.json")
    body = stem.removeprefix("perspectrum-")
    claim_id, rest = body.split(f"-{model}-", 1)
    topology, panel = rest.split("-", 1)
    return claim_id, topology, panel


def token_clock(mention: Any) -> dict[str, Any] | None:
    return mention.provenance.extra.get("token_clock")


def output_facts_by_round(store: FactStore, execution_id: str, rounds: list[int]) -> dict[int, set[str]]:
    output = {round_: set() for round_ in rounds}
    for fact_id, fact in store.facts.items():
        for mention_id in fact.mention_ids:
            provenance = store.mentions[mention_id].provenance
            if (provenance.execution_id == execution_id
                    and provenance.channel == Channel.OUTPUT
                    and provenance.round in output):
                output[provenance.round].add(fact_id)
    return output


def observed_uptake(view: Any, debate: dict[str, Any]) -> tuple[int, int]:
    """First expression after a fact was actually delivered from a prior peer turn."""
    uptake = 0
    novel_total = 0
    for current in view.rounds:
        if current == min(view.rounds):
            continue
        for agent in view.agents:
            prior_self = set().union(*(view.said[(agent, round_)] for round_ in view.rounds if round_ < current))
            novel = view.said[(agent, current)] - prior_self
            novel_total += len(novel)
            delivered = debate["delivery"].get(f"{agent}|{current}", {}).get("peer_turns", [])
            delivered_facts: set[str] = set()
            for slot in delivered:
                source_agent, source_round = slot.split("|")
                delivered_facts |= view.said.get((source_agent, int(source_round)), set())
            uptake += len(novel & delivered_facts)
    return uptake, novel_total


def one_run(
    path: Path,
    model: str,
    counter: VisibleTokenCounter,
    debate_dir: Path | None = None,
) -> dict[str, Any]:
    claim_id, topology, panel = parse_name(path, model)
    debate_name = path.name.removesuffix(".tokens.json") + ".debate.json"
    debate_path = (debate_dir or path.parent) / debate_name
    debate = json.loads(debate_path.read_text())
    store = FactStore.load(str(path))
    clocks = turn_clocks(debate, counter)
    execution_id = debate["execution_id"]
    view = build_view(store, execution_id, debate["roles"])
    # Round 0 is the shared source material, not part of the generated debate.
    rounds = [round_ for round_ in view.rounds if round_ > 0]
    by_round = output_facts_by_round(store, execution_id, rounds)
    all_output = set().union(*by_round.values()) if by_round else set()
    final = by_round[max(rounds)] if rounds else set()
    seen: set[str] = set()
    trajectory: list[dict[str, float]] = []
    for round_ in rounds:
        current = by_round[round_]
        previous = by_round.get(round_ - 1, set())
        trajectory.append({
            "round": round_,
            "alive": len(current),
            "born": len(current - seen),
            "died": len(previous - current),
            "carried": len(previous & current),
            "cumulative": len(seen | current),
        })
        seen |= current

    all_mentions = [
        mention for mention in store.mentions.values()
        if mention.provenance.execution_id == execution_id and token_clock(mention) is not None
    ]
    output_tokens = max(
        (clock.output_before_turn + clock.output_tokens for clock in clocks.values()),
        default=0,
    )
    total_tokens = max(
        (clock.total_before_turn + clock.prompt_tokens + clock.output_tokens for clock in clocks.values()),
        default=0,
    )
    first_token: dict[str, int] = {}
    last_token: dict[str, int] = {}
    for fact_id in all_output:
        times = [
            token_clock(store.mentions[mention_id])["cumulative_visible_output_tokens"]
            for mention_id in store.facts[fact_id].mention_ids
            if token_clock(store.mentions[mention_id]) is not None
        ]
        if times:
            first_token[fact_id], last_token[fact_id] = min(times), max(times)
    uptake, novel_total = observed_uptake(view, debate)
    r1 = by_round.get(min(rounds), set()) if rounds else set()
    span = [last_token[fact_id] - first_token[fact_id] for fact_id in first_token]
    curve = {
        str(budget): {
            "n_facts": sum(time <= budget for time in first_token.values()),
            "eligible": output_tokens >= budget,
        }
        for budget in BUDGETS
    }
    values = {
        "output_tokens": output_tokens,
        "visible_total_tokens": total_tokens,
        "output_facts": len(all_output),
        "facts_per_1k_output_tokens": 1000 * len(all_output) / max(output_tokens, 1),
        "final_facts": len(final),
        "final_facts_per_1k_output_tokens": 1000 * len(final) / max(output_tokens, 1),
        "r2_born": trajectory[1]["born"] if len(trajectory) > 1 else 0,
        "r3_born": trajectory[2]["born"] if len(trajectory) > 2 else 0,
        "r2_died": trajectory[1]["died"] if len(trajectory) > 1 else 0,
        "r3_died": trajectory[2]["died"] if len(trajectory) > 2 else 0,
        "r1_final_survival": len(r1 & final) / max(len(r1), 1),
        "novel_output_facts": novel_total,
        "observed_uptake": uptake,
        "observed_uptake_rate": uptake / max(novel_total, 1),
        "mean_fact_span_tokens": statistics.mean(span) if span else 0.0,
    }
    output_slots = {
        (mention.provenance.agent_id, mention.provenance.round)
        for mention in all_mentions
    }
    return {
        "claim_id": claim_id,
        "panel": panel,
        "topology": topology,
        "path": path.name,
        "factful_output_slots": len(output_slots),
        "expected_output_slots": len(view.agents) * len(rounds),
        "metrics": values,
        "token_curve": curve,
        "trajectory": trajectory,
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def bootstrap_interval(values: list[float], seed: int, samples: int) -> list[float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return [means[int(samples * 0.025)], means[int(samples * 0.975) - 1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--debate-dir", type=Path,
                        help="Directory containing the paired .debate.json files.")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob(f"*{args.model}*.tokens.json"))
    counter = VisibleTokenCounter()
    runs = [one_run(path, args.model, counter, args.debate_dir) for path in paths]
    if not runs:
        parser.error(f"no token-annotated {args.model} stores in {args.input_dir}")
    panels = sorted({run["panel"] for run in runs})
    by_panel = {panel: [run for run in runs if run["panel"] == panel] for panel in panels}
    means = {
        panel: {metric: mean([run["metrics"][metric] for run in panel_runs]) for metric in METRICS}
        for panel, panel_runs in by_panel.items()
    }
    token_curve: dict[str, dict[str, dict[str, float]]] = {}
    for panel, panel_runs in by_panel.items():
        token_curve[panel] = {}
        for budget in BUDGETS:
            eligible = [run["token_curve"][str(budget)]["n_facts"] for run in panel_runs
                        if run["token_curve"][str(budget)]["eligible"]]
            token_curve[panel][str(budget)] = {"n_eligible": len(eligible), "mean_facts": mean(eligible)}

    by_claim_panel = {(run["claim_id"], run["panel"]): run for run in runs}
    paired: dict[str, dict[str, Any]] = {}
    for target in (panel for panel in panels if panel != "neutral"):
        claims = sorted(claim for claim, panel in by_claim_panel
                        if panel == "neutral" and (claim, target) in by_claim_panel)
        paired[target] = {"n_claims": len(claims), "metrics": {}}
        for index, metric in enumerate(METRICS):
            differences = [
                by_claim_panel[(claim, target)]["metrics"][metric]
                - by_claim_panel[(claim, "neutral")]["metrics"][metric]
                for claim in claims
            ]
            paired[target]["metrics"][metric] = {
                "mean_difference": mean(differences),
                "positive": sum(value > 0 for value in differences),
                "negative": sum(value < 0 for value in differences),
                "bootstrap_95": bootstrap_interval(differences, 31 + index, args.bootstrap_samples),
            }

    output = {
        "model": args.model,
        "n_runs": len(runs),
        "n_claims": len({run["claim_id"] for run in runs}),
        "clock": {
            "measurement": "estimated visible text tokens with BAAI/bge-base-en-v1.5",
            "order": "round_then_agent; same-round order is accounting only",
            "excludes": "provider usage and hidden reasoning tokens",
        },
        "metrics": list(METRICS),
        "means": means,
        "token_curve": token_curve,
        "paired_vs_neutral": paired,
        "runs": runs,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2) + "\n")
    print(f"runs={len(runs)} claims={output['n_claims']} -> {args.out_json}")
    for panel in panels:
        row = means[panel]
        print(
            f"{panel:<8} facts={row['output_facts']:.1f} /1k={row['facts_per_1k_output_tokens']:.1f} "
            f"final={row['final_facts']:.1f} uptake={row['observed_uptake_rate']:.3f} "
            f"r1->r3={row['r1_final_survival']:.3f} out_tok={row['output_tokens']:.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
