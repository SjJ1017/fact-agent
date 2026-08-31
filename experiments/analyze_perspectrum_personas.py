#!/usr/bin/env python3
"""Analyze DeepSeek Perspectrum persona conditions from rematched FactStores.

The unit for comparisons is a claim: neutral, lenses, and stance runs on the
same claim are paired. Cross-agent adoption is calculated with agent ids rather
than role labels, since neutral agents intentionally share the same role text.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from factflow.aggregate import build_view, signature, survival_curve, trajectory
from factflow.types import Channel, FactStore


METRICS = (
    "n_facts", "adoption", "persistence", "shared", "multi", "survived",
    "redundancy", "echo", "r1_alive", "r2_born", "r2_died", "r3_born",
    "r3_died", "r3_carried", "r1_to_r3", "r2_to_r3", "source_matched",
)


def parse_name(path: Path, model: str) -> tuple[str, str, str]:
    stem = path.name.removesuffix(".v2.json")
    body = stem.removeprefix("perspectrum-")
    claim_id, rest = body.split(f"-{model}-", 1)
    topology, panel = rest.split("-", 1)
    return claim_id, topology, panel


def agent_flow(view: Any) -> tuple[int, int]:
    """Return (peer adoption, self-persistence) across adjacent rounds."""
    adoption = 0
    persistence = 0
    for previous, current in zip(view.rounds, view.rounds[1:]):
        for target in view.agents:
            earlier = set().union(*(view.said[(target, r)] for r in view.rounds if r < current))
            novel = view.said[(target, current)] - earlier
            adoption += sum(
                any(fact in view.said[(source, previous)] for source in view.agents if source != target)
                for fact in novel
            )
            persistence += len(view.said[(target, current)] & view.said[(target, previous)])
    return adoption, persistence


def one_run(path: Path, model: str) -> dict[str, Any]:
    claim_id, topology, panel = parse_name(path, model)
    store = FactStore.load(str(path))
    execution_id = store.executions()[0]
    debate = json.loads(path.with_name(path.name.removesuffix(".v2.json") + ".debate.json").read_text())
    view = build_view(store, execution_id, debate["roles"])
    sig = signature(view)
    rounds = {row["round"]: row for row in trajectory(view)}
    curve = survival_curve(view)
    adoption, persistence = agent_flow(view)
    output_slots = {
        (mention.provenance.agent_id, mention.provenance.round)
        for mention in store.mentions.values()
        if mention.provenance.channel == Channel.OUTPUT
    }
    relations = Counter(relation.relation for relation in store.relations)
    return {
        "claim_id": claim_id,
        "panel": panel,
        "topology": topology,
        "path": path.name,
        "agents": view.agents,
        "factful_output_slots": len(output_slots),
        "expected_output_slots": len(view.agents) * len(view.rounds),
        "n_facts": sig["n_facts"],
        "shared": sig["spread:shared"],
        "multi": sig["support:multi"],
        "survived": sig["fate:survived"],
        "redundancy": sig["redundancy"],
        "echo": sig["echo_rate"],
        "source_matched": sig["grounding:gold"] + sig["grounding:context"],
        "adoption": adoption,
        "persistence": persistence,
        "r1_alive": rounds[1]["alive"],
        "r2_born": rounds[2]["born"],
        "r2_died": rounds[2]["died"],
        "r3_born": rounds[3]["born"],
        "r3_died": rounds[3]["died"],
        "r3_carried": rounds[3]["carried"],
        "r1_to_r3": curve.get(1, [0.0])[-1],
        "r2_to_r3": curve.get(2, [0.0])[-1],
        "contradictions": relations["CONTRADICTS"],
        "equivalences": relations["EQUIVALENT"],
    }


def bootstrap_interval(values: list[float], seed: int, samples: int) -> list[float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return [means[int(samples * 0.025)], means[int(samples * 0.975) - 1]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob(f"*{args.model}*.v2.json"))
    runs = [one_run(path, args.model) for path in paths]
    panels = sorted({run["panel"] for run in runs})
    by_panel = {panel: [run for run in runs if run["panel"] == panel] for panel in panels}
    means = {
        panel: {metric: statistics.mean(run[metric] for run in panel_runs) for metric in METRICS}
        for panel, panel_runs in by_panel.items()
    }

    by_claim_panel = {(run["claim_id"], run["panel"]): run for run in runs}
    paired: dict[str, dict[str, Any]] = {}
    for target in (panel for panel in panels if panel != "neutral"):
        claims = sorted({claim for claim, panel in by_claim_panel if panel == "neutral" and (claim, target) in by_claim_panel})
        paired[target] = {"n_claims": len(claims), "metrics": {}}
        for index, metric in enumerate(METRICS):
            differences = [by_claim_panel[(claim, target)][metric] - by_claim_panel[(claim, "neutral")][metric]
                           for claim in claims]
            paired[target]["metrics"][metric] = {
                "mean_difference": statistics.mean(differences),
                "median_difference": statistics.median(differences),
                "positive": sum(value > 0 for value in differences),
                "negative": sum(value < 0 for value in differences),
                "zero": sum(value == 0 for value in differences),
                "bootstrap_95": bootstrap_interval(differences, 17 + index, args.bootstrap_samples),
            }

    complete_claims = sorted({run["claim_id"] for run in runs if all(
        by_claim_panel[(run["claim_id"], panel)]["factful_output_slots"]
        == by_claim_panel[(run["claim_id"], panel)]["expected_output_slots"]
        for panel in panels
    )})
    complete_slot_sensitivity: dict[str, dict[str, Any]] = {}
    for target in (panel for panel in panels if panel != "neutral"):
        complete_slot_sensitivity[target] = {"n_claims": len(complete_claims), "metrics": {}}
        for metric in METRICS:
            differences = [by_claim_panel[(claim, target)][metric] - by_claim_panel[(claim, "neutral")][metric]
                           for claim in complete_claims]
            complete_slot_sensitivity[target]["metrics"][metric] = {
                "mean_difference": statistics.mean(differences),
                "positive": sum(value > 0 for value in differences),
                "negative": sum(value < 0 for value in differences),
                "zero": sum(value == 0 for value in differences),
            }

    output = {
        "model": args.model,
        "n_runs": len(runs),
        "n_claims": len({run["claim_id"] for run in runs}),
        "metrics": list(METRICS),
        "means": means,
        "paired_vs_neutral": paired,
        "complete_slot_sensitivity": {
            "claim_ids": complete_claims,
            "paired_vs_neutral": complete_slot_sensitivity,
        },
        "runs": runs,
        "notes": [
            "Adoption means a fact newly expressed by an agent that appeared in another agent's immediately preceding output.",
            "Source matching is diagnostic only: these v2 stores have not received a source-grounding audit.",
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2) + "\n")
    print(f"runs={len(runs)} claims={output['n_claims']} -> {args.out_json}")
    for panel in panels:
        print(panel, " ".join(f"{metric}={means[panel][metric]:.3f}" for metric in METRICS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
