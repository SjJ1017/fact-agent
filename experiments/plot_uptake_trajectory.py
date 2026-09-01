#!/usr/bin/env python3
"""Plot cumulative observed peer-uptake share against visible output tokens."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factflow.aggregate import build_view  # noqa: E402
from factflow.types import Channel, FactStore  # noqa: E402


COLORS = {"neutral": "#377E8C", "lenses": "#C2543D", "stance": "#716B8F"}
LABELS = {"neutral": "Neutral", "lenses": "Epistemic lenses", "stance": "Stance-diverse"}


def parse_name(path: Path, model: str) -> tuple[str, str]:
    body = path.name.removesuffix(".tokens.json").removeprefix("perspectrum-")
    _, rest = body.split(f"-{model}-", 1)
    topology, panel = rest.split("-", 1)
    return topology, panel


def mention_time(store: FactStore, fact_id: str, agent: str, round_: int) -> int | None:
    times = [
        store.mentions[mention_id].provenance.extra["token_clock"]["cumulative_visible_output_tokens"]
        for mention_id in store.facts[fact_id].mention_ids
        if (store.mentions[mention_id].provenance.channel == Channel.OUTPUT
            and store.mentions[mention_id].provenance.agent_id == agent
            and store.mentions[mention_id].provenance.round == round_
            and store.mentions[mention_id].provenance.extra.get("token_clock"))
    ]
    return min(times) if times else None


def events(path: Path, debate_dir: Path, model: str) -> tuple[int, str, list[tuple[int, bool]]]:
    topology, panel = parse_name(path, model)
    debate_path = debate_dir / (path.name.removesuffix(".tokens.json") + ".debate.json")
    debate = json.loads(debate_path.read_text())
    store = FactStore.load(str(path))
    view = build_view(store, debate["execution_id"], debate["roles"])
    output_rounds = sorted(round_ for round_ in view.rounds if round_ > 0)
    output_tokens = max(
        (mention.provenance.extra["token_clock"]["cumulative_visible_output_tokens"]
         for mention in store.mentions.values() if mention.provenance.extra.get("token_clock")),
        default=0,
    )
    observed: list[tuple[int, bool]] = []
    for round_ in output_rounds:
        for agent in view.agents:
            prior_self = set().union(*(view.said[(agent, earlier)] for earlier in output_rounds if earlier < round_))
            novel = view.said[(agent, round_)] - prior_self
            delivered_facts: set[str] = set()
            for peer_turn in debate["delivery"].get(f"{agent}|{round_}", {}).get("peer_turns", []):
                peer, peer_round = peer_turn.split("|")
                delivered_facts |= view.said.get((peer, int(peer_round)), set())
            for fact_id in novel:
                time = mention_time(store, fact_id, agent, round_)
                if time is not None:
                    observed.append((time, fact_id in delivered_facts))
    return output_tokens, panel, observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stores", type=Path)
    parser.add_argument("--debate-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    args = parser.parse_args()

    runs: dict[str, list[tuple[int, list[tuple[int, bool]]]]] = defaultdict(list)
    for path in sorted(args.stores.glob(f"*{args.model}*.tokens.json")):
        total, panel, run_events = events(path, args.debate_dir, args.model)
        runs[panel].append((total, run_events))
    budgets = list(range(100, args.max_output_tokens + 1, args.step))
    fig, axis = plt.subplots(figsize=(9.2, 4.4), constrained_layout=True)
    means: list[float] = []
    for panel in ("neutral", "lenses", "stance"):
        values = []
        for budget in budgets:
            per_run = []
            for total, run_events in runs[panel]:
                if total < budget:
                    continue
                eligible = [uptake for time, uptake in run_events if time <= budget]
                if eligible:
                    per_run.append(sum(eligible) / len(eligible))
            values.append(float(np.mean(per_run)) if per_run else np.nan)
        means.extend(value for value in values if not np.isnan(value))
        axis.plot(budgets, values, label=LABELS[panel], color=COLORS[panel], linewidth=2.4)
    lower, upper = min(means), max(means)
    pad = max(0.025, (upper - lower) * 0.25)
    axis.set_ylim(max(0, lower - pad), min(1, upper + pad))
    axis.set_xlim(min(budgets), max(budgets))
    axis.set_xlabel("Cumulative visible output tokens")
    axis.set_ylabel("Share of novel output facts")
    axis.set_title(f"Observed peer uptake: {args.model}, full topology", loc="left",
                   fontweight="bold", fontsize=11)
    axis.text(0.02, 0.91,
              "Numerator: novel fact previously delivered from a peer; denominator: all novel output facts",
              transform=axis.transAxes, fontsize=7.5, color="#555555")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1),
                borderaxespad=0, fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
