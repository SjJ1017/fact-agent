#!/usr/bin/env python3
"""Plot token-aligned expressed, dead, and survived fact trajectories.

Consumes the local token-clock copies written by ``token_clock.py``.  The plot
uses visible output tokens only, so the horizontal scale remains meaningful
despite the prompt dossier being repeated for every panelist turn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factflow.types import FactStore  # noqa: E402


COLORS = {"neutral": "#377E8C", "lenses": "#C2543D", "stance": "#716B8F"}
LABELS = {"neutral": "Neutral", "lenses": "Epistemic lenses", "stance": "Stance-diverse"}


def panel_name(path: Path) -> str:
    return path.name.removesuffix(".tokens.json").rsplit("-", 1)[1]


def run_data(path: Path, output_tokens: int) -> dict[str, Any]:
    store = FactStore.load(str(path))
    output_facts: dict[str, list[Any]] = {}
    for fact_id, fact in store.facts.items():
        mentions = []
        for mention_id in fact.mention_ids:
            mention = store.mentions[mention_id]
            if mention.provenance.extra.get("token_clock"):
                mentions.append(mention)
        if mentions:
            output_facts[fact_id] = mentions
    final_round = max(
        mention.provenance.round for mentions in output_facts.values() for mention in mentions
    )
    firsts = []
    dead_last = []
    survived_first = []
    for fact_id, mentions in output_facts.items():
        first = store.facts[fact_id].properties["first_output_token_clock"]["cumulative_visible_output_tokens"]
        firsts.append(first)
        last = max(mention.provenance.extra["token_clock"]["cumulative_visible_output_tokens"]
                   for mention in mentions)
        if any(mention.provenance.round == final_round for mention in mentions):
            survived_first.append(first)
        else:
            dead_last.append(last)
    return {
        "total_output": output_tokens,
        "expressed_first": firsts,
        "dead_last": dead_last,
        "survived_first": survived_first,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stores", type=Path)
    parser.add_argument("--analysis-json", type=Path, required=True,
                        help="JSON emitted by analyze_token_lifecycle.py.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-output-tokens", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    args = parser.parse_args()

    paths = sorted(args.stores.glob(f"*{args.model}*.tokens.json"))
    if not paths:
        parser.error(f"no token stores for {args.model} in {args.stores}")
    analysis = json.loads(args.analysis_json.read_text())
    output_tokens = {run["path"]: run["metrics"]["output_tokens"] for run in analysis["runs"]}
    grouped: dict[str, list[dict[str, Any]]] = {panel: [] for panel in LABELS}
    for path in paths:
        grouped[panel_name(path)].append(run_data(path, output_tokens[path.name]))

    budgets = np.arange(0, args.max_output_tokens + args.step, args.step)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True, sharey=False)
    panels = (
        ("expressed_first", "Expressed Facts", ""),
        ("dead_last", "Dead Facts", "Last expression before final absence"),
        ("survived_first", "Survived Facts", "First expression; present in final round"),
    )
    for axis, (field, title, subtitle) in zip(axes, panels):
        for panel in ("neutral", "lenses", "stance"):
            means, lows, highs = [], [], []
            for budget in budgets:
                values = [sum(time <= budget for time in run[field])
                          for run in grouped[panel] if run["total_output"] >= budget]
                means.append(np.mean(values))
                lows.append(np.percentile(values, 2.5))
                highs.append(np.percentile(values, 97.5))
            axis.plot(budgets, means, label=LABELS[panel], color=COLORS[panel], linewidth=2.3)
            axis.fill_between(budgets, lows, highs, color=COLORS[panel], alpha=0.13)
        axis.set_title(title, loc="left", fontweight="bold")
        if subtitle:
            axis.text(0.02, 0.91, subtitle, transform=axis.transAxes, fontsize=8, color="#555555")
        axis.set_xlabel("Cumulative visible output tokens")
        axis.set_xlim(0, args.max_output_tokens)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Unique canonical facts")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle("DeepSeek-v4-flash on Perspectrum, full topology (n=12 claims)", x=0.01,
                 ha="left", fontsize=11, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
