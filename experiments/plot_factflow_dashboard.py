#!/usr/bin/env python3
"""Plot stance and lifecycle trajectories from independent post-hoc artifacts."""

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


def fate_times(path: Path) -> tuple[int, dict[str, list[int]]]:
    store = FactStore.load(str(path))
    by_fact: dict[str, list[Any]] = {}
    for fact_id, fact in store.facts.items():
        mentions = [store.mentions[mention_id] for mention_id in fact.mention_ids
                    if store.mentions[mention_id].provenance.extra.get("token_clock")]
        if mentions:
            by_fact[fact_id] = mentions
    final_round = max(mention.provenance.round for mentions in by_fact.values() for mention in mentions)
    first, dead_last, survived_first = [], [], []
    for mentions in by_fact.values():
        times = [mention.provenance.extra["token_clock"]["cumulative_visible_output_tokens"]
                 for mention in mentions]
        fact_first = min(times)
        first.append(fact_first)
        if any(mention.provenance.round == final_round for mention in mentions):
            survived_first.append(fact_first)
        else:
            dead_last.append(max(times))
    total_output = max(
        mention.provenance.extra["token_clock"]["cumulative_visible_output_tokens"]
        for mentions in by_fact.values() for mention in mentions
    )
    return total_output, {"expressed": first, "dead": dead_last, "survived": survived_first}


def set_mean_ylim(axis: Any, values: list[float], *, floor_zero: bool = False) -> None:
    lower, upper = min(values), max(values)
    pad = max(0.06 if not floor_zero else 3.0, (upper - lower) * 0.22)
    axis.set_ylim(0 if floor_zero else max(0, lower - pad), upper + pad)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stores", type=Path)
    parser.add_argument("stance_analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-output-tokens", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    args = parser.parse_args()

    stance = json.loads(args.stance_analysis.read_text())
    paths = sorted(args.stores.glob(f"*{args.model}*.tokens.json"))
    grouped: dict[str, list[tuple[int, dict[str, list[int]]]]] = {panel: [] for panel in LABELS}
    for path in paths:
        grouped[panel_name(path)].append(fate_times(path))
    budgets = list(range(100, args.max_output_tokens + 1, args.step))
    panels = (
        ("entropy_normalized", "Stance entropy", "Normalized Shannon entropy", False),
        ("polarized_balance", "Polarized balance", "Support vs undermine; neutral excluded", False),
        ("expressed", "Expressed facts", "Cumulative unique canonical facts", True),
        ("dead", "Dead facts", "Last expression before final absence", True),
        ("survived", "Survived facts", "First expression; present in final round", True),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.1, 7.2), constrained_layout=True)
    used_axes = list(axes.flat)
    for axis, (metric, title, subtitle, count_metric) in zip(used_axes, panels):
        plotted: list[float] = []
        for panel in LABELS:
            if metric in ("entropy_normalized", "polarized_balance"):
                values = [stance["summary"][panel]["token_curve"][str(budget)][metric]
                          for budget in budgets]
            else:
                values = []
                for budget in budgets:
                    eligible = [sum(time <= budget for time in fates[metric])
                                for total, fates in grouped[panel] if total >= budget]
                    values.append(float(np.mean(eligible)) if eligible else np.nan)
            plotted.extend(value for value in values if not np.isnan(value))
            axis.plot(budgets, values, label=LABELS[panel], color=COLORS[panel], linewidth=2.35)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.text(0.02, 0.91, subtitle, transform=axis.transAxes, fontsize=8, color="#555555")
        axis.set_xlabel("Cumulative visible output tokens")
        axis.set_xlim(min(budgets), max(budgets))
        set_mean_ylim(axis, plotted, floor_zero=count_metric)
        axis.grid(axis="y", alpha=0.22)
    used_axes[0].set_ylabel("Per-claim mean score")
    used_axes[2].set_ylabel("Per-claim mean unique facts")
    used_axes[3].set_ylabel("Per-claim mean unique facts")
    used_axes[4].set_ylabel("Per-claim mean unique facts")
    used_axes[5].axis("off")
    used_axes[5].legend(*used_axes[0].get_legend_handles_labels(), loc="center", frameon=False, fontsize=10)
    fig.suptitle(f"Post-hoc fact-flow trajectories: {args.model}, Perspectrum full topology (n={stance['n_claims']} claims)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
