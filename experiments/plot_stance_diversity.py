#!/usr/bin/env python3
"""Plot post-hoc fact stance diversity and balance against output-token budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"neutral": "#377E8C", "lenses": "#C2543D", "stance": "#716B8F"}
LABELS = {"neutral": "Neutral", "lenses": "Epistemic lenses", "stance": "Stance-diverse"}


def values_at_budget(runs: list[dict], budget: str, metric: str) -> list[float]:
    return [run["token_curve"][budget][metric] for run in runs
            if run["token_curve"][budget]["eligible"] and run["token_curve"][budget][metric] is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.analysis_json.read_text())
    panels = tuple(LABELS)
    budgets = sorted(int(budget) for budget in data["summary"]["neutral"]["token_curve"])
    grouped = {panel: [run for run in data["runs"] if run["panel"] == panel] for panel in panels}
    figures = (
        ("entropy_normalized", "Stance diversity", "Normalized Shannon entropy across support, undermine, neutral"),
        ("polarized_balance", "Polarized balance", "1 = equal support and undermine; neutral facts excluded"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.1), constrained_layout=True, sharex=True)
    for axis, (metric, title, subtitle) in zip(axes, figures):
        all_means = []
        for panel in panels:
            means = []
            for budget in budgets:
                values = values_at_budget(grouped[panel], str(budget), metric)
                means.append(np.mean(values) if values else np.nan)
            all_means.extend(value for value in means if not np.isnan(value))
            axis.plot(budgets, means, label=LABELS[panel], color=COLORS[panel], linewidth=2.3)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.text(0.02, 0.91, subtitle + "; cumulative unique-fact portfolio",
                  transform=axis.transAxes, fontsize=8, color="#555555")
        axis.set_xlabel("Cumulative visible output tokens")
        axis.set_xlim(100, max(budgets))
        lower, upper = min(all_means), max(all_means)
        pad = max(0.06, (upper - lower) * 0.22)
        axis.set_ylim(max(0, lower - pad), min(1.0, upper + pad))
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Score (per-claim mean)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Fact stance trajectories: DeepSeek-v4-flash, Perspectrum full topology (n=12 claims)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
