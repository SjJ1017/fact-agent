#!/usr/bin/env python3
"""Plot token-aligned stance diversity and polarized balance by persona.

Companion to ``plot_token_lifecycle.py``, which plots how many facts survive.
This plots what kind: counting facts showed no persona effect, while their
stance composition does.

The x-axis is cumulative visible *output* tokens, so conditions are compared at
matched discussion budget rather than at matched round number - a condition that
simply writes more would otherwise look richer for free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"neutral": "#377E8C", "lenses": "#C2543D", "stance": "#716B8F"}
LABELS = {"neutral": "Neutral", "lenses": "Epistemic lenses", "stance": "Stance-diverse"}
ORDER = ("neutral", "lenses", "stance")


def curve(summary: dict, panel: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    tc = summary[panel]["token_curve"]
    budgets = sorted(int(b) for b in tc)
    xs, ys = [], []
    for b in budgets:
        row = tc[str(b)]
        if row.get("n_eligible", 0) >= 8:      # keep the panel comparable across budgets
            xs.append(b)
            ys.append(row[key])
    return np.array(xs), np.array(ys)


def paired_panel(ax, paired: dict, when: str, metric: str, title: str) -> None:
    """Paired difference against neutral, with its bootstrap interval."""
    names = [p for p in ORDER if p in paired]
    for i, p in enumerate(names):
        d = paired[p][when][metric]
        lo, hi = d["bootstrap_95"]
        m = d["mean_difference"]
        excludes_zero = lo > 0 or hi < 0
        ax.plot([lo, hi], [i, i], color=COLORS[p], lw=2.4,
                solid_capstyle="round", alpha=0.9 if excludes_zero else 0.45)
        ax.plot([m], [i], "o", color=COLORS[p], ms=8,
                mec="white", mew=1.2, zorder=3)
        if excludes_zero:
            ax.text(m, i + 0.30, "CI excludes 0", ha="center", va="bottom",
                    fontsize=7.5, color=COLORS[p], fontweight="bold")
    ax.axvline(0, color="#555", lw=1, zorder=0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS[p] for p in names], fontsize=8.5)
    ax.set_ylim(-0.55, len(names) - 0.15)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis", type=Path, help="output of analyze_stance_diversity.py")
    ap.add_argument("--out", type=Path, default=Path("/tmp/factflow-stance-diversity.png"))
    a = ap.parse_args()

    data = json.loads(a.analysis.read_text())
    summary, paired = data["summary"], data["paired_vs_neutral"]

    fig = plt.figure(figsize=(11.2, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.52, wspace=0.42,
                          left=0.115, right=0.985, top=0.9, bottom=0.1)

    for col, (key, title, ylab) in enumerate((
            ("polarized_balance", "Polarized balance", "1 − |n₊ − n₋| / (n₊ + n₋)"),
            ("entropy_normalized", "Stance entropy", "normalized H"),
            ("dual_sided", "Both sides present", "fraction of claims"))):
        ax = fig.add_subplot(gs[0, col])
        for p in ORDER:
            xs, ys = curve(summary, p, key)
            ax.plot(xs, ys, color=COLORS[p], lw=2.1, label=LABELS[p])
            ax.plot(xs[-1:], ys[-1:], "o", color=COLORS[p], ms=5)
        ax.set_title(title, fontsize=10.5, pad=7)
        ax.set_xlabel("cumulative visible output tokens", fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.16, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    paired_panel(fig.add_subplot(gs[1, 0]), paired, "final", "polarized_balance",
                 "Final balance − neutral (paired over 12 claims)")
    paired_panel(fig.add_subplot(gs[1, 1]), paired, "final", "entropy_normalized",
                 "Final entropy − neutral")
    paired_panel(fig.add_subplot(gs[1, 2]), paired, "at_1000", "polarized_balance",
                 "Balance at 1000 tokens − neutral")

    fig.suptitle("Counting facts shows no persona effect; their stance composition does",
                 fontsize=12.5, fontweight="semibold", y=0.975)
    fig.text(0.02, 0.012,
             f"{data['n_runs']} runs · {data['n_claims']} claims × 3 personas · "
             "deepseek-v4-flash · all-to-all · shared dossier · "
             "bars are 95% bootstrap intervals on the paired difference",
             fontsize=7.6, color="#555")
    fig.savefig(a.out, dpi=170)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
