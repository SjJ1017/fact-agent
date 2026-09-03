"""Dataset-agnostic debate cases, and how context is dealt out to agents.

`experiments/run_perspectrum.py` hard-codes one dataset shape: a claim, four
evidence entries, and the same dossier in front of every agent.  None of the
fact-flow measurement needs that shape.  This module holds the parts that
differ between datasets -- what a case is, and who sees which part of it --
so a runner can stay the same across PerspectruM, MMLU-Pro, DelibTrace and
the asymmetric-observation tasks.

The dealing step is not cosmetic.  While every agent holds the whole dossier,
a fact appearing in B's turn is never attributable to transmission: B had it
all along (fix.md defect five).  Dealing disjoint or partial evidence is what
makes "B stated a fact only A was given" an observable event.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Item:
    """One piece of context that can be given to some agents and not others.

    `side` is the stance the item takes on the question when the dataset says
    so ("support" / "undermine"); it is None when the dataset has no such
    notion.  `tags` is free-form and is what role-aligned dealing matches on.
    """

    id: str
    text: str
    side: str | None = None
    tags: tuple[str, ...] = ()

    def render(self) -> str:
        return f"[{self.id}] {self.text}"


@dataclass(frozen=True)
class Case:
    """One debate instance, in whatever dataset it came from.

    `public` is context every agent sees regardless of dealing (the scenario,
    the multiple-choice options, the rules of the task).  `items` is the
    dealable part.  `meta` carries whatever the scorer needs later -- gold
    answers, option letters, the source row id -- and is never shown.
    """

    id: str
    question: str
    public: str = ""
    items: tuple[Item, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def by_id(self, item_id: str) -> Item | None:
        return next((i for i in self.items if i.id == item_id), None)


# --------------------------------------------------------------------------
# dealing


def _round_robin(items: Sequence[Item], agents: Sequence[str],
                 rng: random.Random) -> dict[str, list[Item]]:
    order = list(items)
    rng.shuffle(order)
    out: dict[str, list[Item]] = {a: [] for a in agents}
    for n, item in enumerate(order):
        out[agents[n % len(agents)]].append(item)
    return out


def _sample(items: Sequence[Item], agents: Sequence[str], p: float,
            rng: random.Random) -> dict[str, list[Item]]:
    """Each item to each agent independently with probability `p`.

    Two guarantees are then repaired by hand, because independent sampling
    gives neither: every item reaches at least one agent (otherwise it is not
    in the system at all and cannot be traced), and every agent holds at least
    one item (otherwise it is arguing with nothing).
    """
    out: dict[str, list[Item]] = {a: [] for a in agents}
    for item in items:
        holders = [a for a in agents if rng.random() < p]
        if not holders:
            holders = [rng.choice(list(agents))]
        for a in holders:
            out[a].append(item)
    for a in agents:
        if not out[a] and items:
            out[a].append(rng.choice(list(items)))
    return out


def _role_aligned(items: Sequence[Item], agents: Sequence[str],
                  affinity: Sequence[Sequence[str]]) -> dict[str, list[Item]]:
    """Give each agent the items its role is meant to argue from.

    `affinity[i]` lists the `side`/`tag` values agent `i` receives.  An item
    matching no agent's affinity goes to everyone: it is common ground, not
    somebody's private brief.  An empty affinity list means "everything",
    which is how an adjudicator role is written.
    """
    if len(affinity) != len(agents):
        raise ValueError(
            f"affinity has {len(affinity)} entries for {len(agents)} agents")
    wanted = [set(a) for a in affinity]
    claimed = set().union(*wanted) if wanted else set()
    out: dict[str, list[Item]] = {a: [] for a in agents}
    for item in items:
        keys = {item.side} | set(item.tags) if item.side else set(item.tags)
        for agent, want in zip(agents, wanted):
            if not want or (keys & want) or not (keys & claimed):
                out[agent].append(item)
    return out


def deal(case: Case, agents: Sequence[str], spec: dict[str, Any],
         seed: int) -> dict[str, tuple[Item, ...]]:
    """Decide which items each agent is given.

    Modes:
      full        every agent gets every item (what the 2026-09 corpus did)
      partition   disjoint and covering -- the union is the full item set
      sample      independent per (item, agent) with probability p
      role_aligned  items matching the role's affinity, plus common ground

    The seed is mixed with the case id so two cases in one run do not get the
    same deal, and so a rerun of one case reproduces it exactly.
    """
    mode = spec.get("mode", "full")
    rng = random.Random(f"{seed}|{case.id}")
    if mode == "full":
        dealt = {a: list(case.items) for a in agents}
    elif mode == "partition":
        dealt = _round_robin(case.items, agents, rng)
    elif mode == "sample":
        dealt = _sample(case.items, agents, float(spec.get("p", 0.6)), rng)
    elif mode == "role_aligned":
        dealt = _role_aligned(case.items, agents, spec.get("affinity", []))
    else:
        raise ValueError(f"unknown disclosure mode {mode!r}")
    return {a: tuple(dealt[a]) for a in agents}


def coverage(dealt: dict[str, tuple[Item, ...]], case: Case) -> dict[str, Any]:
    """What the deal actually did, recorded next to the debate.

    Worth storing rather than recomputing: an analysis that asks whether a
    fact moved needs to know who held it at the start, and a deal drawn from
    a seeded rng is only reproducible while the seeding code is unchanged.
    """
    held = {a: [i.id for i in items] for a, items in dealt.items()}
    union = set().union(*(set(v) for v in held.values())) if held else set()
    sizes = [len(v) for v in held.values()]
    return {
        "held": held,
        "n_items": len(case.items),
        "covered": len(union),
        "uncovered": sorted({i.id for i in case.items} - union),
        "mean_per_agent": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "exclusive": sorted(
            i for i in union
            if sum(1 for v in held.values() if i in v) == 1),
    }
