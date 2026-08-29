"""Fact classes, and the compressed spacetime graph they make possible.

The reason this module exists is a asymmetry between traces and facts.

A trace is a sequence of strings of unbounded length. Two runs of the *same*
configuration on the *same* question share no alignment: different wording,
different turn lengths, different numbers of rounds before someone concedes.
There is no operation that averages two traces, so every trace-level study is
stuck reporting either one anecdote or one end-to-end scalar (accuracy), with
nothing in between.

Facts are different. A fact is a typed object, and the types are *run
independent*: whether a fact is gold, whether it was contested, whether two
agents arrived at it separately or one echoed the other. Bin the facts of a run
by those labels and the run collapses to a fixed-length vector regardless of how
long the transcript was. Fixed-length vectors average, subtract, and take
confidence intervals.

So: aggregate over facts, not over text. The compressed spacetime graph is the
same G^t as before, but each edge carries a count *per class* instead of a list
of fact ids, which makes the graph itself comparable across runs.

Four axes, chosen because each answers a different question about the run:

    grounding   where the fact came from        gold / context / injected
    support     how many agents reached it      independent, without being told
    spread      how far it travelled            private / shared
    fate        did it survive to the end       survived / dropped

`support` is the one that is hard to get any other way, and the one that breaks
majority voting. Three agents asserting a fact looks like three-fold agreement;
if two of them only said it after reading the first, it is one source and two
echoes. Only a content-level trace distinguishes those, and they should not be
weighed the same.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

from .types import Channel, FactStore

GROUNDING = ("gold", "context", "injected")
SUPPORT = ("single", "multi")
SPREAD = ("private", "shared")
FATE = ("dropped", "survived")


@dataclass(frozen=True)
class FactClass:
    """Run-independent labels for one fact. The unit of aggregation."""

    grounding: str
    support: str
    spread: str
    fate: str
    contested: bool

    def key(self) -> str:
        c = "contested" if self.contested else "clear"
        return f"{self.grounding}/{self.support}/{self.spread}/{self.fate}/{c}"


@dataclass
class RunView:
    """Everything downstream needs, computed once per (store, execution)."""

    execution_id: str
    agents: list[str]
    rounds: list[int]
    said: dict[tuple[str, int], set[str]]      # (agent, round) -> fact ids expressed
    classes: dict[str, FactClass]              # fact id -> class
    independent: dict[str, set[str]]           # fact id -> agents that derived it
    roles: dict[str, str]                      # agent -> role label


def _grounding(store: FactStore, fid: str) -> str:
    seen_source = False
    for mid in store.facts[fid].mention_ids:
        p = store.mentions[mid].provenance
        if p.channel == Channel.SOURCE:
            seen_source = True
            if p.extra.get("gold"):
                return "gold"
    return "context" if seen_source else "injected"


def _contested(store: FactStore) -> set[str]:
    """Facts on either end of a CONTRADICTS edge, lifted to canonical ids."""
    out: set[str] = set()
    for r in store.relations:
        if r.relation != "CONTRADICTS":
            continue
        fa, fb = store.mention_to_fact.get(r.a), store.mention_to_fact.get(r.b)
        if fa and fb and fa != fb:
            out.add(fa)
            out.add(fb)
    return out


def build_view(
    store: FactStore,
    execution_id: str,
    roles: Optional[dict[str, str]] = None,
) -> RunView:
    agents = store.agents(execution_id)
    rounds = [r for r in store.rounds(execution_id) if r > 0]
    said: dict[tuple[str, int], set[str]] = {(a, r): set() for a in agents for r in rounds}

    for fid, fact in store.facts.items():
        for mid in fact.mention_ids:
            p = store.mentions[mid].provenance
            if p.execution_id != execution_id or p.channel != Channel.OUTPUT:
                continue
            if (p.agent_id, p.round) in said:
                said[(p.agent_id, p.round)].add(fid)

    # An agent asserts a fact *independently* if it expressed the fact at a round
    # by which no peer had expressed it. Anything later is an echo, however
    # confidently phrased. Under full broadcast this is exact; under a sparse
    # topology it is conservative, because a non-neighbour's utterance was never
    # visible and so cannot have been echoed.
    independent: dict[str, set[str]] = defaultdict(set)
    for fid in store.facts:
        first: dict[str, int] = {}
        for a in agents:
            hits = [r for r in rounds if fid in said[(a, r)]]
            if hits:
                first[a] = min(hits)
        if not first:
            continue
        earliest = min(first.values())
        for a, r in first.items():
            if r == earliest:
                independent[fid].add(a)

    contested = _contested(store)
    last = rounds[-1] if rounds else 0
    survivors = set().union(*[said[(a, last)] for a in agents]) if rounds else set()

    classes: dict[str, FactClass] = {}
    for fid in store.facts:
        holders = {a for a in agents if any(fid in said[(a, r)] for r in rounds)}
        if not holders:
            continue  # source-only fact nobody ever uttered; not part of the flow
        classes[fid] = FactClass(
            grounding=_grounding(store, fid),
            support="multi" if len(independent[fid]) > 1 else "single",
            spread="shared" if len(holders) > 1 else "private",
            fate="survived" if fid in survivors else "dropped",
            contested=fid in contested,
        )

    return RunView(
        execution_id=execution_id,
        agents=agents,
        rounds=rounds,
        said=said,
        classes=classes,
        independent=dict(independent),
        roles=roles or {a: a for a in agents},
    )


def compressed_flow(view: RunView, axis: str = "grounding") -> dict[tuple[str, str, str], int]:
    """The averageable spacetime graph: (role_from, role_to, class) -> fact count.

    An edge is laid whenever a fact expressed by `a` at round r is expressed by
    `b` at r+1 having not been expressed by `b` before, i.e. the transmission
    edge of G^t, but counted per class rather than enumerated. Self edges are
    persistence. Keying on *role* rather than agent id is what lets two runs with
    different agent orderings be added together.
    """
    out: dict[tuple[str, str, str], int] = defaultdict(int)
    rounds = view.rounds
    for i, r in enumerate(rounds[:-1]):
        nxt = rounds[i + 1]
        for b in view.agents:
            before = set().union(*[view.said[(b, x)] for x in rounds if x <= r]) if i >= 0 else set()
            for fid in view.said[(b, nxt)]:
                cls = view.classes.get(fid)
                if cls is None:
                    continue
                label = getattr(cls, axis) if axis != "contested" else (
                    "contested" if cls.contested else "clear")
                if fid in before:
                    out[(view.roles[b], view.roles[b], label)] += 1  # persistence
                    continue
                senders = [a for a in view.agents if a != b and fid in view.said[(a, r)]]
                for a in senders:
                    out[(view.roles[a], view.roles[b], label)] += 1
    return dict(out)


def signature(view: RunView) -> dict[str, float]:
    """Fixed-length run vector. Every entry is a fraction, so runs on questions
    of different sizes stay comparable."""
    cls = list(view.classes.values())
    n = len(cls) or 1
    sig: dict[str, float] = {"n_facts": float(len(cls))}
    for axis, values in (("grounding", GROUNDING), ("support", SUPPORT),
                         ("spread", SPREAD), ("fate", FATE)):
        for v in values:
            sig[f"{axis}:{v}"] = sum(getattr(c, axis) == v for c in cls) / n
    sig["contested"] = sum(c.contested for c in cls) / n

    said_total = sum(len(v) for v in view.said.values())
    distinct = len(cls)
    # Redundancy: how much of what was said had already been said. 0 = every
    # utterance was new, 1 would be everyone repeating one fact forever.
    sig["redundancy"] = 1.0 - distinct / said_total if said_total else 0.0
    # Echo rate: of the facts more than one agent expressed, how many had only
    # one independent source. High echo with high apparent agreement is the
    # signature of a debate that converged without evidence.
    shared = [f for f, c in view.classes.items() if c.spread == "shared"]
    sig["echo_rate"] = (sum(len(view.independent.get(f, ())) == 1 for f in shared)
                        / len(shared)) if shared else 0.0
    return sig


def mean_signature(views: Iterable[RunView]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for v in views:
        for k, val in signature(v).items():
            acc[k].append(val)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def mean_flow(views: Iterable[RunView], axis: str = "grounding") -> dict[tuple[str, str, str], float]:
    """Average the compressed graphs. This is the operation traces do not admit."""
    acc: dict[tuple[str, str, str], float] = defaultdict(float)
    n = 0
    for v in views:
        n += 1
        for k, c in compressed_flow(v, axis).items():
            acc[k] += c
    return {k: val / n for k, val in acc.items()} if n else {}


# -- trajectories -----------------------------------------------------------
#
# Transport is only one of the things a round does. An agent that reasons emits
# facts nobody held, and calling those a leak misreads the task: on an inference
# problem the *birth* rate is the signal, not a defect. So the per-round view
# below tracks a population, not a pipeline. Facts are born, carried, and die,
# and a configuration has a characteristic demography.
#
# Rounds align across runs by construction, so these curves average, and the
# cost axis makes them comparable between configurations that buy their rounds
# at different prices.


def trajectory(view: RunView, cost: Optional[dict[tuple[str, int], float]] = None) -> list[dict]:
    """Per-round population accounting for the facts in a run.

        alive     distinct facts anyone expressed this round
        born      alive now, never expressed before
        died      expressed last round, expressed by nobody now
        carried   alive in both rounds

    `died` is not waste by default. A fact that is raised, examined, and dropped
    is what deliberation is supposed to look like; a fact that is dropped and
    was gold is a loss. The split by grounding keeps those apart.

    `cost` maps (agent, round) to whatever the round cost - tokens, characters,
    money. When given, each row carries the running total, so a curve can be
    drawn against spend rather than against round index.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    prev: set[str] = set()
    spend = 0.0

    for r in view.rounds:
        alive = set().union(*[view.said[(a, r)] for a in view.agents]) if view.agents else set()
        born = alive - seen
        died = prev - alive
        if cost is not None:
            spend += sum(cost.get((a, r), 0.0) for a in view.agents)

        def split(ids: set[str], axis: str) -> dict[str, int]:
            out: dict[str, int] = defaultdict(int)
            for f in ids:
                c = view.classes.get(f)
                if c is not None:
                    out[getattr(c, axis)] += 1
            return dict(out)

        rows.append({
            "round": r,
            "alive": len(alive),
            "born": len(born),
            "died": len(died),
            "carried": len(alive & prev),
            "cumulative": len(seen | alive),
            "said": sum(len(view.said[(a, r)]) for a in view.agents),
            "born_by_grounding": split(born, "grounding"),
            "died_by_grounding": split(died, "grounding"),
            "born_contested": sum(1 for f in born if view.classes.get(f)
                                  and view.classes[f].contested),
            # Of what was born this round, how much is still standing at the end.
            "born_survives": sum(1 for f in born if view.classes.get(f)
                                 and view.classes[f].fate == "survived"),
            "cost": spend,
        })
        seen |= alive
        prev = alive
    return rows


def survival_curve(view: RunView) -> dict[int, list[float]]:
    """For facts born at round b, the fraction still expressed at each later round.

    A configuration that generates freely and prunes hard has a steep curve; one
    that accumulates without discarding has a flat one. Two runs can reach the
    same number of surviving facts by either route, and only the curve says
    which happened.
    """
    born_at: dict[str, int] = {}
    alive_at: dict[int, set[str]] = {}
    seen: set[str] = set()
    for r in view.rounds:
        alive = set().union(*[view.said[(a, r)] for a in view.agents]) if view.agents else set()
        alive_at[r] = alive
        for f in alive - seen:
            born_at[f] = r
        seen |= alive

    out: dict[int, list[float]] = {}
    for b in view.rounds:
        cohort = [f for f, r in born_at.items() if r == b]
        if not cohort:
            continue
        out[b] = [sum(f in alive_at[r] for f in cohort) / len(cohort)
                  for r in view.rounds if r >= b]
    return out


def mean_trajectory(views: Iterable[RunView],
                    cost: Optional[dict[str, dict[tuple[str, int], float]]] = None) -> list[dict]:
    """Average the per-round rows. Rounds are the alignment key, so this is
    exactly the operation transcripts do not admit."""
    acc: dict[int, list[dict]] = defaultdict(list)
    for v in views:
        for row in trajectory(v, (cost or {}).get(v.execution_id)):
            acc[row["round"]].append(row)
    scalars = ("alive", "born", "died", "carried", "cumulative", "said",
               "born_contested", "born_survives", "cost")
    out = []
    for r in sorted(acc):
        rows = acc[r]
        m = {"round": r, "n_runs": len(rows)}
        m.update({k: sum(x[k] for x in rows) / len(rows) for k in scalars})
        for field in ("born_by_grounding", "died_by_grounding"):
            agg: dict[str, float] = defaultdict(float)
            for x in rows:
                for k, val in x[field].items():
                    agg[k] += val / len(rows)
            m[field] = dict(agg)
        out.append(m)
    return out
