"""Fact flow as a graph, and what the graph's shape tells you.

Three graphs live inside one run, and they answer different questions. Keeping
them apart matters, because "centrality" on the wrong one is a number with no
referent.

  fact graph        nodes are facts, edges are entailment.
                    Centrality here finds the *crux*: the premise that many
                    conclusions route through. If a run is wrong, the crux is
                    where to look first, and it is usually not the fact the
                    agents argued about most.

  spacetime graph   nodes are (agent, round), edges are facts moving between
                    them. Centrality here finds the *broker*: the agent that
                    other agents' facts have to pass through to reach the end.

  role graph        the spacetime graph collapsed onto role labels and summed.
                    This is the one that survives averaging across runs, so it
                    is the one to compare configurations on.

The measure that does not come from textbook centrality is `load_bearing`: for
each agent, how many gold facts that survived to the final round had *that agent
as their only independent source*. An agent with a high count is a single point
of failure — remove it and the answer loses evidence that nothing else supplies.
An agent with zero is, for this run, decorative: everything it contributed was
also contributed by someone else.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import networkx as nx

from .aggregate import RunView
from .types import FactStore


def fact_graph(store: FactStore) -> nx.DiGraph:
    """Canonical facts joined by entailment; contradictions kept as attributes.

    Mention-level relations are lifted to fact level, which collapses the many
    duplicate judgements between two clusters into one edge with a weight.
    """
    g = nx.DiGraph()
    for fid, fact in store.facts.items():
        g.add_node(fid, text=fact.canonical_text)

    conflict: dict[tuple[str, str], int] = defaultdict(int)
    for r in store.relations:
        fa, fb = store.mention_to_fact.get(r.a), store.mention_to_fact.get(r.b)
        if not fa or not fb or fa == fb:
            continue
        if r.relation == "A_ENTAILS_B":
            g.add_edge(fa, fb, weight=g.get_edge_data(fa, fb, {}).get("weight", 0) + 1)
        elif r.relation == "B_ENTAILS_A":
            g.add_edge(fb, fa, weight=g.get_edge_data(fb, fa, {}).get("weight", 0) + 1)
        elif r.relation == "CONTRADICTS":
            conflict[tuple(sorted((fa, fb)))] += 1
    for (a, b), w in conflict.items():
        g.add_edge(a, b, kind="contradicts", weight=w)
        g.add_edge(b, a, kind="contradicts", weight=w)
    return g


def spacetime_graph(view: RunView) -> nx.DiGraph:
    """(agent, round) nodes; edges weighted by how many facts crossed them.

    Transmission edges only go forward in time, so the graph is a DAG and
    path-based centrality has an unambiguous direction.
    """
    g = nx.DiGraph()
    for a in view.agents:
        for r in view.rounds:
            g.add_node((a, r), agent=a, round=r, role=view.roles.get(a, a),
                       n_said=len(view.said[(a, r)]))

    for i, r in enumerate(view.rounds[:-1]):
        nxt = view.rounds[i + 1]
        for b in view.agents:
            before = set().union(*[view.said[(b, x)] for x in view.rounds if x <= r])
            for fid in view.said[(b, nxt)]:
                if fid in before:
                    _bump(g, (b, r), (b, nxt), "persistence")
                    continue
                for a in view.agents:
                    if a != b and fid in view.said[(a, r)]:
                        _bump(g, (a, r), (b, nxt), "transmission")
    return g


def _bump(g: nx.DiGraph, u: Any, v: Any, kind: str) -> None:
    d = g.get_edge_data(u, v)
    if d is None:
        g.add_edge(u, v, weight=1, kind=kind)
    else:
        d["weight"] += 1
        if d.get("kind") != kind:
            d["kind"] = "mixed"


def role_graph(view: RunView) -> nx.DiGraph:
    """Spacetime graph collapsed onto roles. Fixed shape, so it averages."""
    g = nx.DiGraph()
    st = spacetime_graph(view)
    for (a, _r), (b, _s), d in st.edges(data=True):
        ra, rb = view.roles.get(a, a), view.roles.get(b, b)
        cur = g.get_edge_data(ra, rb, {}).get("weight", 0)
        g.add_edge(ra, rb, weight=cur + d["weight"])
    return g


def crux_facts(store: FactStore, top: int = 10) -> list[tuple[str, float, str]]:
    """Facts ranked by betweenness in the entailment graph.

    A fact with high betweenness sits on many inference paths: several premises
    feed it and several conclusions rest on it. That is the load-bearing
    proposition of the argument, and it is a different thing from the fact that
    was repeated most often, which is what a frequency count would surface.
    """
    g = fact_graph(store)
    if g.number_of_edges() == 0:
        return []
    bc = nx.betweenness_centrality(g, weight=None)
    ranked = sorted(bc.items(), key=lambda kv: -kv[1])[:top]
    return [(fid, score, store.facts[fid].canonical_text) for fid, score in ranked if score > 0]


def broker_scores(view: RunView) -> dict[str, float]:
    """Per-agent betweenness on the spacetime graph, summed over that agent's
    rounds. High means other agents' facts reach the end through this agent."""
    g = spacetime_graph(view)
    if g.number_of_nodes() < 3:
        return {a: 0.0 for a in view.agents}
    bc = nx.betweenness_centrality(g, weight="weight")
    out: dict[str, float] = defaultdict(float)
    for (a, _r), score in bc.items():
        out[a] += score
    return dict(out)


def load_bearing(view: RunView, grounding: str = "gold") -> dict[str, int]:
    """Per agent: surviving facts of this grounding whose ONLY independent
    source was that agent.

    This is the pruning and attribution number. An agent scoring zero added
    nothing that survived which someone else did not also supply, so removing it
    costs the run no evidence. An agent scoring high is a single point of
    failure: if it had stayed silent, or been wrong, no other agent would have
    carried the fact.
    """
    out: dict[str, int] = {a: 0 for a in view.agents}
    for fid, cls in view.classes.items():
        if cls.fate != "survived" or cls.grounding != grounding:
            continue
        src = view.independent.get(fid, set())
        if len(src) == 1:
            out[next(iter(src))] += 1
    return out


def fragility(view: RunView, grounding: str = "gold") -> float:
    """Fraction of surviving facts resting on a single independent source.

    The run-level version of `load_bearing`. A run at 1.0 reached its answer with
    no redundancy at all: every supporting fact came from exactly one agent, and
    a single hallucination anywhere is unopposed. A run near 0 had every fact
    independently reproduced, which is what agreement is supposed to mean.
    """
    surv = [f for f, c in view.classes.items()
            if c.fate == "survived" and c.grounding == grounding]
    if not surv:
        return 0.0
    return sum(len(view.independent.get(f, ())) <= 1 for f in surv) / len(surv)


def summarise(store: FactStore, view: RunView) -> dict[str, Any]:
    st = spacetime_graph(view)
    fg = fact_graph(store)
    return {
        "execution": view.execution_id,
        "agents": view.agents,
        "rounds": view.rounds,
        "spacetime": {"nodes": st.number_of_nodes(), "edges": st.number_of_edges(),
                      "density": nx.density(st) if st.number_of_nodes() > 1 else 0.0},
        "fact_graph": {"nodes": fg.number_of_nodes(), "edges": fg.number_of_edges(),
                       "components": nx.number_weakly_connected_components(fg)},
        "broker": broker_scores(view),
        "load_bearing_gold": load_bearing(view, "gold"),
        "load_bearing_injected": load_bearing(view, "injected"),
        "fragility_gold": fragility(view, "gold"),
        "crux": crux_facts(store, top=5),
    }
