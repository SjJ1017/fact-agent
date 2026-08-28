"""Fact classes and graph metrics, on hand-built stores with known answers.

Every fixture here encodes a situation whose correct reading is obvious to a
human, so a failure means the metric disagrees with the thing it claims to
measure rather than that a number moved.
"""

from __future__ import annotations

import pytest

from factflow.aggregate import build_view, compressed_flow, mean_signature, signature
from factflow import graph as G
from factflow.types import CanonicalFact, Channel, FactMention, FactStore, Provenance, Relation


def _store(utterances, source=(), relations=()):
    """utterances: {(agent, round): [fact_text, ...]}; source: [(text, is_gold)]."""
    st = FactStore()
    for text, gold in source:
        m = FactMention(mention_id=f"s_{abs(hash(text))%10**8}", text=text, quote=text,
                        provenance=Provenance(execution_id="e", channel=Channel.SOURCE,
                                              round=0, extra={"gold": gold}))
        st.add_mentions([m])
        f = st.facts.get(CanonicalFact.make_id(text)) or CanonicalFact(
            fact_id=CanonicalFact.make_id(text), canonical_text=text)
        f.mention_ids.append(m.mention_id)
        st.assign(f)
    for (agent, rnd), texts in utterances.items():
        for text in texts:
            mid = f"m_{agent}{rnd}_{abs(hash(text))%10**8}"
            st.add_mentions([FactMention(mention_id=mid, text=text, quote=text,
                                         provenance=Provenance(execution_id="e", agent_id=agent,
                                                               round=rnd, channel=Channel.OUTPUT))])
            f = st.facts.get(CanonicalFact.make_id(text)) or CanonicalFact(
                fact_id=CanonicalFact.make_id(text), canonical_text=text)
            f.mention_ids.append(mid)
            st.assign(f)
    for a, b, rel in relations:
        st.relations.append(Relation(a=a, b=b, relation=rel))
    return st


def _fid(store, text):
    return CanonicalFact.make_id(text)


# -- independent support vs echo -------------------------------------------

def test_three_agents_saying_it_at_once_is_three_sources():
    st = _store({("A", 1): ["X"], ("B", 1): ["X"], ("C", 1): ["X"]})
    v = build_view(st, "e")
    assert v.independent[_fid(st, "X")] == {"A", "B", "C"}
    assert v.classes[_fid(st, "X")].support == "multi"


def test_two_agents_repeating_the_first_is_one_source():
    st = _store({("A", 1): ["X"], ("B", 2): ["X"], ("C", 2): ["X"]})
    v = build_view(st, "e")
    assert v.independent[_fid(st, "X")] == {"A"}
    assert v.classes[_fid(st, "X")].support == "single"
    # Both runs end with all three asserting X; only support tells them apart.
    assert v.classes[_fid(st, "X")].spread == "shared"


def test_echo_rate_separates_the_two():
    genuine = build_view(_store({("A", 1): ["X"], ("B", 1): ["X"], ("A", 2): ["X"], ("B", 2): ["X"]}), "e")
    echoed = build_view(_store({("A", 1): ["X"], ("B", 2): ["X"], ("A", 2): ["X"]}), "e")
    assert signature(genuine)["echo_rate"] == 0.0
    assert signature(echoed)["echo_rate"] == 1.0


# -- grounding and fate -----------------------------------------------------

def test_gold_context_and_injected_are_distinguished():
    st = _store({("A", 1): ["G", "C", "I"]}, source=[("G", True), ("C", False)])
    v = build_view(st, "e")
    assert v.classes[_fid(st, "G")].grounding == "gold"
    assert v.classes[_fid(st, "C")].grounding == "context"
    assert v.classes[_fid(st, "I")].grounding == "injected"


def test_a_fact_absent_from_the_last_round_is_dropped():
    st = _store({("A", 1): ["X", "Y"], ("A", 2): ["X"]})
    v = build_view(st, "e")
    assert v.classes[_fid(st, "X")].fate == "survived"
    assert v.classes[_fid(st, "Y")].fate == "dropped"


def test_source_facts_nobody_utters_are_not_in_the_flow():
    st = _store({("A", 1): ["X"]}, source=[("NEVER", True)])
    v = build_view(st, "e")
    assert _fid(st, "NEVER") not in v.classes


# -- the signature is fixed length, which is the point ----------------------

def test_signatures_of_different_sized_runs_have_the_same_keys():
    small = build_view(_store({("A", 1): ["X"], ("A", 2): ["X"]}), "e")
    big = build_view(_store({("A", 1): [f"f{i}" for i in range(20)],
                             ("B", 1): [f"f{i}" for i in range(10)],
                             ("A", 2): [f"f{i}" for i in range(15)],
                             ("B", 2): [f"f{i}" for i in range(18)]}), "e")
    assert set(signature(small)) == set(signature(big))
    m = mean_signature([small, big])
    assert set(m) == set(signature(small))


def test_redundancy_is_zero_when_nothing_repeats_and_high_when_it_does():
    fresh = build_view(_store({("A", 1): ["X"], ("A", 2): ["Y"]}), "e")
    stuck = build_view(_store({("A", 1): ["X"], ("A", 2): ["X"], ("B", 1): ["X"], ("B", 2): ["X"]}), "e")
    assert signature(fresh)["redundancy"] == 0.0
    assert signature(stuck)["redundancy"] == pytest.approx(0.75)


# -- graphs -----------------------------------------------------------------

def test_load_bearing_names_the_sole_source_of_a_surviving_gold_fact():
    st = _store({("A", 1): ["G"], ("B", 2): ["G"], ("A", 2): ["G"]}, source=[("G", True)])
    v = build_view(st, "e")
    lb = G.load_bearing(v, "gold")
    assert lb["A"] == 1 and lb["B"] == 0
    assert G.fragility(v, "gold") == 1.0


def test_a_gold_fact_two_agents_found_separately_is_not_fragile():
    st = _store({("A", 1): ["G"], ("B", 1): ["G"], ("A", 2): ["G"], ("B", 2): ["G"]},
                source=[("G", True)])
    assert G.fragility(build_view(st, "e"), "gold") == 0.0


def test_spacetime_edges_split_transmission_from_persistence():
    st = _store({("A", 1): ["X"], ("A", 2): ["X"], ("B", 2): ["X"]})
    g = G.spacetime_graph(build_view(st, "e"))
    assert g[("A", 1)][("A", 2)]["kind"] == "persistence"
    assert g[("A", 1)][("B", 2)]["kind"] == "transmission"


def test_fact_graph_lifts_mention_relations_to_facts():
    st = _store({("A", 1): ["P"], ("B", 1): ["Q"]})
    ma = next(m for m in st.mentions if "A1" in m)
    mb = next(m for m in st.mentions if "B1" in m)
    st.relations.append(Relation(a=ma, b=mb, relation="A_ENTAILS_B"))
    g = G.fact_graph(st)
    assert g.has_edge(_fid(st, "P"), _fid(st, "Q"))


def test_crux_ranks_the_fact_inferences_route_through():
    """P -> R -> Q: R is the middle of the only path, so it is the crux."""
    st = _store({("A", 1): ["P", "R", "Q"]})
    ids = {t: [m for m in st.facts[_fid(st, t)].mention_ids][0] for t in ("P", "R", "Q")}
    st.relations.append(Relation(a=ids["P"], b=ids["R"], relation="A_ENTAILS_B"))
    st.relations.append(Relation(a=ids["R"], b=ids["Q"], relation="A_ENTAILS_B"))
    top = G.crux_facts(st, top=3)
    assert top and top[0][0] == _fid(st, "R")


def test_compressed_flow_keys_on_role_so_runs_can_be_added():
    st = _store({("A", 1): ["X"], ("B", 2): ["X"]})
    v = build_view(st, "e", roles={"A": "critic", "B": "solver"})
    flow = compressed_flow(v, axis="grounding")
    assert flow[("critic", "solver", "injected")] == 1


def test_contested_facts_are_flagged_from_contradiction_edges():
    st = _store({("A", 1): ["X"], ("B", 1): ["NOT X"]})
    ma = next(m for m in st.mentions if "A1" in m)
    mb = next(m for m in st.mentions if "B1" in m)
    st.relations.append(Relation(a=ma, b=mb, relation="CONTRADICTS"))
    v = build_view(st, "e")
    assert v.classes[_fid(st, "X")].contested
    assert v.classes[_fid(st, "NOT X")].contested
