"""The two rules that make matching affordable and stop it collapsing.

Both were paid for in wasted runs. Skipping the pairs whose answer the blocker
already implies is what makes the pass affordable; refusing to chain a weak
edge is what stops union-find fusing a whole debate into one cluster.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fake_llm import FakeLLM

from factflow.match import UNION_MIN_SIMILARITY, cluster, identify, match
from factflow.types import Channel, FactMention, Provenance, Relation


def _m(mid: str, text: str, agent: str = "A", rnd: int = 1) -> FactMention:
    return FactMention(mention_id=mid, text=text, quote=text,
                       provenance=Provenance(execution_id="e", agent_id=agent,
                                             round=rnd, channel=Channel.OUTPUT))


# -- spending the call where the answer is in doubt -------------------------

def test_pairs_below_the_band_are_rejected_without_a_call():
    ms = [_m("m1", "Handguns are concealable."), _m("m2", "Rain fell in April.", "B")]
    llm = FakeLLM(equivalent={frozenset({"Handguns are concealable.", "Rain fell in April."})})
    rels = identify(llm, ms, [(0, 1, 0.40)], auto_reject_below=0.55)
    assert llm.calls == []
    assert [r.relation for r in rels] == ["UNRELATED"]


def test_pairs_above_the_band_are_accepted_without_a_call():
    ms = [_m("m1", "The ban raises pressure."), _m("m2", "The ban increases pressure.", "B")]
    llm = FakeLLM()
    rels = identify(llm, ms, [(0, 1, 0.97)], auto_accept_above=0.95)
    assert llm.calls == []
    assert [r.relation for r in rels] == ["EQUIVALENT"]


def test_pairs_inside_the_band_go_to_the_model():
    ms = [_m("m1", "Handguns are concealable."), _m("m2", "Handguns can be concealed.", "B")]
    llm = FakeLLM(equivalent={frozenset({"Handguns are concealable.",
                                         "Handguns can be concealed."})})
    rels = identify(llm, ms, [(0, 1, 0.80)], auto_reject_below=0.55, auto_accept_above=0.95)
    assert llm.calls == ["IdentityResult"]
    assert [r.relation for r in rels] == ["EQUIVALENT"]


def test_both_shortcuts_are_off_by_default():
    """A caller passing a placeholder rather than a measured similarity must get
    a real judgement, not have the placeholder decide the pair."""
    ms = [_m("m1", "A"), _m("m2", "B", "B")]
    llm = FakeLLM()
    identify(llm, ms, [(0, 1, 1.0)])
    assert llm.calls == ["IdentityResult"]


def test_a_mixed_batch_splits_into_skipped_and_judged():
    ms = [_m("m1", "A"), _m("m2", "B", "B"), _m("m3", "C", "C")]
    llm = FakeLLM(equivalent={frozenset({"A", "B"})})
    rels = identify(llm, ms, [(0, 1, 0.80), (0, 2, 0.30), (1, 2, 0.20)],
                    auto_reject_below=0.55)
    assert llm.calls == ["IdentityResult"]        # one batch, not three calls
    by = {frozenset({r.a, r.b}): r.relation for r in rels}
    assert by[frozenset({"m1", "m2"})] == "EQUIVALENT"
    assert by[frozenset({"m1", "m3"})] == "UNRELATED"
    assert by[frozenset({"m2", "m3"})] == "UNRELATED"


def test_the_note_is_kept_when_the_model_names_a_difference():
    ms = [_m("m1", "A"), _m("m2", "B", "B")]
    rels = identify(FakeLLM(), ms, [(0, 1, 0.80)])
    assert rels[0].rationale == "differs"          # 'none' is dropped, a real note is not


# -- not chaining a weak edge ----------------------------------------------

def _eq(a: str, b: str, sim: float) -> Relation:
    return Relation(a=a, b=b, relation="EQUIVALENT", confidence=sim)


def test_a_strong_edge_merges():
    ms = [_m("m1", "A"), _m("m2", "A too", "B")]
    facts = cluster(ms, [_eq("m1", "m2", 0.95)])
    assert len(facts) == 1


def test_a_weak_edge_is_recorded_but_does_not_merge():
    """The judgement stands for its own pair; it just may not imply a third."""
    ms = [_m("m1", "A"), _m("m2", "A too", "B")]
    facts = cluster(ms, [_eq("m1", "m2", 0.75)])
    assert len(facts) == 2


def test_weak_edges_cannot_chain_unrelated_facts_together():
    """A~B and B~C at low similarity must not produce A~C, which nobody judged.
    Unioning every SAME edge fused six real stores into clusters of up to 111."""
    ms = [_m("m1", "The ban raises pressure."),
          _m("m2", "Pressure rises.", "B"),
          _m("m3", "Rain fell in April.", "C")]
    weak = [_eq("m1", "m2", 0.72), _eq("m2", "m3", 0.71)]
    assert len(cluster(ms, weak)) == 3
    strong = [_eq("m1", "m2", 0.95), _eq("m2", "m3", 0.95)]
    assert len(cluster(ms, strong)) == 1          # the rule is about strength, not shape


def test_the_union_threshold_is_the_documented_one():
    ms = [_m("m1", "A"), _m("m2", "A too", "B")]
    assert len(cluster(ms, [_eq("m1", "m2", UNION_MIN_SIMILARITY)])) == 1
    assert len(cluster(ms, [_eq("m1", "m2", UNION_MIN_SIMILARITY - 0.01)])) == 2


def test_unrelated_edges_never_merge_however_similar():
    ms = [_m("m1", "A"), _m("m2", "B", "B")]
    r = Relation(a="m1", b="m2", relation="UNRELATED", confidence=0.99)
    assert len(cluster(ms, [r])) == 2


# -- the whole path ---------------------------------------------------------

def test_match_registers_every_mention_once():
    ms = [_m("m1", "The ban raises social pressure.", "A", 1),
          _m("m2", "The ban increases social pressure.", "B", 1),
          _m("m3", "Rain fell in April.", "C", 1)]
    eq = {frozenset({"The ban raises social pressure.",
                     "The ban increases social pressure."})}
    store = match(FakeLLM(equivalent=eq), ms, threshold=0.2, top_k=5,
                  auto_accept_above=1.01)
    assert set(store.mentions) == {"m1", "m2", "m3"}
    assert sum(len(f.mention_ids) for f in store.facts.values()) == 3


def test_a_subset_is_not_auto_accepted_as_the_same_fact():
    """candidate_pairs scores max(cosine, containment), so a fact whose tokens
    are all contained in another scores 1.0 - and that is the dropped-scope
    case, which is DIFFERENT. An accept-above rule would merge them unasked."""
    ms = [_m("m1", "UBI reduced poverty by 12.3% in the Kenya pilot."),
          _m("m2", "UBI reduced poverty.", "B")]
    llm = FakeLLM()                                # nothing declared equivalent
    store = match(llm, ms, threshold=0.2, top_k=5, union_min_similarity=0.2)
    assert llm.calls, "the pair was decided without asking"
    assert len(store.facts) == 2
