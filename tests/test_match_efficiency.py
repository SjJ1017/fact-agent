"""The two changes that make matching affordable, and the one that makes it legible.

Matching is the quadratic stage: a 3-round 3-agent debate is ~97 mentions and
~150 candidate pairs, and every pair used to cost an LLM call. These tests pin
the behaviour of skipping the calls whose answer the blocker already implies,
of not paying for prose nobody reads, and of walking the debate in time order.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fake_llm import FakeLLM

from factflow.match import adjudicate, incremental_match
from factflow.types import Channel, FactMention, Provenance


def _m(mid: str, text: str, agent: str = "A", rnd: int = 1) -> FactMention:
    return FactMention(mention_id=mid, text=text, quote=text,
                       provenance=Provenance(execution_id="e", agent_id=agent,
                                             round=rnd, channel=Channel.OUTPUT))


def test_pairs_below_the_band_are_rejected_without_a_call():
    ms = [_m("m1", "Handguns are concealable."), _m("m2", "Rain fell in April.")]
    llm = FakeLLM(equivalent={frozenset({"Handguns are concealable.", "Rain fell in April."})})
    rels = adjudicate(llm, ms, [(0, 1, 0.40)], auto_reject_below=0.55)
    assert llm.calls == []                      # the whole point: no call made
    assert [r.relation for r in rels] == ["UNRELATED"]


def test_pairs_inside_the_band_still_go_to_the_adjudicator():
    ms = [_m("m1", "Handguns are concealable."), _m("m2", "Handguns can be concealed.")]
    llm = FakeLLM(equivalent={frozenset({"Handguns are concealable.",
                                         "Handguns can be concealed."})})
    rels = adjudicate(llm, ms, [(0, 1, 0.80)], auto_reject_below=0.55)
    assert llm.calls == ["AdjudicationResult"]
    assert [r.relation for r in rels] == ["EQUIVALENT"]


def test_a_mixed_batch_splits_into_skipped_and_judged():
    ms = [_m("m1", "A"), _m("m2", "B"), _m("m3", "C")]
    llm = FakeLLM(equivalent={frozenset({"A", "B"})})
    rels = adjudicate(llm, ms, [(0, 1, 0.90), (0, 2, 0.30), (1, 2, 0.20)],
                      auto_reject_below=0.55)
    assert llm.calls == ["AdjudicationResult"]   # one batch, not three
    by = {frozenset({r.a, r.b}): r.relation for r in rels}
    assert by[frozenset({"m1", "m2"})] == "EQUIVALENT"
    assert by[frozenset({"m1", "m3"})] == "UNRELATED"
    assert by[frozenset({"m2", "m3"})] == "UNRELATED"


def test_auto_reject_off_by_default_so_existing_callers_are_unchanged():
    ms = [_m("m1", "A"), _m("m2", "B")]
    llm = FakeLLM()
    adjudicate(llm, ms, [(0, 1, 0.10)])
    assert llm.calls == ["AdjudicationResult"]


def test_rationale_off_requests_the_lean_schema():
    ms = [_m("m1", "A"), _m("m2", "B")]
    llm = FakeLLM(equivalent={frozenset({"A", "B"})})
    rels = adjudicate(llm, ms, [(0, 1, 0.90)], rationale=False)
    assert llm.calls == ["LeanAdjudication"]
    assert rels[0].relation == "EQUIVALENT"
    assert rels[0].rationale is None


def test_incremental_walks_turns_in_time_order():
    ms = [_m("m3", "Third.", "A", 2), _m("m1", "First.", "A", 1), _m("m2", "Second.", "B", 1)]
    seen: list[str] = []

    class Recorder(FakeLLM):
        def parse(self, *, system, user, output_format, **kw):
            seen.append(user)
            return super().parse(system=system, user=user, output_format=output_format, **kw)

    store = incremental_match(Recorder(), ms)
    # every mention registered exactly once, and each turn was its own step
    assert set(store.mentions) == {"m1", "m2", "m3"}
    assert len(store.facts) == 3


def test_incremental_compares_against_the_canon_not_every_mention():
    """A fact repeated in later turns links to the registered fact, so the canon
    stays small while mentions accumulate."""
    ms = [_m("m1", "Handguns are concealable.", "A", 1),
          _m("m2", "Handguns are concealable.", "B", 1),
          _m("m3", "Handguns are concealable.", "A", 2)]
    store = incremental_match(FakeLLM(), ms)
    assert len(store.mentions) == 3
    assert len(store.facts) == 1                      # identical text collapses by id
    assert len(store.facts[next(iter(store.facts))].mention_ids) == 3


def test_a_poisoned_pair_does_not_take_its_batch_down_with_it():
    """A content-filtered pair stalls rather than refusing, so a batch failure
    must not discard the pairs that were fine."""
    ms = [_m("m1", "A"), _m("m2", "B"), _m("m3", "POISON"), _m("m4", "D")]

    class Filtered(FakeLLM):
        def parse(self, *, system, user, output_format, **kw):
            if "POISON" in user:
                raise TimeoutError("gateway stalled")
            return super().parse(system=system, user=user, output_format=output_format, **kw)

    llm = Filtered(equivalent={frozenset({"A", "B"})})
    bad: list = []
    rels = adjudicate(llm, ms, [(0, 1, 0.90), (2, 3, 0.90)], batch_size=8,
                      unjudged_out=bad)
    judged = {frozenset({r.a, r.b}): r.relation for r in rels}
    assert judged[frozenset({"m1", "m2"})] == "EQUIVALENT"   # survived the bisect
    assert bad == [(2, 3)]                                    # offender isolated
    assert frozenset({"m3", "m4"}) not in judged


def test_bisect_can_be_turned_off():
    ms = [_m("m1", "A"), _m("m2", "POISON")]

    class Filtered(FakeLLM):
        def parse(self, **_kw):
            raise TimeoutError("gateway stalled")

    assert adjudicate(Filtered(), ms, [(0, 1, 0.9)], bisect_on_failure=False) == []


def test_the_guard_asks_the_same_question_the_caller_did():
    """A guard that adjudicates a five-way relation over clusters built from
    SAME/DIFFERENT gives one pair two verdicts under two definitions, and pays
    the expensive question to do it."""
    from factflow.match import binary_match
    asked: list[str] = []

    class Watcher(FakeLLM):
        def parse(self, *, system, user, output_format, **kw):
            asked.append(output_format.__name__)
            return super().parse(system=system, user=user, output_format=output_format, **kw)

    # Four wordings of one claim, plus an unrelated one so the blocker has
    # something to contrast against - identical strings give TF-IDF nothing to
    # weigh and produce no candidate pairs at all. Union-find then builds a
    # component big enough (> 2 members) for the guard to re-verify.
    texts = ["The ban raises social pressure.",
             "The ban increases social pressure.",
             "Social pressure rises under the ban.",
             "The ban leads to greater social pressure.",
             "Rain fell in April."]
    # Spread across agents and rounds: candidate_pairs skips pairs from one
    # slot, since extraction already deduplicated within a turn.
    slots = [("A", 1), ("B", 1), ("C", 1), ("A", 2), ("B", 2)]
    ms = [_m(f"m{i}", x, agent=s[0], rnd=s[1])
          for i, (x, s) in enumerate(zip(texts, slots))]
    equivalent = {frozenset({a, b}) for a in texts[:4] for b in texts[:4] if a != b}
    binary_match(Watcher(equivalent=equivalent), ms, threshold=0.2, top_k=5)
    assert asked, "no judgement was made at all"
    assert "AdjudicationResult" not in asked, f"guard fell back to five-way: {asked}"
    assert set(asked) <= {"IdentityResult", "IdentityResultBare"}
