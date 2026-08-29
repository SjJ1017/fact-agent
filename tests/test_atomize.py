"""Splitting facts the extractor left joined.

A third of the Perspectrum pilot's facts carried a conjunction, and the damage
lands in the matcher: two turns asserting the same things never produce an exact
repeat, so what should have been set membership arrives as an entailment
judgement. These tests pin the split itself and, just as importantly, the cases
that must survive it whole.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fake_llm import FakeLLM

from factflow.atomize import atomize, looks_joined
from factflow.types import Channel, FactMention, Provenance


def _m(mid: str, text: str) -> FactMention:
    return FactMention(mention_id=mid, text=text, quote="ORIGINAL SPAN",
                       provenance=Provenance(execution_id="e", agent_id="A", round=1,
                                             channel=Channel.OUTPUT))


# -- the pre-filter ---------------------------------------------------------

def test_prefilter_catches_conjunctions_and_set_quantifiers():
    assert looks_joined("E3 and E4 argue against the ban.")
    assert looks_joined("Both dossiers name the same enzyme.")
    assert looks_joined("It cites unequal treatment, backlash, and discretion.")


def test_prefilter_passes_a_plain_single_claim():
    assert not looks_joined("Handguns are concealable.")
    assert not looks_joined("E2 reports declines in handgun crime.")


def test_prefilter_errs_toward_asking():
    """'Bosnia and Herzegovina' is a false positive by design: one wasted call
    that returns the fact unchanged beats leaving a conjunction in the store."""
    assert looks_joined("Bosnia and Herzegovina is a country.")


# -- splitting --------------------------------------------------------------

def test_a_joined_fact_becomes_one_mention_per_assertion():
    ms = [_m("m1", "E3 opposes the ban and E4 opposes the ban.")]
    out = atomize(FakeLLM(), ms)
    assert [m.text for m in out] == ["E3 opposes the ban", "E4 opposes the ban"]


def test_parts_carry_the_parent_provenance_and_quote():
    ms = [_m("m1", "A holds and B holds.")]
    out = atomize(FakeLLM(), ms)
    assert all(m.quote == "ORIGINAL SPAN" for m in out)
    assert all(m.provenance.agent_id == "A" and m.provenance.round == 1 for m in out)
    assert all(m.provenance.extra["split_from"] == "m1" for m in out)


def test_part_ids_are_derived_from_the_parent():
    out = atomize(FakeLLM(), [_m("m1", "A holds and B holds.")])
    assert [m.mention_id for m in out] == ["m1#1", "m1#2"]


def test_an_atomic_fact_keeps_its_original_id():
    """Re-running over an already-atomic store must be a no-op, or every
    downstream reference breaks on each pass."""
    ms = [_m("m1", "Handguns are concealable.")]
    out = atomize(FakeLLM(), ms)
    assert len(out) == 1
    assert out[0].mention_id == "m1"
    assert out[0] is ms[0]


def test_a_fact_the_model_declines_to_split_keeps_its_id():
    ms = [_m("m1", "Bosnia and Herzegovina is a country.")]
    out = atomize(FakeLLM(keep_whole={"Bosnia and Herzegovina is a country."}), ms)
    assert [m.mention_id for m in out] == ["m1"]


def test_order_is_preserved_around_a_split():
    ms = [_m("m1", "First."), _m("m2", "A holds and B holds."), _m("m3", "Last.")]
    out = atomize(FakeLLM(), ms)
    assert [m.text for m in out] == ["First.", "A holds", "B holds", "Last."]


def test_only_suspect_facts_are_sent():
    llm = FakeLLM()
    atomize(llm, [_m("m1", "Handguns are concealable."), _m("m2", "A holds and B holds.")])
    assert llm.calls == ["AtomizeResult"]          # one batch, not two calls
    atomize(llm, [_m("m3", "Handguns are concealable.")])
    assert llm.calls == ["AtomizeResult"]          # nothing suspect: no call at all


def test_prefilter_can_be_disabled():
    llm = FakeLLM()
    atomize(llm, [_m("m1", "Handguns are concealable.")], prefilter=False)
    assert llm.calls == ["AtomizeResult"]


# -- the cache guard --------------------------------------------------------

def test_an_empty_extraction_is_not_cached():
    """A call that returns nothing has failed; caching it makes the failure
    permanent and silent, which is how a poisoned entry turned a working
    extractor into one that reported zero facts on every later run."""
    from factflow.extract import extract_facts
    from factflow.llm import LLM, LLMConfig
    import tempfile

    class Flaky:
        name = "fake"
        default_model = "m"
        calls = 0

        def generate(self, *, system, user, output_format, model, max_tokens):
            Flaky.calls += 1
            if Flaky.calls == 1:                      # first attempt comes back empty
                return output_format.model_validate({"facts": []})
            return output_format.model_validate(
                {"facts": [{"text": "A holds.", "polarity": "affirm", "quote": "A holds."}]})

    with tempfile.TemporaryDirectory() as d:
        llm = LLM(LLMConfig(model="m", cache_dir=d), backend=Flaky())
        assert extract_facts(llm, "some text") == []
        again = extract_facts(llm, "some text")       # same key: must retry, not replay
        assert [m.text for m in again] == ["A holds."]
