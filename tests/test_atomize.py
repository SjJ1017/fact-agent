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
    assert not looks_joined("The veil creates social pressure.")


def test_prefilter_catches_attribution_even_with_nothing_joined():
    """A wrapper needs stripping whether or not the sentence is also joined."""
    assert looks_joined("E1 shows handguns are concealable.")
    assert looks_joined("Panelist B asserts the dossier contains opinions.")


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

def test_an_empty_extraction_is_retried_and_never_cached():
    """An empty result is a failed call, and the two defences against it are
    linked: it must not be cached, and it must be retried. Caching would make
    the failure permanent; not retrying deleted 5.5% of the corpus silently,
    since 33 of 34 turns that came back empty produced facts on a second ask.

    A backend that fails once and then succeeds must therefore be recovered
    inside a single extract_facts call — which can only happen if the empty
    result was not written to the cache."""
    import tempfile

    import pytest as _pytest

    from factflow.extract import EmptyExtraction, extract_facts
    from factflow.llm import LLM, LLMConfig

    class Flaky:
        name = "fake"
        default_model = "m"
        calls = 0

        def generate(self, *, system, user, output_format, model, max_tokens):
            Flaky.calls += 1
            if Flaky.calls == 1:                      # first attempt comes back empty
                return output_format.model_validate({"facts": []})
            return output_format.model_validate(
                {"facts": [{"text": "A holds.", "polarity": "affirm",
                            "quote": "A holds."}]})

    with tempfile.TemporaryDirectory() as d:
        llm = LLM(LLMConfig(model="m", cache_dir=d), backend=Flaky())
        mentions = extract_facts(llm, "A holds.", attempts=3)
        assert [m.text for m in mentions] == ["A holds."]
        assert Flaky.calls == 2, "the empty result was cached, so the retry never ran"

    class Always:
        name = "fake"
        default_model = "m"
        calls = 0

        def generate(self, *, system, user, output_format, model, max_tokens):
            Always.calls += 1
            return output_format.model_validate({"facts": []})

    with tempfile.TemporaryDirectory() as d:
        llm = LLM(LLMConfig(model="m", cache_dir=d), backend=Always())
        with _pytest.raises(EmptyExtraction):
            extract_facts(llm, "A holds.", attempts=3)
        assert Always.calls == 3, "a text that is genuinely factless is still tried 3x"


def test_a_stripped_single_claim_is_not_mistaken_for_no_change():
    """Unwrapping rewrites a claim in place and still returns one part.
    Treating one part as 'unchanged' silently discarded every strip."""
    class Stripper(FakeLLM):
        def _atomize(self, user, model):
            import json as _j
            out = []
            for item in _j.loads(user):
                txt = item["text"]
                out.append({"fact_id": item["fact_id"],
                            "parts": [txt.split(" shows ", 1)[-1]]})
            return model.model_validate({"facts": out})

    ms = [_m("m1", "E1 shows handguns are concealable.")]
    out = atomize(Stripper(), ms)
    assert [m.text for m in out] == ["handguns are concealable."]
    assert out[0].provenance.extra["split_from"] == "m1"


def test_a_genuinely_untouched_fact_keeps_its_id():
    ms = [_m("m1", "Bosnia and Herzegovina is a country.")]
    out = atomize(FakeLLM(keep_whole={"Bosnia and Herzegovina is a country."}), ms)
    assert out[0].mention_id == "m1"


def test_identical_claims_in_one_turn_collapse():
    """Stripping the attribution makes 'E3 argues X' and 'E4 argues X' identical.
    Blocking never compares two mentions from the same slot, so if atomize does
    not collapse them here nothing downstream ever will."""
    class Splitter(FakeLLM):
        def _atomize(self, user, model):
            import json as _j
            return model.model_validate({"facts": [
                {"fact_id": i["fact_id"], "parts": ["The veil should not be banned.",
                                                    "The veil should not be banned."]}
                for i in _j.loads(user)]})

    out = atomize(Splitter(), [_m("m1", "E3 and E4 argue against banning the veil.")])
    assert [m.text for m in out] == ["The veil should not be banned."]


def test_identical_claims_in_different_turns_are_both_kept():
    """Two agents saying the same thing is the signal, not noise - de-duplication
    is per turn only."""
    class Echo(FakeLLM):
        def _atomize(self, user, model):
            import json as _j
            return model.model_validate({"facts": [
                {"fact_id": i["fact_id"], "parts": ["The veil should not be banned."]}
                for i in _j.loads(user)]})

    ms = [_m("m1", "E3 argues against banning the veil.", ),
          FactMention(mention_id="m2", text="E4 argues against banning the veil.",
                      quote="q", provenance=Provenance(execution_id="e", agent_id="B",
                                                       round=1, channel=Channel.OUTPUT))]
    out = atomize(Echo(), ms)
    assert len(out) == 2
