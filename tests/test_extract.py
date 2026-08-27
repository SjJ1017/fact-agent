"""Extraction must decompose coordinated predicates.

An unsplit conjunction does not stay contained: a later restatement of one
conjunct ("Scott Derrickson is American") then looks like a *weakened* version of
the whole list, and the matcher records a spurious entailment edge. Degradation
counts inherit that error, so under-splitting shows up as a debate phenomenon
that never happened.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fake_llm import FakeLLM

from factflow import Provenance, extract_facts

SENT = "Scott Derrickson is an American director, screenwriter and producer."
SPLIT = [
    {"text": "Scott Derrickson is American.", "polarity": "affirm", "qualifiers": []},
    {"text": "Scott Derrickson is a director.", "polarity": "affirm", "qualifiers": []},
    {"text": "Scott Derrickson is a screenwriter.", "polarity": "affirm", "qualifiers": []},
    {"text": "Scott Derrickson is a producer.", "polarity": "affirm", "qualifiers": []},
]


def test_coordinated_predicate_becomes_separate_facts():
    llm = FakeLLM(extractions={SENT: SPLIT})
    facts = extract_facts(llm, SENT, Provenance(agent_id="A", round=1))
    assert len(facts) == 4
    assert {f.text for f in facts} == {f["text"] for f in SPLIT}


def test_prompt_carries_the_split_rule_and_its_exceptions():
    """The rule is only as good as the examples; pin both directions."""
    from factflow.extract import EXTRACTION_SYSTEM

    assert "ONE predicate per fact" in EXTRACTION_SYSTEM
    assert "screenwriter" in EXTRACTION_SYSTEM, "positive example missing"
    for keep_whole in ("black and\nwhite film", "Bosnia and Herzegovina"):
        assert keep_whole.replace("\n", " ") in EXTRACTION_SYSTEM.replace("\n", " "), (
            f"negative example {keep_whole!r} missing - without it the model over-splits"
        )


def test_duplicate_facts_within_one_text_are_dropped():
    dup = [SPLIT[0], dict(SPLIT[0]), SPLIT[1]]
    llm = FakeLLM(extractions={SENT: dup})
    assert len(extract_facts(llm, SENT, Provenance(agent_id="A", round=1))) == 2


def test_prompt_forbids_unresolved_referents():
    """Rule 2 was being ignored until it carried examples, exactly like rule 1.

    Note "both" is deliberately NOT listed here: a quantified subject is a rule 1
    (atomicity) defect, not a rule 2 (reference) one, and the two want opposite
    fixes. Resolving "Both were American" into a single named fact keeps a
    duplicate of facts already extracted; distributing it removes one.
    """
    from factflow.extract import EXTRACTION_SYSTEM

    flat = " ".join(EXTRACTION_SYSTEM.split())
    assert "The series has 40 books." in flat, "the failing case must appear as an example"
    for forbidden in ("the series", "the film", "a lone surname"):
        assert forbidden in flat, f"{forbidden!r} not named as forbidden"
    assert "handled by rule 1" in flat, "rule 2 must hand quantifiers off to rule 1"


def test_prompt_separates_siblings_from_subsets():
    from factflow.match import ADJUDICATION_SYSTEM

    flat = " ".join(ADJUDICATION_SYSTEM.split())
    assert "SIBLING CATEGORIES ARE NOT ENTAILMENT" in flat
    assert "poodle" in flat, "the genuine-subset counterexample must stay"
    assert "filmmaker" in flat, "the sibling counterexample must stay"


def test_audit_ignores_names_that_contain_and():
    """Long proper names broke both heuristics: "Science Fiction and Fantasy"
    read as a coordinated predicate, and two facts about one long-named entity
    read as near-duplicates. 236 of 279 flags on the first pass were this."""
    from factflow.audit import audit
    from factflow.types import CanonicalFact, Channel, FactMention, FactStore, Provenance

    store = FactStore()
    texts = [
        "The Andre Norton Award for Young Adult Science Fiction and Fantasy is presented by SFWA.",
        "Graphic novels are eligible for the Andre Norton Award for Young Adult Science Fiction and Fantasy.",
        "The core group of teenagers experience the rapture.",
        "The core group of teenagers experience the tribulation.",
        "The film Kiss and Tell was released in 1945.",
    ]
    for i, t in enumerate(texts):
        m = FactMention(mention_id=f"m{i}", text=t,
                        provenance=Provenance(agent_id="A", round=1, channel=Channel.OUTPUT))
        store.add_mentions([m])
        store.assign(CanonicalFact(fact_id=f"f{i}", canonical_text=t, mention_ids=[m.mention_id]))

    kinds = {f.kind for f in audit(store)}
    assert "unsplit-conjunction" not in kinds, "'Science Fiction and Fantasy' is a name, not a conjunction"
    assert "possible-missed-merge" not in kinds, "rapture vs tribulation are different facts"
    assert "unresolved-referent" not in kinds, "'The film Kiss and Tell' names its referent"


def test_quantified_sentence_is_an_atomicity_defect_not_a_reference_one():
    """"Both were American" is not under-specified, it is not atomic.

    It asserts one thing per member, and those per-member facts are normally
    already extracted from the same text - so emitting it too produces a
    duplicate that can never be matched or checked on its own. Resolving the
    referent would keep the duplicate; distributing removes it.
    """
    from factflow.audit import audit
    from factflow.extract import EXTRACTION_SYSTEM
    from factflow.types import CanonicalFact, Channel, FactMention, FactStore, Provenance

    flat = " ".join(EXTRACTION_SYSTEM.split())
    assert "A QUANTIFIER OVER A SET IS NOT AN ATOMIC FACT" in flat
    assert "Never emit the quantified sentence itself alongside them" in flat

    store = FactStore()
    for i, t in enumerate([
        "Both were American.",
        "All three studies found an effect.",
        "Both entries explicitly state their nationality as American.",
        "Scott Derrickson is American.",
    ]):
        m = FactMention(mention_id=f"m{i}", text=t,
                        provenance=Provenance(agent_id="A", round=1, channel=Channel.OUTPUT))
        store.add_mentions([m])
        store.assign(CanonicalFact(fact_id=f"f{i}", canonical_text=t, mention_ids=[m.mention_id]))

    by_fact = {f.fact_id: f.kind for f in audit(store)}
    assert by_fact.get("f0") == "quantified-aggregate"
    assert by_fact.get("f1") == "quantified-aggregate"
    assert "f3" not in by_fact, "a proper atomic fact must not be flagged"


def test_adjudication_prompt_separates_wording_from_information():
    """The defect that made hotpot-1 report 0% gold retention.

    The adjudicator split one fact across four clusters -- "was named ambassador
    to Ghana", "held the position of ambassador to Ghana", "was later named ...",
    "was ... ambassador to Ghana" -- each a defensible technical distinction and
    each wrong for fact tracking. Every gold source fact then read as never
    expressed, because the source phrasing ("Shirley Temple Black ...") sat in
    its own cluster.
    """
    from factflow.match import ADJUDICATION_SYSTEM

    flat = " ".join(ADJUDICATION_SYSTEM.split())
    assert "ENTAILMENT REQUIRES A DIFFERENCE IN INFORMATION, NOT IN WORDING" in flat
    assert "name the specific value, scope, condition, or qualifier" in flat, (
        "the rule needs the naming test, not just the slogan"
    )
    for example in ("held the position of", "was later named", "Shirley Temple Black"):
        assert example in flat, f"{example!r} missing as a worked EQUIVALENT example"
    assert "same row in a database" in flat, "the operational test must stay"
