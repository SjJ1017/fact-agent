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
