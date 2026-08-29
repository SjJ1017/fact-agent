"""End-to-end checks on the three stages, using FakeLLM in place of the API."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fake_llm import FakeLLM  # noqa: E402

from factflow import (  # noqa: E402
    Channel,
    FactStore,
    PropertySpec,
    Relation,
    Provenance,
    TraceRecord,
    annotate_store,
    extract_trace,
    match,
)

# Three phrasings of one proposition, one weakened variant, one contradiction.
F_FULL = "UBI reduced poverty by 12.3% in the Kenya pilot study"
F_PARA = "The Kenya pilot study found a 12.3% poverty reduction under UBI"
F_WEAK = "UBI reduces poverty"
F_OPPOSITE = "UBI did not reduce poverty in the Kenya pilot study"
F_OTHER = "The Kenya pilot study ran for two years"

RAW = {
    "agentA-r1": [
        {"text": F_FULL, "polarity": "affirm", "qualifiers": ["in the Kenya pilot study"]},
        {"text": F_OTHER, "polarity": "affirm", "qualifiers": []},
    ],
    "agentB-r1": [{"text": F_PARA, "polarity": "affirm", "qualifiers": ["Kenya pilot study"]}],
    "agentA-r2": [{"text": F_WEAK, "polarity": "affirm", "qualifiers": []}],
    "agentB-r2": [{"text": F_OPPOSITE, "polarity": "negate", "qualifiers": []}],
}

# The production default (threshold .75) is tuned for real stores; this fixture is
# four short sentences where lexical overlap is low, so the clustering logic is
# exercised at a threshold that lets the pairs through. Blocking itself is tested
# separately in test_blocking.py.
TEST_THRESHOLD = 0.30

FAKE = dict(
    extractions=RAW,
    equivalent={frozenset((F_FULL, F_PARA))},
    entails={(F_FULL, F_WEAK), (F_PARA, F_WEAK)},
    contradicts={frozenset((F_FULL, F_OPPOSITE)), frozenset((F_PARA, F_OPPOSITE))},
)


def build_trace(execution_id: str = "run-0") -> list[TraceRecord]:
    return [
        TraceRecord(
            text=key,
            provenance=Provenance(
                execution_id=execution_id,
                agent_id=key[5],
                round=int(key[-1]),
                channel=Channel.OUTPUT,
            ),
        )
        for key in RAW
    ]


def test_extract_produces_decontextualised_mentions():
    llm = FakeLLM(**FAKE)
    mentions = extract_trace(llm, build_trace())
    assert len(mentions) == 5
    assert {m.provenance.agent_id for m in mentions} == {"A", "B"}
    assert {m.provenance.round for m in mentions} == {1, 2}
    full = next(m for m in mentions if m.text == F_FULL)
    assert full.qualifiers == ["in the Kenya pilot study"]


def test_in_context_and_in_output_lookups():
    llm = FakeLLM(**FAKE)
    store = match(llm, extract_trace(llm, build_trace()), threshold=TEST_THRESHOLD,
                  union_min_similarity=TEST_THRESHOLD)
    fid = next(
        fid
        for fid, f in store.facts.items()
        if {store.mentions[m].text for m in f.mention_ids} >= {F_FULL, F_PARA}
    )
    assert store.in_output(fid, "A", 1, "run-0") is True
    assert store.in_output(fid, "B", 1, "run-0") is True
    assert store.in_output(fid, "A", 2, "run-0") is False  # A weakened it in round 2
    assert store.in_context(fid, "A", 1, "run-0") is False  # no context records supplied


def test_ids_stay_stable_across_executions():
    llm = FakeLLM(**FAKE)
    store = match(llm, extract_trace(llm, build_trace("run-0")), threshold=TEST_THRESHOLD,
                  union_min_similarity=TEST_THRESHOLD)
    before = set(store.facts)

    store = match(llm, extract_trace(llm, build_trace("run-1")), store=store,
                  threshold=TEST_THRESHOLD, union_min_similarity=TEST_THRESHOLD)
    assert set(store.facts) == before, "a repeated run must not mint new fact ids"
    assert store.executions() == ["run-0", "run-1"]

    fid = next(iter(before))
    assert len(store.facts[fid].mention_ids) > 1


def test_store_roundtrips_through_disk(tmp_path):
    llm = FakeLLM(**FAKE)
    store = match(llm, extract_trace(llm, build_trace()), threshold=TEST_THRESHOLD,
                  union_min_similarity=TEST_THRESHOLD)
    path = tmp_path / "store.json"
    store.save(str(path))
    assert FactStore.load(str(path)).facts.keys() == store.facts.keys()


def test_property_annotation_writes_typed_values():
    llm = FakeLLM(
        **FAKE,
        annotator=lambda text: {
            "critical": "12.3%" in text,
            "kind": "measurement" if "12.3%" in text else "other",
        },
    )
    store = match(llm, extract_trace(llm, build_trace()), threshold=TEST_THRESHOLD,
                  union_min_similarity=TEST_THRESHOLD)
    store = annotate_store(
        llm,
        store,
        [
            PropertySpec(name="critical", description="Would omitting it change the answer?"),
            PropertySpec(
                name="kind",
                description="What kind of fact is it?",
                type="choice",
                choices=["measurement", "other"],
            ),
        ],
        instruction="Did the Kenya UBI pilot reduce poverty?",
    )
    props = [f.properties for f in store.facts.values()]
    assert all("critical" in p and "kind" in p for p in props)
    assert any(p["critical"] is True and p["kind"] == "measurement" for p in props)
    assert any(p["critical"] is False for p in props)

