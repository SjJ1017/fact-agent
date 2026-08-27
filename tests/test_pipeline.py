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


def test_paraphrases_merge_but_weakened_variant_does_not():
    llm = FakeLLM(**FAKE)
    store = match(llm, extract_trace(llm, build_trace()), threshold=TEST_THRESHOLD)

    texts_by_fact = {
        fid: {store.mentions[m].text for m in f.mention_ids} for fid, f in store.facts.items()
    }
    merged = [t for t in texts_by_fact.values() if {F_FULL, F_PARA} <= t]
    assert len(merged) == 1, "the two paraphrases must land in one cluster"

    # A weakened restatement is a different canonical fact, not the same one.
    assert F_WEAK not in merged[0]
    # A contradiction is never merged either.
    assert F_OPPOSITE not in merged[0]
    assert len(store.facts) == 4  # {FULL,PARA}, WEAK, OPPOSITE, OTHER


def test_entailment_edges_are_retained_for_degradation_analysis():
    llm = FakeLLM(**FAKE)
    store = match(llm, extract_trace(llm, build_trace()), threshold=TEST_THRESHOLD)
    kinds = {r.relation for r in store.relations}
    assert "A_ENTAILS_B" in kinds or "B_ENTAILS_A" in kinds
    assert "CONTRADICTS" in kinds


def test_in_context_and_in_output_lookups():
    llm = FakeLLM(**FAKE)
    store = match(llm, extract_trace(llm, build_trace()), threshold=TEST_THRESHOLD)
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
    store = match(llm, extract_trace(llm, build_trace("run-0")), threshold=TEST_THRESHOLD)
    before = set(store.facts)

    store = match(llm, extract_trace(llm, build_trace("run-1")), store=store, threshold=TEST_THRESHOLD)
    assert set(store.facts) == before, "a repeated run must not mint new fact ids"
    assert store.executions() == ["run-0", "run-1"]

    fid = next(iter(before))
    assert len(store.facts[fid].mention_ids) > 1


def test_store_roundtrips_through_disk(tmp_path):
    llm = FakeLLM(**FAKE)
    store = match(llm, extract_trace(llm, build_trace()), threshold=TEST_THRESHOLD)
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
    store = match(llm, extract_trace(llm, build_trace()), threshold=TEST_THRESHOLD)
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


def test_guard_keeps_cluster_when_a_judgement_is_missing():
    """A dropped adjudication must not split an already-merged cluster.

    Regression: the guard defaulted to splitting whenever the representative
    check was absent, so one dropped batch shattered a correct cluster. That
    failed silently - the fact count rose and every retention number computed
    from it fell.
    """
    from factflow.match import cluster
    from factflow.types import Channel, FactMention, Provenance

    texts = [
        "Shirley Temple played Corliss Archer in the 1945 film Kiss and Tell",
        "The 1945 film Kiss and Tell starred Shirley Temple as Corliss Archer",
        "Kiss and Tell (1945) cast Shirley Temple in the role of Corliss Archer",
        "In Kiss and Tell, released 1945, Corliss Archer was played by Shirley Temple",
    ]
    mentions = [
        FactMention(
            mention_id=f"m{i}",
            text=t,
            provenance=Provenance(agent_id="ABCD"[i], round=1, channel=Channel.OUTPUT),
        )
        for i, t in enumerate(texts)
    ]
    # Every pair judged EQUIVALENT: one component of four.
    relations = [
        Relation(a=f"m{i}", b=f"m{j}", relation="EQUIVALENT")
        for i in range(4)
        for j in range(i + 1, 4)
    ]

    # A guard that can answer nothing - simulating dropped batches.
    class SilentLLM:
        def parse(self, **_kw):
            raise AssertionError("guard should not need to re-adjudicate known pairs")

        def map(self, fn, items):
            return []

    facts, _extra = cluster(SilentLLM(), mentions, relations, transitivity_guard=True)
    assert len(facts) == 1, f"expected one cluster, got {len(facts)}"
    assert len(facts[0].mention_ids) == 4


def test_guard_still_splits_on_explicit_disagreement():
    from factflow.match import cluster
    from factflow.types import Channel, FactMention, Provenance

    texts = ["X happened in 1945", "X happened in 1945", "Y happened in 1945"]
    mentions = [
        FactMention(
            mention_id=f"m{i}",
            text=t,
            provenance=Provenance(agent_id="ABC"[i], round=1, channel=Channel.OUTPUT),
        )
        for i, t in enumerate(texts)
    ]
    relations = [
        Relation(a="m0", b="m1", relation="EQUIVALENT"),
        Relation(a="m1", b="m2", relation="EQUIVALENT"),  # the bad transitive link
        Relation(a="m0", b="m2", relation="UNRELATED"),  # representative disagrees
    ]

    class SilentLLM:
        def parse(self, **_kw):
            raise AssertionError("no re-adjudication needed")

        def map(self, fn, items):
            return []

    facts, _ = cluster(SilentLLM(), mentions, relations, transitivity_guard=True)
    assert len(facts) == 2, "an explicit disagreement must still split"
