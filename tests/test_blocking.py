"""Blocking must rank degradation pairs above contradictions.

This is the property the whole typed-relation design depends on, and plain
cosine gets it backwards, so it is pinned here rather than left to the
end-to-end test.
"""

from factflow.blocking import TfidfBlocker, candidate_pairs, containment, tokenize
from factflow.types import Channel, FactMention, Provenance

SPECIFIC = "UBI reduced poverty by 12.3% in the Kenya pilot study"
WEAKENED = "UBI reduces poverty"
NEGATED = "UBI did not reduce poverty in the Kenya pilot study"
UNRELATED = "The Kenya pilot study ran for two years"
ALL = [SPECIFIC, WEAKENED, NEGATED, UNRELATED]


def _mentions(texts):
    return [
        FactMention(
            mention_id=f"m{i}",
            text=t,
            provenance=Provenance(agent_id=str(i), round=i, channel=Channel.OUTPUT),
        )
        for i, t in enumerate(texts)
    ]


def test_containment_is_asymmetric_and_catches_nesting():
    assert containment(tokenize(SPECIFIC), tokenize(WEAKENED)) == 1.0
    assert containment(tokenize(WEAKENED), tokenize(UNRELATED)) == 0.0


def test_raw_cosine_ranks_degradation_below_contradiction():
    # The motivating failure: without containment the pair we care about is lost.
    blocker = TfidfBlocker()
    emb = blocker.encode(ALL)
    sim = blocker.similarity(emb, emb)
    assert sim[0][1] < sim[0][2], "cosine puts the negation closer than the weakening"
    assert sim[0][1] < 0.30, "and below a plausible default threshold"


def test_blended_score_recovers_the_degradation_pair():
    pairs = {(i, j): s for i, j, s in candidate_pairs(_mentions(ALL))}
    assert (0, 1) in pairs, "specific x weakened must survive blocking"
    assert pairs[(0, 1)] >= pairs[(0, 2)], "and outrank the contradiction"


def test_same_slot_pairs_are_skipped():
    dup = [
        FactMention(
            mention_id=f"m{i}",
            text=t,
            provenance=Provenance(agent_id="A", round=1, channel=Channel.OUTPUT),
        )
        for i, t in enumerate([SPECIFIC, WEAKENED])
    ]
    assert candidate_pairs(dup) == []
