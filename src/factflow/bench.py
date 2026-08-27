"""A small gold probe set for comparing models on the two judgement tasks.

Scope, stated plainly: these probes were written from defects observed in real
runs, so they are biased toward the ways extraction and adjudication are already
known to break. A model scoring well here is not verified in general - it has
cleared the failures that have actually cost us numbers. That is a lower bar
than a proper benchmark and a much higher bar than nothing, which is what the
pipeline had before.

Two tasks are scored separately because they fail differently and can be served
by different models: extraction is a decomposition task where the failure is
under-splitting, and adjudication is a 5-way classification where the failure is
false EQUIVALENT edges that union-find then amplifies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .extract import extract_facts
from .llm import LLM
from .match import adjudicate
from .types import Channel, FactMention, Polarity, Provenance, RelationType

FORBIDDEN_OPENERS = re.compile(
    r"^(both|he|she|they|it|this|these|those)\b|^the (series|film|study|book|author|documents?)\s+(is|was|are|were|has|had|consists)",
    re.I,
)


@dataclass
class ExtractionProbe:
    name: str
    text: str
    why: str
    exact_facts: int | None = None
    min_facts: int | None = None
    max_facts: int | None = None
    each_appears: Sequence[str] = ()        # each string must appear in SOME fact
    never_appears: Sequence[str] = ()       # must appear in NO fact
    verbatim: Sequence[str] = ()            # must survive character-for-character
    no_unresolved: bool = True
    negated: int | None = None              # expected count of polarity=negate

    def score(self, facts: list[FactMention]) -> tuple[int, int, list[str]]:
        checks: list[tuple[bool, str]] = []
        texts = [f.text for f in facts]
        joined = " || ".join(texts)

        if self.exact_facts is not None:
            checks.append((len(facts) == self.exact_facts, f"expected {self.exact_facts} facts, got {len(facts)}"))
        if self.min_facts is not None:
            checks.append((len(facts) >= self.min_facts, f"expected >={self.min_facts} facts, got {len(facts)}"))
        if self.max_facts is not None:
            checks.append((len(facts) <= self.max_facts, f"expected <={self.max_facts} facts, got {len(facts)}"))
        for s in self.each_appears:
            checks.append((any(s.lower() in t.lower() for t in texts), f"no fact mentions {s!r}"))
        for s in self.never_appears:
            checks.append((not any(s.lower() in t.lower() for t in texts), f"a fact still contains {s!r}"))
        for s in self.verbatim:
            checks.append((s in joined, f"{s!r} was not preserved verbatim"))
        if self.no_unresolved:
            bad = [t for t in texts if FORBIDDEN_OPENERS.match(t)]
            checks.append((not bad, f"unresolved referent: {bad[:1]}"))
        if self.negated is not None:
            n = sum(1 for f in facts if f.polarity == Polarity.NEGATE)
            checks.append((n == self.negated, f"expected {self.negated} negated facts, got {n}"))

        passed = sum(1 for ok, _ in checks if ok)
        return passed, len(checks), [msg for ok, msg in checks if not ok]


@dataclass
class RelationProbe:
    name: str
    a: str
    b: str
    gold: RelationType
    why: str


EXTRACTION_PROBES: list[ExtractionProbe] = [
    ExtractionProbe(
        "split-4-way", "Scott Derrickson is an American director, screenwriter and producer.",
        "the original bug: a coordinated predicate kept whole creates phantom degradation edges",
        min_facts=3, each_appears=["screenwriter", "producer"],
    ),
    ExtractionProbe(
        "split-verb-phrases", "The Kenya trial ran from 2018 to 2020 and paid recipients $22 per month.",
        "coordinated verb phrases are two facts", exact_facts=2, verbatim=["$22", "2018", "2020"],
    ),
    ExtractionProbe(
        "keep-name-whole",
        "The Andre Norton Award is presented by the Science Fiction and Fantasy Writers of America.",
        "'and' inside a proper name is not a conjunction to split",
        max_facts=2, each_appears=["Science Fiction and Fantasy Writers of America"],
    ),
    ExtractionProbe(
        "keep-idiom-whole", "Plan 9 from Outer Space is a black and white film shot in Bosnia and Herzegovina.",
        "non-separable phrases must survive", max_facts=3,
        each_appears=["black and white"], never_appears=["is black."],
    ),
    ExtractionProbe(
        "resolve-both", "Scott Derrickson and Ed Wood are both American. Both were filmmakers.",
        "'Both were filmmakers' is unmatchable unless the referents are named",
        min_facts=4, each_appears=["Scott Derrickson", "Ed Wood"], never_appears=["Both were", "Both are"],
    ),
    ExtractionProbe(
        "resolve-definite", "Animorphs is a book series. The series consists of 40 short novels.",
        "'The series' must be resolved to the named entity",
        min_facts=2, each_appears=["Animorphs"], verbatim=["40"],
    ),
    ExtractionProbe(
        "resolve-pronoun", "Tim Burton directed the biopic. He also produced it.",
        "pronoun subject and object both need resolving",
        min_facts=2, each_appears=["Tim Burton"], never_appears=["He also", "produced it"],
    ),
    ExtractionProbe(
        "numbers-verbatim",
        "Household poverty fell by 12.3% among recipients, from a baseline of 41.7% in March 2019.",
        "rounding a figure is a silent factual error", verbatim=["12.3%", "41.7%", "March 2019"],
    ),
    ExtractionProbe(
        "negation-is-a-fact", "The patient had no fever and no chest pain on admission.",
        "pertinent negatives carry diagnostic information and must survive as facts",
        min_facts=2, negated=2, each_appears=["fever", "chest pain"],
    ),
    ExtractionProbe(
        "qualifier-attached", "UBI reduces poverty, but only in pilot studies with fewer than 500 households.",
        "a claim whose scope condition is dropped becomes a stronger claim than the source made",
        each_appears=["pilot"], verbatim=["500"],
    ),
    ExtractionProbe(
        "apposition", "Ed Wood, an American filmmaker, directed Plan 9 from Outer Space.",
        "apposition packs two assertions into one clause",
        min_facts=2, each_appears=["filmmaker", "Plan 9"],
    ),
    ExtractionProbe(
        "no-meta-discourse", "The documents state that Shirley Temple was ambassador to Ghana.",
        "the fact is about the world, not about the documents",
        never_appears=["The documents state"], each_appears=["Ghana"],
    ),
    ExtractionProbe(
        "unresolvable-still-emitted", "The award was presented at the 2019 ceremony.",
        "dropping an under-specified fact hides it; emitting it leaves it auditable",
        min_facts=1, no_unresolved=False, verbatim=["2019"],
    ),
    ExtractionProbe(
        "no-invention", "Kiss and Tell was released in 1945.",
        "nothing beyond the text may be added", exact_facts=1, verbatim=["1945"],
    ),
    ExtractionProbe(
        "skip-questions", "Should governments implement UBI? Kenya ran a trial in 2018.",
        "a question is not a proposition", exact_facts=1, never_appears=["Should governments"],
    ),
]

RELATION_PROBES: list[RelationProbe] = [
    RelationProbe("paraphrase", "Shirley Temple was named US ambassador to Ghana.",
                  "Shirley Temple was appointed United States ambassador to Ghana.",
                  "EQUIVALENT", "pure rewording"),
    RelationProbe("word-order", "The 1945 film Kiss and Tell starred Shirley Temple as Corliss Archer.",
                  "Shirley Temple played Corliss Archer in the 1945 film Kiss and Tell.",
                  "EQUIVALENT", "reordering is not a content change"),
    RelationProbe("specificity-drop", "Scott Derrickson is an American director.",
                  "Scott Derrickson is a director.", "A_ENTAILS_B", "the modifier is dropped"),
    RelationProbe("qualifier-drop", "The drug reduces mortality in patients over 65.",
                  "The drug reduces mortality.", "A_ENTAILS_B", "the scope condition is dropped"),
    RelationProbe("value-drop", "Revenue grew 12.3% in Q4.", "Revenue grew.",
                  "A_ENTAILS_B", "the figure is dropped"),
    RelationProbe("hypernym", "Rex is a poodle.", "Rex is a dog.",
                  "A_ENTAILS_B", "a genuine subset relation must still be found"),
    RelationProbe("reversed-direction", "Scott Derrickson is a director.",
                  "Scott Derrickson is an American director.",
                  "B_ENTAILS_A", "direction matters and is easy to invert"),
    RelationProbe("siblings-roles", "Ed Wood is a filmmaker.", "Ed Wood is a director.",
                  "UNRELATED", "overlapping professions are not entailment"),
    RelationProbe("siblings-genre", "Animorphs is science fiction.", "Animorphs is fantasy.",
                  "UNRELATED", "adjacent categories are not entailment"),
    RelationProbe("negation", "The patient has a fever.", "The patient has no fever.",
                  "CONTRADICTS", "opposite polarity, same predicate"),
    RelationProbe("negation-scoped", "UBI reduced poverty in the Kenya pilot.",
                  "UBI did not reduce poverty in the Kenya pilot.",
                  "CONTRADICTS", "must not be scored UNRELATED"),
    RelationProbe("incompatible-values", "The trial enrolled 500 households.",
                  "The trial enrolled 5,000 households.",
                  "CONTRADICTS", "incompatible figures for one quantity"),
    RelationProbe("same-entity-diff-predicate", "Shirley Temple was ambassador to Ghana.",
                  "Shirley Temple was ambassador to Czechoslovakia.",
                  "UNRELATED", "same subject, different object"),
    RelationProbe("same-entity-diff-role", "Shirley Temple served as Chief of Protocol.",
                  "Kiss and Tell (1945) starred Shirley Temple as Corliss Archer.",
                  "UNRELATED", "the exact false EQUIVALENT that produced a 20-member blob"),
    RelationProbe("different-subjects", "Jara Hamee's story is narrated by Aldrea.",
                  "The Hork-Bajir Chronicles is narrated by Aldrea.",
                  "UNRELATED", "shared predicate, different subject"),
    RelationProbe("distinct-events", "The core group experience the rapture.",
                  "The core group experience the tribulation.",
                  "UNRELATED", "one content word apart, but different facts"),
    RelationProbe("name-variant", "Shirley Temple Black was named ambassador to Ghana.",
                  "Shirley Temple was named ambassador to Ghana.",
                  "EQUIVALENT", "a name variant is the same person"),
    RelationProbe("abbrev-variant", "Edward Davis Wood Jr. was an American filmmaker.",
                  "Ed Wood was an American filmmaker.",
                  "EQUIVALENT", "full name vs common name"),
    RelationProbe("unrelated-topics", "The patwari is a village accountant in Telangana.",
                  "Shirley Temple played Corliss Archer.",
                  "UNRELATED", "nothing in common"),
    RelationProbe("precision-loss-numeric", "Poverty fell by 12.3%.", "Poverty fell by about 40%.",
                  "CONTRADICTS", "the two figures cannot both describe one quantity"),
    RelationProbe("precision-loss-compatible", "Poverty fell by 12.3%.", "Poverty fell by roughly 12%.",
                  "A_ENTAILS_B", "a compatible rounding is a weakening, not a contradiction"),
]


@dataclass
class BenchResult:
    model: str
    extraction_passed: int = 0
    extraction_total: int = 0
    relation_correct: int = 0
    relation_total: int = 0
    false_equivalent: int = 0        # non-EQUIVALENT gold scored EQUIVALENT: the costly error
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    failures: list[str] = field(default_factory=list)
    per_relation: list[dict[str, Any]] = field(default_factory=list)

    @property
    def extraction_score(self) -> float:
        return self.extraction_passed / self.extraction_total if self.extraction_total else 0.0

    @property
    def relation_score(self) -> float:
        return self.relation_correct / self.relation_total if self.relation_total else 0.0


def run_bench(llm: LLM, model_label: str | None = None) -> BenchResult:
    import time

    label = model_label or llm.model
    res = BenchResult(model=label)
    t0 = time.time()

    for probe in EXTRACTION_PROBES:
        try:
            facts = extract_facts(llm, probe.text, Provenance(agent_id="bench", round=1))
        except Exception as exc:  # noqa: BLE001
            res.extraction_total += 1
            res.failures.append(f"[{probe.name}] call failed: {type(exc).__name__}: {exc}")
            continue
        ok, total, msgs = probe.score(facts)
        res.extraction_passed += ok
        res.extraction_total += total
        for m in msgs:
            res.failures.append(f"[{probe.name}] {m}")

    mentions, pairs = [], []
    for i, probe in enumerate(RELATION_PROBES):
        mentions += [
            FactMention(mention_id=f"a{i}", text=probe.a,
                        provenance=Provenance(agent_id="A", round=1, channel=Channel.OUTPUT)),
            FactMention(mention_id=f"b{i}", text=probe.b,
                        provenance=Provenance(agent_id="B", round=1, channel=Channel.OUTPUT)),
        ]
        pairs.append((2 * i, 2 * i + 1, 1.0))

    try:
        rels = adjudicate(llm, mentions, pairs, batch_size=5)
        got = {frozenset((r.a, r.b)): r.relation for r in rels}
    except Exception as exc:  # noqa: BLE001
        res.failures.append(f"[adjudicate] {type(exc).__name__}: {exc}")
        got = {}

    for i, probe in enumerate(RELATION_PROBES):
        res.relation_total += 1
        actual = got.get(frozenset((f"a{i}", f"b{i}")))
        correct = actual == probe.gold
        res.relation_correct += correct
        if not correct and actual == "EQUIVALENT":
            res.false_equivalent += 1
        res.per_relation.append(
            {"name": probe.name, "gold": probe.gold, "got": actual, "correct": correct, "why": probe.why}
        )

    res.seconds = time.time() - t0
    res.input_tokens = llm.usage.input_tokens
    res.output_tokens = llm.usage.output_tokens
    return res
