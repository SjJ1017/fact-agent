"""Core data model.

The unit of observation is a *mention*: one atomic proposition as it appears in
one specific place (an execution, an agent, a round, and a channel).  Mentions
that express the same proposition are grouped into a `CanonicalFact`, which
carries a stable id across agents, rounds and repeated executions.

That grouping is what makes `InContext(f, A, t)` and `InOutput(f, A, t)`
computable: both are lookups on (canonical fact id, agent, round, channel).
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, Field


class Channel(str, Enum):
    """Where a mention was observed.

    CONTEXT — the proposition was available to the agent (in its prompt/context).
    OUTPUT  — the agent actually expressed the proposition in its own message.
    SOURCE  — the proposition comes from the task's ground-truth material
              (reference document, gold evidence, seeded fact pool).
    """

    CONTEXT = "context"
    OUTPUT = "output"
    SOURCE = "source"


class Polarity(str, Enum):
    """Whether the proposition asserts or denies its predicate.

    Kept explicit because negations ("no fever", "the policy does not cover
    flood damage") are semantically load-bearing but sit very close to their
    affirmative counterpart in embedding space.
    """

    AFFIRM = "affirm"
    NEGATE = "negate"


class Provenance(BaseModel):
    """Address of a mention inside a multi-agent trace."""

    execution_id: str = "exec-0"
    agent_id: Optional[str] = None
    round: Optional[int] = None
    channel: Channel = Channel.OUTPUT
    doc_id: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def key(self) -> tuple:
        return (self.execution_id, self.agent_id, self.round, self.channel.value)


class FactMention(BaseModel):
    """One atomic proposition observed at one place in the trace."""

    mention_id: str
    text: str
    polarity: Polarity = Polarity.AFFIRM
    quote: Optional[str] = None  # supporting span from the source text
    qualifiers: list[str] = Field(default_factory=list)  # conditions/hedges attached to this proposition
    provenance: Provenance = Field(default_factory=Provenance)

    @staticmethod
    def make_id(text: str, provenance: Provenance) -> str:
        payload = f"{text}|{provenance.key()}"
        return "m_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


RelationType = Literal[
    "EQUIVALENT",  # same propositional content
    "A_ENTAILS_B",  # A is strictly more specific; B is a weakened form of A
    "B_ENTAILS_A",  # B is strictly more specific
    "CONTRADICTS",  # mutually incompatible
    "UNRELATED",  # different propositions
]


class Relation(BaseModel):
    """A judged relation between two mentions.

    Only EQUIVALENT edges are used to build canonical clusters.  The entailment
    directions are kept because they are the signal for *degradation*: a fact
    that survives as `B_ENTAILS_A` rather than `EQUIVALENT` is still present but
    weaker (e.g. "12.3%" restated as "about 10%").
    """

    a: str  # mention_id
    b: str  # mention_id
    relation: RelationType
    confidence: float = 1.0
    rationale: Optional[str] = None
    # Whatever the judge produced before the label was derived.  For the NLI
    # form that is the two directional margins at full precision: the label is
    # a thresholded view of them, and re-deriving it under a different cutoff
    # must not require re-running the model.
    properties: dict[str, Any] = Field(default_factory=dict)


class CanonicalFact(BaseModel):
    """A cluster of mentions expressing the same proposition."""

    fact_id: str
    canonical_text: str
    polarity: Polarity = Polarity.AFFIRM
    mention_ids: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_id(canonical_text: str) -> str:
        norm = " ".join(canonical_text.lower().split())
        return "f_" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


class FactStore(BaseModel):
    """Registry of every mention and canonical fact seen so far.

    Persisting this across executions is what keeps fact ids stable between
    repeated runs of the same data entry.
    """

    mentions: dict[str, FactMention] = Field(default_factory=dict)
    facts: dict[str, CanonicalFact] = Field(default_factory=dict)
    mention_to_fact: dict[str, str] = Field(default_factory=dict)
    relations: list[Relation] = Field(default_factory=list)
    accounting: dict[str, Any] = Field(default_factory=dict)

    # -- population ---------------------------------------------------------

    def add_mentions(self, mentions: Iterable[FactMention]) -> None:
        for m in mentions:
            self.mentions[m.mention_id] = m

    def assign(self, fact: CanonicalFact) -> None:
        self.facts[fact.fact_id] = fact
        for mid in fact.mention_ids:
            self.mention_to_fact[mid] = fact.fact_id

    # -- lifecycle queries --------------------------------------------------

    def observed(
        self,
        fact_id: str,
        agent_id: str,
        round_: int,
        channel: Channel,
        execution_id: str = "exec-0",
    ) -> bool:
        """The indicator behind InContext / InOutput."""
        fact = self.facts.get(fact_id)
        if fact is None:
            return False
        for mid in fact.mention_ids:
            p = self.mentions[mid].provenance
            if (
                p.execution_id == execution_id
                and p.agent_id == agent_id
                and p.round == round_
                and p.channel == channel
            ):
                return True
        return False

    def in_context(self, fact_id: str, agent_id: str, round_: int, execution_id: str = "exec-0") -> bool:
        return self.observed(fact_id, agent_id, round_, Channel.CONTEXT, execution_id)

    def in_output(self, fact_id: str, agent_id: str, round_: int, execution_id: str = "exec-0") -> bool:
        return self.observed(fact_id, agent_id, round_, Channel.OUTPUT, execution_id)

    def agents(self, execution_id: str | None = None) -> list[str]:
        out = {
            m.provenance.agent_id
            for m in self.mentions.values()
            if m.provenance.agent_id is not None
            and (execution_id is None or m.provenance.execution_id == execution_id)
        }
        return sorted(out)

    def rounds(self, execution_id: str | None = None) -> list[int]:
        out = {
            m.provenance.round
            for m in self.mentions.values()
            if m.provenance.round is not None
            and (execution_id is None or m.provenance.execution_id == execution_id)
        }
        return sorted(out)

    def executions(self) -> list[str]:
        return sorted({m.provenance.execution_id for m in self.mentions.values()})

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str) -> "FactStore":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate_json(f.read())
