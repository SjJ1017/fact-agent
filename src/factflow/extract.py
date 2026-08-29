"""(1) Text -> atomic facts.

Decontextualization is the load-bearing requirement, not atomicity.  "She is an
actress" cannot be matched against anything; "Bridget Moynahan is an actress"
can.  Every downstream stage assumes each mention stands alone, so the prompt
spends most of its budget on resolving references and preserving exact values.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field

from .llm import LLM
from .types import FactMention, Polarity, Provenance

EXTRACTION_SYSTEM = """\
You decompose text into atomic facts for a fact-tracking system.

An atomic fact is a short, self-contained, verifiable proposition expressing a \
single piece of information.

Rules:
1. ATOMIC - ONE predicate per fact. A coordinated predicate is several facts, not one. Split every list, however natural it reads as a phrase:

   "Scott Derrickson is an American director, screenwriter and producer."
     -> Scott Derrickson is American.
     -> Scott Derrickson is a director.
     -> Scott Derrickson is a screenwriter.
     -> Scott Derrickson is a producer.

   "The trial ran from 2018 to 2020 and paid recipients $22 per month."
     -> The trial ran from 2018 to 2020.
     -> The trial paid recipients $22 per month.

   This applies to apposition too: "Ed Wood, an American filmmaker, directed Plan 9" is two facts - that Ed Wood is an American filmmaker, and that he directed Plan 9.

   A QUANTIFIER OVER A SET IS NOT AN ATOMIC FACT. "Both", "all", "each", "neither", "the two", "they" assert one thing per member, so emit one fact per member, naming it:

   "Scott Derrickson and Ed Wood are both American."
     -> Scott Derrickson is American.
     -> Ed Wood is American.

   Never emit the quantified sentence itself alongside them. "Both were American" adds nothing once the two per-member facts exist, it cannot be matched or checked on its own, and it duplicates content already covered. If the per-member facts are already among the facts you are emitting, emit nothing further for the quantified sentence.

   Do NOT split a phrase whose parts are not separately assertable. "black and white film", "trial and error", "Bosnia and Herzegovina", and a joint action that only holds jointly ("A and B co-founded the company") each stay whole.
2. SELF-CONTAINED - resolve every pronoun, definite description, quantifier, \
and relative time expression against the surrounding text. A reader who sees \
ONLY the fact, with no other context, must be able to check it.

   "The series has 40 books."      -> The Animorphs series has 40 books.
   "It was released last year."    -> Kiss and Tell was released in 1945.
   "He directed the film."         -> Tim Burton directed Ed Wood.

   Bare "the series", "the film", "the study", "the author", "this", "the \
above", "last year", and a lone surname are all FORBIDDEN as the SUBJECT of an \
emitted fact. (Quantified subjects - "both", "all", "they" - are handled by \
rule 1: distribute them, do not resolve them into a single fact.)

   If a referent genuinely cannot be resolved from the text, still emit the \
fact, using the most specific description the text supports ("the award" -> \
"the award presented by SFWA"). Never drop a fact for this reason: a dropped \
fact is invisible downstream, whereas an under-specified one can at least be \
seen and corrected.
3. FAITHFUL - copy numbers, dates, units, names, and magnitudes EXACTLY as \
written. Never round, convert, or approximate. If the source says "12.3%", the \
fact says "12.3%".
4. NEGATION IS INFORMATION - an explicit denial ("no fever was observed", \
"the policy does not cover flood damage") is a fact. Emit it with \
polarity="negate" and phrase the text as the negated proposition.
5. QUALIFIERS STAY ATTACHED - conditions, scope limits, and hedges that change \
the truth of the proposition ("in patients over 65", "only in pilot studies", \
"if funded by tax reform") remain part of the fact text AND are listed \
separately in the `qualifiers` field.
6. NO INVENTION - only propositions supported by the text. Do not add world \
knowledge, do not infer.
7. SKIP non-propositional content: questions, instructions, greetings, and \
pure meta-discourse about the conversation itself.
8. SKIP statements ABOUT the source material rather than about the world. \
"Both entries explicitly state their nationality as American", "The documents \
describe Animorphs as a science fantasy series", and "No other person linked to \
Corliss Archer held a government post in the documents" are claims about what \
the text says, not facts. Emit the underlying world fact if there is one, and \
nothing otherwise.

For each fact also return `quote`: the shortest verbatim span from the source \
text that supports it.
"""

DISCOURSE_CLAUSE = """\

EXCEPTION to rule 7: this text is one turn in a multi-agent conversation. \
Statements about other participants' positions or about the state of the \
discussion ("Agent B argues that X", "the group has not settled whether Y") \
ARE propositions - emit them, phrased with the participant named explicitly.
"""

FOCUS_CLAUSE = """\

The text is being analysed in relation to this question:
{focus}

Extract ALL atomic facts, not only those relevant to the question. Relevance is \
judged in a later stage; over-filtering here destroys information permanently.
"""


class ExtractedFact(BaseModel):
    text: str = Field(description="The self-contained atomic proposition.")
    polarity: Literal["affirm", "negate"] = "affirm"
    qualifiers: list[str] = Field(
        default_factory=list,
        description="Conditions or scope limits that modify this proposition.",
    )
    quote: Optional[str] = Field(
        default=None, description="Shortest verbatim supporting span from the source."
    )


class ExtractionResult(BaseModel):
    facts: list[ExtractedFact]


def extract_facts(
    llm: LLM,
    text: str,
    provenance: Provenance | None = None,
    focus: str | None = None,
    include_discourse: bool = False,
) -> list[FactMention]:
    """Extract atomic facts from a single piece of text.

    `focus` steers phrasing toward a task question without filtering; leave it
    None for a task-agnostic pass.
    """
    if not text or not text.strip():
        return []

    provenance = provenance or Provenance()
    system = EXTRACTION_SYSTEM
    if include_discourse:
        system += DISCOURSE_CLAUSE
    if focus:
        system += FOCUS_CLAUSE.format(focus=focus)

    result = llm.parse(
        system=system,
        user=f"<text>\n{text}\n</text>",
        output_format=ExtractionResult,
        # Zero facts from a non-empty text means the call failed, not that the
        # text was factless. Caching that would make the failure permanent.
        cache_if=lambda r: bool(r.facts),
    )

    mentions: list[FactMention] = []
    seen: set[str] = set()
    for f in result.facts:
        norm = " ".join(f.text.lower().split())
        if not norm or norm in seen:
            continue
        seen.add(norm)
        mentions.append(
            FactMention(
                mention_id=FactMention.make_id(f.text, provenance),
                text=f.text.strip(),
                polarity=Polarity(f.polarity),
                qualifiers=f.qualifiers,
                quote=f.quote,
                provenance=provenance,
            )
        )
    return mentions


class TraceRecord(BaseModel):
    """One observation slot in a trace: some text at a (execution, agent, round, channel)."""

    text: str
    provenance: Provenance


def extract_trace(
    llm: LLM,
    records: Sequence[TraceRecord],
    focus: str | None = None,
    include_discourse: bool = True,
) -> list[FactMention]:
    """Extract from every record in a trace, concurrently."""

    def _one(rec: TraceRecord) -> list[FactMention]:
        return extract_facts(
            llm, rec.text, rec.provenance, focus=focus, include_discourse=include_discourse
        )

    batches = llm.map(_one, list(records))
    return [m for batch in batches for m in batch]
