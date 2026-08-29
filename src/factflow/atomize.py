"""(1b) Splitting facts the extractor left joined.

An extraction prompt that does not insist on atomicity produces facts like

    "E3 and E4 argue against banning the veil on slippery-slope and
     trust-in-government grounds."

which is four claims wearing one sentence. A third of the Perspectrum pilot's
2914 facts carried a conjunction, and the damage lands two stages downstream:
two turns that assert the same things never produce an exact repeat, so the
matcher is handed entailment judgements where it should have been handed set
membership. Once split, "E3 argues against banning the veil" simply recurs, and
recurrence needs no adjudicator at all.

Doing this after extraction rather than inside it is possible because extracted
facts are already decontextualised. Each one stands alone, so splitting it needs
nothing but the sentence itself - which also makes this the cheapest LLM call in
the pipeline, and the one place where a store produced by someone else's prompt
can still be repaired.

Only suspect facts are sent. `looks_joined` is a deliberately loose regex filter:
a false positive costs one call that returns the fact unchanged, a false
negative leaves a conjunction in the store, so it errs toward asking.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from .llm import LLM
from .types import FactMention

ATOMIZE_SYSTEM = """\
You split sentences that assert several things into one sentence per assertion.

A sentence is ATOMIC when it asserts exactly one thing about one subject. Split
anything else, and repeat the shared parts in full so each piece stands alone:

  "E3 and E4 argue against banning the veil on slippery-slope and
   trust-in-government grounds."
    -> E3 argues against banning the veil.
    -> E4 argues against banning the veil.
    -> Banning the veil creates slippery-slope risks.
    -> Banning the veil undermines trust in government.

  "Scott Derrickson is an American director and screenwriter."
    -> Scott Derrickson is American.
    -> Scott Derrickson is a director.
    -> Scott Derrickson is a screenwriter.

  "E2 reports declines in handgun crime and gives no controls for confounds."
    -> E2 reports declines in handgun crime.
    -> E2 gives no controls for confounds.

A set quantifier is not atomic. "Both", "all", "each", "neither", "the two",
"they" assert one thing per member, so name each member:

  "Both dossiers name the same enzyme." -> one fact per dossier, named.

Do NOT split:
  - a name or term that happens to contain a conjunction: "Bosnia and
    Herzegovina", "black and white film", "trust in government"
  - a single condition with several parts: "cleared by CYP3A4 and CYP2C9" stays
    whole when the clearance requires both
  - a range: "ran from 2018 to 2020"

Keep the original wording wherever you can; you are dividing a sentence, not
rewriting it. Never add information that was not there, and never drop a piece.
If a sentence is already atomic, return it unchanged as its single part."""


class SplitFact(BaseModel):
    fact_id: int
    parts: list[str] = Field(description="One sentence per assertion; the original "
                                         "unchanged if it was already atomic.")


class AtomizeResult(BaseModel):
    facts: list[SplitFact]


# A conjunction, a comma list, or a set quantifier. Loose on purpose.
_JOINED = re.compile(
    r"\b(and|or|as well as|along with)\b"
    r"|\b(both|all|each|neither|either|the two|they|these|those)\b"
    r"|,\s*\w+(\s+\w+){0,3}\s*,",
    re.IGNORECASE,
)


def looks_joined(text: str) -> bool:
    """Cheap pre-filter. Two thirds of a typical store never needs a call."""
    return bool(_JOINED.search(text or ""))


def _derive(parent: FactMention, text: str, n: int) -> FactMention:
    """A part inherits everything but its identity and its text.

    `quote` stays the parent's span: the evidence for "E3 argues against banning
    the veil" really is the whole sentence it was cut from, and keeping it makes
    the split auditable rather than something to take on faith.
    """
    return FactMention(
        mention_id=f"{parent.mention_id}#{n}",
        text=text,
        quote=parent.quote,
        polarity=parent.polarity,
        qualifiers=list(parent.qualifiers),
        provenance=parent.provenance.model_copy(
            update={"extra": {**parent.provenance.extra, "split_from": parent.mention_id}}
        ),
    )


def atomize(
    llm: LLM,
    mentions: Sequence[FactMention],
    batch_size: int = 20,
    prefilter: bool = True,
    progress: Optional[str] = None,
) -> list[FactMention]:
    """Return mentions with every conjunction split, order preserved.

    Facts that pass through unchanged keep their original `mention_id`, so a
    store that was already atomic comes back identical and ids stay stable
    across re-runs.
    """
    mentions = list(mentions)
    suspect = [i for i, m in enumerate(mentions)
               if not prefilter or looks_joined(m.text)]
    if not suspect:
        return mentions

    batches = [suspect[s:s + batch_size] for s in range(0, len(suspect), batch_size)]
    done = [0]

    def _one(idxs: list[int]) -> dict[int, list[str]]:
        payload = [{"fact_id": i, "text": mentions[i].text} for i in idxs]
        res = llm.parse(system=ATOMIZE_SYSTEM,
                        user=__import__("json").dumps(payload, ensure_ascii=False),
                        output_format=AtomizeResult)
        done[0] += 1
        if progress:
            print(f"[atomize {progress}] batch {done[0]}/{len(batches)}", flush=True)
        return {f.fact_id: [p.strip() for p in f.parts if p and p.strip()]
                for f in res.facts}

    splits: dict[int, list[str]] = {}
    for part in llm.map(_one, batches):
        splits.update(part)

    out: list[FactMention] = []
    for i, m in enumerate(mentions):
        parts = splits.get(i)
        if not parts or len(parts) == 1:
            # Unchanged, or the model confirmed it was already atomic. Either
            # way keep the original id so nothing downstream has to re-link.
            out.append(m)
            continue
        out.extend(_derive(m, p, n) for n, p in enumerate(parts, 1))
    return out
