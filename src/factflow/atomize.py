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

Only suspect facts are sent. `looks_joined` is a deliberately loose regex
filter: a false positive costs one call that returns the fact unchanged, a
false negative leaves a conjunction in the store, so it errs toward asking. It
also matches attribution wrappers, since those need stripping whether or not
anything is joined.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from .llm import LLM
from .types import FactMention

ATOMIZE_SYSTEM = """\
You rewrite each sentence into the atomic facts about the world that it states.

Two jobs, and every sentence needs both checked:

  1. STRIP the attribution, if any. Do this even when the sentence states only
     one thing - a single claim wrapped in "E1 shows ..." still needs unwrapping.
  2. SPLIT what is left, if it asserts more than one thing. Repeat the shared
     parts in full so each piece stands alone.

JOB 1 - DROP THE ATTRIBUTION. A fact is about the world, not about who mentioned it.
  "E1 shows handguns are concealable" and "handguns are concealable" are one
  fact, and writing the source into the sentence records the same information
  twice - provenance already says which agent said it in which round.

  It also nests without limit. "Panelist B identifies that E2's estimate lacks
  controls" is a fact about a fact about a fact, and the next round can wrap it
  again. Every agent wraps to a different depth, so nothing ever matches. Strip
  the wrapper and keep the proposition:

  "E1 shows handguns are concealable ranged weapons."
    -> Handguns are concealable ranged weapons.

  "Panelist B asserts the dossier contains opinions."
    -> The dossier contains opinions.

  "E3 and E4 argue against banning the veil on slippery-slope and
   trust-in-government grounds."
    -> The veil should not be banned.
    -> Banning the veil creates slippery-slope risks.
    -> Banning the veil undermines trust in government.

  Note the last one: with the attribution gone, E3 and E4 asserting the same
  thing is one fact, not two. Only the grounds are separate claims.

JOB 2 - A COMPLETE PROPOSITION. A ground, reason, or risk named by a bare noun is not
  a fact - say what is actually being claimed. "cites discrimination" becomes
  "the ban would cause discrimination", never "cites discrimination as a
  reason":

  "E3 and E4 cite discrimination, backlash, and government discretion."
    -> Banning the veil would cause discrimination.
    -> Banning the veil would cause backlash.
    -> Banning the veil leaves too much to government discretion.

  Recovering the proposition from a noun phrase means using the rest of the
  sentence, not inventing: the subject of the claim is whatever the sentence was
  about. If the sentence genuinely does not say what the claim is, keep the
  noun phrase as it stands rather than guessing.

  When the sentence is *about* a source rather than merely attributed to one -
  "E2 does not address public buildings", "the dossier contains no controls" -
  that is what it asserts, so keep it. The rule removes a wrapper around a
  claim; it does not delete a claim whose subject happens to be a document.

Do NOT split:
  - a name or term that happens to contain a conjunction: "Bosnia and
    Herzegovina", "black and white film", "trust in government"
  - a single condition with several parts: "cleared by CYP3A4 and CYP2C9" stays
    whole when the clearance requires both
  - a range: "ran from 2018 to 2020"

Keep the original wording wherever you can. Never add information that was not
there, and never drop a claim - the parts together must assert everything the
original asserted about the world, no more.

Return a sentence unchanged only when it is BOTH unattributed AND single. A
sentence that states one thing but names who states it is not unchanged: it
comes back as one part, with the wrapper gone."""


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
    r"|,\s*\w+(\s+\w+){0,3}\s*,"
    # attribution wrappers: stripped even when nothing is joined
    r"|^(E\d+|Panelist\s+\w+|Agent\s+\w+)\s+\w+"
    r"|\b(claims?|shows?|argues?|states?|asserts?|reports?|notes?|identifies)\s+that\b",
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
        if not parts:
            out.append(m)
            continue
        if len(parts) == 1:
            # One part is not the same as no change: stripping an attribution
            # rewrites a single claim in place, and returning the original here
            # silently discarded every unwrap the model performed.
            if _same_text(parts[0], m.text):
                out.append(m)                      # genuinely untouched: keep the id
            else:
                out.append(_derive(m, parts[0], 1))
            continue
        out.extend(_derive(m, p, n) for n, p in enumerate(parts, 1))
    return out


def _same_text(a: str, b: str) -> bool:
    norm = lambda s: " ".join(s.lower().split()).rstrip(".")
    return norm(a) == norm(b)
