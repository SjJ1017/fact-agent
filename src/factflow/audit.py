"""Automatic quality flags on a matched store.

Every defect found in this project so far was found by reading output, not by a
test: extraction keeping "an American director, screenwriter and producer" whole,
extraction emitting "Both were American", the adjudicator tying a Telangana
land-records fact to a Shirley Temple film credit. None of those raise an error -
they produce a plausible-looking store with wrong numbers in it.

These checks are cheap heuristics, deliberately tuned to over-report. They are a
reading aid that puts the suspicious rows first, not a validator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .blocking import TfidfBlocker, containment, tokenize
from .types import Channel, FactStore

UNRESOLVED = re.compile(
    r"^(both|he|she|they|it|this|that|these|those|the (series|film|movie|study|"
    r"book|author|company|article|documents?|position|office|award|trial|program))\b",
    re.I,
)
# "The office of Secretary of State..." and "The series is Animorphs" name their
# referent immediately; only a bare "The office ..." is actually unresolvable.
RESOLVED_INLINE = re.compile(
    r"^[Tt]he\s+\w+\s+(?:(?:of|for|is|was|named|called|titled)\s+)?[\"'\u201c]?[A-Z0-9]"
)
CONJUNCTION = re.compile(r",\s+\w+,?\s+and\s+\w|\b(is|was|are|were)\b[^.]*\b\w+\s+and\s+\w+", re.I)
# A run of >=2 capitalised words is almost always a proper name. Names routinely
# contain "and" ("Science Fiction and Fantasy Writers of America") and, when two
# facts are about the same long-named entity, that shared name dominates any
# bag-of-words similarity. Both heuristics below would otherwise fire constantly
# on exactly the entities the corpus is densest in.
PROPER = re.compile(r"\b(?:[A-Z][\w'’\-]*)(?:\s+(?:of|for|the|and|in|de|von|van)\s+|\s+)(?:[A-Z][\w'’\-]*)(?:(?:\s+(?:of|for|the|and|in|de|von|van)\s+|\s+)[A-Z][\w'’\-]*)*")


def proper_spans(text: str) -> list[str]:
    return [m.group(0) for m in PROPER.finditer(text)]


def strip_shared_names(a: str, b: str) -> tuple[str, str]:
    """Remove proper-name spans the two texts share, so comparison sees the rest."""
    shared = {s.lower() for s in proper_spans(a)} & {s.lower() for s in proper_spans(b)}
    for name in sorted(shared, key=len, reverse=True):
        pat = re.compile(re.escape(name), re.I)
        a, b = pat.sub(" ", a), pat.sub(" ", b)
    return a, b
HEDGE = re.compile(r"\b(the documents?|the (text|passage|context)|according to the (documents?|text))\b", re.I)


FUNCTION_WORDS = frozenset(
    """a an the of in on at to for by with from as is was were are be been being
    later then also that which who whom whose and or but not no it its his her
    their this these those there here when while during had has have do does did
    named called serve served hold held include included""".split()
)


@dataclass
class Flag:
    severity: str  # "high" | "medium"
    kind: str
    fact_id: str
    detail: str
    partner_id: str | None = None


def audit(store: FactStore, near_dup_threshold: float = 0.82) -> list[Flag]:
    flags: list[Flag] = []
    facts = list(store.facts.items())

    for fid, f in facts:
        t = f.canonical_text
        if UNRESOLVED.match(t) and not RESOLVED_INLINE.match(t):
            flags.append(Flag("high", "unresolved-referent", fid,
                              "opens with a referent that was never resolved; cannot be matched"))
        masked = t
        for name in proper_spans(t):
            masked = masked.replace(name, " ")
        if CONJUNCTION.search(masked):
            flags.append(Flag("medium", "unsplit-conjunction", fid,
                              "carries a coordinated predicate that should be several facts"))
        if HEDGE.search(t):
            flags.append(Flag("medium", "meta-discourse", fid,
                              "asserts something about the documents rather than about the world"))

    # Over-merge: a cluster whose members share little vocabulary.
    for fid, f in facts:
        if len(f.mention_ids) < 2:
            continue
        toks = [tokenize(store.mentions[m].text) for m in f.mention_ids]
        worst = min(
            (containment(a, b) for i, a in enumerate(toks) for b in toks[i + 1 :]),
            default=1.0,
        )
        if worst < 0.34:
            flags.append(Flag("high", "suspect-merge", fid,
                              f"two members share only {worst:.0%} of the shorter one's tokens"))

    # Under-merge: two separate facts that look like the same proposition.
    # Scored on what is left after the shared entity name is removed, otherwise
    # every pair of facts about one long-named entity looks like a duplicate.
    if len(facts) > 1:
        texts = [f.canonical_text for _, f in facts]
        blocker = TfidfBlocker()
        emb = blocker.encode(texts)
        sims = blocker.similarity(emb, emb)
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                if float(sims[i][j]) < 0.55:
                    continue
                ra, rb = strip_shared_names(texts[i], texts[j])
                ta, tb = tokenize(ra), tokenize(rb)
                if not ta or not tb:
                    continue
                # Two facts about one subject that differ in a CONTENT word are
                # different facts ("experience the rapture" vs "the tribulation").
                # Only flag pairs whose difference is purely functional wording.
                diff = (ta ^ tb) - FUNCTION_WORDS
                if any(len(w) > 3 for w in diff):
                    continue
                score = max(len(ta & tb) / len(ta | tb), containment(ta, tb))
                if score >= near_dup_threshold:
                    flags.append(Flag("medium", "possible-missed-merge", facts[i][0],
                                      f"{score:.0%} similar once the shared name is removed",
                                      partner_id=facts[j][0]))

    order = {"high": 0, "medium": 1}
    flags.sort(key=lambda f: (order[f.severity], f.kind))
    return flags


def summarise(flags: Iterable[Flag]) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in flags:
        out[f.kind] = out.get(f.kind, 0) + 1
    return out
