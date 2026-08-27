"""(2) Matching mentions across agents, rounds, and executions.

Two design choices carry most of the weight:

* The adjudicator returns a **typed relation**, not a same/different bit.
  Collapsing to a bit throws away the degradation signal - a fact restated less
  precisely ("12.3%" -> "about 10%") is neither the same fact nor a different
  one, it is a weakened survival, and only an entailment direction can say so.

* Clustering runs a **transitivity guard**.  LLM equivalence judgements are not
  transitive: A~B and B~C does not give A~C, and naive union-find on noisy edges
  produces one giant blob that silently merges unrelated facts.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field

from .blocking import Blocker, TfidfBlocker, candidate_pairs, candidate_pairs_against
from .llm import LLM
from .types import CanonicalFact, Channel, FactMention, FactStore, Relation, RelationType

logger = logging.getLogger(__name__)

ADJUDICATION_SYSTEM = """\
You compare pairs of atomic facts and label the logical relation between them.

Labels:
- EQUIVALENT: the two express the same proposition. Either could replace the \
other without changing what is asserted. Paraphrase, reordering, synonyms, and \
different surface framing are all EQUIVALENT.
- A_ENTAILS_B: A asserts everything B asserts, and more. A is strictly more \
specific, precise, or qualified than B.
- B_ENTAILS_A: the reverse.
- CONTRADICTS: the two cannot both be true.
- UNRELATED: different propositions; neither entails the other.

Decision rules, in priority order:

1. PRECISION IS NOT EQUIVALENCE. If one states a value, scope, or condition the \
other omits or blurs, that is entailment, not equivalence. "Revenue grew 12.3%" \
vs "Revenue grew" -> A_ENTAILS_B. "12.3%" vs "about 10%" -> A_ENTAILS_B only if \
the vaguer one is consistent; if the values are incompatible -> CONTRADICTS.
2. A DROPPED QUALIFIER IS NOT EQUIVALENCE. "The drug reduces mortality in \
patients over 65" vs "The drug reduces mortality" -> A_ENTAILS_B.
3. POLARITY. Same predicate, opposite polarity -> CONTRADICTS, never UNRELATED. \
"The patient has a fever" vs "The patient has no fever" -> CONTRADICTS.
4. SAME ENTITY IS NOT ENOUGH. Two facts about the same entity but asserting \
different predicates are UNRELATED.
5. SIBLING CATEGORIES ARE NOT ENTAILMENT. Entailment requires that the truth of \
one FORCES the truth of the other. Two roles, professions, genres, or categories \
that merely overlap or often co-occur are UNRELATED, however related they sound. \
"Ed Wood is a filmmaker" vs "Ed Wood is a director" -> UNRELATED: a filmmaker \
need not be a director, and a director need not be called a filmmaker. Use \
entailment ONLY for a genuine subset relation - "X is a poodle" entails "X is a \
dog"; "X is a dog" does not entail "X is a poodle". If you have to argue for the \
entailment, it is UNRELATED.
6. Judge propositional content only. Ignore differences in tone, length, \
attribution, and word choice that do not change what is asserted.

Answer for every pair you are given, using the pair_id you were given.
"""


class PairJudgement(BaseModel):
    pair_id: int
    relation: Literal["EQUIVALENT", "A_ENTAILS_B", "B_ENTAILS_A", "CONTRADICTS", "UNRELATED"]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    rationale: Optional[str] = Field(default=None, description="One short clause.")


class AdjudicationResult(BaseModel):
    judgements: list[PairJudgement]


def _render_pair(pid: int, a: FactMention, b: FactMention) -> dict:
    def side(m: FactMention) -> dict:
        d = {"text": m.text, "polarity": m.polarity.value}
        if m.qualifiers:
            d["qualifiers"] = m.qualifiers
        return d

    return {"pair_id": pid, "A": side(a), "B": side(b)}


def adjudicate(
    llm: LLM,
    mentions: Sequence[FactMention],
    pairs: Sequence[tuple[int, int, float]],
    batch_size: int = 20,
) -> list[Relation]:
    """Label each candidate pair. Batched to amortise the system prompt."""
    if not pairs:
        return []

    batches: list[list[tuple[int, tuple[int, int, float]]]] = []
    for start in range(0, len(pairs), batch_size):
        chunk = list(enumerate(pairs[start : start + batch_size], start=start))
        batches.append(chunk)

    def _one(chunk) -> list[Relation]:
        payload = [_render_pair(pid, mentions[i], mentions[j]) for pid, (i, j, _) in chunk]
        result = llm.parse(
            system=ADJUDICATION_SYSTEM,
            user=json.dumps(payload, ensure_ascii=False, indent=2),
            output_format=AdjudicationResult,
        )
        by_id = {j.pair_id: j for j in result.judgements}
        out: list[Relation] = []
        for pid, (i, j, _sim) in chunk:
            judged = by_id.get(pid)
            if judged is None:
                continue
            out.append(
                Relation(
                    a=mentions[i].mention_id,
                    b=mentions[j].mention_id,
                    relation=judged.relation,
                    confidence=judged.confidence,
                    rationale=judged.rationale,
                )
            )
        return out

    return [r for batch in llm.map(_one, batches) for r in batch]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _pick_canonical(members: Sequence[FactMention]) -> FactMention:
    """Choose the cluster's representative phrasing.

    Ground-truth (SOURCE) phrasing wins when present - it is the reference the
    trace should be measured against.  Otherwise prefer the most specific
    surviving phrasing, approximated by qualifier count then length.
    """
    source = [m for m in members if m.provenance.channel == Channel.SOURCE]
    pool = source or list(members)
    return max(pool, key=lambda m: (len(m.qualifiers), len(m.text)))


def cluster(
    llm: LLM,
    mentions: Sequence[FactMention],
    relations: Sequence[Relation],
    transitivity_guard: bool = True,
    min_confidence: float = 0.0,
) -> tuple[list[CanonicalFact], list[Relation]]:
    """Group mentions into canonical facts using EQUIVALENT edges only.

    Returns the clusters plus any extra relations produced by the guard.
    """
    index = {m.mention_id: i for i, m in enumerate(mentions)}
    uf = _UnionFind(len(mentions))
    for r in relations:
        if r.relation != "EQUIVALENT" or r.confidence < min_confidence:
            continue
        if r.a in index and r.b in index:
            uf.union(index[r.a], index[r.b])

    components: dict[int, list[int]] = {}
    for i in range(len(mentions)):
        components.setdefault(uf.find(i), []).append(i)

    extra: list[Relation] = []
    if transitivity_guard:
        components, extra = _guard(llm, mentions, relations, components)

    facts: list[CanonicalFact] = []
    for members_idx in components.values():
        members = [mentions[i] for i in members_idx]
        rep = _pick_canonical(members)
        facts.append(
            CanonicalFact(
                fact_id=CanonicalFact.make_id(rep.text),
                canonical_text=rep.text,
                polarity=rep.polarity,
                mention_ids=[m.mention_id for m in members],
            )
        )
    return facts, extra


def _guard(
    llm: LLM,
    mentions: Sequence[FactMention],
    relations: Sequence[Relation],
    components: dict[int, list[int]],
) -> tuple[dict[int, list[int]], list[Relation]]:
    """Re-verify every member of a large component against its representative.

    Transitive closure over noisy pairwise judgements is the main source of
    over-merging, and it degrades silently: the cluster count drops and every
    downstream retention number rises. One extra round of medoid checks removes
    most of it at a small, bounded cost.
    """
    known: dict[frozenset[str], RelationType] = {
        frozenset((r.a, r.b)): r.relation for r in relations
    }
    extra: list[Relation] = []
    out: dict[int, list[int]] = {}
    next_key = max(components) + 1 if components else 0
    splits = 0

    for root, members_idx in components.items():
        if len(members_idx) <= 2:
            out[root] = members_idx
            continue

        rep_idx = index_of(mentions, _pick_canonical([mentions[i] for i in members_idx]))
        to_check = [i for i in members_idx if i != rep_idx]
        unknown = [
            i
            for i in to_check
            if frozenset((mentions[i].mention_id, mentions[rep_idx].mention_id)) not in known
        ]
        if unknown:
            new_rels = adjudicate(
                llm, mentions, [(min(rep_idx, i), max(rep_idx, i), 1.0) for i in unknown]
            )
            extra.extend(new_rels)
            for r in new_rels:
                known[frozenset((r.a, r.b))] = r.relation

        kept = [rep_idx]
        for i in to_check:
            rel = known.get(frozenset((mentions[i].mention_id, mentions[rep_idx].mention_id)))
            # Split only on an EXPLICIT disagreement. A missing judgement (dropped
            # batch, skipped pair_id) is not evidence against the edge union-find
            # already accepted, and defaulting to split silently shatters correct
            # clusters - which inflates the fact count and depresses every
            # retention number computed from it.
            if rel is None or rel == "EQUIVALENT":
                kept.append(i)
            else:
                out[next_key] = [i]
                next_key += 1
                splits += 1
        out[root] = kept

    if splits:
        logger.info("transitivity guard split off %d mention(s)", splits)
    return out, extra


def index_of(mentions: Sequence[FactMention], target: FactMention) -> int:
    for i, m in enumerate(mentions):
        if m.mention_id == target.mention_id:
            return i
    raise ValueError("mention not in pool")


def match(
    llm: LLM,
    mentions: Sequence[FactMention],
    store: FactStore | None = None,
    blocker: Blocker | None = None,
    threshold: float = 0.50,
    top_k: int = 12,
    batch_size: int = 20,
    transitivity_guard: bool = True,
) -> FactStore:
    """Full matching pipeline: block -> adjudicate -> cluster -> register.

    Pass an existing `store` to match a new execution against facts already
    registered; ids of previously-seen facts are preserved so the same
    proposition carries the same id across runs.
    """
    store = store or FactStore()
    mentions = list(mentions)

    # 1. Link new mentions to already-registered facts.
    existing = [(fid, f.canonical_text) for fid, f in store.facts.items()]
    linked: dict[str, str] = {}
    if existing and mentions:
        cross = candidate_pairs_against(
            mentions, [t for _, t in existing], blocker=TfidfBlocker(), threshold=threshold, top_k=top_k
        )
        if cross:
            probe = [
                FactMention(
                    mention_id=f"__reg__{fid}",
                    text=text,
                    provenance=mentions[0].provenance.model_copy(update={"channel": Channel.SOURCE}),
                )
                for fid, text in existing
            ]
            pool = mentions + probe
            offset = len(mentions)
            pool_pairs = [(i, offset + j, sim) for i, j, sim in cross]
            for r in adjudicate(llm, pool, pool_pairs, batch_size=batch_size):
                if r.relation != "EQUIVALENT":
                    continue
                mid, reg = (r.a, r.b) if r.b.startswith("__reg__") else (r.b, r.a)
                linked.setdefault(mid, reg.removeprefix("__reg__"))

    # 2. Cluster the remaining new mentions among themselves.
    fresh = [m for m in mentions if m.mention_id not in linked]
    store.add_mentions(mentions)

    if fresh:
        pairs = candidate_pairs(fresh, blocker=blocker, threshold=threshold, top_k=top_k)
        relations = adjudicate(llm, fresh, pairs, batch_size=batch_size)
        facts, extra = cluster(llm, fresh, relations, transitivity_guard=transitivity_guard)
        store.relations.extend(relations)
        store.relations.extend(extra)
        for f in facts:
            existing_fact = store.facts.get(f.fact_id)
            if existing_fact:
                existing_fact.mention_ids = sorted(set(existing_fact.mention_ids) | set(f.mention_ids))
                store.assign(existing_fact)
            else:
                store.assign(f)

    # 3. Attach mentions that matched the registry.
    for mid, fid in linked.items():
        fact = store.facts[fid]
        if mid not in fact.mention_ids:
            fact.mention_ids.append(mid)
        store.assign(fact)

    return store
