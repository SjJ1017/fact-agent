"""(2) Matching mentions across agents, rounds, and executions.

One question and one rule.

THE QUESTION IS BINARY. An earlier version asked for a typed relation -
equivalent, entails in either direction, contradicts, unrelated - on the
argument that a fact restated less precisely is a weakened survival rather than
a different fact, and only a direction can say so. It was the better description
and the wrong instrument. Deciding a direction of entailment is a much harder
judgement than deciding sameness, and on a reasoning model that shows up as
spend: an eight-pair five-way adjudication burned 15999 reasoning tokens and
returned an empty string, where the same pairs answered SAME/DIFFERENT in 3656.
The distinction was also not buying anything. Once facts are properly atomic,
"E3 argues against the ban" simply recurs, and recurrence needs no adjudicator;
where an entailment does change the claim materially, calling it a different
fact is the honest reading, and that is what DIFFERENT says.

THE RULE IS THAT TRANSITIVITY IS NOT FREE. Union-find over pairwise SAME edges
will chain: A~B and B~C is taken as A~C, and a few plausible-looking edges at
the bottom of the similarity range are enough to fuse a whole run into one
cluster. Measured on six stores, unioning every SAME edge produced clusters of
up to 111 mentions spanning unrelated claims. Requiring the edge to also carry
a blocking similarity of at least .90 caps the largest cluster at 8 across all
six while keeping 18% of mentions merged - the point where the blob disappears
and further tightening only costs recall. Weaker SAME edges are still recorded
as relations, because the judgement stands on its own; they simply do not get
to imply a third pair nobody judged.

An earlier fix for the same problem re-checked every member of a large cluster
against a representative. It cost an LLM call per member - most of a
twenty-four-minute run - and it asked the model to compare each fact against
whichever phrasing happened to be picked, which on source-document sentences is
its own source of noise. Not forming the chain is cheaper and does not depend
on that choice.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from .blocking import Blocker, TfidfBlocker, candidate_pairs, candidate_pairs_against
from .llm import LLM
from .types import CanonicalFact, Channel, FactMention, FactStore, Relation

logger = logging.getLogger(__name__)

# Union only above this blocking similarity. See the module docstring.
UNION_MIN_SIMILARITY = 0.90

IDENTITY_SYSTEM = """\
You decide whether two atomic facts state the same thing.

SAME: they assert the same thing about the same subject. Paraphrase, word order,
synonyms, added or dropped hedges, and differences in how precisely a scope is
worded are all SAME - if both would be recorded as one row in a table of
who-claimed-what, they are SAME.

DIFFERENT: they assert different things, contradict each other, or concern
different subjects. A qualifier that changes *which* cases the claim covers, or
reverses it, makes them DIFFERENT.

For each pair, first name the single difference between a and b in at most eight
words (write "none" if there is none), then rule. A difference in wording is not
a difference. Judge only what is written; do not reason about whether either is
true, and do not deliberate beyond that one short note.

Answer every pair."""


class IdentityJudgement(BaseModel):
    """`diff` before `same`, and the order is the point.

    A model with hidden reasoning spends an unbounded budget before it writes
    anything - measured here at 4000 reasoning tokens and an empty string. A
    short field in the schema buys most of the same deliberation at a length we
    choose: the model must name the difference before it may rule on it, and
    naming it costs a dozen tokens rather than four thousand.

    JSON is generated in field order, so naming the difference first puts it in
    the context the decision token is drawn from. Asking for it afterwards is
    not worthless - a model told it will have to justify itself answers
    differently - but that route works only through anticipation, and this one
    makes the note causally available.
    """

    pair_id: int
    diff: str = Field(description="The one difference between a and b in at most "
                                  "8 words, or 'none'. Wording differences are not "
                                  "differences.")
    same: bool


class IdentityResult(BaseModel):
    judgements: list[IdentityJudgement]


def identify(
    llm: LLM,
    mentions: Sequence[FactMention],
    pairs: Sequence[tuple[int, int, float]],
    batch_size: int = 16,
    auto_reject_below: float = 0.0,
    auto_accept_above: float = 1.01,
    progress: Optional[str] = None,
) -> list[Relation]:
    """SAME/DIFFERENT per candidate pair, emitted as EQUIVALENT/UNRELATED.

    Relations keep their old shape so the store and every view over it are
    unchanged; only two of the labels are ever produced.

    Both shortcuts are measured, on 128 adjudicated pairs relabelled SAME (=
    equivalent or either entailment) / DIFFERENT:

                     SAME median   DIFF median   at .60 loses SAME   saves DIFF
        TF-IDF            .917          .500              4.3%          89.8%
        bge-base          .931          .571              0.0%          94.9%

    Nothing below the reject line was ever judged SAME, so skipping it costs no
    merge. `auto_accept_above` is safe only because the question is binary:
    bge-base above .95 was 81% equivalent and 19% entailment with no unrelated
    pair at all, which under the five-way scheme was unusable and under
    SAME/DIFFERENT is simply correct.

    `auto_accept_above` is off by default, and the reason is a trap worth
    naming. `candidate_pairs` scores a pair as max(cosine, containment), and a
    containment of 1.0 means every token of one fact appears in the other -
    which is exactly the dropped-scope case. "UBI reduced poverty by 12.3% in
    the Kenya pilot" contains "UBI reduced poverty", so an accept-above rule
    merges them without asking, and under SAME/DIFFERENT a claim that drops
    which cases it covers is DIFFERENT. The calibration that made the top band
    look safe had mapped entailment to SAME, which is the mapping this
    pipeline no longer uses.

    A caller that passes a placeholder rather than a measured similarity must
    leave both shortcuts off, or the placeholder decides the pair.
    """
    if not pairs:
        return []

    out: list[Relation] = []
    ask: list[tuple[int, int, float]] = []
    for i, j, sim in pairs:
        if sim < auto_reject_below:
            out.append(Relation(a=mentions[i].mention_id, b=mentions[j].mention_id,
                                relation="UNRELATED", confidence=sim))
        elif sim >= auto_accept_above:
            out.append(Relation(a=mentions[i].mention_id, b=mentions[j].mention_id,
                                relation="EQUIVALENT", confidence=sim))
        else:
            ask.append((i, j, sim))
    if not ask:
        return out

    batches = [list(enumerate(ask[s:s + batch_size], start=s))
               for s in range(0, len(ask), batch_size)]
    done = [0]

    def _one(chunk) -> list[Relation]:
        payload = [{"pair_id": pid, "a": mentions[i].text, "b": mentions[j].text}
                   for pid, (i, j, _) in chunk]
        res = llm.parse(system=IDENTITY_SYSTEM,
                        user=json.dumps(payload, ensure_ascii=False),
                        output_format=IdentityResult)
        done[0] += 1
        if progress:
            print(f"[identify {progress}] batch {done[0]}/{len(batches)}", flush=True)
        by_id = {j.pair_id: j for j in res.judgements}
        got: list[Relation] = []
        for pid, (i, j, sim) in chunk:
            judged = by_id.get(pid)
            if judged is None:
                continue
            note = judged.diff
            got.append(Relation(a=mentions[i].mention_id, b=mentions[j].mention_id,
                                relation="EQUIVALENT" if judged.same else "UNRELATED",
                                confidence=sim,
                                rationale=note if note and note != "none" else None))
        return got

    return out + [r for b in llm.map(_one, batches) for r in b]


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

    Source phrasing wins when present - it is the reference the trace is
    measured against. Otherwise prefer the most specific surviving phrasing,
    approximated by qualifier count then length.
    """
    source = [m for m in members if m.provenance.channel == Channel.SOURCE]
    pool = source or list(members)
    return max(pool, key=lambda m: (len(m.qualifiers), len(m.text)))


def cluster(
    mentions: Sequence[FactMention],
    relations: Sequence[Relation],
    union_min_similarity: float = UNION_MIN_SIMILARITY,
) -> list[CanonicalFact]:
    """Group mentions by SAME edges strong enough to be chained.

    A SAME edge below the threshold is trusted about its own pair and not about
    any third pair it would connect through - see the module docstring.
    """
    index = {m.mention_id: i for i, m in enumerate(mentions)}
    uf = _UnionFind(len(mentions))
    for r in relations:
        if r.relation != "EQUIVALENT" or r.confidence < union_min_similarity:
            continue
        if r.a in index and r.b in index:
            uf.union(index[r.a], index[r.b])

    components: dict[int, list[int]] = {}
    for i in range(len(mentions)):
        components.setdefault(uf.find(i), []).append(i)

    facts: list[CanonicalFact] = []
    for members_idx in components.values():
        members = [mentions[i] for i in members_idx]
        rep = _pick_canonical(members)
        facts.append(CanonicalFact(
            fact_id=CanonicalFact.make_id(rep.text),
            canonical_text=rep.text,
            polarity=rep.polarity,
            mention_ids=[m.mention_id for m in members],
        ))
    return facts


def match(
    llm: LLM,
    mentions: Sequence[FactMention],
    store: FactStore | None = None,
    blocker: Blocker | None = None,
    threshold: float = 0.70,
    top_k: int = 12,
    batch_size: int = 16,
    auto_reject_below: float = 0.0,
    auto_accept_above: float = 1.01,
    union_min_similarity: float = UNION_MIN_SIMILARITY,
    progress: Optional[str] = None,
) -> FactStore:
    """block -> identify -> cluster -> register.

    Pass an existing `store` to match a new batch against facts already
    registered; ids of previously-seen facts are preserved, so the same
    proposition carries the same id across turns and across runs.
    """
    store = store or FactStore()
    mentions = list(mentions)
    if not mentions:
        return store
    blocker = blocker or TfidfBlocker()

    # 1. Link to facts already registered.
    existing = [(fid, f.canonical_text) for fid, f in store.facts.items()]
    linked: dict[str, str] = {}
    if existing:
        cross = candidate_pairs_against(mentions, [x for _, x in existing],
                                        blocker=blocker, threshold=threshold, top_k=top_k)
        if cross:
            probe = [FactMention(mention_id=f"__reg__{fid}", text=text,
                                 provenance=mentions[0].provenance.model_copy(
                                     update={"channel": Channel.SOURCE}))
                     for fid, text in existing]
            pool = mentions + probe
            offset = len(mentions)
            for r in identify(llm, pool, [(i, offset + j, s) for i, j, s in cross],
                              batch_size=batch_size, auto_reject_below=auto_reject_below,
                              auto_accept_above=auto_accept_above, progress=progress):
                if r.relation != "EQUIVALENT" or r.confidence < union_min_similarity:
                    continue
                mid, reg = (r.a, r.b) if r.b.startswith("__reg__") else (r.b, r.a)
                linked.setdefault(mid, reg.removeprefix("__reg__"))

    # 2. Cluster what is left among itself.
    fresh = [m for m in mentions if m.mention_id not in linked]
    store.add_mentions(mentions)
    if fresh:
        pairs = candidate_pairs(fresh, blocker=blocker, threshold=threshold, top_k=top_k)
        relations = identify(llm, fresh, pairs, batch_size=batch_size,
                             auto_reject_below=auto_reject_below,
                             auto_accept_above=auto_accept_above, progress=progress)
        store.relations.extend(relations)
        for f in cluster(fresh, relations, union_min_similarity):
            cur = store.facts.get(f.fact_id)
            if cur:
                cur.mention_ids = sorted(set(cur.mention_ids) | set(f.mention_ids))
                store.assign(cur)
            else:
                store.assign(f)

    # 3. Attach what matched the registry.
    for mid, fid in linked.items():
        fact = store.facts[fid]
        if mid not in fact.mention_ids:
            fact.mention_ids.append(mid)
        store.assign(fact)
    return store
