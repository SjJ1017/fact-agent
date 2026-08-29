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

0. ENTAILMENT REQUIRES A DIFFERENCE IN INFORMATION, NOT IN WORDING. Before \
choosing an entailment label, name the specific value, scope, condition, or \
qualifier that one side states and the other omits. If you cannot name one, the \
answer is EQUIVALENT.

   These are all EQUIVALENT, not entailment:
   - aspect and voice: "was named ambassador to Ghana" / "held the position of \
ambassador to Ghana" / "served as ambassador to Ghana" / "was ambassador to \
Ghana". Appointment and tenure are not separate facts for this purpose.
   - tense and sequence adverbs: "was named X" / "was later named X" / "was \
subsequently named X". "Later" adds no checkable content.
   - name variants of one referent: "Shirley Temple Black" / "Shirley Temple", \
"Edward Davis Wood Jr." / "Ed Wood".
   - reporting verbs: "X is a director" / "X worked as a director".

   Do not reason your way to a technical distinction. If two facts would be \
recorded as the same row in a database of who-did-what, they are EQUIVALENT.

1. PRECISION IS NOT EQUIVALENCE. If one states a value, scope, or condition the \
other omits or blurs, that is entailment, not equivalence. "Revenue grew 12.3%" \
vs "Revenue grew" -> A_ENTAILS_B. "12.3%" vs "about 10%" -> A_ENTAILS_B only if \
the vaguer one is consistent; if the values are incompatible -> CONTRADICTS.
2. A DROPPED QUALIFIER IS NOT EQUIVALENCE. "The drug reduces mortality in \
patients over 65" vs "The drug reduces mortality" -> A_ENTAILS_B.
3. POLARITY. Same predicate, opposite polarity -> CONTRADICTS, never UNRELATED. \
"The patient has a fever" vs "The patient has no fever" -> CONTRADICTS.
4. SAME ENTITY IS NOT ENOUGH. Two facts about the same entity but asserting \
different predicates are UNRELATED. This includes a repeatable attribute with \
two different values: a person can hold several posts and a film can have \
several actors, so "X was ambassador to Ghana" vs "X was ambassador to \
Czechoslovakia" is UNRELATED, not CONTRADICTS. Reserve CONTRADICTS for values \
that genuinely cannot both hold - one birth date, one enrolment count.
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


class LeanJudgement(BaseModel):
    """Same decision without the prose.

    `rationale` is the largest part of the adjudicator's output and nothing
    downstream reads it, so generating one per pair is paid latency for an
    audit trail nobody opens. It stays available behind a flag.
    """

    pair_id: int
    relation: Literal["EQUIVALENT", "A_ENTAILS_B", "B_ENTAILS_A", "CONTRADICTS", "UNRELATED"]
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class LeanAdjudication(BaseModel):
    judgements: list[LeanJudgement]


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
    auto_reject_below: float = 0.0,
    rationale: bool = True,
    progress: Optional[str] = None,
    bisect_on_failure: bool = True,
    unjudged_out: Optional[list] = None,
) -> list[Relation]:
    """Label each candidate pair. Batched to amortise the system prompt.

    `auto_reject_below` spends the LLM budget where the decision is actually in
    doubt. Blocking similarity is not a verdict, but at the extremes it is
    informative enough to skip the call. Measured on 128 adjudicated pairs from
    the Perspectrum pilot (TF-IDF similarity, threshold .50 / top_k 12):

        similarity        n    EQUIVALENT   ENTAILS   UNRELATED
        >= .95           33          73%        27%          0%
        .85 - .95        28          32%        64%           4%
        .75 - .85         3          33%        33%          33%
        .65 - .75         8           0%        50%          50%
        .50 - .55        56           0%         5%          95%

    Nothing below .55 was judged EQUIVALENT, and only 5% entailed - so rejecting
    that band outright costs no merge and removes 44% of the calls. The upper
    end is *not* safe to auto-merge: above .95 more than a quarter are
    entailments rather than equivalences, and merging those would collapse a
    fact with its own refinement, which is exactly the distinction the typed
    relation exists to keep.

    `progress` labels a per-batch line on stderr. Without it a long pass prints
    nothing until every batch is done, which makes slow indistinguishable from
    hung - the failure mode that cost an afternoon.
    """
    if not pairs:
        return []

    skipped: list[Relation] = []
    if auto_reject_below > 0:
        keep = []
        for i, j, sim in pairs:
            if sim < auto_reject_below:
                skipped.append(Relation(a=mentions[i].mention_id, b=mentions[j].mention_id,
                                        relation="UNRELATED", confidence=1.0 - sim))
            else:
                keep.append((i, j, sim))
        pairs = keep
        if not pairs:
            return skipped

    schema = AdjudicationResult if rationale else LeanAdjudication

    batches: list[list[tuple[int, tuple[int, int, float]]]] = []
    for start in range(0, len(pairs), batch_size):
        chunk = list(enumerate(pairs[start : start + batch_size], start=start))
        batches.append(chunk)

    done = [0]
    unjudged: list[tuple[int, int]] = []

    def _call(chunk) -> list[Relation]:
        payload = [_render_pair(pid, mentions[i], mentions[j]) for pid, (i, j, _) in chunk]
        result = llm.parse(
            system=ADJUDICATION_SYSTEM,
            user=json.dumps(payload, ensure_ascii=False, indent=2),
            output_format=schema,
        )
        return _collect(result, chunk)

    def _one(chunk) -> list[Relation]:
        try:
            out = _call(chunk)
        except Exception:
            # A gateway that content-filters does not refuse, it stalls, so one
            # poisoned pair takes the whole batch down with it and the eleven
            # innocent pairs beside it are lost silently. Bisecting isolates the
            # offender in log(n) extra calls and keeps the rest of the batch.
            out = _bisect(chunk)
        done[0] += 1
        if progress:
            print(f"[adjudicate {progress}] batch {done[0]}/{len(batches)}"
                  f"{f'  unjudged={len(unjudged)}' if unjudged else ''}", flush=True)
        return out

    def _bisect(chunk) -> list[Relation]:
        if not bisect_on_failure:
            return []
        if len(chunk) == 1:
            _pid, (i, j, _s) = chunk[0]
            unjudged.append((i, j))
            return []
        mid = len(chunk) // 2
        out: list[Relation] = []
        for half in (chunk[:mid], chunk[mid:]):
            try:
                out += _call(half)
            except Exception:
                out += _bisect(half)
        return out

    def _collect(result, chunk) -> list[Relation]:
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
                    rationale=getattr(judged, "rationale", None),
                )
            )
        return out

    out = skipped + [r for batch in llm.map(_one, batches) for r in batch]
    if unjudged:
        logger.warning("%d pair(s) could not be judged (likely content-filtered)", len(unjudged))
        if unjudged_out is not None:
            unjudged_out.extend(unjudged)
    return out


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
    judge: Optional[Callable[..., list[Relation]]] = None,
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
        components, extra = _guard(llm, mentions, relations, components, judge)

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
    judge: Optional[Callable[..., list[Relation]]] = None,
) -> tuple[dict[int, list[int]], list[Relation]]:
    """Re-verify every member of a large component against its representative.

    `judge` must be the same question the caller asked in the first place. A
    guard that adjudicates a five-way relation over clusters built from
    SAME/DIFFERENT answers gives a pair two different verdicts under two
    different definitions, and pays the expensive question to do it.

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
            new_rels = (judge or adjudicate)(
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
    auto_reject_below: float = 0.0,
    rationale: bool = True,
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
            for r in adjudicate(llm, pool, pool_pairs, batch_size=batch_size,
                                auto_reject_below=auto_reject_below, rationale=rationale):
                if r.relation != "EQUIVALENT":
                    continue
                mid, reg = (r.a, r.b) if r.b.startswith("__reg__") else (r.b, r.a)
                linked.setdefault(mid, reg.removeprefix("__reg__"))

    # 2. Cluster the remaining new mentions among themselves.
    fresh = [m for m in mentions if m.mention_id not in linked]
    store.add_mentions(mentions)

    if fresh:
        pairs = candidate_pairs(fresh, blocker=blocker, threshold=threshold, top_k=top_k)
        relations = adjudicate(llm, fresh, pairs, batch_size=batch_size,
                               auto_reject_below=auto_reject_below, rationale=rationale)
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


def incremental_match(
    llm: LLM,
    mentions: Sequence[FactMention],
    store: FactStore | None = None,
    threshold: float = 0.50,
    top_k: int = 12,
    batch_size: int = 12,
    auto_reject_below: float = 0.55,
    rationale: bool = False,
    progress: Optional[str] = None,
) -> FactStore:
    """Match turn by turn against the canon built so far.

    All-pairs matching is quadratic in mentions: a 3-round 3-agent debate at 8
    facts a turn is ~97 mentions and ~150 candidate pairs, and doubling the
    rounds quadruples the bill. It is also the wrong shape for the data, which
    arrives in time order.

    Feeding one turn at a time into `match` compares each turn's facts against
    the *canonical facts* accumulated so far rather than against every earlier
    mention. Canon grows far slower than mentions do - that is the whole premise
    of tracking facts rather than text - so the cost goes from O(n^2) to O(n*k).

    It also matches how a debate is read, which means the same code can run
    online, mid-run, instead of only as a post-mortem.
    """
    store = store or FactStore()
    turns: dict[tuple[int, str], list[FactMention]] = {}
    for m in mentions:
        p = m.provenance
        turns.setdefault((p.round or 0, p.agent_id or ""), []).append(m)

    for n, (slot, group) in enumerate(sorted(turns.items()), 1):
        label = f"{progress} {slot[1]}|{slot[0]}" if progress else None
        if label:
            print(f"[incremental {label}] {n}/{len(turns)}  {len(group)} mentions, "
                  f"canon={len(store.facts)}", flush=True)
        store = match(llm, group, store=store, threshold=threshold, top_k=top_k,
                      batch_size=batch_size, auto_reject_below=auto_reject_below,
                      rationale=rationale)
    return store


# -- binary identity --------------------------------------------------------
#
# The five-way relation is the right description of what two facts are to each
# other, and the wrong instrument for measuring a debate.
#
# Two costs sank it. Deciding a direction of entailment is a much harder
# judgement than deciding sameness, and on a reasoning model that shows up
# directly as spend: an eight-pair adjudication burned its entire 4000-token
# budget on reasoning and returned an empty string, three times over, which is
# what made a slow pass look like a hung one.
#
# The second cost is that the distinction was not buying anything. "E1 offers no
# data on the actual effects of religious symbols" and "E1 provides no data on
# the actual effects of *visible* religious symbols" are an entailment, strictly
# speaking. For counting whether an argument survived a round they are the same
# claim, and treating them otherwise is what drove cohort survival to 0.04. And
# where an entailment *does* change the claim materially, the honest reading is
# that a new fact was inferred - which the SAME/DIFFERENT split already records
# by calling them different.
#
# So: ask the cheap question. Diversity and survival do not need more resolution
# than this.


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


class IdentityJudgementBare(BaseModel):
    pair_id: int
    same: bool


class IdentityResult(BaseModel):
    judgements: list[IdentityJudgement]


class IdentityResultBare(BaseModel):
    judgements: list[IdentityJudgementBare]


def identify(
    llm: LLM,
    mentions: Sequence[FactMention],
    pairs: Sequence[tuple[int, int, float]],
    batch_size: int = 16,
    auto_reject_below: float = 0.55,
    auto_accept_above: float = 1.01,
    cue: bool = True,
    progress: Optional[str] = None,
    unjudged_out: Optional[list] = None,
) -> list[Relation]:
    """SAME/DIFFERENT for each candidate pair, emitted as EQUIVALENT/UNRELATED.

    Returns `Relation`s so everything downstream - cluster, FactStore, the
    aggregate views - keeps working unchanged; only two of the five labels are
    ever produced.

    Both cut-offs are measured, on 128 adjudicated pairs from the Perspectrum
    pilot relabelled SAME (= EQUIVALENT or either entailment) / DIFFERENT:

                     SAME median   DIFF median   at .60 loses SAME   saves DIFF
        TF-IDF            .917          .500              4.3%          89.8%
        MiniLM            .935          .323              0.0%          88.1%

    The embedding separates them by .61 against TF-IDF's .42, and at .60 it
    drops nothing. At the top, MiniLM >= .95 was 81% equivalent and 19%
    entailment with no unrelated pair at all - which under the five-way scheme
    was unusable, since merging an entailment collapses a fact with its own
    refinement, and under SAME/DIFFERENT is simply correct. Making the question
    binary is what makes the upper cut-off safe.
    """
    if not pairs:
        return []

    out: list[Relation] = []
    ask: list[tuple[int, int, float]] = []
    for i, j, sim in pairs:
        if sim < auto_reject_below:
            out.append(Relation(a=mentions[i].mention_id, b=mentions[j].mention_id,
                                relation="UNRELATED", confidence=1.0 - sim))
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
    unjudged: list[tuple[int, int]] = []

    def _call(chunk) -> list[Relation]:
        payload = [{"pair_id": pid, "a": mentions[i].text, "b": mentions[j].text}
                   for pid, (i, j, _) in chunk]
        res = llm.parse(system=IDENTITY_SYSTEM,
                        user=json.dumps(payload, ensure_ascii=False),
                        output_format=IdentityResult if cue else IdentityResultBare)
        by_id = {j.pair_id: j.same for j in res.judgements}
        notes = {j.pair_id: getattr(j, "diff", None) for j in res.judgements}
        got: list[Relation] = []
        for pid, (i, j, sim) in chunk:
            if pid not in by_id:
                continue
            note = notes.get(pid)
            got.append(Relation(a=mentions[i].mention_id, b=mentions[j].mention_id,
                                relation="EQUIVALENT" if by_id[pid] else "UNRELATED",
                                confidence=sim,
                                rationale=note if note and note != "none" else None))
        return got

    def _bisect(chunk) -> list[Relation]:
        if len(chunk) == 1:
            unjudged.append((chunk[0][1][0], chunk[0][1][1]))
            return []
        mid = len(chunk) // 2
        got: list[Relation] = []
        for half in (chunk[:mid], chunk[mid:]):
            try:
                got += _call(half)
            except Exception:
                got += _bisect(half)
        return got

    def _one(chunk) -> list[Relation]:
        try:
            got = _call(chunk)
        except Exception:
            got = _bisect(chunk)
        done[0] += 1
        if progress:
            print(f"[identify {progress}] batch {done[0]}/{len(batches)}"
                  f"{f'  unjudged={len(unjudged)}' if unjudged else ''}", flush=True)
        return got

    out += [r for b in llm.map(_one, batches) for r in b]
    if unjudged:
        logger.warning("%d pair(s) could not be judged", len(unjudged))
        if unjudged_out is not None:
            unjudged_out.extend(unjudged)
    return out


def _binary_judge(batch_size: int, auto_reject_below: float,
                  auto_accept_above: float, cue: bool):
    """`identify` bound to the caller's settings, shaped like `adjudicate`."""
    def judge(llm, mentions, pairs, **_kw):
        return identify(llm, mentions, pairs, batch_size=batch_size,
                        auto_reject_below=auto_reject_below,
                        auto_accept_above=auto_accept_above, cue=cue)
    return judge


def binary_match(
    llm: LLM,
    mentions: Sequence[FactMention],
    store: FactStore | None = None,
    blocker: Blocker | None = None,
    threshold: float = 0.50,
    top_k: int = 12,
    batch_size: int = 16,
    auto_reject_below: float = 0.55,
    auto_accept_above: float = 1.01,
    cue: bool = True,
    transitivity_guard: bool = True,
    progress: Optional[str] = None,
) -> FactStore:
    """`match` with the cheap question. Same inputs, same FactStore out."""
    store = store or FactStore()
    mentions = list(mentions)
    if not mentions:
        return store

    existing = [(fid, f.canonical_text) for fid, f in store.facts.items()]
    linked: dict[str, str] = {}
    if existing:
        cross = candidate_pairs_against(mentions, [x for _, x in existing],
                                        blocker=blocker or TfidfBlocker(),
                                        threshold=threshold, top_k=top_k)
        if cross:
            probe = [FactMention(mention_id=f"__reg__{fid}", text=text,
                                 provenance=mentions[0].provenance.model_copy(
                                     update={"channel": Channel.SOURCE}))
                     for fid, text in existing]
            pool = mentions + probe
            offset = len(mentions)
            for r in identify(llm, pool, [(i, offset + j, s) for i, j, s in cross],
                              batch_size=batch_size, auto_reject_below=auto_reject_below,
                              auto_accept_above=auto_accept_above, cue=cue,
                              progress=progress):
                if r.relation != "EQUIVALENT":
                    continue
                mid, reg = (r.a, r.b) if r.b.startswith("__reg__") else (r.b, r.a)
                linked.setdefault(mid, reg.removeprefix("__reg__"))

    fresh = [m for m in mentions if m.mention_id not in linked]
    store.add_mentions(mentions)
    if fresh:
        pairs = candidate_pairs(fresh, blocker=blocker, threshold=threshold, top_k=top_k)
        relations = identify(llm, fresh, pairs, batch_size=batch_size,
                             auto_reject_below=auto_reject_below,
                             auto_accept_above=auto_accept_above, cue=cue,
                             progress=progress)
        facts, extra = cluster(llm, fresh, relations, transitivity_guard=transitivity_guard,
                               judge=_binary_judge(batch_size, auto_reject_below,
                                                   auto_accept_above, cue))
        store.relations.extend(relations)
        store.relations.extend(extra)
        for f in facts:
            cur = store.facts.get(f.fact_id)
            if cur:
                cur.mention_ids = sorted(set(cur.mention_ids) | set(f.mention_ids))
                store.assign(cur)
            else:
                store.assign(f)

    for mid, fid in linked.items():
        fact = store.facts[fid]
        if mid not in fact.mention_ids:
            fact.mention_ids.append(mid)
        store.assign(fact)
    return store
