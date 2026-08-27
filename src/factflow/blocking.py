"""Candidate generation for matching.

Adjudicating every pair with an LLM is O(n^2) calls, which is unaffordable past
a few hundred mentions.  Blocking narrows to a shortlist of plausible pairs
cheaply, so the LLM only sees pairs that could plausibly be the same fact.

Recall here is a hard ceiling on the matcher: a true pair that blocking drops
can never be recovered.  Thresholds are therefore deliberately permissive -
precision is the adjudicator's job, not this stage's.

Cosine alone is not enough.  The pair type this package most needs to catch is
*degradation* - a fact restated with its specifics dropped ("reduced poverty by
12.3% in the Kenya pilot" -> "reduces poverty") - and that is precisely the pair
cosine scores lowest, because the weakened side shares few tokens relative to
the combined vocabulary.  Meanwhile a contradiction, which differs by one
negation token, scores near the top.  So the shortlist score is the max of
cosine and token containment, which is asymmetric and reads a short fact nested
inside a longer one as a strong candidate.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

import numpy as np

from .types import FactMention

CandidatePair = tuple[int, int, float]

_TOKEN = re.compile(r"[a-z0-9][a-z0-9.%/-]*")
_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(tok: str) -> str:
    if any(c.isdigit() for c in tok):
        return tok
    for suf in _SUFFIXES:
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def tokenize(text: str) -> frozenset[str]:
    return frozenset(_stem(t) for t in _TOKEN.findall(text.lower()))


def containment(a: frozenset[str], b: frozenset[str]) -> float:
    """|A n B| / min(|A|, |B|) - high when the shorter fact is nested in the longer.

    This is the degradation detector: a general restatement keeps the subject and
    predicate of the specific fact it came from, and loses only the modifiers.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class Blocker(Protocol):
    def encode(self, texts: Sequence[str]): ...
    def similarity(self, a, b) -> np.ndarray: ...


class TfidfBlocker:
    """Lexical blocker: word bigrams + character n-grams, cosine similarity.

    Character n-grams matter because paraphrases of the same fact usually keep
    the entity and number tokens intact even when the framing changes.
    """

    def __init__(self, min_df: int = 1):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df, sublinear_tf=True)
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfBlocker":
        self._word.fit(texts)
        self._char.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]):
        from scipy.sparse import hstack
        from sklearn.preprocessing import normalize

        if not self._fitted:
            self.fit(texts)
        return normalize(hstack([self._word.transform(texts), self._char.transform(texts)]).tocsr())

    def similarity(self, a, b) -> np.ndarray:
        return np.asarray((a @ b.T).todense())


class SbertBlocker:
    """Dense blocker. Needs the optional `sbert` extra; better on paraphrase."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]):
        return self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

    def similarity(self, a, b) -> np.ndarray:
        return np.asarray(a @ b.T)


def candidate_pairs(
    mentions: Sequence[FactMention],
    blocker: Blocker | None = None,
    threshold: float = 0.45,
    top_k: int = 20,
    same_slot_ok: bool = False,
) -> list[CandidatePair]:
    """Shortlist within one pool of mentions.

    `same_slot_ok=False` skips pairs from the same (execution, agent, round,
    channel).  Two facts extracted from one message are distinct by
    construction - extraction already deduplicated - so adjudicating them wastes
    calls and risks collapsing genuinely different facts.
    """
    if len(mentions) < 2:
        return []
    blocker = blocker or TfidfBlocker()
    texts = [m.text for m in mentions]
    emb = blocker.encode(texts)
    slots = [m.provenance.key() for m in mentions]
    toks = [tokenize(t) for t in texts]

    out: list[CandidatePair] = []
    seen: set[tuple[int, int]] = set()
    chunk = 512
    for start in range(0, len(mentions), chunk):
        sims = blocker.similarity(emb[start : start + chunk], emb)
        for local_i, row in enumerate(sims):
            i = start + local_i
            row = np.maximum(row, [containment(toks[i], t) for t in toks])
            row[i] = -1.0
            k = min(top_k, len(row) - 1)
            if k <= 0:
                continue
            for j in np.argpartition(-row, k - 1)[:k]:
                j = int(j)
                sim = float(row[j])
                if sim < threshold:
                    continue
                if not same_slot_ok and slots[i] == slots[j]:
                    continue
                pair = (i, j) if i < j else (j, i)
                if pair in seen:
                    continue
                seen.add(pair)
                out.append((pair[0], pair[1], sim))
    out.sort(key=lambda p: -p[2])
    return out


def candidate_pairs_against(
    new_mentions: Sequence[FactMention],
    existing_texts: Sequence[str],
    blocker: Blocker | None = None,
    threshold: float = 0.45,
    top_k: int = 10,
) -> list[CandidatePair]:
    """Shortlist new mentions against an existing registry.

    Returns (new_index, existing_index, similarity).  This is the path that
    keeps fact ids stable across repeated executions of the same data entry.
    """
    if not new_mentions or not existing_texts:
        return []
    blocker = blocker or TfidfBlocker()
    new_texts = [m.text for m in new_mentions]
    if hasattr(blocker, "fit"):
        blocker.fit(list(new_texts) + list(existing_texts))
    new_emb = blocker.encode(new_texts)
    old_emb = blocker.encode(list(existing_texts))
    new_toks = [tokenize(t) for t in new_texts]
    old_toks = [tokenize(t) for t in existing_texts]

    out: list[CandidatePair] = []
    sims = blocker.similarity(new_emb, old_emb)
    for i, row in enumerate(sims):
        row = np.maximum(row, [containment(new_toks[i], t) for t in old_toks])
        k = min(top_k, len(row))
        if k <= 0:
            continue
        for j in np.argpartition(-row, k - 1)[:k]:
            j = int(j)
            sim = float(row[j])
            if sim >= threshold:
                out.append((i, j, sim))
    out.sort(key=lambda p: -p[2])
    return out
