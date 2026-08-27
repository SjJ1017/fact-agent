"""factflow - atomic facts as a coordinate system for multi-agent traces.

Three stages:

    extract  : text                       -> FactMention[]
    match    : FactMention[]              -> FactStore (canonical facts, stable ids)
    annotate : FactStore + PropertySpec[] -> FactStore (facts carry properties)

Once matched, `store.in_context(f, agent, t)` and `store.in_output(f, agent, t)`
are lookups, which is what any lifecycle metric is built on.
"""

from .blocking import SbertBlocker, TfidfBlocker, candidate_pairs, candidate_pairs_against
from .extract import TraceRecord, extract_facts, extract_trace
from .llm import LLM, LLMConfig
from .match import adjudicate, cluster, match
from .properties import (
    ABSTRACTION,
    CRITICALITY,
    POLARITY_KIND,
    RELEVANCE,
    TRUTH,
    PropertySpec,
    annotate_store,
    annotate_texts,
)
from .types import (
    CanonicalFact,
    Channel,
    FactMention,
    FactStore,
    Polarity,
    Provenance,
    Relation,
)

__version__ = "0.1.0"

__all__ = [
    "LLM",
    "LLMConfig",
    "extract_facts",
    "extract_trace",
    "TraceRecord",
    "match",
    "adjudicate",
    "cluster",
    "candidate_pairs",
    "candidate_pairs_against",
    "TfidfBlocker",
    "SbertBlocker",
    "PropertySpec",
    "annotate_texts",
    "annotate_store",
    "TRUTH",
    "CRITICALITY",
    "RELEVANCE",
    "ABSTRACTION",
    "POLARITY_KIND",
    "FactMention",
    "CanonicalFact",
    "FactStore",
    "Provenance",
    "Channel",
    "Polarity",
    "Relation",
]
