"""(3) Prompt-driven property annotation.

Which properties matter is a per-study decision - truth against a reference,
criticality for a question, relevance, uniqueness, abstraction level - so the
schema is supplied by the caller rather than fixed here.  A `PropertySpec` list
is compiled into a Pydantic model and answered under structured output, which
keeps the results typed and validated no matter what the caller asked for.

Annotation runs against canonical facts, not mentions: "is this true" is a
question about a proposition, not about one phrasing of it.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, Field, create_model

from .llm import LLM
from .types import CanonicalFact, FactStore

PropertyType = Literal["boolean", "choice", "number", "string"]


class PropertySpec(BaseModel):
    """One property to annotate.

    `description` is passed verbatim to the model and is the main lever on
    annotation quality - state the decision rule and the tie-breaker, not just
    the property name.
    """

    name: str
    description: str
    type: PropertyType = "boolean"
    choices: Optional[list[str]] = None

    def annotation(self) -> tuple[Any, Any]:
        if self.type == "boolean":
            return bool, Field(description=self.description)
        if self.type == "number":
            return float, Field(description=self.description)
        if self.type == "choice":
            if not self.choices:
                raise ValueError(f"property {self.name!r} is type 'choice' but has no choices")
            return Literal[tuple(self.choices)], Field(description=self.description)  # type: ignore[valid-type]
        return str, Field(description=self.description)


ANNOTATION_SYSTEM = """\
You annotate atomic facts with properties for a fact-tracking system.

You are given a numbered list of facts and must return one annotation object \
per fact, keyed by the same `fact_index`.

Rules:
- Judge each fact independently, on its own content.
- Use ONLY the reference material provided. If a property cannot be determined \
from it, follow the property's stated fallback; never guess from world knowledge \
unless the property description says to.
- Be consistent: the same fact must receive the same annotation regardless of \
its position in the list.
"""


def _build_model(specs: Sequence[PropertySpec]) -> type[BaseModel]:
    fields: dict[str, tuple[Any, Any]] = {"fact_index": (int, Field(description="Index of the fact."))}
    for s in specs:
        fields[s.name] = s.annotation()
    item = create_model("FactAnnotation", **fields)  # type: ignore[call-overload]
    return create_model("AnnotationResult", annotations=(list[item], ...))  # type: ignore[valid-type]


def annotate_texts(
    llm: LLM,
    items: Sequence[tuple[str, str]],
    specs: Sequence[PropertySpec],
    instruction: str | None = None,
    reference: str | None = None,
    batch_size: int = 15,
) -> dict[str, dict[str, Any]]:
    """Annotate `(id, text)` pairs. Returns {id: {property_name: value}}."""
    if not items or not specs:
        return {}

    result_model = _build_model(specs)
    system = ANNOTATION_SYSTEM
    if instruction:
        system += f"\nTask context:\n{instruction}\n"

    batches = [list(enumerate(items))[i : i + batch_size] for i in range(0, len(items), batch_size)]

    def _one(batch) -> dict[str, dict[str, Any]]:
        listing = "\n".join(f"[{idx}] {text}" for idx, (_id, text) in batch)
        user = ""
        if reference:
            user += f"<reference>\n{reference}\n</reference>\n\n"
        user += f"<facts>\n{listing}\n</facts>"

        parsed = llm.parse(system=system, user=user, output_format=result_model)
        by_index = {a.fact_index: a for a in parsed.annotations}  # type: ignore[attr-defined]
        out: dict[str, dict[str, Any]] = {}
        for idx, (fid, _text) in batch:
            ann = by_index.get(idx)
            if ann is None:
                continue
            out[fid] = {s.name: getattr(ann, s.name) for s in specs}
        return out

    merged: dict[str, dict[str, Any]] = {}
    for part in llm.map(_one, batches):
        merged.update(part)
    return merged


def annotate_store(
    llm: LLM,
    store: FactStore,
    specs: Sequence[PropertySpec],
    instruction: str | None = None,
    reference: str | None = None,
    batch_size: int = 15,
) -> FactStore:
    """Annotate every canonical fact in a store, writing into `fact.properties`."""
    items = [(fid, f.canonical_text) for fid, f in store.facts.items()]
    results = annotate_texts(
        llm, items, specs, instruction=instruction, reference=reference, batch_size=batch_size
    )
    for fid, props in results.items():
        store.facts[fid].properties.update(props)
    return store


# --- Ready-made specs for the studies this package was built for -------------

TRUTH = PropertySpec(
    name="truth",
    description=(
        "Is this fact supported by the reference material? "
        "'supported' if the reference entails it; 'contradicted' if the reference "
        "entails its negation; 'unsupported' if the reference is silent. "
        "Judge only against the reference, never against world knowledge."
    ),
    type="choice",
    choices=["supported", "contradicted", "unsupported"],
)

CRITICALITY = PropertySpec(
    name="critical",
    description=(
        "Would omitting this fact change how the question is interpreted, which "
        "trade-offs are weighed, or what the answer should be? True only if its "
        "omission could change the answer; general background and descriptive "
        "detail are False."
    ),
    type="boolean",
)

RELEVANCE = PropertySpec(
    name="relevant",
    description="Does this fact bear on the question at all, even weakly? Distractors are False.",
    type="boolean",
)

ABSTRACTION = PropertySpec(
    name="abstraction",
    description=(
        "How compressed is this fact? 1 = a specific measurement, name, date, or "
        "condition; 5 = a broad general claim with no concrete detail."
    ),
    type="number",
)

POLARITY_KIND = PropertySpec(
    name="finding_kind",
    description=(
        "'positive' if the fact asserts something is present or occurred; "
        "'negative' if it asserts something is absent or did not occur "
        "(a pertinent negative); 'neither' otherwise."
    ),
    type="choice",
    choices=["positive", "negative", "neither"],
)
