"""A stand-in for `LLM` so the pipeline is testable without network access.

It dispatches on the requested output model, which is exactly the contract the
real wrapper offers, so anything that works against this works against the API.
"""

from __future__ import annotations

import json
from typing import Callable, Sequence

from pydantic import BaseModel


class FakeLLM:
    def __init__(
        self,
        extractions: dict[str, list[dict]] | None = None,
        equivalent: set[frozenset[str]] | None = None,
        entails: set[tuple[str, str]] | None = None,
        contradicts: set[frozenset[str]] | None = None,
        annotator: Callable[[str], dict] | None = None,
    ):
        self.extractions = extractions or {}
        self.equivalent = equivalent or set()
        self.entails = entails or set()
        self.contradicts = contradicts or set()
        self.annotator = annotator or (lambda text: {})
        self.calls: list[str] = []

    def parse(self, *, system: str, user: str, output_format: type[BaseModel], **_kw):
        name = output_format.__name__
        self.calls.append(name)
        if name == "ExtractionResult":
            return self._extract(user, output_format)
        if name in ("AdjudicationResult", "LeanAdjudication"):
            return self._adjudicate(user, output_format)
        if name == "AnnotationResult":
            return self._annotate(user, output_format)
        raise AssertionError(f"FakeLLM has no handler for {name}")

    def map(self, fn, items: Sequence):
        return [fn(i) for i in items]

    # -- handlers -----------------------------------------------------------

    def _extract(self, user: str, model):
        body = user.split("<text>", 1)[1].rsplit("</text>", 1)[0].strip()
        return model.model_validate({"facts": self.extractions.get(body, [])})

    def _adjudicate(self, user: str, model):
        pairs = json.loads(user)
        judgements = []
        for p in pairs:
            a, b = p["A"]["text"], p["B"]["text"]
            if frozenset((a, b)) in self.equivalent:
                rel = "EQUIVALENT"
            elif (a, b) in self.entails:
                rel = "A_ENTAILS_B"
            elif (b, a) in self.entails:
                rel = "B_ENTAILS_A"
            elif frozenset((a, b)) in self.contradicts:
                rel = "CONTRADICTS"
            else:
                rel = "UNRELATED"
            judgements.append({"pair_id": p["pair_id"], "relation": rel, "confidence": 1.0})
        return model.model_validate({"judgements": judgements})

    def _annotate(self, user: str, model):
        listing = user.split("<facts>", 1)[1].rsplit("</facts>", 1)[0].strip()
        annotations = []
        for line in listing.splitlines():
            line = line.strip()
            if not line.startswith("["):
                continue
            idx, text = line[1:].split("]", 1)
            annotations.append({"fact_index": int(idx), **self.annotator(text.strip())})
        return model.model_validate({"annotations": annotations})
